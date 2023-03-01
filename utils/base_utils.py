# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import time
import copy
from enum import IntEnum
import logging

device_info = 'UNKNOWN'


def set_device_info(device_info_input):
    global device_info
    device_info = device_info_input


class BaseUtil:
    def __init__(self):
        self._device = 'NPU'

        self.npu_output_list = []
        self.cpu_output_list = []
        self.npu_grad_list = []
        self.cpu_grad_list = []

    def set_device(self, device):
        self._device = device

    def clear_output_list(self):
        self.npu_output_list.clear()
        self.cpu_output_list.clear()
        self.npu_grad_list.clear()
        self.cpu_grad_list.clear()

    def base_hook_forward_fn(self, module, input, output):
        logging.debug('forward,use {0}: module={1}'.format(self._device, module))
        if self._device == 'npu':
            self.npu_output_list.append(output)
        elif self._device == 'cpu':
            self.cpu_output_list.append(output)
        else:
            raise ValueError

    def base_hook_backward_fn(self, module, grad_in, grad_out):
        logging.debug('backward,use {0}: module={1}'.format(self._device, module))
        if self._device == 'npu':
            self.npu_grad_list.append(grad_in)
        elif self._device == 'cpu':
            self.cpu_grad_list.append(grad_in)
        else:
            raise ValueError

    def set_value_to_device(self, value):
        if isinstance(value, torch.Tensor):
            value = value.to(self._device)
        elif isinstance(value, list):
            value_list = [self.set_value_to_device(each_value) for each_value in value]
            value = value_list
        elif isinstance(value, tuple):
            value_tuple = tuple(self.set_value_to_device(each_value) for each_value in value)
            value = value_tuple
        elif value is None:
            return value
        else:
            raise NotImplementedError('[set_value_to_device] {0} is currently not supported. '.format(type(value)))
        return value

    def do_real_data_backward(self, output, backward_output):
        backward_output = self.set_value_to_device(backward_output)
        if backward_output is None:
            logging.warning('when do_real_data_backward, backward output is None.')
            return
        if isinstance(output, tuple) or isinstance(output, list):
            assert len(output) == len(backward_output)
            for (each_output, each_backward_output) in zip(output, backward_output):
                if isinstance(each_output, torch.Tensor):
                    each_output.backward(each_backward_output, retain_graph=True)
                else:
                    self.do_real_data_backward(each_output, each_backward_output)
        elif isinstance(output, torch.Tensor):
            output.backward(backward_output)
        else:
            raise NotImplementedError(
                '[do_real_data_backward] output {0} is currently not supported. '.format(type(output)))

    def do_auto_backward(self, output):
        if isinstance(output, tuple) or isinstance(output, list):
            for each_output in output:
                self.do_auto_backward(each_output)
        elif isinstance(output, torch.Tensor):
            if output.dtype != torch.float:
                logging.warning('output.dtype is {0}, set to float.'.format(output.dtype))
                output = output.float()
            if output.requires_grad:
                output.mean().backward()
        else:
            raise NotImplementedError('[do_auto_backward] {0} is currently not supported. '.format(type(output)))
        return output

    def run_step(self, module, auto_backward, *input):
        input = self.set_value_to_device(input)

        output = module(*input)
        if auto_backward:
            self.do_auto_backward(output)
        return output

    def run_and_compare_with_cpu_acc(self, module, module_name, *input):
        from utils.acc_utils import accuracy_comparison
        cpu_module = module
        npu_module = copy.deepcopy(module).to('npu')

        cpu_module.register_forward_hook(self.base_hook_forward_fn)
        cpu_module.register_backward_hook(self.base_hook_backward_fn)
        npu_module.register_forward_hook(self.base_hook_forward_fn)
        npu_module.register_backward_hook(self.base_hook_backward_fn)

        self.set_device('cpu')
        logging.info('module {0} start executing on the cpu. '.format(module_name))
        self.run_step(cpu_module, True, *input)

        self.set_device('npu')
        logging.info('module {0} start executing on the npu. '.format(module_name))
        self.run_step(npu_module, True, *input)

        logging.info('start compare forward, module_name={0}'.format(module_name))
        accuracy_comparison(self.npu_output_list, self.cpu_output_list, module_name)

        logging.info('start compare backward, module_name={0}'.format(module_name))
        accuracy_comparison(self.npu_grad_list, self.cpu_grad_list, module_name)

    def run_and_compare_prof(self, module, prof_path, time_threshold, *input):
        from utils.prof_utils import save_time, compare_with_mean_time
        npu_module = copy.deepcopy(module).to('npu')

        time_start = time.time()
        self.set_device('npu')
        self.run_step(npu_module, True, *input)
        time_one_step = time.time() - time_start

        save_time(time_one_step, prof_path)
        compare_with_mean_time(time_one_step, prof_path, time_threshold=time_threshold)

    def run_and_compare_with_cpu_parameters(self, module, module_name=None, *input):
        from utils.acc_utils import accuracy_comparison

        cpu_module = module
        npu_module = copy.deepcopy(module).to('npu')

        cpu_module.register_forward_hook(self.base_hook_forward_fn)
        cpu_module.register_backward_hook(self.base_hook_backward_fn)
        npu_module.register_forward_hook(self.base_hook_forward_fn)
        npu_module.register_backward_hook(self.base_hook_backward_fn)

        self.set_device('cpu')
        logging.info('compare_parameters, module {0} start executing on the cpu. '.format(module_name))
        self.run_step(cpu_module, True, *input)

        self.set_device('npu')
        logging.info('compare_parameters, module {0} start executing on the npu. '.format(module_name))
        self.run_step(npu_module, True, *input)

        for (npu_para_name, npu_para), (cpu_para_name, cpu_para) in \
                zip(npu_module.named_parameters(), cpu_module.named_parameters()):
            logging.debug('compare_parameters, para_name={} '.format(npu_para_name))
            assert npu_para_name == cpu_para_name
            accuracy_comparison(npu_para, cpu_para)
            accuracy_comparison(npu_para.grad, cpu_para.grad)

    def set_params_from_config(self, module, input):
        for k in module.__dict__:
            if k in input['config']['model_dict']:
                module.__dict__[k] = input['config']['model_dict'][k]
        module.__dict__["_is_full_backward_hook"] = True
        return module

    def run_and_compare_with_real_data_acc(self, module, module_name, config):
        forward_input = config['forward']['inputs']
        backward_output = config['backward']['outputs']
        target_forward_output = config['forward']['outputs']
        target_backward_input = config['backward']['inputs']

        assert forward_input
        assert target_forward_output
        from utils.acc_utils import accuracy_comparison

        self.set_device('npu')
        npu_module = copy.deepcopy(module).to('npu')
        logging.info('[real_data] module {0} start executing on the npu. '.format(module_name))
        output_npu = self.run_step(npu_module, False, *forward_input)
        logging.info('start compare forward, module_name={0}'.format(module_name))
        accuracy_comparison(output_npu, target_forward_output, module_name)

        if backward_output:
            logging.info('start compare backward, module_name={0}'.format(module_name))
            npu_module.register_full_backward_hook(self.base_hook_backward_fn)
            self.do_real_data_backward(output_npu, backward_output)
            if not self.npu_grad_list and target_backward_input[0] is None:
                pass
            else:
                accuracy_comparison(self.npu_grad_list[0], target_backward_input, module_name)
        else:
            logging.info('compare with real_data, backward_output is empty.')
            npu_module.register_full_backward_hook(self.base_hook_backward_fn)
            loss_func = config['config']['loss_fn']
            if loss_func:
                final_output = loss_func(output_npu)
                final_output.backward()
            else:
                logging.warning('module {0} loss_func is empty.'.format(module_name))
        # compare parameters
        for name, p in npu_module.named_parameters():
            if p.grad is not None and config['grads'][name] is not None:
                accuracy_comparison(p.grad, config['grads'][name], name)
            else:
                logging.warning('tensor name {0} grads is None.'.format(name))
