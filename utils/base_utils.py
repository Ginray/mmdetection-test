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
            self.npu_grad_list.append(grad_out)
        elif self._device == 'cpu':
            self.cpu_grad_list.append(grad_out)
        else:
            raise ValueError

    def set_input_to_device(self, **input):
        for key, value in input.items():
            if isinstance(value, torch.Tensor):
                value = value.to(self._device)
            elif isinstance(value, list):
                value_list = [each_value.to(self._device) for each_value in value]
                value = value_list
            elif isinstance(value, tuple):
                value_tuple = tuple(each_value.to(self._device) for each_value in value)
                value = value_tuple
            else:
                raise NotImplementedError('[set_input_to_device] {0} is currently not supported. '.format(type(value)))
            input[key] = value
        return input

    def run_step(self, module, **input):
        input = self.set_input_to_device(**input)
        output = module(**input)
        if isinstance(output, tuple):
            if not output[0].requires_grad:
                logging.warning('[run_step] Warning, output[0].requires_grad is False, set to True.')
                output[0].requires_grad_(True)
            output[0].mean().backward()
        elif isinstance(output, torch.Tensor):
            if output.dtype != torch.int:
                logging.warning('output.dtype is {0}, set to float.'.format(output.dtype))
                output = output.float()
            if not output.requires_grad:
                logging.warning('output.requires_grad is False, set to True.')
                output.requires_grad_(True)
            output.mean().backward()
        else:
            raise NotImplementedError('[run_step] {0} is currently not supported. '.format(type(output)))
        return output

    def run_and_compare_acc(self, module, module_name=None, **input):
        from utils.acc_utils import accuracy_comparison

        self.set_device('cpu')
        module = module.to(self._device)
        logging.info('module {0} start executing on the cpu. '.format(module_name))
        self.run_step(module, **input)

        self.set_device('npu')
        module = module.to(self._device)
        logging.info('module {0} start executing on the npu. '.format(module_name))
        self.run_step(module, **input)

        logging.info('start compare forward, module_name={0}'.format(module_name))
        accuracy_comparison(self.npu_output_list, self.cpu_output_list)

        logging.info('start compare backward, module_name={0}'.format(module_name))
        accuracy_comparison(self.npu_grad_list, self.cpu_grad_list)

    def run_and_compare_prof(self, module, prof_path, time_threshold=0.1, **input):
        from utils.prof_utils import save_time, compare_with_best_time

        time_start = time.time()
        self.set_device('npu')
        module = module.to(self._device)
        self.run_step(module, **input)
        time_one_step = time.time() - time_start

        save_time(time_one_step, prof_path)
        compare_with_best_time(time_one_step, prof_path, time_threshold=time_threshold)

    def run_and_compare_parameters(self, module, module_name=None, **input):
        from utils.acc_utils import accuracy_comparison
        import copy

        cpu_module = module.to('cpu')
        npu_module = copy.deepcopy(module).to('npu')

        self.set_device('cpu')
        logging.info('compare_parameters, module {0} start executing on the cpu. '.format(module_name))
        self.run_step(cpu_module, **input)

        self.set_device('npu')
        logging.info('compare_parameters, module {0} start executing on the npu. '.format(module_name))
        self.run_step(npu_module, **input)

        for (npu_para_name, npu_para), (cpu_para_name, cpu_para) in \
                zip(npu_module.named_parameters(), cpu_module.named_parameters()):
            logging.debug('compare_parameters, para_name={} '.format(npu_para_name))
            assert npu_para_name == cpu_para_name
            accuracy_comparison(npu_para, cpu_para)
            accuracy_comparison(npu_para.grad, cpu_para.grad)
