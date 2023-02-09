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


class LoggingLevel(IntEnum):
    debug = 0
    info = 1
    warn = 2
    error = 3


LOGGING_LEVEL = LoggingLevel.debug
DEVICE_INFO = '910B'


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
        if LOGGING_LEVEL <= LoggingLevel.debug:
            print('forward,use {0}: module={1}'.format(self._device, module))
        if self._device == 'npu':
            self.npu_output_list.append(output)
        elif self._device == 'cpu':
            self.cpu_output_list.append(output)
        else:
            raise ValueError

    def base_hook_backward_fn(self, module, grad_in, grad_out):
        if LOGGING_LEVEL <= LoggingLevel.debug:
            print('backward,use {0}: module={1}'.format(self._device, module))
        if self._device == 'npu':
            self.npu_grad_list.append(grad_out)
        elif self._device == 'cpu':
            self.cpu_grad_list.append(grad_out)
        else:
            raise ValueError

    def run_step(self, module, input):
        module = module.to(self._device)
        if isinstance(input, list):
            input_list = [each_input.to(self._device) for each_input in input]
            input = input_list
        elif isinstance(input, torch.Tensor):
            input = input.to(self._device)
        else:
            raise NotImplementedError

        output = module(input)
        if isinstance(output, tuple):
            output[0].mean().backward()
        elif isinstance(output, torch.Tensor):
            output.mean().backward()
        else:
            raise NotImplementedError

        return output

    def run_and_compare_acc(self, module, input, module_name=None):
        from utils.acc_utils import accuracy_comparison

        self.set_device('cpu')
        self.run_step(module, input)
        self.set_device('npu')
        self.run_step(module, input)

        if LOGGING_LEVEL <= LoggingLevel.info:
            print('==> start compare forward, module_name=', module_name)
        accuracy_comparison(self.npu_output_list, self.cpu_output_list)

        if LOGGING_LEVEL <= LoggingLevel.info:
            print('==> start compare backward, module_name=', module_name)
        accuracy_comparison(self.npu_grad_list, self.cpu_grad_list)

    def run_and_compare_prof(self, module, input, prof_path, time_threshold=0.1):
        from utils.prof_utils import save_time, compare_with_best_time

        time_start = time.time()
        self.set_device('npu')
        self.run_step(module, input)
        time_one_step = time.time() - time_start

        save_time(time_one_step, prof_path)
        compare_with_best_time(time_one_step, prof_path, time_threshold=time_threshold)
