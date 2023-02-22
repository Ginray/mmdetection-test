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


def hook_func_forward(name, module, dump_dict):
    def hook_function(module, inputs, outputs):
        print(f'{name} ######inputs', [i if i is not None else i for i in inputs])
        print(f'{name} ######outputs', [i if i is not None else i for i in outputs])

        import copy
        print(module.__dict__.keys())
        dump_dict['config']['init_kwargs'] = copy.deepcopy(module.__dict__)

        pop_list = []
        for k in dump_dict['config']['init_kwargs']:
            if k in ['_backward_hooks', '_forward_hooks']:
                pop_list.append(k)

        for k in pop_list:
            dump_dict['config']['init_kwargs'].pop(k)

        dump_dict['forward']['inputs'] = inputs
        dump_dict['forward']['outputs'] = outputs

    return hook_function


def hook_func_backward(name, module, dump_dict):
    def hook_function(module, inputs, outputs):
        print(f'{name} ######backward inputs', [i if i is not None else i for i in inputs])
        print(f'{name} ######backward outputs', [i if i is not None else i for i in outputs])

        dump_dict['backward']['inputs'] = inputs
        dump_dict['backward']['outputs'] = outputs

    return hook_function


def param_hook_func(name, dump_dict):
    def hook_function(grad):
        print(f'{name} ######param_hook_func')
        dump_dict['named_parameters'][name] = grad
        torch.save(dump_dict, 'dump.pth')
        if name == 'lateral_convs.0.conv.bias':
            import time
            time.sleep(2)
            exit(0)

    return hook_function


def init_dump_dict():
    tmp_dict = {
        "config": {
            "init_kwargs": {},
            "others": {},
        },
        "forward": {
            "inputs": {},
            "outputs": {},
        },
        "backward": {
            "inputs": {},
            "outputs": {},
        },
        "named_parameters": {},
    }
    return tmp_dict


dump_list = init_dump_dict()

for name, module in model.named_modules():
    print('=========>named_modules, name = ', name)
    if name == "neck":
        module.register_forward_hook(hook_func_forward('[forward]:' + name, module, dump_list))
        module.register_full_backward_hook(hook_func_backward('[backward]:' + name, module, dump_list))

for name, p in model.named_parameters():
    print('=========>named_parameters, name = ', name)
    if "neck" in name:
        p.register_hook(param_hook_func(name.replace('neck.', ""), dump_list))
