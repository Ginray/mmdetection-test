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
import copy
import glob
import os.path

import torch


def module_hook_func(name, module, dump_dict, mode="forward"):
    def hook_function(module, inputs, outputs):
        dump_dict[mode]['inputs'] = inputs
        dump_dict[mode]['outputs'] = outputs

    return hook_function


def param_hook_func(name, dump_dict):
    def hook_function(grad):
        dump_dict['grads'][name] = grad

    return hook_function


def init_dump_dict():
    tmp_dict = {
        "name": "",
        "type": None,
        "config": {
            "model_dict": {},
            "args_kwargs": {},
            "state_dict": {},
            "thresholds": {},
            "loss_fn": {},
        },
        "forward": {
            "inputs": {},
            "outputs": {},
        },
        "backward": {
            "inputs": {},
            "outputs": {},
        },
        "grads": {},
    }
    return tmp_dict


def check_dump_dict(dump_dict, full_log=False):
    err_str = ''

    if not dump_dict['config']['model_dict']:
        err_str += 'config::model_dict is empty, please check.\n'

    if not dump_dict['forward']['inputs']:
        err_str += 'forward::inputs is empty, please check.\n'

    if isinstance(dump_dict['forward']['outputs'], dict) and (not dump_dict['forward']['outputs']):
        err_str += 'forward::outputs is empty, please check.\n'

    if not dump_dict['backward']['inputs']:
        err_str += 'backward::inputs is empty, please check.\n'

    if not dump_dict['backward']['outputs']:
        err_str += 'backward::outputs is empty, please check.\n'

    try:
        param_num = sum([len(list(m.parameters())) for m in (dump_dict['config']['model_dict']['_modules'].values())])
    except:
        param_num = 0

    if not dump_dict['backward']['outputs']:
        if param_num > 0:
            err_str += 'grads is empty, please check.\n'

    if err_str:
        if full_log:
            err_str = f"### {dump_dict['name']} check failed. \n" + err_str
        else:
            err_str = f"### {dump_dict['name']} check failed. \n"

    return err_str


def check_dump_dict_dir(dir_path, full_log=False):
    err_str = ""
    for file_path in glob.glob(os.path.join(dir_path, '*.pth')):
        dump_list = torch.load(file_path, map_location='cpu')
        err_str_tmp = check_dump_dict(dump_list, full_log)

        if err_str_tmp:
            err_str += err_str_tmp
    return err_str


def is_torch_module(module):
    return str(type(module)).replace("'", "").split(' ')[1].startswith('torch.')


def dump_hook(model, dump_name_list=[], auto=False):
    model.dump_dict = dump_dict = {}
    for m_name, module in model.named_modules():
        if m_name in dump_name_list or (auto and (not is_torch_module(module))):
            dump_dict[m_name] = init_dump_dict()
            dump_dict[m_name]['name'] = m_name
            dump_dict[m_name]['type'] = str(type(module))
            dump_dict[m_name]['config']['model_dict'] = copy.deepcopy(module.__dict__)
            module.register_forward_hook(
                module_hook_func('[forward]:' + m_name, module, dump_dict[m_name], mode='forward'))
            module.register_full_backward_hook(
                module_hook_func('[forward]:' + m_name, module, dump_dict[m_name], mode='backward'))

            for p_name, p in module.named_parameters():
                if p.requires_grad:
                    p.register_hook(param_hook_func(p_name.replace(f"{m_name}", ""), dump_dict[m_name]))

    return model


def dump_save(dump_dict, save_dir="./dump"):
    os.makedirs(save_dir, exist_ok=True)

    for k in dump_dict:
        err_str = check_dump_dict(dump_dict[k], True)
        if err_str:
            print(err_str)
        torch.save(dump_dict[k], os.path.join(save_dir, f"{k}.pth"))
    exit()


def diff_error(inputs, targets):
    diff_abs = (inputs - targets).abs()
    return diff_abs.max(), diff_abs.sum() / torch.count_nonzero(diff_abs)


def state_dict_remove_prefix(state_dict, prefix):
    new_state_dict = OrderedDict()
    for name, v in state_dict.items():
        if name.startwith(prefix):
            name = name.replace(prefix, '')
        new_state_dict[name] = v
    return new_state_dict


print('==================================>model')
print(model.modules)

dump_name_list = ["backbone.layer1", "backbone.layer2", "backbone.layer3", "backbone.layer4", "rpn_head.loss_cls"]
model = dump_hook(model, dump_name_list, auto=False)
