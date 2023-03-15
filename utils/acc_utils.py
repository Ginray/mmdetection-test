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
import logging


def cos_comparison(outputs, outputs_expected, cos_threshold, module_name):
    assert isinstance(outputs, torch.Tensor)
    assert isinstance(outputs_expected, torch.Tensor)
    outputs, outputs_expected = outputs.float(), outputs_expected.float()
    if outputs.ndim > 1 and outputs_expected.ndim > 1:
        cos = torch.nn.CosineSimilarity(dim=1)
    else:
        cos = torch.nn.CosineSimilarity(dim=0)
    max_value_of_output = outputs_expected.max()
    if max_value_of_output < 1.0:
        scale = 1.0 / max_value_of_output
        outputs = outputs * scale
        outputs_expected = outputs_expected * scale
    cosine_tensor = cos(outputs, outputs_expected)
    cosine_similarity = cosine_tensor.min()
    if torch.isnan(cosine_similarity):
        return
    if cosine_similarity == 0:
        # outputs中存在一行完全为0，用1替换之后再进行相似度计算
        cosine_similarity = torch.where(cosine_tensor == 0, torch.ones_like(cosine_tensor), cosine_tensor).min()
    logging.info('===>module_name={0}, cosine_similarity={1}, cos_threshold={2}'.
                 format(module_name, cosine_similarity, cos_threshold))
    assert cosine_similarity >= cos_threshold, \
        "cosine_similarity={0}, cos_threshold={1}".format(cosine_similarity, cos_threshold)


def value_comparison(outputs, outputs_expected, value_threshold, module_name):
    assert isinstance(outputs, torch.Tensor)
    assert isinstance(outputs_expected, torch.Tensor)
    value_similarity = (outputs - outputs_expected).to(torch.float).abs().max()
    logging.info(
        '=====>module_name={0}, value_similarity={1}, value_threshold={2}'.
        format(module_name, value_similarity, value_threshold))
    assert value_similarity <= value_threshold, \
        "value_similarity={0}, value_threshold={1}".format(value_similarity, value_threshold)


class ComparisonHook(object):

    def __init__(self):
        self.comparison_fn_map = {}
        self.default_threshold = {'cos': 0.999,
                                  'value': 0.01}
        self.threshold = self.default_threshold.copy()
        self.threshold_module = {}

    def register_comparison_hook(self, name, comparison_fn, threshold=None):
        self.comparison_fn_map[name] = comparison_fn
        if name not in self.default_threshold.keys():
            assert threshold is not None, "Please enter threshold for hook{0}. ".format(name)
        if threshold is not None:
            self.threshold[name] = threshold

    def delete_comparison_hook(self, name):
        if name in self.comparison_fn_map.keys():
            del self.comparison_fn_map[name]
        if name in self.threshold.keys():
            del self.threshold[name]

    def update_threshold_all_module(self, name, threshold):
        """
        设置compare function的所有阈值，例如设置'cos'=0.98
        :param name: compare function的名字
        :param threshold: 希望设置的compare function的阈值
        """
        self.threshold[name] = threshold

    def update_threshold_for_module(self, compare_func_name, module_config):
        """
        为单独的module设置compare function阈值，例如设置'cos_fpn_convs.2.conv.weight'=0.98、‘cos_FPN’=0.998
        :param compare_func_name: compare function的名字
        :param module_config: dict; key为module名称，value为该module的阈值，例如{"fpn_convs.2.conv.weight": 0.998}
        """
        assert isinstance(module_config, dict)
        for module_name, threshold in module_config.items():
            self.threshold_module[compare_func_name + '_' + module_name] = threshold

    def rollback_threshold(self):
        self.threshold = self.default_threshold.copy()
        self.threshold_module = {}

    def reset_default_hook(self):
        self.comparison_fn_map = {}
        self.threshold = self.default_threshold.copy()
        self.threshold_module = {}
        comparison_hook.register_comparison_hook('cos', cos_comparison, self.default_threshold['cos'])
        comparison_hook.register_comparison_hook('value', value_comparison, self.default_threshold['value'])

    def compare(self, outputs, outputs_expected, module_name):
        for name, each_compare in self.comparison_fn_map.items():
            threshold = self.threshold[name]
            if module_name is not None and name + '_' + module_name in self.threshold_module.keys():
                threshold = self.threshold_module[name + '_' + module_name]
            if outputs is None and outputs_expected is None:
                logging.warning('when compare {0}-{1} , outputs is None.'.format(module_name, name))
                continue
            each_compare(outputs, outputs_expected, threshold, module_name)
        logging.info(" ")


comparison_hook = ComparisonHook()
comparison_hook.reset_default_hook()


def accuracy_comparison(outputs, outputs_expected, module_name=None):
    assert type(outputs) == type(outputs_expected)
    if outputs is None and outputs_expected is None:
        return
    if isinstance(outputs, torch.Tensor):
        comparison_hook.compare(outputs.cpu(), outputs_expected.cpu(), module_name)
    elif isinstance(outputs, list) or isinstance(outputs, tuple):
        assert len(outputs) == len(outputs_expected)
        for each_output, each_output_expected in zip(outputs, outputs_expected):
            accuracy_comparison(each_output, each_output_expected, module_name)
    else:
        raise NotImplementedError(
            'Only supports Tensor/tuple and list of Tensor, type of outputs is {0}'.format(type(outputs)))
