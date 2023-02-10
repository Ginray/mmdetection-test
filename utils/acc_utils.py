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


class ComparisonHook(object):

    def __init__(self):
        self.comparison_fn_map = {}
        self.default_threshold = {'cos': 0.999,
                                  'value': 0.01}
        self.threshold = self.default_threshold.copy()

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

    def compare(self, outputs, outputs_expected):
        for name, each_compare in self.comparison_fn_map.items():
            each_compare(outputs, outputs_expected, self.threshold[name])
        logging.info(" ")

    def update_threshold(self, name, threshold):
        self.threshold[name] = threshold

    def rollback_threshold(self):
        self.threshold = self.default_threshold.copy()


comparison_hook = ComparisonHook()


def cos_comparison(outputs, outputs_expected, cos_threshold):
    assert isinstance(outputs, torch.Tensor)
    assert isinstance(outputs_expected, torch.Tensor)
    cos = torch.nn.CosineSimilarity(dim=1)
    max_value_of_output = outputs_expected.max()
    if max_value_of_output < 1.0:
        scale = 1.0 / max_value_of_output
        outputs = outputs * scale
        outputs_expected = outputs_expected * scale
    cosine_similarity = cos(outputs, outputs_expected).min()
    logging.info('==> cosine_similarity={0}, cos_threshold={1}'.format(cosine_similarity, cos_threshold))
    assert cosine_similarity >= cos_threshold, \
        "cosine_similarity={0}, cos_threshold={1}".format(cosine_similarity, cos_threshold)


def value_comparison(outputs, outputs_expected, value_threshold):
    assert isinstance(outputs, torch.Tensor)
    assert isinstance(outputs_expected, torch.Tensor)
    value_similarity = (outputs - outputs_expected).abs().max()
    logging.info('====> value_similarity={0}, value_threshold={1}'.format(value_similarity, value_threshold))
    assert value_similarity <= value_threshold, \
        "value_similarity={0}, value_threshold={1}".format(value_similarity, value_threshold)


comparison_hook.register_comparison_hook('cos', cos_comparison)
comparison_hook.register_comparison_hook('value', value_comparison)


def accuracy_comparison(outputs, outputs_expected):
    if isinstance(outputs, torch.Tensor) and isinstance(outputs_expected, torch.Tensor):
        outputs, outputs_expected = [outputs], [outputs_expected]
    if isinstance(outputs, list) and isinstance(outputs_expected, list):
        assert len(outputs) == len(outputs_expected)
        for each_val in zip(outputs, outputs_expected):
            each_output, each_output_expected = each_val[0], each_val[1]
            if isinstance(each_output, tuple) and isinstance(each_output_expected, tuple):
                # used to compare gradients.
                for each_tuple_val in zip(each_output, each_output_expected):
                    comparison_hook.compare(each_tuple_val[0].cpu(), each_tuple_val[1].cpu())
            else:
                comparison_hook.compare(each_output.cpu(), each_output_expected.cpu())
    else:
        raise NotImplementedError('Only supports Tensor and list of Tensor.')
