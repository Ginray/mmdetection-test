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

import pytest
import torch
import logging
from mmdet.models.losses import CrossEntropyLoss
from utils.acc_utils import comparison_hook
from utils.base_utils import BaseUtil


def cos_comparison_dim_0(outputs, outputs_expected, cos_threshold):
    assert isinstance(outputs, torch.Tensor)
    assert isinstance(outputs_expected, torch.Tensor)
    cos = torch.nn.CosineSimilarity(dim=0)
    max_value_of_output = outputs_expected.max()
    if max_value_of_output < 1.0:
        scale = 1.0 / max_value_of_output
        outputs = outputs * scale
        outputs_expected = outputs_expected * scale
    cosine_similarity = cos(outputs, outputs_expected).min()
    logging.info('==> cosine_similarity={0}, cos_threshold={1}'.format(cosine_similarity, cos_threshold))
    assert cosine_similarity >= cos_threshold, \
        "cosine_similarity={0}, cos_threshold={1}".format(cosine_similarity, cos_threshold)


class TestCrossEntropyLossTestCase:
    def setup_class(self):
        self.base_util = BaseUtil()

        self.loss = CrossEntropyLoss(
            use_sigmoid=False,
            use_mask=False,
            reduction='mean',
            class_weight=None,
            ignore_index=None,
            loss_weight=1.0,
            avg_non_ignore=False)

    def setup_method(self):
        self.base_util.clear_output_list()

    def teardown_method(self):
        comparison_hook.reset_default_hook()

    @pytest.mark.acc
    def test_cross_entropy_acc(self):
        self.loss.register_forward_hook(self.base_util.base_hook_forward_fn)
        self.loss.register_backward_hook(self.base_util.base_hook_backward_fn)

        cls_score = torch.rand([15, 10])
        label = torch.tensor([1, 0, 4, 8, 4, 7, 9, 3, 2, 5, 3, 6, 2, 7, 9])

        comparison_hook.delete_comparison_hook('cos')
        comparison_hook.register_comparison_hook('cos_dim_0', cos_comparison_dim_0, threshold=0.999)
        self.base_util.run_and_compare_acc(self.loss, 'cross_entropy', cls_score=cls_score, label=label)

    @pytest.mark.prof
    def test_cross_entropy_prof(self):
        cls_score = torch.rand([15, 10])
        label = torch.tensor([1, 0, 4, 8, 4, 7, 9, 3, 2, 5, 3, 6, 2, 7, 9])
        prof_path = './data/prof_time_summary/others/cross_entropy/cross_entropy_prof.csv'
        self.base_util.run_and_compare_prof(self.loss, prof_path, time_threshold=0.1, cls_score=cls_score, label=label)