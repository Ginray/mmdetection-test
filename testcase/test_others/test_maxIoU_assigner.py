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
from mmdet.core.bbox import MaxIoUAssigner
from utils.acc_utils import comparison_hook
from utils.base_utils import BaseUtil
from torch.nn import Module


class MaxIoUAssignerModule(Module):
    def __init__(self):
        super().__init__()
        self.assigner = MaxIoUAssigner(
            pos_iou_thr=0.5,
            neg_iou_thr=0.5
        )

    def forward(self, bboxes, gt_bboxes):
        assign_result = self.assigner.assign(bboxes=bboxes, gt_bboxes=gt_bboxes)
        return assign_result.gt_inds


class TestMaxIoUAssignerTestCase:
    def setup_class(self):
        self.base_util = BaseUtil()
        self.maxIoU_assigner = MaxIoUAssignerModule()

    def setup_method(self):
        self.base_util.clear_output_list()

    def teardown_method(self):
        comparison_hook.reset_default_hook()

    @pytest.mark.acc
    def test_maxIoU_assigner_acc(self):
        bboxes = torch.Tensor([[0, 0, 10, 10], [10, 10, 20, 20],
                               [3, 3, 6, 6], [2, 2, 3, 3]])
        gt_bboxes = torch.Tensor([[0, 0, 10, 9], [10, 10, 19, 19],
                                  [10, 10, 15, 15], [3, 3, 4, 4]])

        self.base_util.run_and_compare_acc(self.maxIoU_assigner, 'maxIoU_assigner', bboxes=bboxes, gt_bboxes=gt_bboxes)

    @pytest.mark.prof
    def test_maxIoU_assigner_prof(self):
        bboxes = torch.Tensor([[0, 0, 10, 10], [10, 10, 20, 20],
                               [3, 3, 6, 6], [2, 2, 3, 3]])
        gt_bboxes = torch.Tensor([[0, 0, 10, 9], [10, 10, 19, 19],
                                  [10, 10, 15, 15], [3, 3, 4, 4]])
        prof_path = './data/prof_time_summary/others/maxIoU_assigner/maxIoU_assigner_prof.csv'

        self.base_util.run_and_compare_prof(self.maxIoU_assigner, prof_path, time_threshold=0.1, bboxes=bboxes,
                                            gt_bboxes=gt_bboxes)
