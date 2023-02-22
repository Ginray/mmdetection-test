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

from mmdet.models.roi_heads.roi_extractors import SingleRoIExtractor
from utils.acc_utils import comparison_hook
from utils.base_utils import BaseUtil


class TestSingleRoIExtractorTestCase:
    def setup_class(self):
        self.base_util = BaseUtil()
        cfg = dict(
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32])

        self.roi_extractor = SingleRoIExtractor(**cfg)

    def setup_method(self):
        self.base_util.clear_output_list()

    def teardown_method(self):
        comparison_hook.rollback_threshold()

    @pytest.mark.acc
    @pytest.mark.skip(reason='roi_align currently not supported on npu, fix it before 2023/03/31.')
    def test_roi_extractor_acc(self):
        feats = (
            torch.rand((1, 256, 200, 336)),
            torch.rand((1, 256, 100, 168)),
            torch.rand((1, 256, 50, 84)),
            torch.rand((1, 256, 25, 42)),
        )
        rois = torch.tensor([[0.0000, 587.8285, 52.1405, 886.2484, 341.5644]])

        self.base_util.run_and_compare_acc(self.roi_extractor, 'RoIExtractor', feats, rois)

    @pytest.mark.prof
    @pytest.mark.skip(reason='roi_align currently not supported on npu, fix it before 2023/03/31.')
    def test_roi_extractor_prof(self):
        feats = (
            torch.rand((1, 256, 200, 336)),
            torch.rand((1, 256, 100, 168)),
            torch.rand((1, 256, 50, 84)),
            torch.rand((1, 256, 25, 42)),
        )
        rois = torch.tensor([[0.0000, 587.8285, 52.1405, 886.2484, 341.5644]])
        prof_path = './data/prof_time_summary/heads/roi_extractor/single_roi_extractor_prof.csv'
        self.base_util.run_and_compare_prof(self.roi_extractor, prof_path, 0.1, feats, rois)
