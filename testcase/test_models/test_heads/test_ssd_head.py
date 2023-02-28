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

from mmdet.models.dense_heads import SSDHead
from utils.acc_utils import comparison_hook
from utils.base_utils import BaseUtil


class TestSSDHeadTestCase:
    def setup_class(self):
        self.base_util = BaseUtil()
        self.ssd_head = SSDHead(
            in_channels=(256, 256),
            num_classes=4,
            anchor_generator=dict(
                type='SSDAnchorGenerator',
                scale_major=False,
                input_size=300,
                basesize_ratio_range=(0.15, 0.9),
                strides=[8, 16, 32, 64, 100, 300],
                ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]]),
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2]))

    def setup_method(self):
        self.base_util.clear_output_list()

    def teardown_method(self):
        comparison_hook.rollback_threshold()

    @pytest.mark.acc
    def test_ssd_head_acc(self):
        feats = (
            torch.rand((1, 256, 20, 33)),
            torch.rand((1, 256, 10, 16)),
            torch.rand((1, 256, 5, 8)),
            torch.rand((1, 256, 2, 4)),
        )

        self.base_util.run_and_compare_with_cpu_acc(self.ssd_head, 'SSDHead', feats)

    @pytest.mark.acc
    def test_ssd_head_acc_real_data(self):
        ssd_head = SSDHead(
            in_channels=(512, 1024, 512, 256, 256, 256),
            num_classes=80,
            anchor_generator=dict(
                type='SSDAnchorGenerator',
                scale_major=False,
                input_size=300,
                basesize_ratio_range=(0.15, 0.9),
                strides=[8, 16, 32, 64, 100, 300],
                ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]]),
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2]))

        pt_path = './data/pt_dump/heads/ssd_head/ssd_head.pth'
        config = torch.load(pt_path, map_location=torch.device('cpu'))
        ssd_head = self.base_util.set_params_from_config(ssd_head, config)

        if config['config']['thresholds']:
            comparison_hook.update_threshold_all_module('value', float(config['config']['thresholds']))
        self.base_util.run_and_compare_with_real_data_acc(ssd_head, 'SSDHead', config)

    @pytest.mark.prof
    def test_ssd_head_prof(self):
        feats = (
            torch.rand((1, 256, 20, 33)),
            torch.rand((1, 256, 10, 16)),
            torch.rand((1, 256, 5, 8)),
            torch.rand((1, 256, 2, 4)),
        )
        prof_path = './data/prof_time_summary/heads/ssd_head/single_roi_extractor_prof.csv'
        self.base_util.run_and_compare_prof(self.ssd_head, prof_path, 0.1, feats)
