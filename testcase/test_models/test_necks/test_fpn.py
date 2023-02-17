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
from mmdet.models.necks import FPN
from utils.acc_utils import comparison_hook
from utils.base_utils import BaseUtil


class TestFPNTestCase:
    def setup_class(self):
        self.base_util = BaseUtil()

        s = 64
        self.in_channels = [8, 16, 32, 64]
        self.feat_sizes = [s // 2 ** i for i in range(4)]  # [64, 32, 16, 8]
        out_channels = 8
        self.fpn_model = FPN(
            in_channels=self.in_channels,
            out_channels=out_channels,
            start_level=1,
            add_extra_convs=True,
            num_outs=5)

    def setup_method(self):
        self.base_util.clear_output_list()

    def teardown_method(self):
        comparison_hook.rollback_threshold()

    @pytest.mark.acc
    def test_fpn_acc(self):
        feats = [
            torch.rand(1, self.in_channels[i], self.feat_sizes[i], self.feat_sizes[i])
            for i in range(len(self.in_channels))
        ]

        comparison_hook.update_threshold('value', 0.015)
        self.base_util.run_and_compare_acc(self.fpn_model, 'FPN', inputs=feats)

    @pytest.mark.acc
    def test_resnet_basic_block_acc_real(self):
        self.fpn_model = FPN(
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            num_outs=5,
            start_level=0,
            end_level=-1)
        comparison_hook.update_threshold('value', 0.03)
        comparison_hook.delete_comparison_hook('cos')

        pt_path = './data/pt_dump/necks/fpn/'
        forward_input = torch.load(pt_path + 'fpn_forward_input.pt', map_location=torch.device('cpu'))
        backward_output = torch.load(pt_path + 'fpn_backward_output.pt', map_location=torch.device('cpu'))

        self.base_util.run_and_compare_real_data(self.fpn_model, 'FPN', forward_input=dict(inputs=forward_input),
                                                 backward_output=backward_output)

    @pytest.mark.prof
    def test_fpn_prof(self):
        feats = [
            torch.rand(1, self.in_channels[i], self.feat_sizes[i], self.feat_sizes[i])
            for i in range(len(self.in_channels))
        ]
        prof_path = './data/prof_time_summary/necks/fpn/fpn_prof.csv'
        self.base_util.run_and_compare_prof(self.fpn_model, prof_path, time_threshold=0.1, inputs=feats)
