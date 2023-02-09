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
from mmdet.models.backbones.resnet import BasicBlock
from utils.acc_utils import COMPARISON_HOOK
from utils.base_utils import BaseUtil


class TestResnetTestCase:
    def setup_class(self):
        self.base_util = BaseUtil()
        self.block = BasicBlock(64, 64)

    def setup_method(self):
        self.base_util.clear_output_list()

    def teardown_method(self):
        COMPARISON_HOOK.rollback_threshold()

    @pytest.mark.acc
    def test_resnet_basic_block_acc(self):
        self.block.register_forward_hook(self.base_util.base_hook_forward_fn)
        self.block.register_backward_hook(self.base_util.base_hook_backward_fn)

        input = torch.rand(1, 64, 56, 56)
        self.base_util.run_and_compare_acc(self.block, 'Resnet', x=input)

    @pytest.mark.prof
    def test_resnet_basic_block_prof(self):
        # test BasicBlock structure and forward
        input = torch.rand(1, 64, 56, 56)
        prof_path = './data/prof_time_summary/backbones/resnet/resnet.csv'
        self.base_util.run_and_compare_prof(self.block, prof_path, time_threshold=0.4, x=input)