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
from utils.acc_utils import comparison_hook, cos_comparison_dim_0
from utils.base_utils import BaseUtil


class TestResnetTestCase:
    def setup_class(self):
        self.base_util = BaseUtil()
        self.block = BasicBlock(64, 64)

    def setup_method(self):
        self.base_util.clear_output_list()

    def teardown_method(self):
        comparison_hook.reset_default_hook()

    @pytest.mark.acc
    def test_resnet_basic_block_parameters(self):
        comparison_hook.update_threshold('value', 0.02)
        comparison_hook.delete_comparison_hook('cos')
        comparison_hook.register_comparison_hook('cos_dim_0', cos_comparison_dim_0, threshold=0.99)

        self.block = BasicBlock(3, 3)
        input = torch.load('./data/pt_dump/backbones/resnet/Resnet_input.pt', map_location=torch.device('cpu'))
        self.base_util.run_and_compare_parameters(self.block, 'Resnet', x=input)

    @pytest.mark.acc
    def test_resnet_basic_block_acc(self):
        self.block.register_forward_hook(self.base_util.base_hook_forward_fn)
        self.block.register_backward_hook(self.base_util.base_hook_backward_fn)

        self.block = BasicBlock(64, 64)
        input = torch.rand(1, 64, 56, 56)
        self.base_util.run_and_compare_acc(self.block, 'Resnet', x=input)

    @pytest.mark.acc
    def test_resnet_basic_block_acc_2(self):
        self.block.register_forward_hook(self.base_util.base_hook_forward_fn)
        self.block.register_backward_hook(self.base_util.base_hook_backward_fn)

        # todo 所有路径统一到配置文件中
        self.block = BasicBlock(3, 3)
        input = torch.load('./data/pt_dump/backbones/resnet/Resnet_input.pt', map_location=torch.device('cpu'))
        self.base_util.run_and_compare_acc(self.block, 'Resnet', x=input)

    @pytest.mark.prof
    def test_resnet_basic_block_prof(self):
        # test BasicBlock structure and forward
        self.block = BasicBlock(64, 64)
        input = torch.rand(1, 64, 56, 56)
        prof_path = './data/prof_time_summary/backbones/resnet/resnet_prof.csv'
        self.base_util.run_and_compare_prof(self.block, prof_path, time_threshold=0.4, x=input)

    @pytest.mark.prof
    def test_resnet_basic_block_prof_2(self):
        self.block = BasicBlock(3, 3)
        input = torch.load('./data/pt_dump/backbones/resnet/Resnet_input.pt', map_location=torch.device('cpu'))
        prof_path = './data/prof_time_summary/backbones/resnet/resnet_prof_dump.csv'
        self.base_util.run_and_compare_prof(self.block, prof_path, time_threshold=0.4, x=input)
