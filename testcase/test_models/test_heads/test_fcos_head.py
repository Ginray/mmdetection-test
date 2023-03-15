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

from mmdet.models.dense_heads import FCOSHead
from utils.acc_utils import comparison_hook
from utils.base_utils import BaseUtil


class TestFCOSHeadTestCase:
    def setup_class(self):
        self.base_util = BaseUtil()
        self.fcos_head = FCOSHead(3, 16)

    def setup_method(self):
        self.base_util.clear_output_list()

    def teardown_method(self):
        comparison_hook.rollback_threshold()

    @pytest.mark.acc
    def test_fcos_head_acc(self):
        x = [torch.rand(2, 16, 3, 3) for _ in range(3)]

        self.base_util.run_and_compare_with_cpu_acc(self.fcos_head, 'FCOSHead', x)

    @pytest.mark.prof
    def test_fcos_head_prof(self):
        x = [torch.rand(2, 16, 3, 3) for _ in range(3)]
        prof_path = './data/prof_time_summary/heads/fcos_head/fcos_head_prof.csv'
        self.base_util.run_and_compare_prof(self.fcos_head, prof_path, 0.3, x)
