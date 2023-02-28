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

from mmdet.models.dense_heads import RetinaHead
from utils.acc_utils import comparison_hook
from utils.base_utils import BaseUtil


class TestRetinaHeadTestCase:
    def setup_class(self):
        self.base_util = BaseUtil()
        self.retina_head = RetinaHead(2, 7)

    def setup_method(self):
        self.base_util.clear_output_list()

    def teardown_method(self):
        comparison_hook.rollback_threshold()

    @pytest.mark.acc
    def test_retina_head_acc(self):
        x = torch.rand(2, 2, 7, 3, 3)
        comparison_hook.update_threshold_all_module('value', 1e-4)
        self.base_util.run_and_compare_with_cpu_acc(self.retina_head, 'RetinaHead', x)

    @pytest.mark.acc
    def test_retina_head_acc_real_data(self):
        retina_head = RetinaHead(11, 7)
        pt_path = './data/pt_dump/heads/retina_head/retina_head.pth'
        config = torch.load(pt_path, map_location=torch.device('cpu'))
        retina_head = self.base_util.set_params_from_config(retina_head, config)

        if config['config']['thresholds']:
            comparison_hook.update_threshold_all_module('value', float(config['config']['thresholds']))
        self.base_util.run_and_compare_with_real_data_acc(retina_head, 'RetinaHead', config)

    @pytest.mark.prof
    def test_retina_head_prof(self):
        x = torch.rand(2, 2, 7, 3, 3)
        prof_path = './data/prof_time_summary/heads/retina_head/retina_head_prof.csv'
        self.base_util.run_and_compare_prof(self.retina_head, prof_path, 0.4, x)
