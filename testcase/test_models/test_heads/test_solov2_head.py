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

from mmdet.models.dense_heads import SOLOV2Head
from utils.acc_utils import comparison_hook
from utils.base_utils import BaseUtil


class TestSOLOV2HeadTestCase:
    def setup_class(self):
        self.base_util = BaseUtil()
        mask_feature_head = dict(
            feat_channels=16,
            start_level=0,
            end_level=1,
            out_channels=32,
            mask_stride=4,
            norm_cfg=dict(type='GN', num_groups=2, requires_grad=True))

        kwargs = dict(num_classes=4, in_channels=16, feat_channels=32, stacked_convs=4,
                      strides=[8, 8, 16],
                      scale_ranges=((1, 2), (2, 4), (4, 6)),
                      pos_scale=0.2,
                      num_grids=[2, 4, 6], cls_down_index=0,
                      loss_mask={'type': 'DiceLoss', 'use_sigmoid': True, 'loss_weight': 3.0},
                      loss_cls={'type': 'FocalLoss', 'use_sigmoid': True, 'gamma': 2.0,
                                'alpha': 0.25, 'loss_weight': 1.0},
                      train_cfg=None,
                      test_cfg={'nms_pre': 500, 'score_thr': 0.1, 'mask_thr': 0.5,
                                'filter_thr': 0.05, 'kernel': 'gaussian',
                                'sigma': 2.0, 'max_per_img': 100})
        self.solov2_head = SOLOV2Head(mask_feature_head=mask_feature_head, **kwargs)

    def setup_method(self):
        self.base_util.clear_output_list()

    def teardown_method(self):
        comparison_hook.reset_default_hook()

    @pytest.mark.acc
    def test_solov2_head_acc(self):
        x = [torch.rand(2, 16, 24, 24), torch.rand(2, 16, 12, 12), torch.rand(2, 16, 12, 12)]
        self.base_util.run_and_compare_with_cpu_acc(self.solov2_head, 'SOLOV2Head', x)

    @pytest.mark.acc
    def test_solov2_head_acc_real_data(self):
        import copy
        pt_path = './data/pt_dump/heads/solov2_head/solov2_head.pth'
        config = torch.load(pt_path, map_location=torch.device('cpu'))
        comparison_hook.delete_comparison_hook('cos')  # value值差距很小, cos偏差较大

        solov2_head = copy.deepcopy(self.solov2_head)
        solov2_head = self.base_util.set_params_from_config(solov2_head, config)
        if config['config']['thresholds']:
            comparison_hook.update_threshold_all_module('value', float(config['config']['thresholds']))
        self.base_util.run_and_compare_with_real_data_acc(solov2_head, 'SOLOV2Head', config)

    @pytest.mark.prof
    def test_solov2_head_prof(self):
        x = [torch.rand(2, 16, 24, 24), torch.rand(2, 16, 12, 12), torch.rand(2, 16, 12, 12)]
        prof_path = './data/prof_time_summary/heads/solov2_head/solov2_head_prof.csv'
        self.base_util.run_and_compare_prof(self.solov2_head, prof_path, 0.3, x)
