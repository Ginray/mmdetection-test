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
from mmdet.models.roi_heads.bbox_heads import ConvFCBBoxHead
from utils.acc_utils import comparison_hook
from utils.base_utils import BaseUtil


class TestShared2FCBBoxHeadTestCase:
    def setup_class(self):
        self.base_util = BaseUtil()
        '''
            Shared2FCBBoxHead继承自ConvFCBBoxHead类,固定了ConvFCBBoxHead的入参，
            因此进行ConvFCBBoxHead的测试，主要参数如下:
            (1) num_shared_convs: 共享卷积层数量;
            (2) num_shared_fcs: 共享全连接层数量;
            (3) num_cls_convs: 分类卷积层数量;
            (4) num_cls_fcs: 分类全连接层数量;
            (5) num_reg_convs: 回归卷积层的数量;
            (6) num_reg_fcs: 回归全连接层的数量;
            (7) fc_out_channels: 全连接层后输出层的数量,默认值为1024.
        '''
        cfg = dict(num_shared_convs=2,
                   num_shared_fcs=2,
                   num_cls_convs=0,
                   num_cls_fcs=0,
                   num_reg_convs=0,
                   num_reg_fcs=0,
                   )

        self.box_head = ConvFCBBoxHead(**cfg)

    def setup_method(self):
        self.base_util.clear_output_list()

    def teardown_method(self):
        comparison_hook.rollback_threshold()

    @pytest.mark.acc
    def test_shared2FCB_box_acc(self):
        input = torch.rand(1, 256, 7, 7)
        self.base_util.run_and_compare_acc(self.box_head, 'Shared2FCB', x=input)

    @pytest.mark.prof
    def test_shared2FCB_box_prof(self):
        input = torch.rand(1, 256, 7, 7)
        prof_path = './data/prof_time_summary/heads/shared2FCB_box/shared2FCB_box_prof.csv'
        self.base_util.run_and_compare_prof(self.box_head, prof_path, time_threshold=0.1, x=input)
