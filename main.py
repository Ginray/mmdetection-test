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

import random

import numpy as np
import pytest
import argparse
import torch
import torch_npu
from utils.base_utils import set_device_info


def set_seed(seed=0):
    torch.npu.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_parser():
    """Construct the parser."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--scope",
        type=str,
        default="all",
        choices=["acc", "prof", "all", "single"],
        help="Test the case marked prof/acc.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="UNKNOWN",
        choices=["UNKNOWN", "910B", "910ProB", "910A"],
        help="The device name of NPU.",
    )
    parser.add_argument(
        "--case_id",
        type=int,
        default=0,
        help="ID of the test case, only used when the scope is single.",
    )

    return parser


test_cases = [
    "./testcase/test_models/test_backbones/test_resnet.py",
    "./testcase/test_models/test_necks/test_fpn.py",
    "./testcase/test_models/test_heads/test_single_roi_extractor.py",
    "./testcase/test_models/test_heads/test_shared2FCB_box_head.py",
    "./testcase/test_models/test_heads/test_yolo_x_head.py",
    "./testcase/test_models/test_heads/test_yolo_v3_head.py",
    "./testcase/test_models/test_heads/test_retinanet_head.py",
    "./testcase/test_models/test_heads/test_ssd_head.py",
    "./testcase/test_others/test_cross_entropy_loss.py",
    "./testcase/test_others/test_maxIoU_assigner.py"
]


def main():
    parser = get_parser()
    args = parser.parse_args()
    set_device_info(args.device)
    set_seed()
    if args.scope == "all":
        pytest.main(["-s", "./testcase/"])
    elif args.scope == "acc":
        pytest.main(["-m acc ", "-s", "./testcase/"])
    elif args.scope == "prof":
        pytest.main(["-m prof ", "-s", "./testcase/"])
    elif args.scope == "single":
        pytest.main(["-s", test_cases[args.case_id]])


if __name__ == "__main__":
    main()
