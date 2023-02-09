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

import os

from utils.base_utils import LOGGING_LEVEL, DEVICE_INFO, LoggingLevel
import pandas as pd
from datetime import datetime
import pytz


def save_time(time_to_save, path):
    local_time = datetime.now(pytz.timezone('Asia/Shanghai'))
    time_format = "%Y-%m-%d %H:%M:%S"
    data = {'date': [local_time.strftime(time_format)], 'device_info': DEVICE_INFO, 'one_step_time(s)': [time_to_save]}
    data = pd.DataFrame(data)

    if os.path.exists(path):
        try:
            data_ori = pd.read_csv(path)
            data_to_save = data_ori.append(data)
            data_to_save.to_csv(path, index=False)
        except pd.errors.EmptyDataError:
            data.to_csv(path, index=False)
    else:
        data.to_csv(path, index=False)


def compare_with_best_time(time, path, time_threshold=0.1):
    assert os.path.exists(path)
    data_ori = pd.read_csv(path)
    best_time = data_ori['one_step_time(s)'].loc[data_ori['device_info'] == DEVICE_INFO].min()

    gap = (time - best_time) / best_time
    if LOGGING_LEVEL <= LoggingLevel.info:
        print('=====> best_time={0}, gap={1}, time_threshold={2}'.format(best_time, gap, time_threshold))

    assert gap <= time_threshold, \
        "compare_with_best_time, best_time={0}, gap={1}, time_threshold={2}".format(best_time, gap, time_threshold)
