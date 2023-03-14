# MMDetection-NPU测试用例

## 说明：

本仓库主要包含[MMDetection](https://mmdetection.readthedocs.io/zh_CN/latest/)相关的测试用例，用于验证MMDetection中的模型在Ascend
NPU上的精度和性能。

## 使用方式:

+ 准备环境：

  在执行测试用例前，请按照以下指导文档正确安装mmcv和mmdetection:

  mmcv： https://mmcv.readthedocs.io/zh_CN/latest/get_started/build.html#id6

  mmdetection： https://mmdetection.readthedocs.io/zh_CN/latest/get_started.html#mmdetection

  请均按照文档命令使用源码安装。

  注意：安装mmcv框架时，如果遇到拉取代码的git链接无法下载时，可使用以下指令下载：

   ```
   git pull https://github.com/open-mmlab/mmcv.git
   ```


+ 准备数据：

  需要准备的数据主要为mmdetection使用真实用例执行时的dump数据，当前下载目录：

  ```
  90.90.192.17： /home/syl/code/mmdet-test/data
  ```

  将下载的data文件夹替换mmdetection-test/data 目录：

  文件结构如下所示：

   ```
   mmdetection-test
       ├── data
       ├── LICENSE
       ├── main.py
       ├── pytest.ini
       ├── README.md
       ├── requirements.txt
       ├── testcase
       └── utils
   ```


+ 测试用例执行方式：

  ```
  python3 main.py --device=910B  --scope=acc                   # 测试所有用例精度.
  ```


+ 参数:

  | 名称    | 可选项                                 | 默认值  | 作用                                           |
    | ------- | -------------------------------------- | ------- | ---------------------------------------------- |
  | scope   | ["acc", "prof", "all", "single"]       | all     | 指定测试精度/性能/单个用例.                    |
  | device  | ["UNKNOWN", "910B", "910ProB", "910A"] | UNKNOWN | 指定当前NPU设备的名称，用于和相同设备对比性能. |
  | case_id | int                                    | 0       | 指定测试用例的ID值, 只在scope为single时生效.   |

## 已测试module :

### backbones:

| case_id | module名称 | 性能 | 精度 |
|--------| ----------- | -------- | -------- |
| 0 | ResNet | ✔           | ✔       |

### necks:

| case_id | module名称 | 性能     | 精度 |
| ------------------ | ----------- | -------- | -------- |
| 1 | FPN | ✔           | ✔       |

### heads:

| case_id | module名称 | 性能 | 精度 | 说明 |
|--------------------| ----------- | -------- | -------- | -------- |
| 2 | Shared2FCBBoxHead  | ✔           | ✔       |        |
| 3 | SingleRoIExtractor | ❌ | ❌ | 当前版本不支持，计划330支持 |
| 4           | SSDHead            | ✔          | ✔       |        |
| 5        | RetinaHead         | ✔          | ✔       |        |
| 6        | YOLOV3Head         | ✔          | ✔       |        |
| 7         | YOLOXHead          | ✔          | ✔       |        |
| 8     | CenterNetHead      | ✔          | ✔       |        |
| 9          | FCOSHead           | ✔          | ✔       |        |
| 10       | SOLOV2Head         | ✔          | ✔       |        |

### others:

| case_id | module名称 | 性能 | 精度 |
|------------------| ----------- | -------- | -------- |
| 11 | CrossEntropyLoss | ✔           | ✔        |
| 12 | MaxIoUAssigner   | ✔          |  ✔      |
