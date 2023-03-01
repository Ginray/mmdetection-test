# Testcases for MMDetection

## Usage:

+ parameters

  | name    | choices                                | default | usage                                                    |
    | ------- | -------------------------------------- | ------- | -------------------------------------------------------- |
  | scope   | ["acc", "prof", "all", "single"]       | all     | Test the case marked prof/acc.                           |
  | device  | ["UNKNOWN", "910B", "910ProB", "910A"] | UNKNOWN | The device name of NPU.                                  |
  | case_id | int                                    | 0       | ID of the test case, only used when the scope is single. |


+ example:

```bash
python3 main.py --device=910B  --scope=all                   # Test all the cases.
python3 main.py --device=910B  --scope=acc                   # Test cases marked acc.
python3 main.py --device=910B  --scope=single --case_id=4    # Test the case with id=4.
```

## The interfaces that have been tested on NPU :

### backbones:

| module name   | performance | accuracy |
|--------| ----------- | -------- |
| ResNet | ✔           | ✔       |

### necks:

| module name   | performance | accuracy |
| ------------------ | ----------- | -------- |
| FPN | ✔           | ✔       |

### heads:

| module name        | performance | accuracy |
|--------------------| ----------- | -------- |
| Shared2FCBBoxHead  | ✔           | ✔       |
| SingleRoIExtractor | ❌          | ❌       |
| SSDHead            | ✔          | ✔       |
| RetinaHead         | ✔          | ✔       |
| YOLOV3Head         | ✔          | ✔       |
| YOLOXHead          | ✔          | ✔       |
| CenterNetHead      | ✔          | ✔       |
| FCOSHead           | ✔          | ✔       |
| SOLOV2Head         | ✔          | ✔       |

### others:

| module name      | performance | accuracy |
|------------------| ----------- | -------- |
| CrossEntropyLoss | ✔           | ✔        |
| MaxIoUAssigner   | ✔          |  ✔      |
