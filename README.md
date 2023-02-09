# Testcases for MMDetection

## The interfaces that have been tested on NPU :

### backbones:

| module name   | performance | accuracy |
|--------| ----------- | -------- |
| ResNet | ✔           | ✔       |
|        |             |          |
|        |             |          |

### necks:

| module name   | performance | accuracy |
| ------------------ | ----------- | -------- |
| FPN | ✔           | ✔       |
|                    |             |          |
|                    |             |          |

### heads:

| module name        | performance | accuracy |
|--------------------| ----------- | -------- |
| Shared2FCBBoxHead  | ✔           | ✔       |
| SingleRoIExtractor | ❌          | ❌       |
|                    |             |          |
