# CNN 训练与环境日志

## 2026-11-11 八人数据准备

P01–P08 各有四段原始视频。`01.mp4`、`02.mp4` 用于训练，`03.mp4` 用于验证，`04.mp4` 用于测试。P02 补齐后，四段视频共抽检 73 帧，全部通过单脸裁剪，注册照也通过特征提取检查。

| 划分 | 原始视频数 | 裁剪图片数 |
| --- | ---: | ---: |
| `train` | 16 | 358 |
| `val` | 8 | 98 |
| `test` | 8 | 104 |

裁剪输出位于 `data/cnn/`，三个划分均包含 P01–P08。当前素材数量见 [README](../README.md)。

## 2026-11-11 试训与正式训练

两次运行均使用原交通检测项目的 WSL 虚拟环境，实际设备为 RTX 4060 Laptop GPU。正式训练使用 `SmallFaceCNN`、20 轮、批大小 16、学习率 0.001、随机种子 42。

| 实验 | 轮数 | 最佳轮次 | 最佳验证准确率 |
| --- | ---: | ---: | ---: |
| `cnn_smoke_s42` | 1 | 1 | 12.24%（12/98） |
| `cnn_baseline_s42` | 20 | 14 | 77.55%（76/98） |

正式训练第 20 轮的验证准确率为 71.43%（70/98）。程序按 `val_acc` 选择权重，因此保存了第 14 轮。训练指标、配置和模型记录分别见 [history.csv](../outputs/training/cnn_baseline_s42/history.csv)、[settings.json](../outputs/training/cnn_baseline_s42/settings.json)、[best.pt](../models/train_cnn/cnn_baseline_s42/best.pt)。

## 2026-11-11 建立项目内训练环境

在 `project_11/.venv-train` 新建 Ubuntu / WSL 虚拟环境，从原环境复制训练和裁剪依赖。包文件合计约 4.78 GiB。新解释器与关键依赖均从 `project_11/.venv-train` 加载。

| 项目 | 验证结果 |
| --- | --- |
| Python | 3.12.3 |
| PyTorch / torchvision | `2.12.1+cu130` / `0.27.1+cu130` |
| OpenCV | 4.13.0，YuNet 和 SFace 模型均可加载 |
| CUDA | 13.0，识别到 RTX 4060 Laptop GPU |
| CNN 验证 | `SmallFaceCNN` 在 GPU 上完成合成输入的前向和反向计算 |
| 依赖检查 | `python -m pip check` 通过 |

两次历史训练的日志仍归属于原实验。新环境的 GPU 检查使用合成输入，正式模型后续可在这个环境中加载和测试。

```bash
cd /mnt/c/Users/14363/workplace/project_11
source .venv-train/bin/activate
which python
```

## 2026-11-14 原始亮度闭集测试

使用 `cnn_baseline_s42` 第 14 轮的最佳权重评估 104 张测试图，共认对 67 张，准确率为 **64.42%**。该轮的原始亮度验证准确率为 **77.55%（76/98）**。两个成绩对应不同数据划分，可分别查阅 [训练记录](../outputs/training/cnn_baseline_s42/history.csv) 和 [测试指标](../outputs/training/cnn_baseline_s42/test_metrics.json)。

- P01 认对 0/15，15 张全被预测为 P02。
- P02 认对 0/14，其中 12 张被预测为 P06，2 张被预测为 P03。

每人的测试图来自各自第 `04.mp4` 段视频。逐图结果和误判去向可查 [测试预测](../outputs/training/cnn_baseline_s42/test_predictions.csv) 与 [测试混淆矩阵](../outputs/training/cnn_baseline_s42/confusion_matrix.csv)。

## 2026-11-15 光照对照

| 方法 | 真人验证 | 真人测试 |
| --- | ---: | ---: |
| 八人 CNN 基线，原图 | 76/98（77.55%） | 67/104（64.42%） |
| 基线加自动 Gamma | 63/98（64.29%） | 未记录 |
| 基线加 CLAHE | 44/98（44.90%） | 31/104（29.81%） |
| 训练时扩大亮度变化，原图测试 | 78/98（79.59%） | 49/104（47.12%） |

Gamma 和 CLAHE 都降低了基线成绩。训练亮度增强使用 `ColorJitter(brightness=(0.5, 1.2), contrast=0.1)`，第 15 轮在验证集略有提升，但在独立测试上下降，P01 仍为 0/15。后续模型对照统一使用原图。 [Gamma 验证](../outputs/training/cnn_baseline_s42/auto_gamma/val_metrics.json) · [CLAHE 测试](../outputs/training/cnn_baseline_s42/clahe/test_metrics.json) · [亮度增强测试](../outputs/training/cnn_darkaug_s42/test_metrics.json)

## 2026-11-15 DigiFace 数据划分与从头训练

DigiFace-1M 的 `subjects_0-1999_72_imgs` 分包包含 **2,000 个合成身份**，每个身份 72 张，共 144,000 张。用种子 42，对每个身份的图随机划分为训练、验证和测试。三组采用同一套身份编号，每张原图仅分到一组。 [DigiFace 官方数据说明](https://github.com/microsoft/DigiFace1M#downloading-the-dataset)

| 划分 | 每个身份的张数 | 总张数 |
| --- | ---: | ---: |
| `data/digiface/train` | 58 | 116,000 |
| `data/digiface/val` | 7 | 14,000 |
| `data/digiface/test` | 7 | 14,000 |

`cnn_digiface_s42` 用 SmallFaceCNN 从随机权重训练 20 轮，分类头输出 2,000 类，输入尺寸为 112×112。使用 WSL 的 RTX 4060、批大小 64、学习率 0.001、种子 42、`--data-root data/digiface`。从配置写入到第 20 轮记录落盘约 **5 小时 40 分钟**。 [参数记录](../outputs/training/cnn_digiface_s42/settings.json) · [训练记录](../outputs/training/cnn_digiface_s42/history.csv)

| 轮次 | 训练准确率 | 验证准确率 |
| --- | ---: | ---: |
| 1 | 0.47% | 1.19% |
| 18（最佳权重） | 93.34% | **87.36%（12,230/14,000）** |
| 20 | 94.21% | 85.46% |

第 18 轮权重在合成测试集上认对 **12,237/14,000（87.41%）**。这是 2,000 个合成身份的闭集分类结果。真人数据通过下文的特征模板评估。 [训练曲线](../outputs/training/cnn_digiface_s42/curves.png) · [最佳权重](../models/train_cnn/cnn_digiface_s42/best.pt) · [合成测试指标](../outputs/training/cnn_digiface_s42/test_metrics.json)

## 2026-11-16 真人模板对照与 P02 排查

用原有裁剪图做八人闭集匹配。先提取 358 张真人训练图的特征，逐张 L2 归一化，再按身份取均值并重新归一化。验证和测试分别使用原 `val` 和 `test`，每张图按最高余弦相似度分到八类之一。

| 特征模型与模板 | 真人验证 | 真人测试 | P02 测试 |
| --- | ---: | ---: | ---: |
| 自训 DigiFace，第 18 轮，训练图平均模板 | 97/98（98.98%） | 91/104（87.50%） | 1/14 |
| 预训练 SFace，训练图平均模板 | 98/98（100%） | 104/104（100%） | 14/14 |

DigiFace 在 P01 上为 15/15，13 个测试错误都来自 P02 的 `04.mp4`，其中 7 张错为 P01、3 张错为 P04、3 张错为 P06。P02 与自身模板的相似度中位数由验证段约 0.92 降至测试段约 0.60。测试画面里人脸更小且偏红，脸部中央灰度与第二段训练视频接近。色偏、距离及姿态同时变化，各因素的影响仍待独立验证。SFace 的数字来自使用相同数据和匹配流程的本地复算。 [DigiFace 真人验证](../outputs/training/cnn_digiface_s42/real_templates/val_confusion_matrix.csv) · [DigiFace 真人测试](../outputs/training/cnn_digiface_s42/real_templates/test_confusion_matrix.csv)

在原测试集上，全员换成单张注册照时，DigiFace 真人测试为 **31/104**。保留全部平均模板、只给 P02 加一份注册照模板时，验证和测试结果与当时的平均模板基线一致，P02 仍为 1/14。平均模板基线保存在 [原测试集混淆矩阵](../outputs/training/cnn_digiface_s42/real_templates/test_confusion_matrix.csv)。

另用现有 SFace 刷脸程序读取注册库，在 P02 的原测试视频上逐帧运行。新注册照启用后 **111/111 帧**识别为 P02。该结果属于一段原视频的识别演示，与上表裁剪图闭集评估使用不同模板。 [视频识别记录](../outputs/basic/p02_new_enroll.csv) · [后续步骤](experiment-guide.md#system-next)

## 2026-11-16 P02 补拍与真人复测

用 P02 新拍的常规光线视频更新 `data/training_raw/P02/04.mp4`，该段按原采样方式得到 22 张可用裁剪图。P02 原偏红视频及对应的 14 张裁剪图已归档。更新后，真人训练、验证和测试分别为 358、98 和 112 张。 [原测试素材](../data/archive/p02_original_test_2026-11-16/)

用 `evaluate_digiface_real.py` 加载第 18 轮权重并沿用平均模板加 P02 注册照模板的配置，验证为 **97/98**，新测试集为 **112/112**，其中 P02 为 **22/22**。原测试集的 **91/104** 和本轮的 **112/112** 对应不同的 P02 测试视频，权重未更新。 [新版验证矩阵](../outputs/training/cnn_digiface_s42/p02_extra_template/val_confusion_matrix.csv) · [新版测试矩阵](../outputs/training/cnn_digiface_s42/p02_extra_template/test_confusion_matrix.csv)

SFace 视频演示中，P02 新视频首帧即识别正确，累计 **326/326 帧**为 P02。未注册的 P09 全段 **195/195 帧**为 `UNKNOWN`。 [P02 记录](../outputs/basic/p02_new_test_demo.csv) · [P09 记录](../outputs/basic/p09_unknown_demo.csv)

## 2026-11-16 P01 首刷实验与课程收尾

P01 的首刷评估集包含 10 段独立视频。用 SFace 默认阈值 0.363 运行，按视频清单标出的 2 秒窗口判断每次尝试。两组共 10 次均首刷通过。样本只有一位参与者，部分片段的站位和整体光线有变化，结果适用于本次采集。 [数据清单](../data/development/first_pass/P01/manifest.json) · [首刷实验报告](first-pass-report.md)

| 条件 | 首刷通过 | 需要重试 | 误认 |
| --- | ---: | ---: | ---: |
| 正常光 | 5/5 | 0 | 0 |
| 镜片反光 | 5/5 | 0 | 0 |

本轮课程项目以步骤 17 收尾。演示使用 SFace，DigiFace 保留为自训模型及离线评估，步骤 18 的继续微调和系统接入不列入本轮任务。 [实验 Guide](experiment-guide.md#first-pass-test)

## 2026-11-17 仓库整理与实时演示状态

自训 SmallFaceCNN 已接入 `face_live_demo.py`，使用第 18 轮权重导出的 ONNX 和独立图库。程序默认使用自训模型，SFace 可通过 `--model sface` 选择。默认相似度 0.72、分差 0.02，窗口显示每道门槛与拒识原因。

此前 Windows 回放检查中，P02 为 326/326 帧正确，P09 第二段为 204/204 帧拒识。P01 为 148/173 帧正确、24 帧拒识、1 帧误认。结果来自现有视频，现场摄像头稳定性尚无完整统计。原阶段收尾记录保留作为历史进度。

本次复核文件确认真人 train/val/test 为 358/98/112 张，当前真人模板混淆矩阵为验证 97/98、测试 112/112。首刷窗口重新统计为正常光 5/5、反光 5/5。发布范围为代码、文档及汇总训练指标，原始素材、特征库、模型二进制和虚拟环境保存在本地。
