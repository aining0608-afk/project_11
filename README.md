# 人脸识别课程实验

更新于 2026-11-17。

本项目面向校园进校场景，包含 SFace 人脸识别原型、自训模型的离线评估，以及正常光和镜片反光下的首刷实验。

**课程实验已完成。现场演示使用 `scripts/face_live_demo.py`，默认运行自训 SmallFaceCNN，追加 `--model sface` 可切换到 SFace。**

课程初稿于 2026-11-30 提交，汇报安排在 2026-12-02。实验过程和结果见下方文档入口。

## 文档入口

| 文档 | 用途 |
| --- | --- |
| [分步实验教程](words/experiment-guide.md) | 当前执行主线、WSL/Windows 命令、步骤 1—18、常见问题 |
| [核心算法展示](words/core-algorithms.md) | 按使用顺序展示数据准备、CNN 训练与摄像头识别代码 |
| [CNN 训练日志](words/cnn-training-log.md) | 数据划分、两次训练、最佳权重和独立训练环境验证 |
| [CNN 参数说明](words/cnn-training-parameters.md) | 命令参数、损失、准确率和权重选择依据 |

`words/` 用于存放输出的教程、说明、实验记录和报告等文本资料。

## 实际进度

| 内容 | 当前状态 |
| --- | --- |
| WSL 基础环境 | 已有 `.venv`，使用 headless OpenCV 处理文件 |
| WSL 训练环境 | `.venv-train` 已验证 CUDA、CNN 前向和反向计算，以及 OpenCV 模型加载 |
| Windows 摄像头环境 | 已有独立 `.venv-win`，用于弹窗显示 |
| 检测与特征模型 | YuNet、SFace 的 ONNX 文件已存在 |
| 照片比对与注册 | `face_lab.py` 可用于 `smoke`、`enroll`；注册库含 P01–P08，特征形状为 `(8, 128)` |
| 基础识别入口 | `face_basic.py` 已实现逐帧匹配、拒识、自动方向校正、结果显示与保存 |
| 原型视频 | P01、P02、P09 各两段，共 6 段，存于 `data/development/` |
| CNN 原始素材 | P01–P08 的 32 段视频和 8 张注册照已通过素材检查 |
| CNN 裁剪数据 | `data/cnn/` 中有 358 张训练图、98 张验证图、112 张测试图（原测试集为 104 张） |
| 反光素材 | 两段 P01 视频已归档到 `data/archive/reflection/P01/` |
| 自定义网络训练 | `cnn_smoke_s42` 已完成 1 轮试训，`cnn_baseline_s42` 已完成 20 轮正式训练 |
| 闭集测试 | 已完成，结果见 [训练日志](words/cnn-training-log.md) |
| 自训练模型 | DigiFace 已完成离线评估，可以用独立注册库做实时演示 |
| 眼镜反光 | P01 的 10 次首刷对照已完成，见 [首刷报告](words/first-pass-report.md) |

当前实时流程：YuNet 检测与关键点、对齐裁剪、SmallFaceCNN 特征提取、L2 归一化、注册库余弦匹配、阈值与最高分差值拒识。可通过参数切换为 SFace 特征。

SmallFaceCNN 使用 DigiFace 的 2,000 个合成身份从随机权重训练，最佳权重来自第 18 轮。SFace 是预训练对照模型。

## 已有结果

以下为 2026-11-02 检查 `outputs/basic/` 中既有 CSV 的统计，未重新运行或覆盖结果：

| 结果文件 | 帧数 | 输出记录 |
| --- | --- | --- |
| `P01_normal_01.csv` | 149 | 全部输出 P01 |
| `P02_normal_01.csv` | 108 | 全部输出 P02 |
| `P09_normal_01.csv` | 195 | 全部输出 UNKNOWN |
| `P09_threshold045.csv` | 195 | 全部输出 UNKNOWN |
| `webcam_auto.csv` | 740 | P01：718，UNKNOWN：19，NO_FACE：3；记录的旋转角度为 270° |

这些是少量既有片段的逐帧输出。相邻帧不是独立人员样本，摄像头日志也未逐帧标注真实身份，因此不能将上述计数直接称为独立测试准确率。P09 在两个阈值下均被拒识，也不足以确定正式阈值。

较早的 `webcam_demo.csv` 记录为 1351 帧 NO_FACE；之后增加了自动方向校正，新的结果另存为 `webcam_auto`。

## 从公开仓库开始

仓库提供代码、教程、训练曲线和汇总指标。真人素材、注册特征、模型文件与虚拟环境保存在本地，因此克隆后需要准备模型和自己的注册素材。文档中指向这些本地文件的路径用于说明实验来源。

在仓库根目录的 Windows PowerShell 中创建演示环境。

```powershell
py -3.13 -m venv .venv-win
.\.venv-win\Scripts\python.exe -m pip install -r requirements-demo.txt
.\.venv-win\Scripts\python.exe scripts\face_live_demo.py --help
```

按 [实验教程步骤 2–3](words/experiment-guide.md) 准备 YuNet 和 SFace ONNX 文件，放入 `models/`，校验值见 `qa/model-sha256.txt`。SFace 的注册照放在 `data/enrollment/<编号>/enroll.jpg`，运行 `python scripts/face_lab.py enroll` 创建图库。

自训模式需要 `models/train_cnn/cnn_digiface_s42/best.pt`、`data/cnn/train/<编号>/` 下的对齐训练图和 `data/enrollment/P02/enroll.jpg`。在训练环境安装 `requirements-training.txt` 的依赖后，运行 `python scripts/prepare_digiface_demo.py` 生成 ONNX 和专用图库。该模板构建程序沿用本实验的 P02 附加模板设计。训练环境与 Windows 演示环境分别安装 headless 和桌面 OpenCV。

下面保留原实验机器的运行命令。其他机器请将项目目录改成自己的克隆位置。当前依赖清单来自本地环境，尚未在全新环境执行完整训练复现。

## 运行基础系统

现有环境和模型可以继续使用，无需每次重装。首次配置、缺依赖或遇到代理问题时，查看分步教程。

### 1. WSL：进入项目并激活基础环境

以下命令在 **Ubuntu / WSL** 中执行：

```bash
cd /mnt/c/Users/14363/workplace/project_11
source .venv/bin/activate
which python
python scripts/face_basic.py --help
```

`which python` 应指向本项目的 `.venv/bin/python`。

### 2. 识别已有正常视频

```bash
python scripts/face_basic.py \
  --source data/development/P01/normal_01.mp4 \
  --name P01_check_20261102
```

检查未知人员时使用：

```bash
python scripts/face_basic.py \
  --source data/development/P09/normal_01.mp4 \
  --name P09_check_20261102
```

结果保存到 `outputs/basic/<name>.mp4` 和同名 CSV。在 Windows 中打开导出视频查看结果。重复使用同一 `--name` 会更新同名文件，需保留旧结果时使用新名称。

图片也可作为 `--source` 输入，此时保存标注图片和 CSV。

### 3. Windows：实时摄像头演示

以下命令在 **Windows PowerShell** 中执行，默认使用自训的 **SmallFaceCNN**。

```powershell
cd C:\Users\14363\workplace\project_11
.\.venv-win\Scripts\python.exe scripts\face_live_demo.py
```

需要手动切换到 SFace 时，追加 `--model sface`。

```powershell
.\.venv-win\Scripts\python.exe scripts\face_live_demo.py --model sface
```

两种模式都用 YuNet 定位人脸、OpenCV 对齐人脸，窗口标题会显示当前识别模型。SmallFaceCNN 使用 DigiFace 合成数据训练的第 18 轮权重导出文件 `models/train_cnn/cnn_digiface_s42/embedding.onnx`，读取 `work/gallery_digiface.npz`。这份图库包含 P01–P08 训练图的平均特征，并给 P02 加一份注册照模板。SFace 模式读取 `work/gallery.npz`。

SmallFaceCNN 演示的默认相似度阈值为 0.72，前两名分差至少为 0.02。切换到 SFace 时默认使用阈值 0.363、分差 0.0。P09 的第一段视频用于确定 SmallFaceCNN 的相似度门槛，第二段视频 204 帧均被拒识。用现有 P02 测试视频检查时，326 帧均正确识别。P01 测试视频中，148/173 帧识别正确，24 帧被拒识，1 帧误认成 P06。这些数字仅描述现有视频逐帧结果。

请一人入镜，窗口会显示识别编号或未知人员提示。按 `Q`、`Esc` 或关闭窗口退出。预览仅在本机显示，不保存视频和 CSV。若要重新导出自训模型和注册库，在装有 `onnx` 的 WSL 训练环境运行 `python scripts/prepare_digiface_demo.py`。Windows 演示环境直接读取导出的模型。

有多个摄像头时可追加 `--camera 1`。如果自动校正后的画面方向仍不合适，可追加 `--rotate 270` 固定为逆时针旋转 90°。

需要用 SFace 录制标注视频及 CSV 时，使用录制入口：

```powershell
.\.venv-win\Scripts\python.exe scripts\face_basic.py --source 0 --show --rotate auto --name webcam_recording
```

### 4. 添加或替换注册照

注册照按 `data/enrollment/<人员编号>/enroll.jpg` 保存。当前 P01–P08 已注册，P09 用于未知人员检查。

只有注册照发生变化时，才在 WSL 项目环境重新执行：

```bash
python scripts/face_lab.py enroll
```

这会重建 `work/gallery.npz`。人员变化后需要重新检查识别和拒识表现。

## face_basic.py 参数和输出含义

| 参数 | 默认值 | 用途 |
| --- | --- | --- |
| `--source` | 必填 | 图片、视频路径或摄像头编号 |
| `--name` | `basic_demo` | 输出文件名前缀，不含目录 |
| `--threshold` | `0.363` | 身份匹配最低分数；当前是入门初始值，尚未通过正式本地实验确定 |
| `--margin` | `0.0` | 最高分与第二名的最低分差 |
| `--rotate` | `auto` | 自动校正，或固定顺时针 0/90/180/270 度 |
| `--show` | 关闭 | 弹窗显示，需支持桌面显示的 OpenCV |
| `--max-frames` | `0` | 限制处理帧数；0 表示不限制 |

自动模式记住上次成功方向，首次找不到脸或连续漏检 5 帧时搜索其他方向，选择检测置信度最高的方向。`270` 表示顺时针 270°，即逆时针 90°。

CSV 包含帧编号、源视频时间戳、人脸数、输出身份、匹配分数、最高分差值和旋转角度。可能的状态有注册编号、`UNKNOWN`、`NO_FACE`、`MULTIPLE_FACES`、`INVALID_FEATURE`。无脸和多脸情况同样保留在日志中。

离线 CSV 记录读取器返回的源视频时间戳；摄像头时间戳留空。导出 MP4 用于回放，不能直接用其播放时长推算真实通行延迟。匹配分数也不是身份概率。

## WSL 训练环境

在 Ubuntu / WSL 中激活本项目的训练环境：

```bash
cd /mnt/c/Users/14363/workplace/project_11
source .venv-train/bin/activate
which python
```

`which python` 应显示 `project_11/.venv-train/bin/python`。此环境用于抽帧裁剪、CNN 训练和测试。旧环境用于两次已完成的历史实验，新环境验证记录见 [CNN 训练日志](words/cnn-training-log.md)。

已完成的闭集测试、视频演示和首刷实验见 [实验教程第 15–17 步](words/experiment-guide.md#closed-set-test)。

## 目录结构

```text
project_11/
├─ README.md
├─ words/                   # 教程、算法说明、笔记、文本报告
├─ scripts/
│  ├─ face_live_demo.py      # 现场摄像头实时预览
│  ├─ face_basic.py          # 当前视频/摄像头基础识别入口
│  ├─ face_lab.py            # 照片比对、注册及旧融合实验入口
│  ├─ video_to_phto.py    # 视频抽帧与对齐裁剪
│  └─ train_face_cnn.py       # SmallFaceCNN 训练与闭集测试
├─ data/
│  ├─ smoke/                # 照片比对材料
│  ├─ enrollment/           # 注册照片
│  ├─ development/          # P01/P02/P09 的 6 段原型视频
│  ├─ training_raw/         # 按人员编号存放 01–04 原始视频
│  ├─ cnn/                  # train、val、test 三组裁剪图
│  ├─ test/                 # 为后续独立测试预留
│  └─ archive/reflection/   # 已归档的反光视频
├─ models/                  # YuNet、SFace ONNX 和 CNN 权重
├─ work/                    # gallery.npz、对齐调试图等
├─ outputs/                 # 基础识别输出、CNN 训练日志和曲线
├─ qa/                      # 初始依赖版本、模型 SHA256
├─ configs/                 # 为后续配置预留
├─ src/                     # 为后续模块拆分预留
├─ .venv/                   # WSL 基础识别环境
├─ .venv-train/             # WSL CUDA 训练环境
└─ .venv-win/               # Windows 摄像头环境
```

原始人脸图片、视频与特征库包含参与者信息，保存在本地；公开代码时排除这些素材和虚拟环境。实验中使用 P01 等编号，演示素材按参与者同意的范围使用。
