# 人脸识别基础系统：按步骤完成课程原型

更新于 2026-11-17。项目目录为 `C:\Users\14363\workplace\project_11`。

已完成基础识别、自训 CNN 和真人裁剪图评估。核心代码解释与训练程序见 [core-algorithms.md](core-algorithms.md)。

## 现在从哪里继续

**现在整理课程报告和演示。** [步骤 17 的首刷实验](#first-pass-test)已完成，使用 P01 的 10 段独立视频。准备演示时使用步骤 16 的识别和拒识录像，报告里加入步骤 15 的自训结果与步骤 17 的首刷对照。自训模型已接入实时摄像头，默认运行 SmallFaceCNN，启动和阈值调整见 [README](../README.md)。

SFace 的已注册人员和陌生人视频演示已通过。DigiFace 第 18 轮权重的合成测试为 **87.41%**，更换 P02 测试视频后，真人闭集复测为 **112/112**。现有结果和文件入口见[步骤 15](#closed-set-test)与[步骤 16](#system-next)。

### 当前文件检查结果

| 内容 | 本次实际检查到的状态 | 现在是否重做 |
| --- | --- | --- |
| WSL 环境 | `.venv` 可用，能导入 OpenCV 和 NumPy | 不用 |
| WSL 训练环境 | `.venv-train` 已通过 RTX 4060 和 CNN 计算检查 | 从步骤 10 激活 |
| 两个 ONNX 模型 | YuNet、SFace 模型文件已存在 | 不用 |
| 照片比对材料 | 已有 `work/smoke_a.jpg`、`smoke_b.jpg` | 可复核，不必从头做 |
| 注册照 | P01–P08 的 `enroll.jpg` 均已通过特征提取检查 | 可用于注册 |
| SFace 注册库 | `work/gallery.npz` 有 P01–P08，特征矩阵为 `(8, 128)` | 适用于现有视频演示 |
| 原型视频 | 基础目录保留 6 段正常视频，另有 2 段反光视频已归档 | 可复跑原型 |
| CNN 原始视频 | P01–P08 各有 `01.mp4` 至 `04.mp4`，共 32 段已通过检查 | 已完成抽帧 |
| CNN 裁剪图 | `data/cnn/` 已有 358 张训练图、98 张验证图、112 张测试图 | 可直接使用 |
| CNN 正式训练 | `cnn_baseline_s42` 完成 20 轮，第 14 轮验证准确率最高，为 77.55% | 可复用最佳权重 |
| CNN 闭集测试 | 原测试集 67/104（64.42%），P01 和 P02 在各自测试视频中全错 | 保留为原测试集基线 |
| 预训练 SFace 对照 | 裁剪图闭集匹配验证 98/98、原测试集 104/104 | 保留为原测试集对照 |
| 合成人脸 DigiFace | 2,000 个身份，训练 116,000 张、验证 14,000 张、测试 14,000 张 | 可直接读取 `data/digiface` 的三个划分 |
| DigiFace 从头训练 | `cnn_digiface_s42` 第 18 轮最佳权重，合成验证 87.36%、测试 87.41% | 已完成 |
| DigiFace 真人模板匹配 | 验证 97/98，新测试集 112/112，其中 P02 为 22/22 | 已完成闭集复测 |
| SFace 视频演示 | P02 为 326/326 帧，未注册的 P09 为 195/195 帧 `UNKNOWN` | 已完成 |
| P01 首刷实验 | 正常光和镜片反光各 5/5，重试和误认均为 0 | 已完成，结果可写入课程报告 |

素材检查记录了解码、时间戳和单脸裁剪结果。各组训练、验证、测试与数据排查结果见 [CNN 训练日志](cnn-training-log.md)。

## 步骤 1：重启后回到项目

在 Windows 的 PowerShell 或终端中输入：

```powershell
wsl -d Ubuntu
```

进入 Ubuntu 后执行：

```bash
cd /mnt/c/Users/14363/workplace/project_11
source .venv/bin/activate
which python
```

Python 路径应位于本项目的 `.venv/bin/`。每次打开新终端通常需要重新激活；已经装好的依赖和下载好的模型不必重复处理。

**验收：** 当前目录和 Python 环境正确。

## 步骤 2：复核基础依赖

当前环境已经可用，先检查，不重新安装：

```bash
python - <<'PY'
import cv2
import numpy as np
print('OpenCV:', cv2.__version__)
print('NumPy:', np.__version__)
assert hasattr(cv2, 'FaceDetectorYN')
assert hasattr(cv2, 'FaceRecognizerSF')
print('接口检查通过')
PY
```

现有 WSL 环境使用 headless OpenCV，适合读取文件、保存图片和视频。先用导出的视频查看结果。摄像头弹窗演示在第八步单独处理。

只有在新建环境、依赖确实缺失时，才参考下面的安装命令：

```bash
python -m pip install 'opencv-python-headless>=4.10,<5' numpy
```

此前的 pip 代理故障记录保留在文末附录 B，供同样问题再次发生时参考。不要把安装依赖当成每次启动的步骤。

**验收：** 两个接口存在。

## 步骤 3：复核模型，不重复下载

```bash
ls -lh models/face_detection_yunet_2023mar.onnx
ls -lh models/face_recognition_sface_2021dec.onnx
```

你已经有这两个文件。后面的程序会实际加载它们。

模型来源与用途：

- [YuNet 官方模型说明](https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/README.md)：找人脸和关键点。
- [SFace 官方模型说明](https://github.com/opencv/opencv_zoo/blob/main/models/face_recognition_sface/README.md)：提取可比较的身份特征。
- [OpenCV 官方实操教程](https://docs.opencv.org/4.13.0/d0/dd4/tutorial_dnn_face.html)：查看检测、对齐、特征与匹配的调用流程。

**验收：** 两个文件存在，后续加载无异常。

## 步骤 4：理解并复核照片比对

先理解最短流程：找到脸，按关键点对齐，把脸变成特征向量，再比较特征的相似度。

现有 `scripts/face_lab.py` 保留。若需要重新查看同人与异人分数，可以使用原命令，前提是对应照片存在：

```bash
python scripts/face_lab.py smoke data/smoke/a1.jpg data/smoke/a2.jpg
python scripts/face_lab.py smoke data/smoke/a1.jpg data/smoke/b1.jpg
```

`a1`、`a2` 应是同一人的不同照片，`b1` 是另一位参与者。程序输出相似度，并更新 `work/smoke_a.jpg`、`work/smoke_b.jpg`。分数不是身份概率，不能把 0.8 直接理解成“80% 是本人”。

**验收：** 理解同人与异人比较的目的，看过对齐后的图像。已经完成就跳过。

## 步骤 5：确认注册库

P01–P08 都有各自的注册照，格式为 `data/enrollment/<人员编号>/enroll.jpg`，例如：

```text
data/enrollment/
├─ P01/enroll.jpg
└─ P02/enroll.jpg
```

注册库检查命令：

```bash
python - <<'PY'
import numpy as np
with np.load('work/gallery.npz') as z:
    print('已注册人员:', z['ids'].tolist())
    print('特征矩阵:', z['features'].shape)
PY
```

当前预期为 P01–P08 和 `(8, 128)`。P09 留作未知人员。注册程序会读取 `data/enrollment/` 下所有符合路径格式的照片。

只有添加、替换注册照后，才重新执行：

```bash
python scripts/face_lab.py enroll
```

**验收：** 注册库已建立，知道添加人员后如何更新。

## 步骤 6：新增基础识别入口，先识别正常视频

复跑基础识别时直接使用现有的 `scripts/face_basic.py`。

`face_lab.py` 保留用于注册、照片比对和旧融合实验。基础识别入口位于：

```text
C:\Users\14363\workplace\project_11\scripts\face_basic.py
```

可先检查参数说明：

```bash
python scripts/face_basic.py --help
```

然后先跑一段 P01 的正常视频：

```bash
python scripts/face_basic.py \
  --source data/development/P01/normal_01.mp4 \
  --name P01_normal_01
```

程序逐帧读取、检测并识别，保存：

```text
outputs/basic/P01_normal_01.mp4
outputs/basic/P01_normal_01.csv
```

用 Windows 文件资源管理器打开 `project_11\outputs\basic`，播放导出视频。画面会显示人脸框、身份编号和相似度；CSV 保留每帧的输出。重复运行同一 `--name` 会更新同名结果，需保留旧版本时换名称。

第一版每帧独立决策。先观察是否能稳定显示正确编号，不加入多帧平滑或质量权重。

**验收：** 视频能读取，导出文件能打开，能够看到人脸框与身份输出。若显示 `UNKNOWN`，先按第七步分析分数，不要立刻加复杂算法。

## 步骤 7：验证两名已注册人员与一名未知人员

分别运行：

```bash
python scripts/face_basic.py \
  --source data/development/P02/normal_01.mp4 \
  --name P02_normal_01

python scripts/face_basic.py \
  --source data/development/P09/normal_01.mp4 \
  --name P09_normal_01
```

查看三个视频，按事先提供的人员编号判断结果，记录到实验笔记：

| 输入 | 预期行为 | 实际结果 |
| --- | --- | --- |
| P01 正常视频 | 主要显示 P01 | 实际填写 |
| P02 正常视频 | 主要显示 P02 | 实际填写 |
| P09 正常视频 | 主要显示 UNKNOWN | 实际填写 |

不要预先把“预期行为”写成已验证结果。若画面没有人脸，输出 `NO_FACE`；出现多张脸时输出 `MULTIPLE_FACES`，这版先按单人使用场景处理。

### 阈值先怎么用

附录默认匹配阈值为 **0.363**，来源于 [OpenCV 教程给出的 SFace 在 LFW 上的示例门槛](https://docs.opencv.org/4.13.0/d0/dd4/tutorial_dnn_face.html)。它只作为跑通基础系统的初始值，不是为本项目注册库验证过的最优阈值。

程序还保留最高分与第二名的差值门槛，初始为 0。先记下已注册和未注册视频的分数范围。如果要尝试阈值变化，只调整一个参数，保存新名称，例如：

```bash
python scripts/face_basic.py \
  --source data/development/P09/normal_01.mp4 \
  --threshold 0.45 \
  --name P09_threshold045
```

这里 0.45 只是试验值，不保证合适。提高阈值通常会增加拒识，降低阈值可能增加误认。不要只看 P01 能否通过，还要一起检查 P02 与 P09。

这一阶段做功能验证，不要求十人采集或复杂的参数搜索。最后确定一个初版阈值后，再拍几段新的正常视频做独立检查；不能把调参视频的表现直接叫作泛化准确率。

**验收：** 已注册人员能被正确区分，未知人员能被拒识；若存在错误，已经记录实际案例和分数。

## 步骤 8：接入摄像头并显示结果

先完成视频版，再验证现场摄像头。`--source 0` 表示访问运行程序所在系统的摄像头编号 0；如果失败，需检查设备访问，而不是重训模型。

### 8.1 WSL 先继续用视频，摄像头可在 Windows 单独演示

你当前 WSL 环境是 headless，不支持本教程中的弹窗预览。为避免改坏已能工作的环境，现场显示可以放在独立的 Windows 虚拟环境中。

只有做到摄像头演示时，才在 **Windows PowerShell** 执行下面的环境命令：

```powershell
cd C:\Users\14363\workplace\project_11
py -3 -m venv .venv-win
.\.venv-win\Scripts\python.exe -m pip install 'opencv-python>=4.10,<5' numpy
```

这里直接调用 Windows 虚拟环境内的 Python，不必执行激活脚本。若 `py` 不存在，先检查 Windows 是否安装 Python；WSL 里的 Python 不等于 Windows 原生 Python。

然后运行：

```powershell
.\.venv-win\Scripts\python.exe scripts\face_basic.py --source 0 --show --name webcam_demo
```

请关掉可能占用摄像头的“相机”应用。窗口出现后，先让 P01 或 P02 入镜，再让未注册参与者入镜；按 `q` 退出。若编号 0 打不开，可检查设备是否连接，再尝试编号 1。

已有模型文件、注册库和源码可以直接共用，不必在 Windows 重新录入全部人员。若安装依赖遇到网络错误，先处理网络或代理，视频版成果仍然保留。

### 8.2 画面横着、有脸却显示 NO_FACE

基础程序默认使用 `--rotate auto`。每帧先检测原画面，再按需要依次试 90°、180°、270°，采用第一个检测到人脸的角度。四个角度都检测不到时，返回原画面并标为 `NO_FACE`。

2026-11-02 的截图复核结果：原方向检测为 0 张脸，逆时针旋转 90° 后检测到 1 张脸。这项验证说明方向校正解决了该截图的检测问题，尚不代表摄像头身份识别已全部通过。

先在旧程序窗口按 `q` 退出，再在 Windows PowerShell 重启程序：

```powershell
.\.venv-win\Scripts\python.exe scripts\face_basic.py --source 0 --show --rotate auto --name webcam_auto
```

也可以手动固定方向。角度按顺时针计算，`270` 等于逆时针 90°，与这次截图所需的校正方向一致：

```powershell
.\.venv-win\Scripts\python.exe scripts\face_basic.py --source 0 --show --rotate 270 --name webcam_rot270
```

`--rotate 0` 关闭自动方向校正；`90`、`180`、`270` 固定相应旋转角度。画面上的 `rot` 与 CSV 的 `rotation_deg` 记录当前旋转角度。旋转使用 [OpenCV 的 rotate 接口](https://docs.opencv.org/4.13.0/d2/de8/group__core__array.html)，属于基础摄像头输入处理，不涉及反光评分或多帧身份融合。

**验收：** 摄像头窗口有方向正确的画面，能显示编号或 UNKNOWN，能够正常退出。尚未完成现场摄像头时，应把当前成果称为“视频版基础原型”。

## 步骤 9：整理一份最小演示与基础报告

基础系统先交代清楚：

1. 目标：对已注册人员做身份匹配，对未注册人员拒识。
2. 流程：YuNet 检测、关键点对齐、SFace 特征、余弦匹配、阈值决策。
3. 实现：注册入口、基础识别入口、显示与保存结果。
4. 测试：P01/P02/P09 的正常场景，成功与失败都记录。
5. 局限：人员少、光照与姿态变化、默认阈值仍需更多验证。

先准备一段能展示注册人员识别与未知人员拒识的录像，再逐步整理成完整课程报告和 6 分钟 PPT。单帧的识别状态可能跳动，先如实记录，不立即扩展成多帧算法实验。

笔记里的“自己实现质量多帧融合”目前是此前计划中的内容，不应直接作为已完成贡献。写报告时按真实进展表述，例如：“当前完成注册与逐帧阈值识别，质量评估与多帧融合列为后续扩展。”原笔记作为历史记录保留，必要时另加进展说明。

**基础系统完成标准：**

- [ ] 有可更新的人员注册库。
- [ ] 能识别正常图片或视频里的注册人员。
- [ ] 能对未注册人员输出 UNKNOWN。
- [ ] 有摄像头现场显示，或明确标注当前只完成视频版。
- [ ] 结果可以保存、回放，代码与命令可复现。
- [ ] 能讲清特征、相似度、阈值和拒识的含义。

满足这些后，继续文末步骤 10，进入自己训练网络的阶段。训练、验证与模型接入完成后，再决定是否增加反光处理。

## 基础阶段只看这些资料

1. [OpenCV 官方检测与识别教程](https://docs.opencv.org/4.13.0/d0/dd4/tutorial_dnn_face.html)：先看 Python 调用链，能对应自己的代码即可。
2. [SFace 论文页面](https://arxiv.org/abs/2205.12010)，[PDF](https://arxiv.org/pdf/2205.12010)：先读摘要、示意图和结论，理解预训练特征的来源。
3. [YuNet 官方模型说明](https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/README.md)：了解人脸框与关键点的作用。

QAN、SER-FIQ、反光相关质量指标和多帧融合论文暂时不设为必读。

## 已拍视频与旧实验代码怎么处理

基础阶段使用 `data/development/P01`、`P02`、`P09` 下的 6 段 `normal` 视频，人员编号和文件名保持不变。两段 `front` 视频已移到 `data/archive/reflection/P01/`，作为以后可用的素材，当前不参与基础功能检查，也不要求继续标注或补拍。

原 `face_lab.py` 保留用于 `smoke`、`enroll`，其 `cache/calibrate/evaluate` 属于后续实验路线。本教程新增的 `face_basic.py` 不依赖清单、质量权重或融合缓存。早先发现的可变帧率问题不再作为“先做反光实验”的理由；基础版逐帧处理全段视频，不统计 3 秒通行成功率。

基础版 CSV 记录读取器返回的视频时间戳，导出 MP4 按平均帧率编码供回放。保留原视频，当前不使用这个导出视频测量精确的真实通行耗时。

## 附录 A：基础识别程序

直接使用项目里的识别脚本。下方程序用于对照阅读。 [基础识别源码](../scripts/face_basic.py)

```python
from pathlib import Path
import argparse
import csv

import cv2 as cv
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def rotate_frame(frame, degrees):
    rotations = {90: cv.ROTATE_90_CLOCKWISE, 180: cv.ROTATE_180,
                 270: cv.ROTATE_90_COUNTERCLOCKWISE}
    return frame if degrees == 0 else cv.rotate(frame, rotations[degrees])


class FrameOrientation:
    """选定自动模式或固定角度，例如 FrameOrientation('auto')。"""

    def __init__(self, mode='auto'):
        self.angles = (0, 90, 180, 270) if mode == 'auto' else (int(mode),)

    @staticmethod
    def detect(frame, detector):
        """用 YuNet 检测当前画面，返回人脸框、关键点和置信度。"""
        detector.setInputSize((frame.shape[1], frame.shape[0]))
        _, faces = detector.detect(frame)
        return faces

    def apply(self, frame, detector):
        """返回 (旋转后的画面, 人脸检测结果, 顺时针角度)。"""
        for angle in self.angles:
            oriented = rotate_frame(frame, angle)
            faces = self.detect(oriented, detector)
            # 固定模式直接返回；自动模式采用第一个检测到脸的角度。
            if len(self.angles) == 1 or (faces is not None and len(faces) > 0):
                return oriented, faces, angle
        return frame, None, 0


def fit_output_frame(frame, size):
    """Keep the recorded video size fixed when the camera orientation changes."""
    width, height = size
    h, w = frame.shape[:2]
    if (w, h) == size:
        return frame
    scale = min(width / w, height / h)
    resized = cv.resize(frame, (max(1, round(w * scale)), max(1, round(h * scale))))
    canvas = np.zeros((height, width, 3), dtype=frame.dtype)
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def choose_identity(feature, ids, gallery, threshold, margin):
    scores = gallery @ feature
    order = np.argsort(scores)
    best = int(order[-1])
    top = float(scores[best])
    gap = top - float(scores[order[-2]]) if len(ids) > 1 else 2.0
    identity = ids[best] if top >= threshold and gap >= margin else 'UNKNOWN'
    return identity, top, gap


def recognize(frame, detector, recognizer, ids, gallery, threshold, margin,
              orientation=None):
    h, w = frame.shape[:2]
    if max(h, w) > 640:
        scale = 640 / max(h, w)
        frame = cv.resize(frame, (round(w * scale), round(h * scale)))
    orientation = orientation if orientation is not None else FrameOrientation()
    frame, faces, angle = orientation.apply(frame, detector)
    count = 0 if faces is None else len(faces)
    result = dict(face_count=count, identity='NO_FACE', score='', gap='',
                  rotation_deg=angle)
    if count == 1:
        aligned = recognizer.alignCrop(frame, faces[0])
        feature = recognizer.feature(aligned).copy().reshape(-1)
        norm = float(np.linalg.norm(feature))
        if norm > 1e-12 and np.isfinite(feature).all():
            identity, top, gap = choose_identity(
                feature / norm, ids, gallery, threshold, margin)
            result.update(identity=identity, score=top, gap=gap)
        else:
            result['identity'] = 'INVALID_FEATURE'
    elif count > 1:
        result['identity'] = 'MULTIPLE_FACES'
    if faces is not None:
        for face in faces:
            x, y, w, h = face[:4].astype(int)
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 255), 2)
    label = result['identity']
    if result['score'] != '':
        label += f" score={result['score']:.3f}"
    label += f' rot={angle}'
    cv.putText(frame, label, (8, 25), cv.FONT_HERSHEY_SIMPLEX,
               0.6, (0, 255, 0), 2, cv.LINE_AA)
    return frame, result


def main():
    parser = argparse.ArgumentParser(description='Basic per-frame face recognition')
    parser.add_argument('--source', required=True, help='Image/video path or camera number')
    parser.add_argument('--threshold', type=float, default=0.363)
    parser.add_argument('--margin', type=float, default=0.0)
    parser.add_argument('--name', default='basic_demo', help='Output filename stem')
    parser.add_argument('--show', action='store_true', help='Requires desktop OpenCV')
    parser.add_argument('--max-frames', type=int, default=0, help='0 means no frame limit')
    parser.add_argument('--rotate', choices=['auto', '0', '90', '180', '270'],
                        default='auto', help='Automatic orientation or clockwise degrees')
    args = parser.parse_args()
    if args.max_frames < 0:
        parser.error('--max-frames must be nonnegative')
    if not -1 <= args.threshold <= 1 or not 0 <= args.margin <= 2:
        parser.error('threshold must be in [-1,1], margin in [0,2]')
    if not args.name or Path(args.name).name != args.name or args.name in ('.', '..'):
        parser.error('--name must be a filename stem without directories')
    camera = args.source.isdigit()
    if camera and not args.show and args.max_frames == 0:
        parser.error('For a camera, use --show or set --max-frames')
    source = int(args.source) if camera else str(ROOT / args.source)
    with np.load(ROOT / 'work/gallery.npz') as z:
        ids = z['ids'].tolist()
        gallery = z['features'].astype(np.float32)
    if not ids or gallery.ndim != 2 or gallery.shape[0] != len(ids):
        raise ValueError('Invalid or empty gallery')
    norms = np.linalg.norm(gallery, axis=1, keepdims=True)
    if not np.isfinite(gallery).all() or np.any(norms <= 1e-12):
        raise ValueError('Invalid gallery features')
    gallery = gallery / norms
    detector = cv.FaceDetectorYN.create(
        str(ROOT / 'models/face_detection_yunet_2023mar.onnx'),
        '', (320, 320), 0.9, 0.3, 5000)
    recognizer = cv.FaceRecognizerSF.create(
        str(ROOT / 'models/face_recognition_sface_2021dec.onnx'), '')
    output = ROOT / 'outputs/basic'
    output.mkdir(parents=True, exist_ok=True)
    fields = ['frame', 'source_time_ms', 'face_count', 'identity', 'score', 'gap',
              'rotation_deg']
    image = not camera and Path(args.source).suffix.lower() in (
        '.jpg', '.jpeg', '.png', '.bmp', '.webp')
    cap, writer = None, None
    writer_size = None
    orientation = FrameOrientation(args.rotate)
    total = 0
    try:
        with (output / f'{args.name}.csv').open(
                'w', encoding='utf-8-sig', newline='') as f:
            log = csv.DictWriter(f, fieldnames=fields)
            log.writeheader()
            if image:
                frame = cv.imread(source)
                if frame is None:
                    raise ValueError(f'Cannot read image: {source}')
                annotated, result = recognize(
                    frame, detector, recognizer, ids, gallery,
                    args.threshold, args.margin, orientation)
                if not cv.imwrite(str(output / f'{args.name}.jpg'), annotated):
                    raise RuntimeError('Image output failed')
                log.writerow(dict(frame=0, source_time_ms='', **result))
                print(result)
                return
            cap = cv.VideoCapture(source)
            if not cap.isOpened():
                raise ValueError(f'Cannot open video/camera: {source}')
            fps = float(cap.get(cv.CAP_PROP_FPS))
            fps = fps if np.isfinite(fps) and fps > 0 else 25.0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                timestamp = '' if camera else float(cap.get(cv.CAP_PROP_POS_MSEC))
                if timestamp != '' and (not np.isfinite(timestamp) or timestamp < 0):
                    timestamp = ''
                annotated, result = recognize(
                    frame, detector, recognizer, ids, gallery,
                    args.threshold, args.margin, orientation)
                if writer is None:
                    writer_size = (annotated.shape[1], annotated.shape[0])
                    writer = cv.VideoWriter(str(output / f'{args.name}.mp4'),
                                            cv.VideoWriter_fourcc(*'mp4v'), fps, writer_size)
                    if not writer.isOpened():
                        raise RuntimeError('Video output failed')
                writer.write(fit_output_frame(annotated, writer_size))
                log.writerow(dict(frame=total, source_time_ms=timestamp, **result))
                total += 1
                if args.show:
                    cv.imshow('Basic face recognition - q to quit', annotated)
                    if cv.waitKey(1) & 0xFF == ord('q'):
                        break
                if args.max_frames and total >= args.max_frames:
                    break
            if total == 0:
                raise ValueError('No frames could be read')
    finally:
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv.destroyAllWindows()
    print(f'Processed {total} frames. Outputs: {output}')


if __name__ == '__main__':
    main()
```

读代码时按 `choose_identity`、`recognize`、`main` 的顺序。先明白身份分数如何变成输出，再看视频读取和显示，不必一开始改写所有函数。

## 附录 B：既有 pip 代理故障记录

下面保留此前已经整理的代理处理记录。当前环境已可用时跳过；只有再次出现同类错误才使用。记录中的旧“下一步”编号属于历史文字，当前执行顺序以本文主线为准。

### 本次故障记录（2026-11-02）：pip 代理连接失败

**发生位置：** 执行 `python -m pip install --upgrade pip` 和安装 OpenCV 等依赖时。

**截图中的关键报错：**

```text
ProxyError('Cannot connect to proxy.' ...)
Failed to establish a new connection: [Errno 111] Connection refused
Could not find a version that satisfies the requirement opencv-python-headless<5,>=4.10
No matching distribution found for opencv-python-headless<5,>=4.10
```

**原因：** pip 尝试通过一个连接不上的代理访问软件源，代理连接被拒绝，反复重试后仍然无法取得软件包版本列表，随后出现“找不到匹配版本”的提示。本次应优先处理代理连接，不能仅凭最后两行判断 Python 或 OpenCV 版本不兼容。截图本身没有确定代理的具体地址和配置来源。

**解决步骤：** 以下命令仍在 Ubuntu / WSL 终端执行。

1. 如果终端还在不断重试，先按 `Ctrl+C` 停止；已经回到命令提示符则直接继续。
2. 保留现有虚拟环境，进入项目并激活它：

   ```bash
   cd /mnt/c/Users/14363/workplace/project_11
   source .venv/bin/activate
   ```

3. 暂时跳过升级 pip。截图里的 pip 24.0 已能执行安装命令；先处理联网失败，完成依赖安装。整段复制以下命令：

   ```bash
   env -u http_proxy -u https_proxy -u all_proxy \
       -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u PIP_PROXY \
       PIP_CONFIG_FILE=/dev/null \
       python -m pip install \
       --index-url https://pypi.org/simple \
       --timeout 20 --retries 1 \
       'opencv-python-headless>=4.10,<5' numpy pandas matplotlib
   ```

   `env -u` 为本条命令移除代理相关环境变量，`PIP_CONFIG_FILE=/dev/null` 为本次调用禁用 pip 配置文件加载，`--index-url` 指定官方软件源。它们不会永久修改系统或终端的代理设置。配置文件处理方式见 [pip 官方配置说明](https://pip.pypa.io/en/stable/topics/configuration/#pip-config-file)。

4. 安装命令成功后，验证依赖和接口：

   ```bash
   python - <<'PY'
   import cv2
   import numpy
   import pandas
   import matplotlib
   assert hasattr(cv2, 'FaceDetectorYN')
   assert hasattr(cv2, 'FaceRecognizerSF')
   print('安装与接口检查通过，OpenCV 版本：', cv2.__version__)
   PY
   ```

5. 验证通过后，创建所需目录并重新记录实际依赖版本：

   ```bash
   mkdir -p qa work/scores words outputs models scripts
   python -m pip freeze > qa/requirements-initial.txt
   ```

**成功后接着做什么：** 不必重新执行原来报错的两条 pip 安装命令，它们可能再次使用失效代理。完成本节下方的环境检查，然后直接进入“步骤 3：只下载两个模型”。无需重新创建虚拟环境。

**如果仍然失败：** 若错误变为连接超时或域名解析失败，说明绕过代理后仍存在网络访问问题；若仍为 `ProxyError`，则需要进一步定位代理来源。保留最后约 15 行错误，依据新错误继续排查。安装与验证未成功前，先停留在步骤 2；`pip freeze` 能生成文件，并不代表依赖已经安装成功。

---

<a id="training-stage"></a>

## 第二阶段：自己定义、训练和验证人脸识别网络

本节是新增主线，接在基础阶段步骤 9 后执行。前面的附录是查阅材料，不必重新执行一遍。

目标是亲自完成网络定义、前向计算、交叉熵损失、反向传播、参数更新、验证与权重保存。本次先训练一个小型 CNN，之后可选做迁移学习，最后才讨论反光。训练从正常条件开始。

### 步骤 10：进入已验证可用的 WSL GPU 环境

`project_11/.venv-train` 已准备好，包含 Python 3.12.3、PyTorch `2.12.1+cu130`、torchvision `0.27.1+cu130` 和 OpenCV 4.13.0。已验证 RTX 4060 上的 CNN 前向和反向计算，以及 OpenCV 人脸模型加载。

在 **Ubuntu / WSL** 运行：

```bash
cd /mnt/c/Users/14363/workplace/project_11
source .venv-train/bin/activate
which python
python - <<'PY'
import torch
import torchvision
print('torch:', torch.__version__)
print('torchvision:', torchvision.__version__)
print('CUDA:', torch.cuda.is_available())
assert torch.cuda.is_available(), 'CUDA 不可用，请检查 WSL 训练环境'
print('GPU:', torch.cuda.get_device_name(0))
PY
```

WSL 的 `.venv` 用于基础识别，`.venv-train` 用于裁剪、训练和测试，Windows 的 `.venv-win` 用于摄像头演示。最早的八人试训和基线使用旧项目环境，DigiFace 正式训练已在 `.venv-train` 完成。 [CNN 训练日志](cnn-training-log.md)

**验收** `which python` 指向 `project_11/.venv-train/bin/python`，终端显示 `CUDA: True` 和 RTX 4060。

### 步骤 11：确认正常视频的分组

原始视频存放于 `data/training_raw/<人员编号>/`，每人四段。注册照单独放在 `data/enrollment/<人员编号>/enroll.jpg`。

| 文件名 | 目标时长 | 数据划分 | 用途 |
| --- | --- | --- | --- |
| `01.mp4` | 约 10 秒 | `train` | 更新网络权重 |
| `02.mp4` | 约 10 秒 | `train` | 更新网络权重 |
| `03.mp4` | 约 5 秒 | `val` | 选择最佳轮次与参数 |
| `04.mp4` | 约 5 秒 | `test` | 最终一次评估 |

P01–P08 已通过素材检查。首批素材中，训练段实测 10.03–11.77 秒，验证与测试段实测 5.17–7.44 秒。P02 的常规光线测试视频后来补拍，现用的 `04.mp4` 约 10.9 秒，原片与原裁剪图已归档。P09、P10 等未知人员留作单独的拒识检查。[P02 原测试素材](../data/archive/p02_original_test_2026-11-16/)

按整段视频和拍摄批次确定划分，同一段视频的相邻帧、镜像和增强版本归入同一组。验证与测试是否独立录制，以实际采集记录为准。素材数量与当前划分见 [README](../README.md)。

**验收** 每位参与本轮 CNN 实验的人员都有两段训练、一段验证和一段测试视频。三组使用同一批已知身份。

### 步骤 12：检测、校正方向并裁剪人脸

已用 `scripts/video_to_phto.py` 完成 32 段视频的裁剪。补拍后 P02 的测试图由 14 张更新为 22 张，目前三个划分共 568 张图片。[视频裁剪函数与参数](core-algorithms.md#prepare-cnn-faces)

以下命令记录了八人的批量抽帧方式，适用于对应裁剪结果尚未生成时。每段使用不同的 `--clip` 值，保留帧的来源。

```bash
(
  set -e
  for person in P01 P02 P03 P04 P05 P06 P07 P08; do
    for number in 01 02 03 04; do
      case "$number" in
        01|02) split=train ;;
        03) split=val ;;
        04) split=test ;;
      esac
      python scripts/video_to_phto.py \
        --source "data/training_raw/$person/$number.mp4" \
        --split "$split" --person "$person" --clip "${person}_${number}"
    done
  done
)
```

脚本遇到已有同一 clip 的裁剪文件会报错。中途失败后，按尚未完成的人员和片段继续执行。

最后结构为：

```text
data/cnn/
├─ train/
│  ├─ P01/*.jpg
│  └─ P02–P08 的人员目录
├─ val/
│  ├─ P01/*.jpg
│  └─ P02–P08 的人员目录
└─ test/
   ├─ P01/*.jpg
   └─ P02–P08 的人员目录
```

所有 split 的人员目录应一致、每个目录非空。扩展人员时，三个 split 一起补齐。脚本使用已有 YuNet 与对齐接口，没有用 SFace 特征替代训练图片。

人工打开各批次若干裁剪图，确认方向、脸部范围和标签正确。记录脚本报告的 sampled/saved/no_single_face。训练阶段可以跳过未成功裁剪的图片，但系统级评估仍要统计检测失败。

**验收：** 裁剪的人脸正确，来源可追溯，没有跨组重复图像。正式训练前先核对这些，再调学习率。

### 步骤 13：保存自己的网络与训练程序，读懂五行训练循环

现有 `scripts/train_face_cnn.py` 已用于试训和正式训练。[CNN 训练代码与参数](core-algorithms.md#train-cnn-code)

重点阅读：

1. `SmallFaceCNN`：四个卷积块、128 维特征层、C 类分类头。
2. `run_epoch`：前向计算、交叉熵、`backward`、`step`。
3. 验证阶段：`model.eval()` 与关闭梯度。
4. 最佳模型保存：只根据验证集选择，测试集不参与。

它从随机权重开始训练全部卷积层，输入是 RGB 人脸图像，经 112×112 缩放与固定归一化。分类头输出 logits，交叉熵之前不额外 Softmax。对应资料：[官方优化教程](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)、[CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)。

**验收：** 能解释本次到底哪些权重被更新；不能把这个网络称为自己训练的 SFace 或 ArcFace。

### 步骤 14：训练并保存最佳权重

八人基线和 DigiFace 正式训练都已完成，已有权重可以直接评估。需要重跑 DigiFace 时，在步骤 10 的 `.venv-train` 中用一个新实验名，例如：

```bash
python scripts/train_face_cnn.py \
  --mode train --data-root data/digiface --run cnn_digiface_repeat_s42 \
  --epochs 20 --batch 64 --lr 0.001 --seed 42 --device cuda
```

这组训练从随机权重开始，使用 2,000 个合成身份和 112×112 输入，原运行约耗时 5 小时 40 分钟。脚本按验证准确率保存 `best.pt`，同名运行会报错。 [DigiFace 训练记录](../outputs/training/cnn_digiface_s42/history.csv) · [CNN 训练参数](cnn-training-parameters.md)

<a id="closed-set-test"></a>

### 步骤 15：读取合成测试与真人闭集结果

<a id="digiface-next"></a>
#### DigiFace 合成测试

第 18 轮的合成验证为 **12,230/14,000（87.36%）**，独立合成测试为 **12,237/14,000（87.41%）**。两组都在 2,000 个合成身份中分类。 [合成测试指标](../outputs/training/cnn_digiface_s42/test_metrics.json)

需要复核时，在 WSL 激活 `.venv-train` 并运行下面的命令。

```bash
python scripts/train_face_cnn.py --mode test --data-root data/digiface \
  --run cnn_digiface_s42 --batch 64 --device cuda --no-auto-gamma
```

使用 `--no-auto-gamma` 保持原图输入。复跑会更新该实验目录下的合成测试指标、混淆矩阵和逐图预测。

<a id="resume-after-lighting"></a>
#### P01–P08 真人对照

| 方法 | 验证集 | 原测试集 | 原 P02 测试 |
| --- | ---: | ---: | ---: |
| 八人 CNN 直接分类 | 76/98 | 67/104 | 0/14 |
| 自训 DigiFace，用真人训练图建平均模板 | 97/98 | 91/104 | 1/14 |
| 预训练 SFace，用真人训练图建平均模板 | 98/98 | 104/104 | 14/14 |

DigiFace 对真人使用 `model.encode()` 提取特征。逐张归一化后按身份求均值作为模板，验证和测试图按最高余弦相似度匹配。测试混淆矩阵的行是真实身份，列是预测身份。DigiFace 认对了 P01 的 15 张图，全部 13 个误判集中在 P02 的一段测试视频。 [真人验证矩阵](../outputs/training/cnn_digiface_s42/real_templates/val_confusion_matrix.csv) · [真人测试矩阵](../outputs/training/cnn_digiface_s42/real_templates/test_confusion_matrix.csv)

更换 P02 的 `04.mp4` 后，用当前 `evaluate_digiface_real.py` 复测，验证为 **97/98**，新测试集为 **112/112**，其中 P02 为 **22/22**。这次使用第 18 轮权重与现有模板配置，即训练图平均模板加一份 P02 注册照模板。报告中将 **91/104** 标为原测试集对照，将 **112/112** 标为换片后的复测。两次分数对应不同拍摄素材，反映的是这两段视频在现有模型下的表现。 [新测试混淆矩阵](../outputs/training/cnn_digiface_s42/p02_extra_template/test_confusion_matrix.csv) · [训练日志](cnn-training-log.md)

<a id="system-next"></a>
### 步骤 16：刷脸视频演示（已完成）

SFace 使用 P01–P08 的注册库完成了下面三段视频的逐帧检查。每一行是一段视频。

| 视频 | 结果 | 演示文件 |
| --- | --- | --- |
| P02 新拍的常规光线视频 | 第一帧即识别为 P02，326/326 帧正确 | [标注视频](../outputs/basic/p02_new_test_demo.mp4) · [逐帧记录](../outputs/basic/p02_new_test_demo.csv) |
| 未注册的 P09 | 195/195 帧为 `UNKNOWN` | [标注视频](../outputs/basic/p09_unknown_demo.mp4) · [逐帧记录](../outputs/basic/p09_unknown_demo.csv) |
| P02 原偏红视频 | 111/111 帧识别为 P02 | [逐帧记录](../outputs/basic/p02_new_enroll.csv) |

<a id="first-pass-test"></a>
### 步骤 17：比较正常光和眼镜反光下的首刷表现（已完成）

本次使用 P01 的 10 段独立视频，每段对应一次刷脸。正常光和镜片反光组各 5 段，文件在 `data/development/first_pass/P01/` 的 `normal/` 和 `glare/` 目录。每段的 2 秒判定区间已标在清单里。 [数据集说明](../data/development/first_pass/P01/README.md) · [视频清单](../data/development/first_pass/P01/manifest.json)

需要复现时，在 WSL 的项目根目录运行下面的命令。

```bash
source .venv/bin/activate
for condition in normal glare; do
  for video in data/development/first_pass/P01/"$condition"/*.mp4; do
    clip=$(basename "$video" .mp4)
    python scripts/face_basic.py --source "$video" --name "p01_first_pass_${condition}_$clip"
  done
done
```

用 `outputs/basic/` 中的 `p01_first_pass_*.csv` 统计结果。按清单时间筛选 `source_time_ms`，判断区间内第一次出现的已注册身份。先出现 `P01` 算首刷通过，先出现其他编号算误认，2 秒内没有已注册编号算需要重试。

| 条件 | 独立尝试 | 首刷通过 | 需要重试 | 误认 |
| --- | ---: | --- | --- | --- |
| 正常光 | 5 | 5 | 0 | 0 |
| 镜片反光 | 5 | 5 | 0 | 0 |

报告使用本次 P01 的实测结果，结论限于这 10 次尝试。 [首刷实验报告](first-pass-report.md)

<a id="improve-digiface"></a>
### 步骤 18：自训模型已接入，可选继续改进

1. 用归档的 P02 偏红视频作为困难场景对照。若要验证色偏或远距离的影响，每次固定其他条件，只改变一项。微调用新增的训练视频，验证和最终复测各留独立片段。
2. 以 DigiFace 第 18 轮权重为起点，用小学习率微调最后一个卷积块和特征层。现有训练脚本从随机权重起步，需要另加加载初始权重的入口。优先对比色偏增强与低分辨率增强，每次只加一种。
3. 用同一个模板匹配流程对比微调前后，记录 P02 正确数、全体准确率和其他人被误认成 P02 的次数。阈值使用独立的已知和未知人员素材校准。 [PyTorch 迁移学习教程](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

自训模型已通过 `prepare_digiface_demo.py` 导出特征网络并建立独立图库。`face_live_demo.py` 默认使用 SmallFaceCNN，`--model sface` 切换对照，`--model digiface` 保留为自训模型别名。默认阈值 0.72、分差 0.02，界面显示拒识原因。窗口程序及现有视频回放检查已通过，现场摄像头条件下的稳定性仍需实测。

课程报告可以依次展示基础系统、自训网络和训练方法、独立测试结果，以及 P02 的失败案例。自训权重接入、反光处理和多帧融合按实际完成情况写进演示或后续计划。
