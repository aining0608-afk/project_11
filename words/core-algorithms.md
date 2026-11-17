# 人脸识别核心代码

按数据准备、模型训练、模板建立、模型导出和摄像头识别的使用顺序展示核心代码。以下为源码节选，省略命令行校验、日志和界面样式；完整参数和实现见 [scripts 目录](../scripts/)。

## 1. 视频抽帧与人脸对齐

<a id="prepare-cnn-faces"></a>

源码：[video_to_phto.py](../scripts/video_to_phto.py)。真人视频按整段划分 train、val、test，约每 0.5 秒采样一帧。

```python
# --split、--person、--clip 分别指定数据划分、身份和文件名前缀。
out = ROOT / "data/cnn" / args.split / args.person

# frame 来自视频；自动旋转后，仅保存检测到单张人脸的画面。
frame, faces, _ = orientation.apply(frame, detector)
if faces is None or len(faces) != 1:
    skipped += 1
else:
    aligned = recognizer.alignCrop(frame, faces[0])
    target = out / f"{args.clip}_{saved:04d}.jpg"
    cv.imwrite(str(target), aligned)
    saved += 1
```

## 2. CNN 数据加载

<a id="train-cnn-code"></a>

源码：[train_face_cnn.py](../scripts/train_face_cnn.py)。训练数据由 `--data-root` 指定；本项目自训模型使用 DigiFace 数据。

```python
def transform(training, gamma_reference=None, clahe=False):
    steps = [transforms.Resize(FACE_SIZE)]  # 统一为 112×112。
    if training:
        steps += [
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=(0.5, 1.2), contrast=0.1),
        ]
    elif clahe:
        steps.append(enhance_shadows)
    elif gamma_reference is not None:
        steps.append(transforms.Lambda(
            lambda image: auto_gamma(image, gamma_reference)
        ))
    return transforms.Compose(steps + [
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),  # RGB 归一化到 [-1,1]。
    ])


# 目录名作为身份标签；训练时增强，验证时保持原始预处理。
train = datasets.ImageFolder(data / "train", transform=transform(True))
val = datasets.ImageFolder(data / "val", transform=transform(False))
train_loader = DataLoader(train, args.batch, shuffle=True, num_workers=0)
val_loader = DataLoader(val, args.batch, shuffle=False, num_workers=0)
```

## 3. SmallFaceCNN 网络

```python
class SmallFaceCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        blocks = []
        channels = [3, 32, 64, 128, 256]

        # 四个卷积块，每次池化使宽高减半：112 → 56 → 28 → 14 → 7。
        for source, target in zip(channels[:-1], channels[1:]):
            blocks.extend([
                nn.Conv2d(source, target, 3, padding=1, bias=False),
                nn.BatchNorm2d(target),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ])

        self.features = nn.Sequential(
            *blocks, nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.embedding = nn.Linear(256, 128)
        self.classifier = nn.Linear(128, num_classes)

    def encode(self, x):
        return self.embedding(self.features(x))  # [B,3,112,112] → [B,128]。

    def forward(self, x):
        return self.classifier(self.encode(x))   # 输出训练身份的分类分数。
```

## 4. 训练、验证与权重保存

```python
def run_epoch(model, loader, criterion, device, optimizer=None):
    # 传入 optimizer 则训练，省略则评估。
    model.train(optimizer is not None)
    total_loss, correct, total = 0.0, 0, 0
    targets, predictions = [], []

    with torch.set_grad_enabled(optimizer is not None):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if optimizer is not None:
                optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)
            if optimizer is not None:
                loss.backward()   # 计算梯度。
                optimizer.step()  # 更新参数。

            pred = logits.argmax(1)
            total_loss += loss.item() * len(labels)
            correct += (pred == labels).sum().item()
            total += len(labels)
            targets.extend(labels.detach().cpu().tolist())
            predictions.extend(pred.detach().cpu().tolist())

    if total == 0:
        raise ValueError("Empty data loader")
    return total_loss / total, correct / total, targets, predictions
```

```python
model = SmallFaceCNN(len(train.classes)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    model.parameters(), lr=args.lr, weight_decay=1e-4,
)

best = -1.0
for epoch in range(1, args.epochs + 1):
    tl, ta, _, _ = run_epoch(model, train_loader, criterion, device, optimizer)
    vl, va, _, _ = run_epoch(model, val_loader, criterion, device)

    if va > best:  # 按验证准确率保存最佳权重。
        best = va
        torch.save(dict(
            state_dict=model.state_dict(), classes=train.classes,
            epoch=epoch, val_acc=va, architecture="SmallFaceCNN_v1",
            input_size=112, normalize_mean=[0.5] * 3,
            normalize_std=[0.5] * 3,
        ), weights)
```

权重保存至 `models/train_cnn/<run>/best.pt`。训练期间验证使用 `transform(False)`；单独运行 val/test 时需加 `--no-auto-gamma` 才使用相同预处理。

## 5. 加载权重与建立真人模板

<a id="digiface-eval-code"></a>

源码：[evaluate_digiface_real.py](../scripts/evaluate_digiface_real.py)。用自训网络的 128 维特征匹配 P01–P08，注册库来自真人训练图及 P02 注册照。

```python
def load_model(device):
    checkpoint = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    model = SmallFaceCNN(len(checkpoint["classes"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, checkpoint["epoch"]


@torch.inference_mode()
def extract_features(model, dataset, device):
    features, labels = [], []
    for images, targets in DataLoader(dataset, batch_size=64, shuffle=False):
        vectors = model.encode(images.to(device))
        features.append(F.normalize(vectors, dim=1).cpu())
        labels.append(targets)
    return torch.cat(features), torch.cat(labels)
```

```python
# build_templates() 节选：逐人计算平均特征，再归一化。
train = ImageFolder(DATA / "train", transform=preprocess)
train_features, train_labels = extract_features(model, train, device)
templates = torch.stack([
    train_features[train_labels == i].mean(dim=0)
    for i in range(len(train.classes))
])
templates = F.normalize(templates, dim=1)

# aligned 为 P02 注册照对齐后的人脸，生成其第二份模板。
rgb = cv.cvtColor(aligned, cv.COLOR_BGR2RGB)
photo = preprocess(Image.fromarray(rgb)).unsqueeze(0).to(device)
with torch.inference_mode():
    p02_extra = F.normalize(model.encode(photo), dim=1).cpu()[0]
```

## 6. 导出摄像头模型与注册库

源码：[prepare_digiface_demo.py](../scripts/prepare_digiface_demo.py)。

```python
device = torch.device("cpu")
model, epoch = load_model(device)
train, templates, p02_extra = build_templates(model, transform(False), device)

# 只导出特征网络，移除合成身份分类头。
encoder = nn.Sequential(model.features, model.embedding).eval()
torch.onnx.export(
    encoder, torch.zeros(1, 3, 112, 112), str(DIGIFACE_ONNX),
    input_names=["face"], output_names=["embedding"],
    opset_version=13, dynamo=False,
)

# 八个身份、九行模板；P02 的两份模板使用同一编号。
features = torch.cat([templates, p02_extra.unsqueeze(0)]).numpy()
np.savez_compressed(
    DIGIFACE_GALLERY, ids=[*train.classes, "P02"], features=features,
)
```

输出：`models/train_cnn/cnn_digiface_s42/embedding.onnx` 与 `work/gallery_digiface.npz`。

## 7. 摄像头模型初始化

源码：[face_live_demo.py](../scripts/face_live_demo.py)。默认 SmallFaceCNN，可用 `--model sface` 切换。

```python
class DigiFaceRecognizer:
    def __init__(self, aligner):
        self.aligner = aligner
        self.net = cv.dnn.readNetFromONNX(str(DIGIFACE_ONNX))

    def alignCrop(self, frame, face):
        return self.aligner.alignCrop(frame, face)

    def feature(self, aligned):
        # BGR → RGB，缩放至 112×112，并与训练预处理保持一致。
        image = cv.dnn.blobFromImage(
            aligned, 1.0 / 127.5, (112, 112), (127.5,) * 3, swapRB=True,
        )
        self.net.setInput(image)
        return self.net.forward()


# 默认模式：YuNet 检测，OpenCV 对齐，SmallFaceCNN 提取特征。
ids, gallery = load_gallery(DIGIFACE_GALLERY)
detector, aligner = engines()
recognizer = DigiFaceRecognizer(aligner)
```

## 8. 摄像头循环与逐帧识别

以下保留摄像头调用顺序，省略状态面板和窗口关闭检测。

```python
orientation = FrameOrientation(args.rotate)
camera = cv.VideoCapture(args.camera)  # 默认 camera=0。
try:
    if not camera.isOpened():
        raise RuntimeError("Cannot open camera")
    while True:
        ok, frame = camera.read()
        if not ok or frame is None:
            break

        annotated, result = recognize(
            frame, detector, recognizer, ids, gallery,
            args.threshold, args.margin, orientation,
        )
        cv.imshow("Face demo", annotated)
        if cv.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
            break
finally:
    camera.release()
    cv.destroyAllWindows()
```

`recognize()` 的核心计算来自 [face_basic.py](../scripts/face_basic.py)，默认相似度门槛为 0.72，身份分差门槛为 0.02。

```python
# 自动模式每帧依次尝试 0°、90°、180°、270°，采用第一个检测到脸的方向。
frame, faces, angle = orientation.apply(frame, detector)

if faces is not None and len(faces) == 1:
    aligned = recognizer.alignCrop(frame, faces[0])
    feature = recognizer.feature(aligned).copy().reshape(-1)
    norm = float(np.linalg.norm(feature))
    if norm > 1e-12 and np.isfinite(feature).all():
        identity, top, gap = choose_identity(
            feature / norm, ids, gallery, threshold, margin,
        )
```

```python
def choose_identity(feature, ids, gallery, threshold, margin):
    scores = gallery @ feature  # 单位向量点积，即余弦相似度。

    if len(set(ids)) != len(ids):
        best_per_id = {}
        for identity, score in zip(ids, scores):
            best_per_id[identity] = max(score, best_per_id.get(identity, -np.inf))
        ids = list(best_per_id)
        scores = np.array(list(best_per_id.values()))

    # 同一人的模板先取最高分，再计算不同身份之间的分差。
    order = np.argsort(scores)
    best = int(order[-1])
    top = float(scores[best])
    gap = top - float(scores[order[-2]]) if len(ids) > 1 else 2.0
    identity = ids[best] if top >= threshold and gap >= margin else "UNKNOWN"
    return identity, top, gap
```

完整 `recognize()` 还处理无脸、多脸、无效特征，并返回带标注画面和识别结果字典。
