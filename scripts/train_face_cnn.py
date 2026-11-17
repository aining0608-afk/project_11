from pathlib import Path
import argparse
import csv
import json
import math
import random

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parents[1]
FACE_SIZE = (112, 112)
MIN_GAMMA = 0.65
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (4, 4)


class SmallFaceCNN(nn.Module):
    def __init__(self, num_classes):
        """用训练时的身份数构造网络。"""
        super().__init__()
        blocks = []

        channels = [3, 32, 64, 128, 256]
        for source, target in zip(channels[:-1], channels[1:]):


            blocks.extend([nn.Conv2d(source, target, 3, padding=1, bias=False),
                           nn.BatchNorm2d(target), nn.ReLU(), nn.MaxPool2d(2)])

        self.features = nn.Sequential(*blocks, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.embedding = nn.Linear(256, 128)
        self.classifier = nn.Linear(128, num_classes)

    def encode(self, x):
        """把已预处理的图像批次变成一个特征批次。"""

        return self.embedding(self.features(x))

    def forward(self, x):
        """给每张图输出训练类别的原始分数。"""

        return self.classifier(self.encode(x))


def face_brightness(image):
    """传入一张 RGB PIL 图片，返回 0～1 的灰度中位数。"""

    gray = np.asarray(TF.rgb_to_grayscale(image))
    return float(np.median(gray)) / 255.0


def auto_gamma(image, reference):
    """传入 RGB PIL 图与一个参考亮度，返回经过亮度处理的 PIL 图。"""
    brightness = face_brightness(image)

    if brightness >= reference:
        return image


    gamma = math.log(reference) / math.log(max(brightness, 1 / 255))
    return TF.adjust_gamma(image, max(MIN_GAMMA, min(1.0, gamma)))


def enhance_shadows(image):
    """接收一张 PIL 图片，返回 RGB 的 PIL 图片。"""
    import cv2



    rgb = np.asarray(image.convert('RGB'))
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))


def transform(training, gamma_reference=None, clahe=False):
    """创建一个处理图片的流水线，返回值本身可以当函数使用。"""

    steps = [transforms.Resize(FACE_SIZE)]
    if training:

        steps += [transforms.RandomHorizontalFlip(),
                  transforms.ColorJitter(brightness=(0.5,1.2), contrast=0.1)]
    elif clahe:
        steps.append(enhance_shadows)
    elif gamma_reference is not None:
        steps.append(transforms.Lambda(lambda image: auto_gamma(image, gamma_reference)))


    return transforms.Compose(steps + [transforms.ToTensor(),
                              transforms.Normalize([0.5] * 3, [0.5] * 3)])


def run_epoch(model, loader, criterion, device, optimizer=None):
    """用 loader 中的所有批次完成一轮，返回 (平均损失, 准确率, 标签列表, 预测列表)。"""

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
                loss.backward()
                optimizer.step()
            pred = logits.argmax(1)

            total_loss += loss.item() * len(labels)
            correct += (pred == labels).sum().item()
            total += len(labels)

            targets.extend(labels.detach().cpu().tolist())
            predictions.extend(pred.detach().cpu().tolist())
    if total == 0:
        raise ValueError('Empty data loader')
    return total_loss / total, correct / total, targets, predictions


def main():
    """按命令行选训练、验证或测试。"""
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['train', 'val', 'test'], default='train')
    p.add_argument('--run', required=True)
    p.add_argument('--data-root', default='data/cnn')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--auto-gamma', action=argparse.BooleanOptionalAction, default=None,
                   help='Use auto gamma in val and test instead of default CLAHE. '
                        'Use --no-auto-gamma to evaluate original images.')
    args = p.parse_args()


    if args.mode == 'train' and args.auto_gamma:
        p.error('--auto-gamma is for validation and testing with saved weights')
    use_clahe = args.mode != 'train' and args.auto_gamma is None
    args.auto_gamma = bool(args.auto_gamma)
    if not args.run or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.run):
        p.error('Use a simple run name, without directories')
    if args.epochs < 1 or args.batch < 1 or args.lr <= 0:
        p.error('epochs, batch and lr must be positive')
    if args.device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA unavailable; check the WSL training environment')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    data = ROOT / args.data_root

    output = ROOT / 'outputs/training' / args.run
    weights = ROOT / 'models/train_cnn' / args.run / 'best.pt'
    criterion = nn.CrossEntropyLoss()
    if args.mode == 'train':
        if output.exists() or weights.exists():
            p.error('Run already exists; choose a new --run name')

        train = datasets.ImageFolder(data / 'train', transform=transform(True))
        val = datasets.ImageFolder(data / 'val', transform=transform(False))
        if len(train.classes) < 2 or train.class_to_idx != val.class_to_idx:
            raise ValueError('Need >=2 classes, identical class mapping in train and val')
        model = SmallFaceCNN(len(train.classes)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

        train_loader = DataLoader(train, args.batch, shuffle=True, num_workers=0)
        val_loader = DataLoader(val, args.batch, shuffle=False, num_workers=0)
        output.mkdir(parents=True)
        weights.parent.mkdir(parents=True, exist_ok=True)
        settings = dict(vars(args), classes=train.classes, torch_version=str(torch.__version__),
                        device_name=torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU')
        (output / 'settings.json').write_text(json.dumps(settings, indent=2), encoding='utf-8')
        history, best = [], -1.0
        for epoch in range(1, args.epochs + 1):


            tl, ta, _, _ = run_epoch(model, train_loader, criterion, device, optimizer)
            vl, va, _, _ = run_epoch(model, val_loader, criterion, device)
            row = dict(epoch=epoch, train_loss=tl, train_acc=ta, val_loss=vl, val_acc=va)
            history.append(row)
            print(row)
            with (output / 'history.csv').open('w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(history)
            if va > best:

                best = va
                torch.save(dict(state_dict=model.state_dict(), classes=train.classes,
                                epoch=epoch, val_acc=va, architecture='SmallFaceCNN_v1',
                                input_size=112, normalize_mean=[0.5] * 3,
                                normalize_std=[0.5] * 3), weights)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        epochs = [r['epoch'] for r in history]
        for ax, names in zip(axes, [('train_loss', 'val_loss'), ('train_acc', 'val_acc')]):
            for name in names:
                ax.plot(epochs, [r[name] for r in history], label=name)
            ax.set_xlabel('epoch')
            ax.legend()
        fig.tight_layout()
        fig.savefig(output / 'curves.png', dpi=160)
        plt.close(fig)
        print('Best checkpoint:', weights)
        return

    checkpoint = torch.load(weights, map_location='cpu', weights_only=True)
    reference = None
    if use_clahe:
        output = output / 'clahe'
    elif args.auto_gamma:

        reference_data = datasets.ImageFolder(
            data / 'train',
            transform=transforms.Compose([transforms.Resize(FACE_SIZE), face_brightness]),
        )
        reference = float(np.median([brightness for brightness, _ in reference_data]))
        output = output / 'auto_gamma'

    split = args.mode
    evaluation = datasets.ImageFolder(
        data / split, transform=transform(False, reference, clahe=use_clahe),
    )
    if evaluation.classes != checkpoint['classes']:

        raise ValueError(f'{split} class mapping differs from checkpoint')
    model = SmallFaceCNN(len(evaluation.classes)).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    loader = DataLoader(evaluation, args.batch, shuffle=False, num_workers=0)
    loss, acc, targets, predictions = run_epoch(model, loader, criterion, device)

    matrix = np.zeros((len(evaluation.classes), len(evaluation.classes)), dtype=int)
    np.add.at(matrix, (targets, predictions), 1)
    report = {f'{split}_loss': loss, f'{split}_acc': acc, 'samples': len(evaluation),
              'classes': evaluation.classes, 'checkpoint_epoch': checkpoint['epoch']}
    if args.auto_gamma:
        report.update(gamma_reference=reference, min_gamma=MIN_GAMMA)
    if use_clahe:
        report.update(clahe_clip_limit=CLAHE_CLIP_LIMIT, clahe_tile_grid_size=CLAHE_TILE_GRID_SIZE)
    if split == 'val' or args.auto_gamma or use_clahe:
        report['per_class'] = {
            name: {'correct': int(matrix[i, i]), 'total': int(matrix[i].sum())}
            for i, name in enumerate(evaluation.classes)
        }
    output.mkdir(parents=True, exist_ok=True)

    (output / f'{split}_metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    with (output / f'{split}_predictions.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'true_class', 'predicted_class'])
        for (path, _), true, pred in zip(evaluation.samples, targets, predictions):
            writer.writerow([path, evaluation.classes[true], evaluation.classes[pred]])
    matrix_name = 'confusion_matrix.csv' if split == 'test' else 'val_confusion_matrix.csv'
    np.savetxt(output / matrix_name, matrix, fmt='%d', delimiter=',',
               header=','.join(evaluation.classes), comments='')
    print(report)


if __name__ == '__main__':
    main()
