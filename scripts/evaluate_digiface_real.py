import csv

import cv2 as cv
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from face_lab import align_face, engines
from train_face_cnn import ROOT, SmallFaceCNN, transform

RUN = 'cnn_digiface_s42'
WEIGHTS = ROOT / 'models/train_cnn' / RUN / 'best.pt'
DATA = ROOT / 'data/cnn'

OUTPUT = ROOT / 'outputs/training' / RUN / 'p02_extra_template'


@torch.inference_mode()
def extract_features(model, dataset, device):
    """用已有 CNN 给一个数据集逐图提特征，返回 (features, labels)。"""

    features, labels = [], []
    for images, targets in DataLoader(dataset, batch_size=64, shuffle=False):

        vectors = model.encode(images.to(device))
        features.append(F.normalize(vectors, dim=1).cpu())
        labels.append(targets)

    return torch.cat(features), torch.cat(labels)


def load_model(device):
    """加载自训 CNN 权重，并切换到评估模式。"""
    checkpoint = torch.load(WEIGHTS, map_location='cpu', weights_only=True)


    model = SmallFaceCNN(len(checkpoint['classes']))
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device).eval()
    return model, checkpoint['epoch']


def build_templates(model, preprocess, device):
    """用每人的真人训练图建平均模板，并为 P02 添加一份注册照模板。"""
    train = ImageFolder(DATA / 'train', transform=preprocess)
    people = train.classes
    train_features, train_labels = extract_features(model, train, device)


    templates = torch.stack([
        train_features[train_labels == i].mean(dim=0) for i in range(len(people))
    ])
    templates = F.normalize(templates, dim=1)


    photo_path = ROOT / 'data/enrollment/P02/enroll.jpg'
    image = cv.imread(str(photo_path))
    if image is None:
        raise ValueError(f'Could not read P02 enrollment photo: {photo_path}')
    detector, aligner = engines()
    aligned = align_face(image, detector, aligner)
    if aligned is None:
        raise ValueError(f'Could not align P02 enrollment photo: {photo_path}')


    rgb = cv.cvtColor(aligned, cv.COLOR_BGR2RGB)

    photo = preprocess(Image.fromarray(rgb)).unsqueeze(0).to(device)
    with torch.inference_mode():
        p02_extra = F.normalize(model.encode(photo), dim=1).cpu()[0]
    print(f'P02 extra template: {photo_path}', flush=True)
    return train, templates, p02_extra


def main():
    """在 .venv-train 中，用真人 train 建模板，用 val 和 test 分别评估。"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, epoch = load_model(device)
    print(f'Checkpoint epoch: {epoch} | Device: {device}', flush=True)
    preprocess = transform(False)
    train, templates, p02_extra = build_templates(model, preprocess, device)
    people = train.classes
    p02_index = people.index('P02')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for split in ('val', 'test'):
        dataset = ImageFolder(DATA / split, transform=preprocess)

        if dataset.class_to_idx != train.class_to_idx:
            raise ValueError(f'{split} identity folders do not match the training set')

        features, labels = extract_features(model, dataset, device)

        scores = features @ templates.T

        scores[:, p02_index] = torch.maximum(
            scores[:, p02_index], features @ p02_extra
        )
        predictions = scores.argmax(dim=1)
        correct = (predictions == labels).sum().item()
        print(f'\n{split}: {correct}/{len(dataset)} ({correct / len(dataset):.2%})')


        matrix = torch.zeros((len(people), len(people)), dtype=torch.int64)
        for actual, predicted in zip(labels.tolist(), predictions.tolist()):
            matrix[actual, predicted] += 1
        for i, person in enumerate(people):
            print(f'{person}: {matrix[i, i].item()}/{matrix[i].sum().item()}')

        false_p02 = (matrix[:, p02_index].sum() - matrix[p02_index, p02_index]).item()
        other_people = len(dataset) - matrix[p02_index].sum().item()
        print(f'Other identities predicted as P02: {false_p02}/{other_people}')

        path = OUTPUT / f'{split}_confusion_matrix.csv'
        with path.open('w', newline='', encoding='utf-8-sig') as file:
            writer = csv.writer(file)
            writer.writerow(['true / predicted', *people])
            for person, row in zip(people, matrix.tolist()):
                writer.writerow([person, *row])
        print(f'Confusion matrix: {path}', flush=True)


if __name__ == '__main__':
    main()
