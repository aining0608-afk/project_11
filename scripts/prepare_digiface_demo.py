import numpy as np
import torch
from torch import nn

from evaluate_digiface_real import build_templates, load_model, transform
from face_live_demo import DIGIFACE_GALLERY, DIGIFACE_ONNX


def main():
    """导出 SmallFaceCNN 的 ONNX 模型，并生成真人注册库。"""
    torch.set_num_threads(4)
    device = torch.device('cpu')
    model, epoch = load_model(device)
    print(f'Checkpoint epoch: {epoch}', flush=True)
    train, templates, p02_extra = build_templates(model, transform(False), device)


    encoder = nn.Sequential(model.features, model.embedding).eval()
    torch.onnx.export(
        encoder, torch.zeros(1, 3, 112, 112), str(DIGIFACE_ONNX),
        input_names=['face'], output_names=['embedding'], opset_version=13, dynamo=False,
    )
    features = torch.cat([templates, p02_extra.unsqueeze(0)]).numpy()
    DIGIFACE_GALLERY.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DIGIFACE_GALLERY, ids=[*train.classes, 'P02'], features=features)
    print(f'ONNX: {DIGIFACE_ONNX}\nGallery: {DIGIFACE_GALLERY}')


if __name__ == '__main__':
    main()
