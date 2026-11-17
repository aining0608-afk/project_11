import argparse

import cv2 as cv
import numpy as np

from face_basic import FrameOrientation, fit_output_frame, recognize
from face_lab import DET, GAL, REC, ROOT, engines


DIGIFACE_ONNX = ROOT / 'models/train_cnn/cnn_digiface_s42/embedding.onnx'
DIGIFACE_GALLERY = ROOT / 'work/gallery_digiface.npz'

MODEL_NAMES = {'smallface': 'SmallFaceCNN', 'sface': 'SFace', 'digiface': 'SmallFaceCNN'}
STATUS_TEXT = {
    'UNKNOWN': 'Unknown person',
    'NO_FACE': 'Step into view',
    'MULTIPLE_FACES': 'One person at a time',
    'INVALID_FEATURE': 'Please try again',
}


class DigiFaceRecognizer:

    def __init__(self, aligner):
        """保存人脸对齐器，并加载 SmallFaceCNN 的 ONNX 模型。"""
        self.aligner = aligner
        self.net = cv.dnn.readNetFromONNX(str(DIGIFACE_ONNX))

    def alignCrop(self, frame, face):
        """根据人脸关键点裁剪并对齐人脸。"""
        return self.aligner.alignCrop(frame, face)

    def feature(self, aligned):

        """预处理对齐的人脸，返回 CNN 提取的特征。"""
        image = cv.dnn.blobFromImage(
            aligned, 1.0 / 127.5, (112, 112), (127.5,) * 3, swapRB=True,
        )
        self.net.setInput(image)
        return self.net.forward()


def load_gallery(path=GAL):
    """读入所选模型的注册库，并按行归一化特征。"""
    with np.load(path, allow_pickle=False) as data:
        ids = data['ids'].tolist()
        features = data['features'].astype(np.float32)
    if not isinstance(ids, list) or not ids or features.shape != (len(ids), 128):
        raise ValueError(f'Expected a 128-D feature for each gallery row: {path}')
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= 1e-12):
        raise ValueError(f'Invalid gallery features: {path}')
    return ids, features / norms


def show_camera(args, detector, recognizer, ids, gallery):
    """打开摄像头，逐帧识别人脸并显示结果。"""
    model_name = MODEL_NAMES[args.model]
    window = f'{model_name} | Live face recognition | Q / Esc to quit'
    orientation = FrameOrientation(args.rotate)
    camera = cv.VideoCapture(args.camera)
    window_opened = False
    try:
        if not camera.isOpened():
            raise RuntimeError(f'Cannot open camera {args.camera}. Close other camera apps or try --camera 1.')

        cv.namedWindow(window, cv.WINDOW_NORMAL)
        window_opened = True
        cv.resizeWindow(window, 960, 720)
        print(f'{model_name} | Camera {args.camera} | {len(set(ids))} enrolled participants | '
              f'Threshold {args.threshold:.3f} | Margin {args.margin:.3f} | Q / Esc to quit')

        while True:
            ok, frame = camera.read()
            if not ok or frame is None or frame.size == 0:
                raise RuntimeError(f'Cannot read frames from camera {args.camera}.')

            annotated, result = recognize(
                frame, detector, recognizer, ids, gallery,
                args.threshold, args.margin, orientation,
            )

            preview = fit_output_frame(annotated, (960, 650))
            panel = np.full((70, 960, 3), (30, 32, 38), dtype=np.uint8)
            identity = result['identity']
            if identity in ids:
                status, color = f'Recognized: {identity}', (130, 240, 140)
            else:
                status, color = STATUS_TEXT[identity], (105, 205, 255)
            score, gap = result['score'], result['gap']
            if identity == 'UNKNOWN':
                reasons = []
                if score < args.threshold:
                    reasons.append('score too low')
                if gap < args.margin:
                    reasons.append('gap too small')
                status += ' | ' + ' and '.join(reasons)
            cv.putText(panel, status, (24, 30), cv.FONT_HERSHEY_SIMPLEX,
                       0.85, color, 2, cv.LINE_AA)
            if score != '':
                details = (f'Score {score:.4f} (min {args.threshold:.4f})  |  '
                           f'Gap {gap:.4f} (min {args.margin:.4f})')
                cv.putText(panel, details, (24, 57), cv.FONT_HERSHEY_SIMPLEX,
                           0.55, (220, 220, 220), 1, cv.LINE_AA)
            cv.putText(panel, 'Q / Esc: exit', (752, 30), cv.FONT_HERSHEY_SIMPLEX,
                       0.65, (220, 220, 220), 1, cv.LINE_AA)
            cv.imshow(window, cv.vconcat([preview, panel]))

            key = cv.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                break

            try:
                if cv.getWindowProperty(window, cv.WND_PROP_VISIBLE) < 1:
                    break
            except cv.error:
                break
    finally:
        camera.release()
        if window_opened:
            cv.destroyAllWindows()


def main():
    """读取命令参数，加载所选模型与注册库，再启动摄像头演示。"""
    parser = argparse.ArgumentParser(description='Live face recognition with YuNet')
    parser.add_argument('--model', type=str.lower, choices=MODEL_NAMES, default='smallface',
                        help='Recognition model, default: smallface; digiface is an alias for smallface')
    parser.add_argument('--camera', type=int, default=0, help='Camera index, default: 0')
    parser.add_argument('--rotate', choices=('auto', '0', '90', '180', '270'), default='auto',
                        help='Automatic orientation or a fixed clockwise angle')
    parser.add_argument('--threshold', type=float,
                        help='Minimum similarity, default: 0.72 (SmallFaceCNN) or 0.363 (SFace)')
    parser.add_argument('--margin', type=float,
                        help='Minimum gap between two identities, default: 0.02 (SmallFaceCNN) or 0 (SFace)')
    args = parser.parse_args()
    use_digiface = args.model in ('smallface', 'digiface')
    if args.threshold is None:
        args.threshold = 0.72 if use_digiface else 0.363
    if args.margin is None:
        args.margin = 0.02 if use_digiface else 0.0
    if args.camera < 0:
        parser.error('--camera must be nonnegative')
    if not -1 <= args.threshold <= 1 or not 0 <= args.margin <= 2:
        parser.error('--threshold must be in [-1,1] and --margin in [0,2]')

    gallery_path = DIGIFACE_GALLERY if use_digiface else GAL
    required = (DET, REC, gallery_path, DIGIFACE_ONNX) if use_digiface else (DET, REC, GAL)
    for path in required:
        if not path.is_file():
            hint = ''
            if use_digiface and path in (DIGIFACE_ONNX, DIGIFACE_GALLERY):
                hint = ' Run scripts/prepare_digiface_demo.py in the WSL training environment.'
            parser.exit(1, f'Missing project file: {path}.{hint}\n')

    try:
        ids, gallery = load_gallery(gallery_path)
        detector, recognizer = engines()
        if use_digiface:
            recognizer = DigiFaceRecognizer(recognizer)
        show_camera(args, detector, recognizer, ids, gallery)
    except KeyboardInterrupt:
        print('\nDemo stopped.')
    except (OSError, ValueError, KeyError, RuntimeError, cv.error) as error:
        parser.exit(1, f'Demo error: {error}\n')


if __name__ == '__main__':
    main()
