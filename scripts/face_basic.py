from pathlib import Path
import argparse
import csv

import cv2 as cv
import numpy as np



ROOT = Path(__file__).resolve().parents[1]


def rotate_frame(frame, degrees):
    """把 OpenCV 的一帧按指定角度顺时针旋转后返回。"""

    rotations = {90: cv.ROTATE_90_CLOCKWISE, 180: cv.ROTATE_180,
                 270: cv.ROTATE_90_COUNTERCLOCKWISE}
    return frame if degrees == 0 else cv.rotate(frame, rotations[degrees])


class FrameOrientation:

    def __init__(self, mode='auto'):
        """设置自动尝试的旋转角度，或使用指定的固定角度。"""
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

            if len(self.angles) == 1 or (faces is not None and len(faces) > 0):
                return oriented, faces, angle
        return frame, None, 0


def fit_output_frame(frame, size):
    """把已经画好标注的一帧调整为固定的输出分辨率。"""
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
    """给定归一化的人脸特征与注册库，返回 (身份, 最高分, 领先第二名的分差)。"""


    scores = gallery @ feature
    if len(set(ids)) != len(ids):

        best_per_id = {}
        for identity, score in zip(ids, scores):
            best_per_id[identity] = max(score, best_per_id.get(identity, -np.inf))
        ids = list(best_per_id)
        scores = np.array(list(best_per_id.values()))
    order = np.argsort(scores)


    best = int(order[-1])
    top = float(scores[best])


    gap = top - float(scores[order[-2]]) if len(ids) > 1 else 2.0


    identity = ids[best] if top >= threshold and gap >= margin else 'UNKNOWN'
    return identity, top, gap


def recognize(frame, detector, recognizer, ids, gallery, threshold, margin,
              orientation=None):
    """处理一帧，返回 (带框和文字的 BGR 画面, 本帧的识别结果字典)。"""

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
    """从命令行读取输入源、注册库和识别参数，然后保存识别结果。"""


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
