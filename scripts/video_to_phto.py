from pathlib import Path
import argparse
import math
import cv2 as cv

from face_basic import ROOT, FrameOrientation


def main():
    """从一个带身份标签的视频生成对齐人脸 JPG。"""
    p = argparse.ArgumentParser()


    p.add_argument('--source', required=True)
    p.add_argument('--split', choices=['train', 'val', 'test'], required=True)
    p.add_argument('--person', required=True)
    p.add_argument('--clip', required=True, help='Unique clip/session identifier')
    args = p.parse_args()

    for name in (args.person, args.clip):
        if not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name):
            p.error('person/clip must use letters, digits, underscore or hyphen')
    out = ROOT / 'data/cnn' / args.split / args.person
    out.mkdir(parents=True, exist_ok=True)

    if list(out.glob(args.clip + '_*.jpg')):
        raise ValueError('This clip already has outputs; use a new clip ID')
    detector = cv.FaceDetectorYN.create(
        str(ROOT / 'models/face_detection_yunet_2023mar.onnx'),
        '', (320, 320), 0.9, 0.3, 5000)

    recognizer = cv.FaceRecognizerSF.create(
        str(ROOT / 'models/face_recognition_sface_2021dec.onnx'), '')
    orientation = FrameOrientation()
    cap = cv.VideoCapture(str(ROOT / args.source))
    if not cap.isOpened():
        raise ValueError('Cannot open source video')

    next_time, previous = 0.0, -1.0

    frames, sampled, saved, skipped = 0, 0, 0, 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            t = float(cap.get(cv.CAP_PROP_POS_MSEC)) / 1000
            if not math.isfinite(t) or t < 0 or (frames > 0 and t <= previous):
                raise ValueError('Video timestamps unavailable; convert to constant frame rate first')
            previous = t
            frames += 1

            if t + 1e-8 < next_time:
                continue
            sampled += 1
            next_time = t + 0.5

            h, w = frame.shape[:2]
            scale = min(1.0, 640 / max(h, w))
            frame = cv.resize(frame, (round(w * scale), round(h * scale)))
            frame, faces, _ = orientation.apply(frame, detector)

            if faces is None or len(faces) != 1:
                skipped += 1
                continue
            aligned = recognizer.alignCrop(frame, faces[0])

            target = out / f'{args.clip}_{saved:04d}.jpg'
            if not cv.imwrite(str(target), aligned):
                raise RuntimeError('Image write failed')
            saved += 1
    finally:
        cap.release()

    print(dict(sampled=sampled, saved=saved, no_single_face=skipped, output=str(out)))


if __name__ == '__main__':
    main()
