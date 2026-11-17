from pathlib import Path
import argparse
import csv
import hashlib
import json
import re
import time

import cv2 as cv
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

DET = ROOT / 'models/face_detection_yunet_2023mar.onnx'
REC = ROOT / 'models/face_recognition_sface_2021dec.onnx'
GAL = ROOT / 'work/gallery.npz'

CFG = dict(window_s=3.0, sample_hz=5, min_frames=3, q_min=0.1,
           quality_version='lap150_dark15_bright245_v1')


def sha(path):
    """传入已有文件的路径（str 或 Path），返回其 SHA256 十六进制字符串。"""

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint():
    """读取 DET、REC、GAL 和 CFG，返回当前离线融合实验的版本字符串。"""

    text = ''.join(sha(p) for p in (DET, REC, GAL))
    text += json.dumps(CFG, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()


def engines():
    """从 models 目录加载两个 OpenCV 模型，返回 (YuNet 检测器, SFace 识别器)。"""

    detector = cv.FaceDetectorYN.create(str(DET), '', (320, 320), 0.9, 0.3, 5000)
    recognizer = cv.FaceRecognizerSF.create(str(REC), '')
    return detector, recognizer


def align_face(image, detector, recognizer):
    """对图中唯一一张人脸进行对齐，无法定位单张脸时返回 None。"""
    if image is None:
        raise ValueError('Image could not be read')
    h, w = image.shape[:2]
    if max(h, w) > 640:
        scale = 640 / max(h, w)
        image = cv.resize(image, (round(w * scale), round(h * scale)))
    h, w = image.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(image)
    if faces is None or len(faces) != 1:
        return None
    return recognizer.alignCrop(image, faces[0])


def embedding(image, detector, recognizer):
    """从一张 BGR 图提取单人的特征和对齐图，返回 (feature, aligned)。"""
    aligned = align_face(image, detector, recognizer)
    if aligned is None:
        return None, None

    feature = recognizer.feature(aligned).copy().reshape(-1)
    norm = float(np.linalg.norm(feature))
    if norm <= 1e-12 or not np.isfinite(feature).all():
        return None, None

    return feature / norm, aligned


def quality(aligned):
    """把 embedding 返回的 aligned 人脸图传进来，得到 0～1 的质量分。"""


    gray = cv.cvtColor(aligned, cv.COLOR_BGR2GRAY)
    clarity = np.clip(cv.Laplacian(gray, cv.CV_64F).var() / 150, 0, 1)
    extreme = np.mean((gray < 15) | (gray > 245))
    return float(clarity * (1 - extreme))


def read_manifest(split, ids):
    """按划分读取离线融合实验的 data/manifest.csv，返回试验字典列表。"""

    required = {'trial_id', 'person_id', 'registered', 'split', 'condition',
                'glasses', 'glare', 'attempt', 'path'}
    with (ROOT / 'data/manifest.csv').open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('Manifest is missing required columns')
        all_rows = list(reader)
    names = [r['trial_id'] for r in all_rows]

    if len(names) != len(set(names)):
        raise ValueError('Duplicate trial_id')
    rows = [r for r in all_rows if r['split'] == split]
    if not rows:
        raise ValueError(f'No rows for split: {split}')
    for r in rows:
        if not re.fullmatch(r'[A-Za-z0-9_-]+', r['trial_id']):
            raise ValueError('Invalid trial_id')
        if r['registered'] not in ('0', '1') or r['glasses'] not in ('0', '1'):
            raise ValueError('registered and glasses must be 0 or 1')
        if (r['person_id'] in ids) != (r['registered'] == '1'):
            raise ValueError(f'Registration label mismatch: {r["trial_id"]}')
        if int(r['attempt']) not in range(1, 6):

            raise ValueError('attempt must be between 1 and 5')
    return rows


def save_csv(path, rows):
    """把非空的字典列表写入 CSV；每条字典对应表中的一行。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def create_cache(row, ids, gallery, detector, recognizer, signature):
    """为清单中的一条试验生成逐帧匹配缓存，函数返回 None。"""

    path = ROOT / row['path']
    cap = cv.VideoCapture(str(path))
    try:
        fps = float(cap.get(cv.CAP_PROP_FPS))
        count = float(cap.get(cv.CAP_PROP_FRAME_COUNT))
        if not cap.isOpened() or not np.isfinite(fps) or fps < CFG['sample_hz']:
            raise ValueError(f'Unreadable video or invalid frame rate: {path}')
        if not np.isfinite(count) or count / fps < CFG['window_s'] - 0.01:
            raise ValueError(f'Video shorter than 3 seconds: {path}')
        scores, qs, times, costs = [], [], [], []
        frame_no = 0
        samples = int(CFG['window_s'] * CFG['sample_hz'])
        while len(scores) < samples:
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f'Decoding stopped before all samples: {path}')

            t = frame_no / fps
            frame_no += 1
            if t + 1e-8 < len(scores) / CFG['sample_hz']:
                continue
            started = time.perf_counter()
            feature, aligned = embedding(frame, detector, recognizer)

            score = np.full(len(ids), np.nan) if feature is None else gallery @ feature
            q = 0.0 if aligned is None else quality(aligned)

            costs.append((time.perf_counter() - started) * 1000)
            scores.append(score)
            qs.append(q)
            times.append(t)
        out = ROOT / 'work/scores' / row['split'] / (row['trial_id'] + '.npz')
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, scores=scores, quality=qs, times=times,
                            processing_ms=costs, ids=ids, fingerprint=signature)
    finally:
        cap.release()


def load_trials(rows, ids, signature):
    """把清单行和对应的缓存配成 (row, arrays) 列表。"""

    trials = []
    for row in rows:
        path = ROOT / 'work/scores' / row['split'] / (row['trial_id'] + '.npz')
        with np.load(path) as z:
            if str(z['fingerprint']) != signature or z['ids'].tolist() != ids:
                raise ValueError(f'Cache fingerprint mismatch: {path}')
            arrays = {k: z[k].copy() for k in ('scores', 'quality', 'times', 'processing_ms')}
        trials.append((row, arrays))
    return trials


def decide(data, method, threshold, margin, ids):
    """给单条视频试验判定一次身份，返回 (编号或 UNKNOWN, 决策秒数)。"""


    used = []
    for i, current in enumerate(data['scores']):
        if not np.isfinite(current).all():

            used = []
            continue
        if method == 'D' and data['quality'][i] < CFG['q_min']:

            continue
        if method == 'B':
            fused = current
        else:
            used.append(i)
            if len(used) < CFG['min_frames']:
                continue
            weights = None if method == 'C' else data['quality'][used]

            fused = np.average(data['scores'][used], axis=0, weights=weights)
        order = np.argsort(fused)
        best, second = int(order[-1]), int(order[-2])

        if fused[best] >= threshold and fused[best] - fused[second] >= margin:
            return ids[best], float(data['times'][i])
    return 'UNKNOWN', CFG['window_s']


def predict(trials, method, params, ids):
    """用同一识别方法为一批视频试验出结果。"""

    out = []
    for row, data in trials:
        pred, t = decide(data, method, params['threshold'], params['margin'], ids)
        known = row['registered'] == '1'
        if known:
            status = 'correct' if pred == row['person_id'] else (
                'reject' if pred == 'UNKNOWN' else 'wrong')
        else:

            status = 'unknown_reject' if pred == 'UNKNOWN' else 'false_accept'
        out.append(dict(row, method=method, prediction=pred, status=status,
                        decision_time_s=t,
                        mean_processing_ms=float(data['processing_ms'].mean())))
    return out


def summarize(rows, method):
    """统计 predict 产生的结果，返回某种方法的指标字典。"""

    selected = [r for r in rows if r['method'] == method]
    known = [r for r in selected if r['registered'] == '1']
    unknown = [r for r in selected if r['registered'] == '0']
    first = [r for r in known if int(r['attempt']) == 1]

    n = lambda group, status: sum(r['status'] == status for r in group)
    rate = lambda value, total: value / total if total else ''
    return dict(method=method, known_n=len(known), unknown_n=len(unknown),
                correct_n=n(known, 'correct'), wrong_n=n(known, 'wrong'),
                reject_n=n(known, 'reject'), false_accept_n=n(unknown, 'false_accept'),
                correct_rate=rate(n(known, 'correct'), len(known)),
                wrong_rate=rate(n(known, 'wrong'), len(known)),
                reject_rate=rate(n(known, 'reject'), len(known)),
                false_accept_rate=rate(n(unknown, 'false_accept'), len(unknown)),
                first_attempt_n=len(first),
                first_attempt_correct_n=n(first, 'correct'),
                first_attempt_correct_rate=rate(n(first, 'correct'), len(first)))


def main():
    """按命令行子命令执行照片比对、注册或旧版的三秒融合实验。"""
    parser = argparse.ArgumentParser()

    sub = parser.add_subparsers(dest='command', required=True)
    smoke = sub.add_parser('smoke')
    smoke.add_argument('image_a')
    smoke.add_argument('image_b')
    sub.add_parser('enroll')
    cache = sub.add_parser('cache')
    cache.add_argument('split', choices=['development', 'test'])
    sub.add_parser('calibrate')
    sub.add_parser('evaluate')
    args = parser.parse_args()
    (ROOT / 'work').mkdir(exist_ok=True)
    if args.command in ('smoke', 'enroll'):
        detector, recognizer = engines()
        if args.command == 'smoke':
            features = []
            for tag, name in [('a', args.image_a), ('b', args.image_b)]:
                feature, aligned = embedding(cv.imread(str(ROOT / name)), detector, recognizer)
                if feature is None:
                    raise ValueError(f'Exactly one detectable face required: {name}')
                cv.imwrite(str(ROOT / f'work/smoke_{tag}.jpg'), aligned)
                features.append(feature)

            print('Cosine similarity:', float(features[0] @ features[1]))
            return
        ids, features = [], []
        for path in sorted((ROOT / 'data/enrollment').glob('*/enroll.jpg')):

            feature, _ = embedding(cv.imread(str(path)), detector, recognizer)
            if feature is None:
                raise ValueError(f'Exactly one detectable face required: {path}')
            ids.append(path.parent.name)
            features.append(feature)
        if len(ids) < 2 or 'UNKNOWN' in ids:

            raise ValueError('Need at least two identities, without reserved name UNKNOWN')

        np.savez_compressed(GAL, ids=ids, features=np.stack(features))
        print('Enrolled:', ids)
        return
    with np.load(GAL) as z:
        ids, gallery = z['ids'].tolist(), z['features'].copy()
    signature = fingerprint()
    split = args.split if args.command == 'cache' else (
        'development' if args.command == 'calibrate' else 'test')

    rows = read_manifest(split, ids)
    if args.command == 'cache':
        detector, recognizer = engines()
        for row in rows:
            create_cache(row, ids, gallery, detector, recognizer, signature)
            print('Cached:', row['trial_id'])
        return
    trials = load_trials(rows, ids, signature)
    if args.command == 'calibrate':

        if {r['registered'] for r in rows} != {'0', '1'}:
            raise ValueError('Development split needs known and unknown participants')
        frozen = dict(settings=CFG, fingerprint=signature, methods={})
        for method in ('B', 'C', 'D'):
            best_key, best = None, None
            for threshold in list(np.round(np.arange(0.2, 1.001, 0.02), 3)) + [1.01]:
                for margin in (0.0, 0.05, 0.1):
                    params = dict(threshold=float(threshold), margin=margin)
                    predictions = predict(trials, method, params, ids)
                    stats = summarize(predictions, method)

                    if stats['wrong_n'] or stats['false_accept_n']:
                        continue
                    key = (stats['correct_n'], float(threshold), margin)

                    if best_key is None or key > best_key:
                        best_key, best = key, dict(params, development_stats=stats)
            frozen['methods'][method] = best
            print(method, best)
        path = ROOT / 'configs/frozen.json'
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(frozen, indent=2), encoding='utf-8')
        return

    frozen = json.loads((ROOT / 'configs/frozen.json').read_text(encoding='utf-8'))
    if frozen['fingerprint'] != signature or frozen['settings'] != CFG:
        raise ValueError('Frozen parameters do not match current models/gallery/settings')
    predictions = []
    for method in ('B', 'C', 'D'):
        predictions.extend(predict(trials, method, frozen['methods'][method], ids))
    summaries = [summarize(predictions, method) for method in ('B', 'C', 'D')]
    save_csv(ROOT / 'outputs/test_predictions.csv', predictions)
    save_csv(ROOT / 'outputs/test_summary.csv', summaries)
    for summary in summaries:
        print(summary)


if __name__ == '__main__':
    main()
