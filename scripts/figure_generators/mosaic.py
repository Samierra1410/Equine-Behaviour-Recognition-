import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

BASE_DIR   = Path(__file__).resolve().parent
VIDEO_DIR  = BASE_DIR / "og_videos"
ONNX_PATH  = BASE_DIR / "ap10k_hrnet_w32.onnx"
OUTPUT_PDF = BASE_DIR / "images" / "fig_mosaic_final.pdf"
OUTPUT_PNG = BASE_DIR / "images" / "fig_mosaic_final.png"

ROWS = [
    ("Icelandic_05_20_HorseHuman_v3_short", "Video B"),
    ("Icelandic_05_20_HorseHuman_v8_short", "Video C"),
    ("Icelandic_05_20_HorseHuman_v7_short", "Video A"),
]

HARDCODED = {
    "Video B": {
        "affiliative": 2032,
        "neutral":     500,
        "avoidant":    2468,
    },
    "Video C": {
        "affiliative": 2440,
        "neutral":     2120,
        "avoidant":    7480,
    },
    "Video A": {
        "affiliative": 801,
        "neutral":     0,
        "avoidant":    929,
    },
}

BEH_LABELS = {
    "affiliative": "Human: Active Engagement",
    "neutral":     "Horse: Passive Engagement",
    "avoidant":    "Horse: Leaving/Avoiding Humans",
}

ZOOM_ROWS = {"Video B"}
ZOOM_PAD  = 0.40

TARGET_H, TARGET_W = 420, 640

COLORS = {
    'yolo_human': (243, 150,  33),
    'yolo_horse': ( 80, 175,  76),
    'mediapipe':  (255, 152,   0),
    'ap10k':      (233,  30,  99),
}
MCOLORS = {k: (v[2]/255, v[1]/255, v[0]/255) for k, v in COLORS.items()}

KP_SKELETON = [
    (0,2),(1,2),(2,3),(3,5),(3,8),(3,4),
    (5,6),(6,7),(8,9),(9,10),
    (5,11),(8,14),(11,12),(12,13),(14,15),(15,16),(11,4),(14,4),
]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
KP_CONF_THR   = 0.1

_ort  = None
_yolo = None


def get_ort():
    global _ort
    if _ort is None:
        import onnxruntime as ort
        _ort = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
        print(f"  ONNX loaded: {ONNX_PATH.name}")
    return _ort


def get_yolo():
    global _yolo
    if _yolo is None:
        from ultralytics import YOLO
        p = BASE_DIR / 'yolov8n.pt'
        _yolo = YOLO(str(p) if p.exists() else 'yolov8n.pt')
    return _yolo


def run_yolo(frame):
    results = get_yolo()(frame, verbose=False, conf=0.25)
    human = horse = None
    for r in results:
        for box in r.boxes:
            cls, conf = int(box.cls[0]), float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            if cls == 0 and (human is None or conf > human['conf']):
                human = {'xyxy': xyxy, 'conf': conf}
            elif cls == 17 and (horse is None or conf > horse['conf']):
                horse = {'xyxy': xyxy, 'conf': conf}
    return human, horse


def run_mediapipe(frame):
    try:
        import mediapipe as mp
        with mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.4
        ) as pose:
            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            return res.pose_landmarks if res.pose_landmarks else None
    except Exception:
        return None


def run_ap10k(frame, horse_box, padding=0.20):
    if horse_box is None:
        return None, None
    h_img, w_img = frame.shape[:2]
    x1, y1, x2, y2 = horse_box['xyxy']
    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * padding), int(bh * padding)
    x1p = max(0, x1 - px); y1p = max(0, y1 - py)
    x2p = min(w_img, x2 + px); y2p = min(h_img, y2 + py)
    crop = frame[y1p:y2p, x1p:x2p]
    if crop.size == 0:
        return None, None
    ch, cw = crop.shape[:2]
    rgb = cv2.cvtColor(cv2.resize(crop, (256, 256)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    tensor = ((rgb - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)[np.newaxis].astype(np.float32)
    sess = get_ort()
    hm = sess.run(None, {sess.get_inputs()[0].name: tensor})[0][0]
    _, hm_h, hm_w = hm.shape
    kps = np.zeros((17, 2), dtype=np.float32)
    scores = np.zeros(17, dtype=np.float32)
    for i in range(17):
        flat = np.argmax(hm[i])
        iy, ix = divmod(int(flat), hm_w)
        kps[i] = [ix / hm_w * cw + x1p, iy / hm_h * ch + y1p]
        scores[i] = float(hm[i, iy, ix])
    return kps, scores


def draw_overlays(frame, human_box, horse_box, mp_lms, kps, scores):
    img = frame.copy()
    h, w = img.shape[:2]

    for det, color, lbl in [
        (human_box, COLORS['yolo_human'], 'person'),
        (horse_box, COLORS['yolo_horse'], 'horse'),
    ]:
        if det:
            x1, y1, x2, y2 = det['xyxy']
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            txt = f"{lbl} {det['conf']:.2f}"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(img, txt, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    if mp_lms:
        import mediapipe as mp
        mpp = mp.solutions.pose
        conns = [
            (mpp.PoseLandmark.NOSE,           mpp.PoseLandmark.LEFT_SHOULDER),
            (mpp.PoseLandmark.NOSE,           mpp.PoseLandmark.RIGHT_SHOULDER),
            (mpp.PoseLandmark.LEFT_SHOULDER,  mpp.PoseLandmark.RIGHT_SHOULDER),
            (mpp.PoseLandmark.LEFT_SHOULDER,  mpp.PoseLandmark.LEFT_ELBOW),
            (mpp.PoseLandmark.LEFT_ELBOW,     mpp.PoseLandmark.LEFT_WRIST),
            (mpp.PoseLandmark.RIGHT_SHOULDER, mpp.PoseLandmark.RIGHT_ELBOW),
            (mpp.PoseLandmark.RIGHT_ELBOW,    mpp.PoseLandmark.RIGHT_WRIST),
            (mpp.PoseLandmark.LEFT_SHOULDER,  mpp.PoseLandmark.LEFT_HIP),
            (mpp.PoseLandmark.RIGHT_SHOULDER, mpp.PoseLandmark.RIGHT_HIP),
            (mpp.PoseLandmark.LEFT_HIP,       mpp.PoseLandmark.RIGHT_HIP),
            (mpp.PoseLandmark.LEFT_HIP,       mpp.PoseLandmark.LEFT_KNEE),
            (mpp.PoseLandmark.RIGHT_HIP,      mpp.PoseLandmark.RIGHT_KNEE),
            (mpp.PoseLandmark.LEFT_KNEE,      mpp.PoseLandmark.LEFT_ANKLE),
            (mpp.PoseLandmark.RIGHT_KNEE,     mpp.PoseLandmark.RIGHT_ANKLE),
        ]
        for c in conns:
            l1 = mp_lms.landmark[c[0].value]
            l2 = mp_lms.landmark[c[1].value]
            if l1.visibility > 0.3 and l2.visibility > 0.3:
                cv2.line(img,
                         (int(l1.x * w), int(l1.y * h)),
                         (int(l2.x * w), int(l2.y * h)),
                         COLORS['mediapipe'], 3, cv2.LINE_AA)
        for idx in [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:
            lm = mp_lms.landmark[idx]
            if lm.visibility > 0.3:
                pt = (int(lm.x * w), int(lm.y * h))
                cv2.circle(img, pt, 5, COLORS['mediapipe'], -1, cv2.LINE_AA)
                cv2.circle(img, pt, 6, (255, 255, 255), 1, cv2.LINE_AA)

    if kps is not None and scores is not None:
        for s, e in KP_SKELETON:
            if scores[s] > KP_CONF_THR and scores[e] > KP_CONF_THR:
                cv2.line(img,
                         tuple(kps[s].astype(int)),
                         tuple(kps[e].astype(int)),
                         COLORS['ap10k'], 2, cv2.LINE_AA)
        for i in range(17):
            if scores[i] > KP_CONF_THR:
                pt = tuple(kps[i].astype(int))
                cv2.circle(img, pt, 5, COLORS['ap10k'], -1, cv2.LINE_AA)
                cv2.circle(img, pt, 6, (255, 255, 255), 1, cv2.LINE_AA)

    return img


def compute_shared_crop(results_list, frame_shape, pad=ZOOM_PAD):
    h_img, w_img = frame_shape[:2]
    xs1, ys1, xs2, ys2 = [], [], [], []
    for res in results_list:
        for det in [res[2], res[3]]:
            if det:
                x1, y1, x2, y2 = det['xyxy']
                xs1.append(x1); ys1.append(y1)
                xs2.append(x2); ys2.append(y2)
    if not xs1:
        return None
    x1, y1, x2, y2 = min(xs1), min(ys1), max(xs2), max(ys2)
    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * pad), int(bh * pad)
    return (max(0, x1 - px), max(0, y1 - py), min(w_img, x2 + px), min(h_img, y2 + py))


def apply_crop(img, box):
    if box is None:
        return img
    cx1, cy1, cx2, cy2 = box
    return img[cy1:cy2, cx1:cx2]


def process_hardcoded(vf, frame_num, cat, label):
    cap = cv2.VideoCapture(str(vf))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"  Could not read frame {frame_num}")
        return None

    human, horse = run_yolo(frame)
    if human is None or horse is None:
        print(f"  Detection failed at frame {frame_num}")
        return None

    mp_lms = run_mediapipe(frame)
    kps, kp_sc = run_ap10k(frame, horse)
    n_kp = int((kp_sc > KP_CONF_THR).sum()) if kp_sc is not None else 0

    h_img, w_img = frame.shape[:2]
    hb, eb = human['xyxy'], horse['xyxy']
    hcx = (hb[0] + hb[2]) / 2; hcy = (hb[1] + hb[3]) / 2
    ecx = (eb[0] + eb[2]) / 2; ecy = (eb[1] + eb[3]) / 2
    dist = ((hcx - ecx) ** 2 + (hcy - ecy) ** 2) ** 0.5 / ((w_img ** 2 + h_img ** 2) ** 0.5)
    sz   = ((hb[2] - hb[0]) * (hb[3] - hb[1]) + (eb[2] - eb[0]) * (eb[3] - eb[1])) / (h_img * w_img)

    beh = BEH_LABELS.get(cat, cat)
    print(f"  {cat}: frame={frame_num}  d={dist:.2f}  kp={n_kp}/17")
    return (0.0, frame, human, horse, mp_lms, kps, kp_sc, dist, sz, n_kp, beh, float(frame_num))


def main():
    print("=" * 70)
    print(" Generating mosaic — AP-10K ONNX inference")
    print("=" * 70)
    get_ort()
    all_row_data = []

    for vname, label in ROWS:
        print(f"\n{'─' * 60}")
        print(f"  Row: {label}  ({vname})")

        matches = [f for f in VIDEO_DIR.iterdir() if vname in f.stem]
        if not matches:
            print(f"  Video not found: {vname}")
            continue
        vf = matches[0]

        best = {}
        hc = HARDCODED.get(label, {})
        for cat in ['affiliative', 'neutral', 'avoidant']:
            fn = hc.get(cat)
            if fn is None:
                print(f"  No hardcoded frame for {label}/{cat}")
                continue
            result = process_hardcoded(vf, fn, cat, label)
            if result:
                best[cat] = result

        missing = [c for c in ['affiliative', 'neutral', 'avoidant'] if c not in best]
        if missing:
            print(f"  Skipping {label}: missing {missing}")
            continue
        all_row_data.append((best, label))

    if not all_row_data:
        print("\nNo rows to render.")
        return

    n_rows = len(all_row_data)
    fig, axes = plt.subplots(n_rows, 3, figsize=(19, 6.5 * n_rows))
    if n_rows == 1:
        axes = [axes]
    col_titles = ['Affiliative', 'Neutral', 'Avoidant']

    for ri, (best, label) in enumerate(all_row_data):
        do_zoom = label in ZOOM_ROWS
        shared_crop = None
        if do_zoom:
            row_results = [best[c] for c in ['affiliative', 'neutral', 'avoidant']]
            shared_crop = compute_shared_crop(row_results, row_results[0][1].shape, pad=ZOOM_PAD)

        for ci, cat in enumerate(['affiliative', 'neutral', 'avoidant']):
            ax = axes[ri][ci]
            sc, frame, hu, ho, mp_lms, kps, kp_sc, dist, sz, n_kp, beh, ts = best[cat]

            annotated = draw_overlays(frame, hu, ho, mp_lms, kps, kp_sc)
            if do_zoom and shared_crop is not None:
                annotated = apply_crop(annotated, shared_crop)

            panel = cv2.resize(annotated, (TARGET_W, TARGET_H))
            ax.imshow(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
            ax.axis('off')

            if ri == 0:
                ax.set_title(col_titles[ci], fontsize=16, fontweight='bold', pad=10)
            if ci == 0:
                ax.set_ylabel(label, fontsize=11, fontweight='bold', rotation=90, labelpad=18)

            ax.text(0.5, 0.02, f"d = {dist:.2f}  |  {beh[:35]}",
                    transform=ax.transAxes, ha='center', va='bottom',
                    fontsize=10, color='white',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.72))

    legend_handles = [
        patches.Patch(fc='none', ec=MCOLORS['yolo_human'], lw=2.5, label='YOLOv8 — Person'),
        patches.Patch(fc='none', ec=MCOLORS['yolo_horse'], lw=2.5, label='YOLOv8 — Horse'),
        plt.Line2D([0], [0], marker='o', color=MCOLORS['mediapipe'], ms=7, ls='-', lw=1.5,
                   label='MediaPipe skeleton'),
        plt.Line2D([0], [0], marker='o', color=MCOLORS['ap10k'], ms=7, ls='-', lw=1.5,
                   label='AP-10K keypoints (real)'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=4, fontsize=12,
               frameon=True, fancybox=True, bbox_to_anchor=(0.5, 0.005), edgecolor='#CCC')
    plt.tight_layout(rect=[0.06, 0.04, 1, 0.98])

    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(OUTPUT_PDF), dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(str(OUTPUT_PNG), dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"\n{'=' * 70}")
    print(f"  Saved: {OUTPUT_PDF}")
    print(f"  Saved: {OUTPUT_PNG}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()