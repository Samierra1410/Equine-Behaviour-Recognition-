"""
generate_fig_sample_frames.py — Figure 5: three-panel affiliative/neutral/avoidant
with YOLOv8, MediaPipe skeleton, AP-10K keypoints. Faces blurred.

Usage:
    cd <project_root>
    python scripts/figure_generators/generate_fig_sample_frames.py
"""

import sys, cv2, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

BASE_DIR   = Path(__file__).resolve().parent.parent.parent
VIDEO_DIR  = BASE_DIR / "og_videos"
ONNX_PATH  = BASE_DIR / "models" / "ap10k_hrnet_w32.onnx"
OUTPUT_DIR = BASE_DIR / "images"
OUTPUT_PDF = OUTPUT_DIR / "fig_sample_frames.pdf"
OUTPUT_PNG = OUTPUT_DIR / "fig_sample_frames.png"

SELECTED_FRAMES = {
    'affiliative': ("Icelandic_05_20_HorseHuman_v7_short.MP4", 801),
    'neutral':     ("Icelandic_05_20_HorseHuman_v5_short.MP4",   0),
    'avoidant':    ("Icelandic_05_20_HorseHuman_v7_short.MP4", 929),
}

KP_NAMES = ['L_Eye','R_Eye','Nose','Neck','root_of_tail','L_Shoulder','L_Elbow','L_F_Paw',
            'R_Shoulder','R_Elbow','R_F_Paw','L_Hip','L_Knee','L_B_Paw','R_Hip','R_Knee','R_B_Paw']
KP_SKELETON = [(0,2),(1,2),(2,3),(3,5),(3,8),(3,4),(5,6),(6,7),(8,9),(9,10),(5,11),(8,14),(11,12),(12,13),(14,15),(15,16),(11,4),(14,4)]
COLORS = {'yolo_human':(243,150,33),'yolo_horse':(80,175,76),'mediapipe':(255,152,0),'ap10k':(233,30,99)}
MCOLORS = {k:(v[2]/255,v[1]/255,v[0]/255) for k,v in COLORS.items()}
IMAGENET_MEAN = np.array([0.485,0.456,0.406],dtype=np.float32)
IMAGENET_STD = np.array([0.229,0.224,0.225],dtype=np.float32)
KP_CONF_THR = 0.1
_ort_session = None
_yolo_model = None


def blur_all_faces(frame, all_human_boxes, blur_strength=51):
    if not all_human_boxes: return frame
    img = frame.copy(); h_img, w_img = img.shape[:2]
    for hbox in all_human_boxes:
        x1,y1,x2,y2 = hbox['xyxy']; bh = y2-y1; face_y2 = y1+int(bh*0.45)
        fx1,fy1 = max(0,x1-5),max(0,y1-5); fx2,fy2 = min(w_img,x2+5),min(h_img,face_y2)
        if fy2<=fy1 or fx2<=fx1: continue
        img[fy1:fy2,fx1:fx2] = cv2.GaussianBlur(img[fy1:fy2,fx1:fx2],(blur_strength,blur_strength),30)
    return img

def get_ort_session():
    global _ort_session
    if _ort_session is not None: return _ort_session
    if not ONNX_PATH.exists(): print(f"ONNX not found: {ONNX_PATH}"); return None
    import onnxruntime as ort
    _ort_session = ort.InferenceSession(str(ONNX_PATH), providers=['CPUExecutionProvider'])
    return _ort_session

def get_yolo():
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        p = BASE_DIR/'yolov8n.pt'; _yolo_model = YOLO(str(p) if p.exists() else 'yolov8n.pt')
    return _yolo_model

def run_yolo(frame):
    results = get_yolo()(frame, verbose=False, conf=0.25)
    human=horse=None; all_humans=[]
    for r in results:
        for box in r.boxes:
            cls,conf = int(box.cls[0]),float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            if cls==0:
                all_humans.append({'xyxy':xyxy,'conf':conf})
                if human is None or conf>human['conf']: human={'xyxy':xyxy,'conf':conf}
            elif cls==17 and (horse is None or conf>horse['conf']): horse={'xyxy':xyxy,'conf':conf}
    return human, horse, all_humans

def run_mediapipe(frame):
    try:
        import mediapipe as mp
        with mp.solutions.pose.Pose(static_image_mode=True,model_complexity=2,min_detection_confidence=0.4) as pose:
            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            return res.pose_landmarks if res.pose_landmarks else None
    except Exception: return None

def run_ap10k(frame, horse_box, padding=0.20):
    if horse_box is None: return None
    sess = get_ort_session()
    if sess is None: return None
    h_img,w_img = frame.shape[:2]; x1,y1,x2,y2 = horse_box['xyxy']
    bw,bh = x2-x1,y2-y1; px,py = int(bw*padding),int(bh*padding)
    x1p,y1p = max(0,x1-px),max(0,y1-py); x2p,y2p = min(w_img,x2+px),min(h_img,y2+py)
    crop = frame[y1p:y2p,x1p:x2p]
    if crop.size==0: return None
    ch,cw = crop.shape[:2]
    rgb = cv2.cvtColor(cv2.resize(crop,(256,256)),cv2.COLOR_BGR2RGB).astype(np.float32)/255
    tensor = ((rgb-IMAGENET_MEAN)/IMAGENET_STD).transpose(2,0,1)[np.newaxis].astype(np.float32)
    hm = sess.run(None,{sess.get_inputs()[0].name:tensor})[0][0]; _,hm_h,hm_w = hm.shape
    kp_dict = {}
    for i,name in enumerate(KP_NAMES):
        flat=np.argmax(hm[i]); iy,ix=divmod(int(flat),hm_w); conf=float(hm[i,iy,ix])
        if conf<KP_CONF_THR: continue
        kp_dict[name] = (int(ix/hm_w*cw+x1p),int(iy/hm_h*ch+y1p))
    return kp_dict if kp_dict else None

def draw_overlays(frame, human_box, horse_box, mp_lms, horse_kps):
    img = frame.copy(); h,w = img.shape[:2]
    for det,color,label in [(human_box,COLORS['yolo_human'],'person'),(horse_box,COLORS['yolo_horse'],'horse')]:
        if det:
            x1,y1,x2,y2 = det['xyxy']; cv2.rectangle(img,(x1,y1),(x2,y2),color,3)
            txt=f"{label} {det['conf']:.2f}"; (tw,th),_=cv2.getTextSize(txt,cv2.FONT_HERSHEY_SIMPLEX,0.7,2)
            cv2.rectangle(img,(x1,y1-th-10),(x1+tw+6,y1),color,-1)
            cv2.putText(img,txt,(x1+3,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
    if mp_lms:
        import mediapipe as mp; mp_pose=mp.solutions.pose
        conns=[(mp_pose.PoseLandmark.NOSE,mp_pose.PoseLandmark.LEFT_SHOULDER),(mp_pose.PoseLandmark.NOSE,mp_pose.PoseLandmark.RIGHT_SHOULDER),
               (mp_pose.PoseLandmark.LEFT_SHOULDER,mp_pose.PoseLandmark.RIGHT_SHOULDER),(mp_pose.PoseLandmark.LEFT_SHOULDER,mp_pose.PoseLandmark.LEFT_ELBOW),
               (mp_pose.PoseLandmark.LEFT_ELBOW,mp_pose.PoseLandmark.LEFT_WRIST),(mp_pose.PoseLandmark.RIGHT_SHOULDER,mp_pose.PoseLandmark.RIGHT_ELBOW),
               (mp_pose.PoseLandmark.RIGHT_ELBOW,mp_pose.PoseLandmark.RIGHT_WRIST),(mp_pose.PoseLandmark.LEFT_SHOULDER,mp_pose.PoseLandmark.LEFT_HIP),
               (mp_pose.PoseLandmark.RIGHT_SHOULDER,mp_pose.PoseLandmark.RIGHT_HIP),(mp_pose.PoseLandmark.LEFT_HIP,mp_pose.PoseLandmark.RIGHT_HIP),
               (mp_pose.PoseLandmark.LEFT_HIP,mp_pose.PoseLandmark.LEFT_KNEE),(mp_pose.PoseLandmark.RIGHT_HIP,mp_pose.PoseLandmark.RIGHT_KNEE),
               (mp_pose.PoseLandmark.LEFT_KNEE,mp_pose.PoseLandmark.LEFT_ANKLE),(mp_pose.PoseLandmark.RIGHT_KNEE,mp_pose.PoseLandmark.RIGHT_ANKLE)]
        for c in conns:
            l1,l2=mp_lms.landmark[c[0].value],mp_lms.landmark[c[1].value]
            if l1.visibility>0.3 and l2.visibility>0.3:
                cv2.line(img,(int(l1.x*w),int(l1.y*h)),(int(l2.x*w),int(l2.y*h)),COLORS['mediapipe'],4,cv2.LINE_AA)
        for idx in [0,11,12,13,14,15,16,23,24,25,26,27,28]:
            lm=mp_lms.landmark[idx]
            if lm.visibility>0.3:
                pt=(int(lm.x*w),int(lm.y*h)); cv2.circle(img,pt,6,COLORS['mediapipe'],-1,cv2.LINE_AA); cv2.circle(img,pt,7,(255,255,255),1,cv2.LINE_AA)
    if horse_kps:
        for i,j in KP_SKELETON:
            a,b=KP_NAMES[i],KP_NAMES[j]
            if a in horse_kps and b in horse_kps: cv2.line(img,horse_kps[a],horse_kps[b],COLORS['ap10k'],3,cv2.LINE_AA)
        for pt in horse_kps.values(): cv2.circle(img,pt,6,COLORS['ap10k'],-1,cv2.LINE_AA); cv2.circle(img,pt,7,(255,255,255),1,cv2.LINE_AA)
    return img


def main():
    print("Generating fig_sample_frames")
    print(f"  Project root: {BASE_DIR}")
    if get_ort_session() is None: sys.exit(1)

    processed = {}
    for cat in ['affiliative','neutral','avoidant']:
        video_name, frame_num = SELECTED_FRAMES[cat]
        print(f"\n[{cat.upper()}] {video_name} frame={frame_num}")
        matches = [f for f in VIDEO_DIR.iterdir() if video_name.replace('.MP4','').replace('.mp4','') in f.stem]
        if not matches: print(f"  Video not found"); continue
        cap = cv2.VideoCapture(str(matches[0])); cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read(); cap.release()
        if not ret: print(f"  Could not read frame"); continue

        human, horse, all_humans = run_yolo(frame)
        print(f"  Person: {round(human['conf'],2) if human else 'N/A'}  Horse: {round(horse['conf'],2) if horse else 'N/A'}")
        frame = blur_all_faces(frame, all_humans)
        print(f"  Blurred {len(all_humans)} face(s)")
        mp_lms = run_mediapipe(frame)
        horse_kps = run_ap10k(frame, horse)
        processed[cat] = draw_overlays(frame, human, horse, mp_lms, horse_kps)

    if not processed: print("\nNo frames processed."); sys.exit(1)

    cats = [c for c in ['affiliative','neutral','avoidant'] if c in processed]
    fig, axes = plt.subplots(1, len(cats), figsize=(6.5*len(cats), 5.5))
    if len(cats)==1: axes=[axes]
    titles = {'affiliative':'(a) Affiliative','neutral':'(b) Neutral','avoidant':'(c) Avoidant'}
    for i,cat in enumerate(cats):
        axes[i].imshow(cv2.cvtColor(processed[cat],cv2.COLOR_BGR2RGB))
        axes[i].set_title(titles[cat],fontsize=14,fontweight='bold',pad=10); axes[i].axis('off')

    legend_handles = [
        patches.Patch(fc='none',ec=MCOLORS['yolo_human'],lw=2.5,label='YOLOv8 — Person'),
        patches.Patch(fc='none',ec=MCOLORS['yolo_horse'],lw=2.5,label='YOLOv8 — Horse'),
        plt.Line2D([0],[0],marker='o',color=MCOLORS['mediapipe'],ms=7,ls='-',lw=1.5,label='MediaPipe skeleton'),
        plt.Line2D([0],[0],marker='o',color=MCOLORS['ap10k'],ms=7,ls='-',lw=1.5,label='AP-10K keypoints'),
    ]
    fig.legend(handles=legend_handles,loc='lower center',ncol=4,fontsize=10,frameon=True,fancybox=True,bbox_to_anchor=(0.5,-0.02),edgecolor='#CCC')
    plt.tight_layout(rect=[0,0.06,1,1])
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    plt.savefig(str(OUTPUT_PDF),dpi=300,bbox_inches='tight',facecolor='white')
    plt.savefig(str(OUTPUT_PNG),dpi=200,bbox_inches='tight',facecolor='white')
    plt.close()
    print(f"\n  Saved: {OUTPUT_PDF}\n  Saved: {OUTPUT_PNG}")

if __name__ == '__main__':
    main()