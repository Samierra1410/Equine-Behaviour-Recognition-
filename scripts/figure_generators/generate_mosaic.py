"""
generate_mosaic.py — 18-panel pose validation mosaic (9 good + 9 failure cases).
All faces blurred for privacy. Self-contained.

Usage:
    cd <project_root>
    python scripts/figure_generators/generate_mosaic.py
"""

import sys, cv2, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
VIDEO_DIR    = PROJECT_ROOT / 'og_videos'
BORIS_DIR    = PROJECT_ROOT / 'data' / 'boris_csvs'
ONNX_PATH    = PROJECT_ROOT / 'models' / 'ap10k_hrnet_w32.onnx'
YOLO_PATH    = PROJECT_ROOT / 'yolov8n.pt'

COLORS = {
    'yolo_human': (243, 150,  33),
    'yolo_horse': ( 80, 175,  76),
    'mediapipe':  (255, 152,   0),
    'ap10k':      (233,  30,  99),
}
MCOLORS = {k: (v[2]/255, v[1]/255, v[0]/255) for k, v in COLORS.items()}

KP_SKELETON = [
    (0,2),(1,2),(2,3),(3,5),(3,8),(3,4),(5,6),(6,7),(8,9),(9,10),
    (5,11),(8,14),(11,12),(12,13),(14,15),(15,16),(11,4),(14,4),
]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
KP_CONF_THR   = 0.1
_ort  = None
_yolo = None

GOOD_ROWS = [
    {'video': 'Icelandic_05_20_HorseHuman_v3_short', 'label': 'Video B',
     'frames': {'affiliative': 2032, 'neutral': 500, 'avoidant': 2468}},
    {'video': 'Icelandic_05_20_HorseHuman_v8_short', 'label': 'Video C',
     'frames': {'affiliative': 2440, 'neutral': 2120, 'avoidant': 7480}},
    {'video': 'Icelandic_05_20_HorseHuman_v7_short', 'label': 'Video A',
     'frames': {'affiliative': 801, 'neutral': 390, 'avoidant': 929}},
]
BEH_LABELS = {
    'affiliative': 'Human: Active Engagement',
    'neutral':     'Horse: Passive Engagement',
    'avoidant':    'Horse: Leaving/Avoiding Humans',
}
GOOD_VIDEO_NAMES = {r['video'] for r in GOOD_ROWS}
BAD_AFF = ['environment', 'engaging with env']
FAILURE_MODES = [
    'no_mediapipe', 'sparse_kp', 'very_sparse_kp', 'no_horse',
    'no_human', 'low_horse_conf', 'low_human_conf', 'multi_horse',
    'distant', 'both_low_conf',
]
FAILURE_CAPTIONS = {
    'no_mediapipe':   lambda d: "MediaPipe skeleton failed",
    'sparse_kp':      lambda d: f"Sparse equine pose: {d['na']}/17 keypoints",
    'very_sparse_kp': lambda d: f"Very sparse equine pose: {d['na']}/17 keypoints",
    'no_horse':       lambda d: "Horse not detected by YOLOv8",
    'no_human':       lambda d: "Human not detected by YOLOv8",
    'low_horse_conf': lambda d: f"Low horse detection confidence: {d['ho']['conf']:.2f}",
    'low_human_conf': lambda d: f"Low human detection confidence: {d['hu']['conf']:.2f}",
    'multi_horse':    lambda d: f"Multiple horses detected ({d['n_horses']} found)",
    'distant':        lambda d: f"Distant subjects ({d['sz']*100:.1f}% of frame)",
    'both_low_conf':  lambda d: f"Low detection confidence: human {d['hu']['conf']:.2f}, horse {d['ho']['conf']:.2f}",
}


def get_ort():
    global _ort
    if _ort is None:
        import onnxruntime as ort
        _ort = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    return _ort

def get_yolo():
    global _yolo
    if _yolo is None:
        from ultralytics import YOLO
        _yolo = YOLO(str(YOLO_PATH) if YOLO_PATH.exists() else 'yolov8n.pt')
    return _yolo

def run_yolo(frame, conf=0.25):
    results = get_yolo()(frame, verbose=False, conf=conf)
    human = horse = None; n_horses = 0; n_humans = 0; all_human_boxes = []
    for r in results:
        for box in r.boxes:
            cls, c = int(box.cls[0]), float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            if cls == 0:
                n_humans += 1; all_human_boxes.append({'xyxy': xyxy, 'conf': c})
                if human is None or c > human['conf']: human = {'xyxy': xyxy, 'conf': c}
            elif cls == 17:
                n_horses += 1
                if horse is None or c > horse['conf']: horse = {'xyxy': xyxy, 'conf': c}
    return human, horse, n_humans, n_horses, all_human_boxes

def run_mediapipe(frame):
    try:
        import mediapipe as mp
        with mp.solutions.pose.Pose(static_image_mode=True, model_complexity=2, min_detection_confidence=0.4) as pose:
            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            return res.pose_landmarks if res.pose_landmarks else None
    except Exception:
        return None

def run_ap10k(frame, horse_box, padding=0.20):
    if horse_box is None: return None, None
    h_img, w_img = frame.shape[:2]
    x1, y1, x2, y2 = horse_box['xyxy']
    bw, bh = x2-x1, y2-y1; px, py = int(bw*padding), int(bh*padding)
    x1p, y1p = max(0,x1-px), max(0,y1-py)
    x2p, y2p = min(w_img,x2+px), min(h_img,y2+py)
    crop = frame[y1p:y2p, x1p:x2p]
    if crop.size == 0: return None, None
    ch, cw = crop.shape[:2]
    rgb = cv2.cvtColor(cv2.resize(crop,(256,256)), cv2.COLOR_BGR2RGB).astype(np.float32)/255
    tensor = ((rgb-IMAGENET_MEAN)/IMAGENET_STD).transpose(2,0,1)[np.newaxis].astype(np.float32)
    hm = get_ort().run(None, {get_ort().get_inputs()[0].name: tensor})[0][0]
    _, hm_h, hm_w = hm.shape
    kps = np.zeros((17,2), dtype=np.float32); scores = np.zeros(17, dtype=np.float32)
    for i in range(17):
        flat = np.argmax(hm[i]); iy, ix = divmod(int(flat), hm_w)
        kps[i] = [ix/hm_w*cw+x1p, iy/hm_h*ch+y1p]; scores[i] = float(hm[i,iy,ix])
    return kps, scores

def draw_overlays(frame, human_box, horse_box, mp_lms, kps, scores):
    img = frame.copy(); h, w = img.shape[:2]
    for det, color, lbl in [(human_box,COLORS['yolo_human'],'person'),(horse_box,COLORS['yolo_horse'],'horse')]:
        if det:
            x1,y1,x2,y2 = det['xyxy']; cv2.rectangle(img,(x1,y1),(x2,y2),color,3)
            txt = f"{lbl} {det['conf']:.2f}"
            (tw,th),_ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.rectangle(img,(x1,y1-th-8),(x1+tw+4,y1),color,-1)
            cv2.putText(img,txt,(x1+2,y1-4),cv2.FONT_HERSHEY_SIMPLEX,0.65,(255,255,255),2)
    if mp_lms:
        import mediapipe as mp; mpp = mp.solutions.pose
        conns = [(mpp.PoseLandmark.NOSE,mpp.PoseLandmark.LEFT_SHOULDER),(mpp.PoseLandmark.NOSE,mpp.PoseLandmark.RIGHT_SHOULDER),
                 (mpp.PoseLandmark.LEFT_SHOULDER,mpp.PoseLandmark.RIGHT_SHOULDER),(mpp.PoseLandmark.LEFT_SHOULDER,mpp.PoseLandmark.LEFT_ELBOW),
                 (mpp.PoseLandmark.LEFT_ELBOW,mpp.PoseLandmark.LEFT_WRIST),(mpp.PoseLandmark.RIGHT_SHOULDER,mpp.PoseLandmark.RIGHT_ELBOW),
                 (mpp.PoseLandmark.RIGHT_ELBOW,mpp.PoseLandmark.RIGHT_WRIST),(mpp.PoseLandmark.LEFT_SHOULDER,mpp.PoseLandmark.LEFT_HIP),
                 (mpp.PoseLandmark.RIGHT_SHOULDER,mpp.PoseLandmark.RIGHT_HIP),(mpp.PoseLandmark.LEFT_HIP,mpp.PoseLandmark.RIGHT_HIP),
                 (mpp.PoseLandmark.LEFT_HIP,mpp.PoseLandmark.LEFT_KNEE),(mpp.PoseLandmark.RIGHT_HIP,mpp.PoseLandmark.RIGHT_KNEE),
                 (mpp.PoseLandmark.LEFT_KNEE,mpp.PoseLandmark.LEFT_ANKLE),(mpp.PoseLandmark.RIGHT_KNEE,mpp.PoseLandmark.RIGHT_ANKLE)]
        for c in conns:
            l1,l2 = mp_lms.landmark[c[0].value], mp_lms.landmark[c[1].value]
            if l1.visibility>0.3 and l2.visibility>0.3:
                cv2.line(img,(int(l1.x*w),int(l1.y*h)),(int(l2.x*w),int(l2.y*h)),COLORS['mediapipe'],3,cv2.LINE_AA)
        for idx in [0,11,12,13,14,15,16,23,24,25,26,27,28]:
            lm = mp_lms.landmark[idx]
            if lm.visibility>0.3:
                pt=(int(lm.x*w),int(lm.y*h)); cv2.circle(img,pt,5,COLORS['mediapipe'],-1,cv2.LINE_AA); cv2.circle(img,pt,6,(255,255,255),1,cv2.LINE_AA)
    if kps is not None and scores is not None:
        for s,e in KP_SKELETON:
            if scores[s]>KP_CONF_THR and scores[e]>KP_CONF_THR:
                cv2.line(img,tuple(kps[s].astype(int)),tuple(kps[e].astype(int)),COLORS['ap10k'],2,cv2.LINE_AA)
        for i in range(17):
            if scores[i]>KP_CONF_THR:
                pt=tuple(kps[i].astype(int)); cv2.circle(img,pt,5,COLORS['ap10k'],-1,cv2.LINE_AA); cv2.circle(img,pt,6,(255,255,255),1,cv2.LINE_AA)
    return img

def blur_all_faces(frame, all_human_boxes, blur_strength=51):
    if not all_human_boxes: return frame
    img = frame.copy(); h_img, w_img = img.shape[:2]
    for hbox in all_human_boxes:
        x1,y1,x2,y2 = hbox['xyxy']; bh = y2-y1; face_y2 = y1+int(bh*0.45)
        fx1,fy1 = max(0,x1-5),max(0,y1-5); fx2,fy2 = min(w_img,x2+5),min(h_img,face_y2)
        if fy2<=fy1 or fx2<=fx1: continue
        img[fy1:fy2,fx1:fx2] = cv2.GaussianBlur(img[fy1:fy2,fx1:fx2],(blur_strength,blur_strength),30)
    return img

def is_bad_aff(n): return any(t in n.lower() for t in BAD_AFF)
def find_video(vn):
    matches = [f for f in VIDEO_DIR.iterdir() if vn in f.stem]
    return matches[0] if matches else None
def find_boris(vn):
    vc = vn.lower().replace('_','').replace('-','').replace(' ','')
    for bf in BORIS_DIR.glob('*.csv'):
        bc = bf.stem.lower().replace('_','').replace('-','').replace(' ','')
        if vc in bc or bc in vc: return bf
    return None

def parse_boris_all(bf):
    try:
        df = pd.read_csv(bf, encoding='utf-8-sig')
        df.columns = df.columns.str.replace('\ufeff','').str.strip()
        tc=bc=cc=tyc=None
        for col in df.columns:
            cl=col.lower().strip()
            if cl=='time' and 'offset' not in cl: tc=col
            elif cl=='behavior': bc=col
            elif cl=='behavioral category': cc=col
            elif cl=='behavior type': tyc=col
        if not all([tc,bc,cc]): return []
        df[tc]=pd.to_numeric(df[tc],errors='coerce'); df=df.dropna(subset=[tc]).sort_values(tc)
        events,opens=[],{}
        for _,row in df.iterrows():
            ts=float(row[tc]); beh=str(row[bc]).strip(); cat=str(row[cc]).strip().lower()
            evt=str(row.get(tyc,'POINT')).strip().upper() if tyc else 'POINT'
            if cat not in ['affiliative','neutral','avoidant']: continue
            key=(beh,cat)
            if evt=='START': opens[key]=ts
            elif evt=='STOP' and key in opens:
                s=opens.pop(key); events.append({'start':s,'end':ts,'mid':(s+ts)/2,'category':cat,'dur':ts-s,'beh':beh})
            elif evt=='POINT': events.append({'start':ts,'end':ts+1,'mid':ts,'category':cat,'dur':1,'beh':beh})
        return events
    except Exception: return []

def analyze_frame(vf, frame_num):
    cap = cv2.VideoCapture(str(vf)); cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read(); cap.release()
    if not ret: return None
    hu,ho,n_humans,n_horses,all_humans = run_yolo(frame, conf=0.15)
    kps,sc = run_ap10k(frame,ho) if ho else (None,None)
    na = int((sc>0.3).sum()) if sc is not None else 0
    h,w = frame.shape[:2]; dist=None; sz=0.0
    if hu and ho:
        hb,eb = hu['xyxy'],ho['xyxy']
        dist = ((((hb[0]+hb[2])/2-(eb[0]+eb[2])/2)**2+((hb[1]+hb[3])/2-(eb[1]+eb[3])/2)**2)**0.5/((w**2+h**2)**0.5))
        sz = ((hb[2]-hb[0])*(hb[3]-hb[1])+(eb[2]-eb[0])*(eb[3]-eb[1]))/(h*w)
    return {'frame':frame,'hu':hu,'ho':ho,'kps':kps,'sc':sc,'dist':dist,'sz':sz,'na':na,'frame_num':frame_num,'n_horses':n_horses,'n_humans':n_humans,'all_humans':all_humans}

def full_process_frame(vf, frame_num):
    cap = cv2.VideoCapture(str(vf)); cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read(); cap.release()
    if not ret: return None
    hu,ho,n_humans,n_horses,all_humans = run_yolo(frame, conf=0.15)
    mp = run_mediapipe(frame) if hu else None
    kps,sc = run_ap10k(frame,ho) if ho else (None,None)
    na = int((sc>0.3).sum()) if sc is not None else 0
    h,w = frame.shape[:2]; dist=None; sz=0.0
    if hu and ho:
        hb,eb = hu['xyxy'],ho['xyxy']
        dist = ((((hb[0]+hb[2])/2-(eb[0]+eb[2])/2)**2+((hb[1]+hb[3])/2-(eb[1]+eb[3])/2)**2)**0.5/((w**2+h**2)**0.5))
        sz = ((hb[2]-hb[0])*(hb[3]-hb[1])+(eb[2]-eb[0])*(eb[3]-eb[1]))/(h*w)
    return {'frame':frame,'hu':hu,'ho':ho,'mp':mp,'kps':kps,'sc':sc,'dist':dist,'sz':sz,'na':na,'frame_num':frame_num,'n_horses':n_horses,'n_humans':n_humans,'all_humans':all_humans}

def classify_failure(d):
    modes=[]; hu,ho,na,sz = d['hu'],d['ho'],d['na'],d['sz']
    if not ho: modes.append('no_horse')
    if not hu: modes.append('no_human')
    if hu and ho:
        if 0.005<sz<0.08: modes.append('distant')
        if 3<=na<=5: modes.append('sparse_kp')
        if na<=2: modes.append('very_sparse_kp')
        if ho['conf']<0.45: modes.append('low_horse_conf')
        if hu['conf']<0.45: modes.append('low_human_conf')
        if hu['conf']<0.55 and ho['conf']<0.55: modes.append('both_low_conf')
    if d['n_horses']>=2: modes.append('multi_horse')
    return modes


def main():
    print("="*70)
    print("  MOSAIC: 9 Good + 9 Challenging")
    print("="*70)
    print(f"  Project root: {PROJECT_ROOT}")
    if not ONNX_PATH.exists(): print(f"FATAL: ONNX not found at {ONNX_PATH}"); sys.exit(1)

    all_videos = sorted([f for f in VIDEO_DIR.iterdir() if f.suffix.upper() in ['.MP4','.MOV','.AVI']])
    print(f"  Videos found: {len(all_videos)}")

    print("\n--- GOOD FRAMES ---")
    good_data = []
    for row in GOOD_ROWS:
        vf = find_video(row['video'])
        if not vf: print(f"  Not found: {row['video']}"); continue
        label = row['label']; best = {}
        for cat in ['affiliative','neutral','avoidant']:
            d = full_process_frame(vf, row['frames'][cat])
            if not d or not d['hu'] or not d['ho']: print(f"  {label}/{cat}: failed"); continue
            d['frame'] = blur_all_faces(d['frame'], d['all_humans'])
            best[cat] = d
            print(f"  {label} {cat}: frame={row['frames'][cat]} d={d['dist']:.2f} kp={d['na']}/17 blurred={len(d['all_humans'])}")
        if len(best)==3: good_data.append((best, label))

    print("\n--- SCANNING FOR FAILURES ---")
    mode_candidates = {m:[] for m in FAILURE_MODES}
    for vf in all_videos:
        vn = vf.stem
        if vn in GOOD_VIDEO_NAMES: continue
        bf = find_boris(vn)
        if not bf: continue
        events = parse_boris_all(bf)
        if not events: continue
        cap = cv2.VideoCapture(str(vf)); fps = cap.get(cv2.CAP_PROP_FPS); cap.release()
        if fps<=0: continue
        for cat in ['affiliative','neutral','avoidant']:
            cat_events = [e for e in events if e['category']==cat]
            if cat=='affiliative': cat_events=[e for e in cat_events if not is_bad_aff(e['beh'])]
            for evt in sorted(cat_events, key=lambda e:e['dur'], reverse=True)[:3]:
                for frac in [0.5,0.3,0.7]:
                    fn = int((evt['start']+evt['dur']*frac)*fps)
                    d = analyze_frame(vf, fn)
                    if not d: continue
                    for mode in classify_failure(d):
                        score = d['sz'] if d['sz']>0 else 0.01
                        mode_candidates[mode].append((score,vn,vf,fn,evt['beh'],d))

    bad_picks=[]; used_keys=set()
    for mode in FAILURE_MODES:
        if len(bad_picks)>=9: break
        cands = mode_candidates[mode]
        if not cands: continue
        cands.sort(key=lambda x:x[0], reverse=(mode!='distant'))
        for score,vn,vf,fn,beh,d in cands:
            if (vn,fn) not in used_keys:
                used_keys.add((vn,fn)); bad_picks.append((mode,vn,vf,fn,beh))
                print(f"    {mode}: {vn[:35]} frame={fn}"); break
    print(f"  Selected {len(bad_picks)}/9 failures")

    bad_frames=[]
    for mode,vn,vf,fn,beh in bad_picks:
        d = full_process_frame(vf, fn)
        if not d: continue
        if mode=='no_horse' and not d['ho']: d['kps']=None; d['sc']=None; d['na']=0
        if mode=='no_human' and not d['hu']: d['mp']=None
        if d['hu'] and d['ho'] and d['mp'] is None:
            if mode not in ['no_horse','no_human']:
                if 'no_mediapipe' not in [bf[0] for bf in bad_frames]: mode='no_mediapipe'
        d['frame'] = blur_all_faces(d['frame'], d['all_humans'])
        d['beh']=beh; bad_frames.append((mode,d))

    if not good_data: print("\nNo good rows."); return

    print("\n--- RENDERING ---")
    n_good=len(good_data); n_bad=len(bad_frames); n_bad_rows=max((n_bad+2)//3,1); n_rows=n_good+n_bad_rows
    fig_h=5.5*n_rows+1.8
    fig,axes = plt.subplots(n_rows,3,figsize=(19,fig_h))
    if n_rows==1: axes=[axes]

    fig.text(0.5,1-0.008,'Successful Detection Examples',ha='center',va='top',fontsize=16,fontweight='bold',fontstyle='italic',color='#2E7D32')
    if n_bad>0:
        fig.text(0.5,1-((n_good*5.5+0.4)/fig_h),'Challenging / Failure Cases',ha='center',va='top',fontsize=16,fontweight='bold',fontstyle='italic',color='#C62828')

    col_titles=['Affiliative','Neutral','Avoidant']
    for ri,(best,label) in enumerate(good_data):
        for ci,cat in enumerate(['affiliative','neutral','avoidant']):
            ax=axes[ri][ci]; d=best[cat]
            ann=draw_overlays(d['frame'],d['hu'],d['ho'],d['mp'],d['kps'],d['sc'])
            ax.imshow(cv2.cvtColor(cv2.resize(ann,(640,420)),cv2.COLOR_BGR2RGB)); ax.axis('off')
            if ri==0: ax.set_title(col_titles[ci],fontsize=18,fontweight='bold',pad=12)
            if ci==0: ax.set_ylabel(label,fontsize=12,fontweight='bold',rotation=90,labelpad=20)
            for sp in ax.spines.values(): sp.set_edgecolor('#4CAF50');sp.set_linewidth(2);sp.set_visible(True)
            ax.text(0.5,0.02,f"d = {d['dist']:.2f}  |  {BEH_LABELS.get(cat,cat)}",transform=ax.transAxes,ha='center',va='bottom',fontsize=10,color='white',bbox=dict(boxstyle='round,pad=0.3',facecolor='#2E7D32',alpha=0.80))

    for idx,(mode,d) in enumerate(bad_frames):
        ri,ci=n_good+idx//3,idx%3; ax=axes[ri][ci]
        ann=draw_overlays(d['frame'],d['hu'],d['ho'],d.get('mp'),d['kps'],d['sc'])
        ax.imshow(cv2.cvtColor(cv2.resize(ann,(640,420)),cv2.COLOR_BGR2RGB)); ax.axis('off')
        for sp in ax.spines.values(): sp.set_edgecolor('#E53935');sp.set_linewidth(2);sp.set_visible(True)
        caption=FAILURE_CAPTIONS.get(mode,lambda d:mode)(d)
        ax.text(0.5,0.02,caption,transform=ax.transAxes,ha='center',va='bottom',fontsize=10,color='white',bbox=dict(boxstyle='round,pad=0.3',facecolor='#C62828',alpha=0.80))

    for idx in range(n_bad,n_bad_rows*3):
        ri,ci=n_good+idx//3,idx%3; axes[ri][ci].axis('off')
        for sp in axes[ri][ci].spines.values(): sp.set_visible(False)

    leg=[patches.Patch(fc='none',ec=MCOLORS['yolo_human'],lw=2.5,label='YOLOv8 — Person'),
         patches.Patch(fc='none',ec=MCOLORS['yolo_horse'],lw=2.5,label='YOLOv8 — Horse'),
         plt.Line2D([0],[0],marker='o',color=MCOLORS['mediapipe'],ms=8,ls='-',lw=2,label='MediaPipe skeleton'),
         plt.Line2D([0],[0],marker='o',color=MCOLORS['ap10k'],ms=8,ls='-',lw=2,label='AP-10K keypoints')]
    fig.legend(handles=leg,loc='lower center',ncol=4,fontsize=12,frameon=True,fancybox=True,bbox_to_anchor=(0.5,0.003),edgecolor='#CCC')
    plt.tight_layout(rect=[0.05,0.03,1,0.97])

    out_pdf=PROJECT_ROOT/'fig_mosaic_good_bad.pdf'; out_png=PROJECT_ROOT/'fig_mosaic_good_bad.png'
    plt.savefig(str(out_pdf),dpi=300,bbox_inches='tight',facecolor='white')
    plt.savefig(str(out_png),dpi=200,bbox_inches='tight',facecolor='white')
    plt.close()
    print(f"\n  Saved: {out_pdf}\n  Saved: {out_png}")

if __name__ == '__main__':
    main()