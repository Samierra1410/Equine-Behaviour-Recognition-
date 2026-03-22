import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import pandas as pd

print("="*80)
print("AP-10K HORSE POSE EXTRACTION (WORKING)")
print("="*80)

# Check imports
try:
    from mmpose.apis import init_model, inference_topdown
    from mmpose.structures import merge_data_samples
    print("✓ mmpose imported")
except ImportError as e:
    print(f"❌ mmpose import failed: {e}")
    exit(1)

BASE_DIR = Path(__file__).resolve().parent.parent
VIDEO_DIR = BASE_DIR / "og_videos"
BORIS_DIR = BASE_DIR / "data/boris_csvs"
OUTPUT_DIR = BASE_DIR / "data/horse_pose_features"

OUTPUT_DIR.mkdir(exist_ok=True)

# Model paths
MODEL_PATH = BASE_DIR / "models" / "hrnet_w32_ap10k.pth"

# Create proper config
CONFIG_PATH = BASE_DIR / "ap10k_proper_config.py"

print("\n📝 Creating proper config...")

config_content = """
# AP-10K HRNet Config with test_dataloader
dataset_type = 'AP10KDataset'
data_mode = 'topdown'
data_root = 'data/ap10k/'

# codec settings
codec = dict(
    type='MSRAHeatmap',
    input_size=(256, 256),
    heatmap_size=(64, 64),
    sigma=2)

# model settings
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='HRNet',
        in_channels=3,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(32, 64)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(32, 64, 128)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(32, 64, 128, 256)))),
    head=dict(
        type='HeatmapHead',
        in_channels=32,
        out_channels=17,
        deconv_out_channels=None,
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=codec),
    test_cfg=dict(
        flip_test=True,
        flip_mode='heatmap',
        shift_heatmap=True))

# test dataloader (REQUIRED!)
test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/ap10k-val-split1.json',
        data_prefix=dict(img='data/'),
        test_mode=True,
        pipeline=[
            dict(type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(type='TopdownAffine', input_size=codec['input_size']),
            dict(type='PackPoseInputs')
        ]))

# test evaluator
test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/ap10k-val-split1.json')
"""

with open(CONFIG_PATH, 'w') as f:
    f.write(config_content)

print(f"✓ Config created: {CONFIG_PATH}")

# Load model
print("\n📦 Loading AP-10K model...")
try:
    model = init_model(str(CONFIG_PATH), str(MODEL_PATH), device='cpu')
    print("✅ AP-10K loaded!")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    exit(1)

# BORIS loading
def load_boris_events(boris_file):
    """Load BORIS timestamps"""
    try:
        df = pd.read_csv(boris_file, encoding='utf-8-sig')
        df.columns = df.columns.str.strip()
        
        time_col = None
        for col in df.columns:
            if 'time' in col.lower() and 'offset' in col.lower():
                time_col = col
                break
        
        if not time_col:
            return []
        
        timestamps = pd.to_numeric(df[time_col], errors='coerce')
        timestamps = timestamps.dropna()
        timestamps = timestamps[timestamps >= 0]
        
        return sorted(timestamps.values)
    except:
        return []

def extract_horse_features(pose_result):
    """Extract features from AP-10K pose"""
    
    if len(pose_result) == 0:
        return None
    
    pred_instances = pose_result[0].pred_instances
    
    if len(pred_instances) == 0:
        return None
    
    # Get keypoints and scores
    keypoints = pred_instances.keypoints[0]  # (17, 2)
    scores = pred_instances.keypoint_scores[0]  # (17,)
    
    features = {}
    
    # Only use confident keypoints
    conf_threshold = 0.3
    valid_kpts = scores > conf_threshold
    
    if valid_kpts.sum() < 3:
        return None
    
    # Extract key features
    # 0: left_eye, 1: right_eye, 2: nose, 3: neck, 4: root_of_tail
    
    if scores[2] > conf_threshold and scores[3] > conf_threshold:
        # Head angle
        head_vec = keypoints[2] - keypoints[3]
        features['head_angle'] = float(np.arctan2(head_vec[1], head_vec[0]))
        features['head_lowered'] = float(keypoints[2][1] > keypoints[3][1])
    
    if scores[0] > conf_threshold and scores[1] > conf_threshold:
        # Ear distance (proxy for ear position)
        ear_dist = np.linalg.norm(keypoints[0] - keypoints[1])
        features['ear_distance'] = float(ear_dist)
    
    if scores[3] > conf_threshold and scores[4] > conf_threshold:
        # Body posture
        body_vec = keypoints[4] - keypoints[3]
        features['body_angle'] = float(np.arctan2(body_vec[1], body_vec[0]))
        features['body_length'] = float(np.linalg.norm(body_vec))
    
    if scores[4] > conf_threshold:
        # Tail height
        features['tail_y'] = float(keypoints[4][1])
    
    # Overall confidence
    features['pose_confidence'] = float(scores.mean())
    features['n_keypoints'] = int(valid_kpts.sum())
    
    return features

# Find BORIS file
def find_boris_file(video_name, boris_files):
    v_clean = video_name.lower().replace('_', '').replace('-', '')
    
    for f in boris_files:
        if video_name.lower() == f.stem.lower():
            return f
    
    for f in boris_files:
        f_clean = f.stem.lower().replace('_', '').replace('-', '')
        if v_clean in f_clean or f_clean in v_clean:
            return f
    
    return None

# Process videos
video_files = list(VIDEO_DIR.glob("*.mp4")) + list(VIDEO_DIR.glob("*.MP4"))
boris_files = list(BORIS_DIR.glob("*.csv"))

print(f"\n✓ {len(video_files)} videos, {len(boris_files)} BORIS files")

WINDOW_BEFORE = 2.0
WINDOW_AFTER = 2.0
SAMPLE_RATE = 5

total_samples = 0
videos_processed = 0

for video_file in tqdm(video_files, desc="Processing"):
    
    # Find BORIS
    boris_file = find_boris_file(video_file.stem, boris_files)
    if not boris_file:
        continue
    
    boris_timestamps = load_boris_events(boris_file)
    if not boris_timestamps:
        continue
    
    # Open video
    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        tqdm.write(f"❌ {video_file.name}: Can't open")
        continue
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    all_features = []
    
    for boris_time in boris_timestamps:
        center_frame = int(boris_time * fps)
        start_frame = max(0, int((boris_time - WINDOW_BEFORE) * fps))
        end_frame = int((boris_time + WINDOW_AFTER) * fps)
        
        for frame_idx in range(start_frame, end_frame, SAMPLE_RATE):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            try:
                # Run inference
                pose_result = inference_topdown(model, frame)
                
                # Extract features
                horse_features = extract_horse_features(pose_result)
                
                if horse_features:
                    horse_features['frame'] = frame_idx
                    horse_features['timestamp'] = float(frame_idx / fps)
                    horse_features['boris_event_timestamp'] = float(boris_time)
                    all_features.append(horse_features)
            
            except Exception as e:
                
                continue
    
    cap.release()
    
    # Save if we got anything
    if all_features:
        output_file = OUTPUT_DIR / f"{video_file.stem}_horse_pose.json"
        
        with open(output_file, 'w') as f:
            json.dump({
                'video': video_file.name,
                'n_frames': len(all_features),
                'features': all_features
            }, f, indent=2)
        
        total_samples += len(all_features)
        videos_processed += 1
        tqdm.write(f"✓ {video_file.stem}: {len(all_features)} frames")

print(f"\n{'='*80}")
print("✅ COMPLETE!")
print(f"   Videos processed: {videos_processed}/{len(video_files)}")
print(f"   Total samples: {total_samples}")
print(f"   Output: {OUTPUT_DIR}")

print("="*80)