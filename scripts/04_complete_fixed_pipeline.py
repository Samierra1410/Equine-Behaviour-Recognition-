import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
from collections import Counter
from ultralytics import YOLO

print("="*80)
print("COMPLETE FIXED PIPELINE: YOLO + MEDIAPIPE + BORIS")
print("="*80)

BASE_DIR = Path(__file__).resolve().parent.parent
VIDEO_DIR = BASE_DIR / "og_videos"
BORIS_DIR = BASE_DIR / "data/boris_csvs"
MEDIAPIPE_DIR = BASE_DIR / "data/mediapipe_features"
OUTPUT_DIR = BASE_DIR / "data/yolo_mediapipe_combined"

OUTPUT_DIR.mkdir(exist_ok=True)

# Settings
SAMPLE_FPS = 3  # Sample 3 frames per second within behavior duration
PADDING = 1.0   # 1 second padding around events

# YOLO classes
PERSON_CLASS = 0
HORSE_CLASS = 17

# Load YOLO
print("\n📦 Loading YOLO...")
try:
    yolo_model = YOLO('yolov8n.pt')
    print("✓ YOLO loaded")
except:
    print("❌ YOLO not available")
    yolo_model = None

# ============================================================================
# BORIS PARSER - FIXED TO USE 'Time' COLUMN!
# ============================================================================

def parse_boris_with_start_stop(boris_file):
    """
    Parse BORIS CSV correctly:
    - Use 'Time' column (NOT 'Time offset' which is always 0!)
    - Handle START/STOP pairs to get behavior durations
    - Capture ALL categories including Avoidant
    """
    try:
        df = None
        for enc in ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252']:
            try:
                df = pd.read_csv(boris_file, encoding=enc)
                break
            except:
                continue
        
        if df is None or len(df) == 0:
            return [], "Empty file"
        
        df.columns = df.columns.str.replace('\ufeff', '').str.strip()
        
        # Find columns
        time_col = None
        behavior_col = None
        category_col = None
        type_col = None
        
        for col in df.columns:
            cl = col.lower().strip()
            
            
            
            if cl == 'time' and 'offset' not in cl:
                time_col = col
            elif cl == 'behavior':
                behavior_col = col
            elif cl == 'behavioral category':
                category_col = col
            elif cl == 'behavior type':
                type_col = col
        
        # If no 'Time' found, try alternatives
        if time_col is None:
            for col in df.columns:
                cl = col.lower().strip()
                if cl == 'time':
                    time_col = col
                    break
        
        if not all([time_col, behavior_col, category_col]):
            return [], f"Missing columns: time={time_col}, behavior={behavior_col}, category={category_col}"
        
        # Parse events with START/STOP handling
        events = []
        valid_cats = ['affiliative', 'neutral', 'avoidant']
        
        # Convert time to numeric
        df[time_col] = pd.to_numeric(df[time_col], errors='coerce')
        df = df.dropna(subset=[time_col])
        df = df.sort_values(time_col).reset_index(drop=True)
        
        if len(df) == 0:
            return [], "No valid timestamps"
        
        # Track open START events
        open_events = {}  # key: (behavior, category) -> start_time
        
        for _, row in df.iterrows():
            try:
                ts = float(row[time_col])
                behavior = str(row[behavior_col]).strip()
                category = str(row[category_col]).strip().lower()
                event_type = str(row.get(type_col, 'POINT')).strip().upper() if type_col else 'POINT'
                
                if category not in valid_cats:
                    continue
                
                key = (behavior, category)
                
                if event_type == 'START':
                    open_events[key] = ts
                    
                elif event_type == 'STOP':
                    if key in open_events:
                        start_time = open_events.pop(key)
                        events.append({
                            'start': start_time,
                            'end': ts,
                            'duration': ts - start_time,
                            'behavior': behavior,
                            'category': category,
                            'type': 'duration'
                        })
                    else:
                        # STOP without START - treat as point
                        events.append({
                            'start': max(0, ts - 1),
                            'end': ts + 1,
                            'duration': 2,
                            'behavior': behavior,
                            'category': category,
                            'type': 'point'
                        })
                        
                elif event_type == 'POINT':
                    events.append({
                        'start': max(0, ts - 1),
                        'end': ts + 1,
                        'duration': 2,
                        'behavior': behavior,
                        'category': category,
                        'type': 'point'
                    })
                    
            except Exception as e:
                continue
        
        # Handle unclosed START events
        for key, start_time in open_events.items():
            behavior, category = key
            events.append({
                'start': start_time,
                'end': start_time + 3,
                'duration': 3,
                'behavior': behavior,
                'category': category,
                'type': 'unclosed'
            })
        
        # Sort by start time
        events.sort(key=lambda x: x['start'])
        
        return events, f"Found {len(events)} events"
        
    except Exception as e:
        return [], str(e)

# ============================================================================
# BEHAVIOR MAPPING
# ============================================================================

def map_to_grouped_behavior(behavior, category):
    """Map to 6 grouped behaviors"""
    bl = str(behavior).lower()
    cl = str(category).lower()
    
    is_horse = 'horse' in bl
    is_active = any(w in bl for w in ['active', 'engagement', 'nose', 'joint'])
    
    if cl == 'affiliative':
        return 'affiliative-active' if is_active else 'affiliative-subtle'
    elif cl == 'neutral':
        return 'neutral-horse' if is_horse else 'neutral-human'
    elif cl == 'avoidant':
        return 'avoidant-horse' if is_horse else 'avoidant-human'
    return 'unknown'

# ============================================================================
# YOLO FEATURE EXTRACTION
# ============================================================================

def extract_yolo_features(frame, frame_w, frame_h):
    """Extract YOLO features from frame"""
    if yolo_model is None:
        return {}
    
    small = cv2.resize(frame, (640, 480))
    results = yolo_model(small, verbose=False)[0]
    
    humans, horses = [], []
    
    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        if conf < 0.25:
            continue
        
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        sx, sy = frame_w / 640, frame_h / 480
        
        bbox = {
            'cx': (x1 + x2) / 2 * sx,
            'cy': (y1 + y2) / 2 * sy,
            'w': (x2 - x1) * sx,
            'h': (y2 - y1) * sy,
            'area': (x2 - x1) * sx * (y2 - y1) * sy,
            'conf': conf
        }
        
        if cls == PERSON_CLASS:
            humans.append(bbox)
        elif cls == HORSE_CLASS:
            horses.append(bbox)
    
    feat = {}
    diag = np.sqrt(frame_w**2 + frame_h**2)
    
    if humans and horses:
        hu = max(humans, key=lambda b: b['area'])
        ho = max(horses, key=lambda b: b['area'])
        
        dx = hu['cx'] - ho['cx']
        dy = hu['cy'] - ho['cy']
        
        feat = {
            'yolo_distance': np.sqrt(dx**2 + dy**2) / diag,
            'yolo_dx': dx / frame_w,
            'yolo_dy': dy / frame_h,
            'yolo_human_x': hu['cx'] / frame_w,
            'yolo_human_y': hu['cy'] / frame_h,
            'yolo_human_area': hu['area'] / (frame_w * frame_h),
            'yolo_horse_x': ho['cx'] / frame_w,
            'yolo_horse_y': ho['cy'] / frame_h,
            'yolo_horse_area': ho['area'] / (frame_w * frame_h),
            'yolo_human_conf': hu['conf'],
            'yolo_horse_conf': ho['conf'],
            'yolo_both_detected': 1.0,
        }
    elif humans:
        hu = max(humans, key=lambda b: b['area'])
        feat = {
            'yolo_human_x': hu['cx'] / frame_w,
            'yolo_human_y': hu['cy'] / frame_h,
            'yolo_human_area': hu['area'] / (frame_w * frame_h),
            'yolo_human_conf': hu['conf'],
            'yolo_both_detected': 0.0,
        }
    elif horses:
        ho = max(horses, key=lambda b: b['area'])
        feat = {
            'yolo_horse_x': ho['cx'] / frame_w,
            'yolo_horse_y': ho['cy'] / frame_h,
            'yolo_horse_area': ho['area'] / (frame_w * frame_h),
            'yolo_horse_conf': ho['conf'],
            'yolo_both_detected': 0.0,
        }
    
    return feat

# ============================================================================
# MEDIAPIPE LOADER
# ============================================================================

def load_mediapipe(video_name):
    """Load MediaPipe features"""
    vc = video_name.lower().replace('_', '').replace('-', '')
    
    for f in MEDIAPIPE_DIR.glob("*.csv"):
        fc = f.stem.lower().replace('_', '').replace('-', '').replace('features', '')
        if vc in fc or fc in vc:
            try:
                return pd.read_csv(f)
            except:
                pass
    return None

def get_mp_features(mp_df, timestamp):
    """Get MediaPipe features at timestamp"""
    if mp_df is None or 'timestamp' not in mp_df.columns:
        return {}
    
    diffs = np.abs(mp_df['timestamp'] - timestamp)
    idx = diffs.idxmin()
    
    if diffs.loc[idx] > 0.5:
        return {}
    
    row = mp_df.loc[idx]
    
    feat_cols = ['nose_x', 'nose_y', 'left_shoulder_x', 'left_shoulder_y',
                 'right_shoulder_x', 'right_shoulder_y', 'left_wrist_x', 'left_wrist_y',
                 'right_wrist_x', 'right_wrist_y', 'center_x', 'center_y',
                 'shoulder_width', 'body_height', 'speed']
    
    feat = {}
    for c in feat_cols:
        if c in row.index and pd.notna(row[c]):
            feat[f'mp_{c}'] = float(row[c])
    
    return feat

# ============================================================================
# VIDEO PROCESSING
# ============================================================================

def find_boris_file(video_name, boris_files):
    """Find matching BORIS file"""
    vc = video_name.lower().replace('_', '').replace('-', '').replace(' ', '')
    
    for bf in boris_files:
        bc = bf.stem.lower().replace('_', '').replace('-', '').replace(' ', '')
        if vc in bc or bc in vc:
            return bf
    return None

def process_video(video_file, boris_file):
    """Process video with BORIS events"""
    
    events, msg = parse_boris_with_start_stop(boris_file)
    
    if not events:
        return None, msg
    
    mp_df = load_mediapipe(video_file.stem)
    
    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        return None, "Can't open video"
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0
    
    samples = []
    
    for event in events:
        # Add padding
        start = max(0, event['start'] - PADDING)
        end = min(duration, event['end'] + PADDING)
        
        if end <= start:
            continue
        
        # Calculate samples
        event_duration = end - start
        n_samples = max(6, int(event_duration * SAMPLE_FPS))
        times = np.linspace(start, end, n_samples)
        
        for t in times:
            frame_idx = int(t * fps)
            if frame_idx >= total_frames or frame_idx < 0:
                continue
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            # Extract features
            yolo_feat = extract_yolo_features(frame, frame_w, frame_h)
            mp_feat = get_mp_features(mp_df, t)
            
            sample = {
                'video': video_file.stem,
                'timestamp': float(t),
                'frame': frame_idx,
                'behavior': event['behavior'],
                'category': event['category'],
                'grouped_behavior': map_to_grouped_behavior(event['behavior'], event['category']),
                'event_type': event['type'],
            }
            sample.update(yolo_feat)
            sample.update(mp_feat)
            samples.append(sample)
    
    cap.release()
    
    if not samples:
        return None, "No samples extracted"
    
    behavior_counts = Counter([s['grouped_behavior'] for s in samples])
    category_counts = Counter([s['category'] for s in samples])
    
    return pd.DataFrame(samples), {'behaviors': behavior_counts, 'categories': category_counts}

# ============================================================================
# MAIN
# ============================================================================

def main():
    
    
    video_files = list(VIDEO_DIR.glob("*.mp4")) + list(VIDEO_DIR.glob("*.MP4"))
    boris_files = list(BORIS_DIR.glob("*.csv"))
    
    print(f"✓ {len(video_files)} videos")
    print(f"✓ {len(boris_files)} BORIS files")
    
    # First, analyze BORIS files
    print("\n📊 Analyzing BORIS files...")
    all_events = []
    valid_boris = 0
    
    for bf in boris_files:
        events, msg = parse_boris_with_start_stop(bf)
        if events:
            valid_boris += 1
            all_events.extend(events)
    
    print(f"   Valid BORIS files: {valid_boris}/{len(boris_files)}")
    print(f"   Total events: {len(all_events)}")
    
    if all_events:
        cat_counts = Counter([e['category'] for e in all_events])
        print(f"\n   Events by category:")
        for cat in ['affiliative', 'neutral', 'avoidant']:
            count = cat_counts.get(cat, 0)
            status = "✅" if count > 0 else "❌"
            print(f"      {cat}: {count} {status}")
    
    # Process videos
    print(f"\n{'='*80}")
    print("PROCESSING VIDEOS")
    print("="*80)
    
    all_samples = []
    overall_behaviors = Counter()
    overall_categories = Counter()
    successful = 0
    
    for video_file in tqdm(video_files, desc="Processing"):
        boris_file = find_boris_file(video_file.stem, boris_files)
        
        if boris_file is None:
            continue
        
        result_df, info = process_video(video_file, boris_file)
        
        if result_df is not None and len(result_df) > 0:
            all_samples.append(result_df)
            successful += 1
            
            if isinstance(info, dict):
                overall_behaviors.update(info.get('behaviors', {}))
                overall_categories.update(info.get('categories', {}))
            
            tqdm.write(f"✓ {video_file.stem[:40]}: {len(result_df)} samples | {dict(info.get('categories', {}))}")
    
    if not all_samples:
        print("\n❌ No samples extracted!")
        return
    
    # Combine
    combined = pd.concat(all_samples, ignore_index=True).fillna(0)
    
    # Save
    output_file = OUTPUT_DIR / "all_features.csv"
    combined.to_csv(output_file, index=False)
    
    feat_cols = [c for c in combined.columns if c.startswith(('yolo_', 'mp_'))]
    
    # Summary
    print(f"\n{'='*80}")
    print("✅ COMPLETE")
    print("="*80)
    print(f"   Videos: {successful}/{len(video_files)}")
    print(f"   Samples: {len(combined)}")
    print(f"   Features: {len(feat_cols)}")
    
    print(f"\n   📊 Categories:")
    for cat in ['affiliative', 'neutral', 'avoidant']:
        count = overall_categories.get(cat, 0)
        pct = 100 * count / len(combined) if count > 0 else 0
        status = "✅" if count > 0 else "❌ MISSING"
        print(f"      {cat:15s}: {count:5d} ({pct:5.1f}%) {status}")
    
    print(f"\n   📊 Behaviors:")
    for beh in ['affiliative-active', 'affiliative-subtle', 'neutral-horse',
                'neutral-human', 'avoidant-horse', 'avoidant-human']:
        count = overall_behaviors.get(beh, 0)
        pct = 100 * count / len(combined) if count > 0 else 0
        status = "✅" if count > 0 else "❌"
        print(f"      {beh:20s}: {count:5d} ({pct:5.1f}%) {status}")
    
    # Check for avoidant
    has_avoidant = overall_categories.get('avoidant', 0) > 0
    
    if has_avoidant:
        print(f"\n   🎉 SUCCESS! All 3 categories captured!")
    else:
        print(f"\n   ⚠️ Still missing Avoidant. Check BORIS files manually.")
    
    # Save summary
    with open(OUTPUT_DIR / 'summary.json', 'w') as f:
        json.dump({
            'samples': len(combined),
            'features': len(feat_cols),
            'categories': dict(overall_categories),
            'behaviors': dict(overall_behaviors),
            'has_all_categories': has_avoidant
        }, f, indent=2)
    
    print(f"\n   📁 Output: {output_file}")
    print("="*80)
    
    
if __name__ == "__main__":
    main()
