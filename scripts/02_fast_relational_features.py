import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import pandas as pd
from ultralytics import YOLO

print("="*80)
print("STEP 1: FAST RELATIONAL FEATURES (BORIS-GUIDED)")
print("="*80)

BASE_DIR = Path(__file__).resolve().parent.parent
VIDEO_DIR = BASE_DIR / "og_videos"
BORIS_DIR = BASE_DIR / "data/boris_csvs"
OUTPUT_DIR = BASE_DIR / "data/relational_features"

OUTPUT_DIR.mkdir(exist_ok=True)

# Load YOLO
print("\n📦 Loading YOLO...")
try:
    yolo_model = YOLO('yolov8n.pt')  # Nano = fastest
    print("✓ YOLO loaded")
except Exception as e:
    print(f"❌ Error loading YOLO: {e}")
    print("   Run: pip install ultralytics")
    exit(1)

PERSON_CLASS = 0
HORSE_CLASS = 17

# Window around each BORIS event (in seconds)
WINDOW_BEFORE = 2.0  # 2 seconds before
WINDOW_AFTER = 2.0   # 2 seconds after
SAMPLE_RATE = 5      # Sample every 5th frame (6 fps instead of 30 fps)

def load_boris_events(boris_file):
    
    
    try:
        # Try different encodings
        df = None
        for encoding in ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252']:
            try:
                df = pd.read_csv(boris_file, encoding=encoding)
                break
            except:
                continue
        
        if df is None or len(df) == 0:
            return []
        
        # Clean column names
        df.columns = df.columns.str.replace('\ufeff', '').str.strip()
        
        
        time_col = None
        for col in df.columns:
            col_lower = col.lower().strip()
            
            if 'time' in col_lower and 'offset' in col_lower:
                time_col = col
                break
            elif col_lower == 'time':
                time_col = col
                break
        
        if time_col is None:
            print(f"      ⚠️  No time column found in {boris_file.name}")
            print(f"      Columns: {list(df.columns)[:10]}")
            return []
        
        # Extract timestamps
        timestamps = pd.to_numeric(df[time_col], errors='coerce')
        timestamps = timestamps.dropna()
        timestamps = timestamps[timestamps >= 0]
        
        if len(timestamps) == 0:
            return []
        
        return sorted(timestamps.values)
    
    except Exception as e:
        print(f"      ❌ Error loading {boris_file.name}: {e}")
        return []

def extract_features_at_timestamp(cap, timestamp, fps, frame_width, frame_height):
    """Extract features for frames around a timestamp"""
    
    # Calculate frame indices
    center_frame = int(timestamp * fps)
    start_frame = max(0, int((timestamp - WINDOW_BEFORE) * fps))
    end_frame = int((timestamp + WINDOW_AFTER) * fps)
    
    features = []
    
    for frame_idx in range(start_frame, end_frame, SAMPLE_RATE):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        # Resize for faster YOLO processing
        frame_small = cv2.resize(frame, (640, 480))
        
        # YOLO detection
        results = yolo_model(frame_small, verbose=False)[0]
        
        human_boxes = []
        horse_boxes = []
        
        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            if conf < 0.3:
                continue
            
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            # Scale back to original size
            scale_x = frame_width / 640
            scale_y = frame_height / 480
            
            x1 *= scale_x
            x2 *= scale_x
            y1 *= scale_y
            y2 *= scale_y
            
            bbox = {
                'center_x': float((x1 + x2) / 2),
                'center_y': float((y1 + y2) / 2),
                'width': float(x2 - x1),
                'height': float(y2 - y1),
                'area': float((x2 - x1) * (y2 - y1)),
                'conf': conf
            }
            
            if cls == PERSON_CLASS:
                human_boxes.append(bbox)
            elif cls == HORSE_CLASS:
                horse_boxes.append(bbox)
        
        # Compute relational features
        frame_features = {
            'frame': frame_idx,
            'timestamp': float(frame_idx / fps),
            'boris_event_timestamp': float(timestamp),
        }
        
        if human_boxes and horse_boxes:
            # Use largest detections
            human = max(human_boxes, key=lambda b: b['area'])
            horse = max(horse_boxes, key=lambda b: b['area'])
            
            # Distance features
            dx = human['center_x'] - horse['center_x']
            dy = human['center_y'] - horse['center_y']
            distance = np.sqrt(dx**2 + dy**2)
            
            frame_diagonal = np.sqrt(frame_width**2 + frame_height**2)
            
            frame_features.update({
                'human_horse_distance': float(distance / frame_diagonal),
                'human_horse_dx': float(dx / frame_width),
                'human_horse_dy': float(dy / frame_height),
                'human_area_ratio': float(human['area'] / (frame_width * frame_height)),
                'horse_area_ratio': float(horse['area'] / (frame_width * frame_height)),
                'human_left_of_horse': float(human['center_x'] < horse['center_x']),
                'human_above_horse': float(human['center_y'] < horse['center_y']),
                'human_center_x': float(human['center_x'] / frame_width),
                'human_center_y': float(human['center_y'] / frame_height),
                'horse_center_x': float(horse['center_x'] / frame_width),
                'horse_center_y': float(horse['center_y'] / frame_height),
                'human_conf': float(human['conf']),
                'horse_conf': float(horse['conf']),
                'both_detected': 1.0,
            })
        
        elif human_boxes:
            human = max(human_boxes, key=lambda b: b['area'])
            frame_features.update({
                'human_detected': 1.0,
                'horse_detected': 0.0,
                'human_center_x': float(human['center_x'] / frame_width),
                'human_center_y': float(human['center_y'] / frame_height),
                'human_area_ratio': float(human['area'] / (frame_width * frame_height)),
                'human_conf': float(human['conf']),
                'both_detected': 0.0,
            })
        
        elif horse_boxes:
            horse = max(horse_boxes, key=lambda b: b['area'])
            frame_features.update({
                'human_detected': 0.0,
                'horse_detected': 1.0,
                'horse_center_x': float(horse['center_x'] / frame_width),
                'horse_center_y': float(horse['center_y'] / frame_height),
                'horse_area_ratio': float(horse['area'] / (frame_width * frame_height)),
                'horse_conf': float(horse['conf']),
                'both_detected': 0.0,
            })
        
        else:
            frame_features.update({
                'human_detected': 0.0,
                'horse_detected': 0.0,
                'both_detected': 0.0,
            })
        
        features.append(frame_features)
    
    return features

def find_boris_file(video_name, boris_files):
    """Find matching BORIS file"""
    video_clean = video_name.lower().replace('_', '').replace('-', '').replace('.mp4', '')
    
    # Try exact match first
    for f in boris_files:
        if video_name.lower() == f.stem.lower():
            return f
    
    # Try partial match
    for f in boris_files:
        boris_clean = f.stem.lower().replace('_', '').replace('-', '')
        if video_clean in boris_clean or boris_clean in video_clean:
            return f
    
    return None

# Main processing
def main():
    video_files = list(VIDEO_DIR.glob("*.mp4")) + list(VIDEO_DIR.glob("*.MP4"))
    boris_files = list(BORIS_DIR.glob("*.csv"))
    
    print(f"\n✓ Found {len(video_files)} videos")
    print(f"✓ Found {len(boris_files)} BORIS files")
    
    if len(video_files) == 0:
        print(f"\n❌ No videos found in {VIDEO_DIR}")
        return
    
    total_frames_processed = 0
    total_events_processed = 0
    videos_processed = 0
    
    for video_file in tqdm(video_files, desc="Processing videos"):
        
        # Find matching BORIS file
        boris_file = find_boris_file(video_file.stem, boris_files)
        
        if not boris_file:
            tqdm.write(f"⚠️  {video_file.stem}: No BORIS file")
            continue
        
        # Load BORIS events
        event_timestamps = load_boris_events(boris_file)
        
        if not event_timestamps:
            tqdm.write(f"⚠️  {video_file.stem}: No events in BORIS")
            continue
        
        # Open video
        cap = cv2.VideoCapture(str(video_file))
        
        if not cap.isOpened():
            tqdm.write(f"❌ {video_file.stem}: Can't open video")
            continue
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Process each BORIS event
        all_features = []
        
        for timestamp in event_timestamps:
            features = extract_features_at_timestamp(cap, timestamp, fps, frame_width, frame_height)
            all_features.extend(features)
            total_frames_processed += len(features)
        
        cap.release()
        
        total_events_processed += len(event_timestamps)
        videos_processed += 1
        
        # Save
        if all_features:
            output_file = OUTPUT_DIR / f"{video_file.stem}_relational.json"
            
            with open(output_file, 'w') as f:
                json.dump({
                    'video': video_file.name,
                    'n_boris_events': len(event_timestamps),
                    'n_frames': len(all_features),
                    'fps': fps,
                    'features': all_features
                }, f, indent=2)
            
            tqdm.write(f"✓ {video_file.stem}: {len(event_timestamps)} events → {len(all_features)} frames")
    
    print(f"\n{'='*80}")
    print(f"✅ FAST EXTRACTION COMPLETE")
    print(f"{'='*80}")
    print(f"   Videos processed: {videos_processed}/{len(video_files)}")
    print(f"   BORIS events: {total_events_processed}")
    print(f"   Frames processed: {total_frames_processed}")
    print(f"   Avg frames/event: {total_frames_processed / max(total_events_processed, 1):.1f}")
    print(f"   Output: {OUTPUT_DIR}")
    
    
    print("="*80)

if __name__ == "__main__":
    main()