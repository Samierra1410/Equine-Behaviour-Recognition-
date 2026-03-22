import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from pathlib import Path
import json
from tqdm import tqdm

class MediaPipeExtractor:
   
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def extract_landmarks(self, frame):
      
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Key landmarks (13 points)
            data = {
                'nose_x': landmarks[0].x,
                'nose_y': landmarks[0].y,
                'nose_vis': landmarks[0].visibility,
                'left_shoulder_x': landmarks[11].x,
                'left_shoulder_y': landmarks[11].y,
                'right_shoulder_x': landmarks[12].x,
                'right_shoulder_y': landmarks[12].y,
                'left_elbow_x': landmarks[13].x,
                'left_elbow_y': landmarks[13].y,
                'right_elbow_x': landmarks[14].x,
                'right_elbow_y': landmarks[14].y,
                'left_wrist_x': landmarks[15].x,
                'left_wrist_y': landmarks[15].y,
                'right_wrist_x': landmarks[16].x,
                'right_wrist_y': landmarks[16].y,
                'left_hip_x': landmarks[23].x,
                'left_hip_y': landmarks[23].y,
                'right_hip_x': landmarks[24].x,
                'right_hip_y': landmarks[24].y,
                'left_knee_x': landmarks[25].x,
                'left_knee_y': landmarks[25].y,
                'right_knee_x': landmarks[26].x,
                'right_knee_y': landmarks[26].y,
            }
            
            # Derived features
            data['center_x'] = (data['left_hip_x'] + data['right_hip_x']) / 2
            data['center_y'] = (data['left_hip_y'] + data['right_hip_y']) / 2
            data['shoulder_width'] = abs(data['right_shoulder_x'] - data['left_shoulder_x'])
            
            # Hand distances from shoulders
            data['left_hand_dist'] = np.sqrt(
                (data['left_wrist_x'] - data['left_shoulder_x'])**2 + 
                (data['left_wrist_y'] - data['left_shoulder_y'])**2
            )
            data['right_hand_dist'] = np.sqrt(
                (data['right_wrist_x'] - data['right_shoulder_x'])**2 + 
                (data['right_wrist_y'] - data['right_shoulder_y'])**2
            )
            data['hands_extended'] = int(data['left_hand_dist'] + data['right_hand_dist'] > 0.5)
            
            # Body metrics
            data['body_height'] = abs(data['nose_y'] - data['center_y'])
            data['is_crouching'] = int(data['body_height'] < 0.3)
            
            # Lean (forward/back)
            shoulder_center_y = (data['left_shoulder_y'] + data['right_shoulder_y']) / 2
            data['lean_forward'] = data['nose_y'] - shoulder_center_y
            
            return data
        
        return None
    
    def interpolate_missing(self, df):
       
        
        # Identify valid frames (non-zero)
        valid_mask = (df != 0).any(axis=1)
        
        if valid_mask.sum() < 2:
            print("  ⚠️  Too few valid frames for interpolation")
            return df
        
        # Linear interpolation
        df_interp = df.copy()
        for col in df.columns:
            if col not in ['frame', 'timestamp']:
                df_interp[col] = df[col].replace(0, np.nan).interpolate(
                    method='linear', limit_direction='both'
                ).fillna(0)
        
        return df_interp
    
    def add_velocity_features(self, df, fps=30):
        """Add velocity and acceleration"""
        
        df = df.copy()
        
        # Center velocity
        df['center_vel_x'] = df['center_x'].diff().fillna(0) * fps
        df['center_vel_y'] = df['center_y'].diff().fillna(0) * fps
        df['speed'] = np.sqrt(df['center_vel_x']**2 + df['center_vel_y']**2)
        
        # Hand velocities
        df['left_hand_vel'] = df['left_hand_dist'].diff().fillna(0).abs() * fps
        df['right_hand_vel'] = df['right_hand_dist'].diff().fillna(0).abs() * fps
        
        return df
    
    def process_video(self, video_path):
        """Process entire video"""
        
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        rows = []
        frame_idx = 0
        
        with tqdm(total=total_frames, desc=f"  Processing") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                timestamp = frame_idx / fps
                
                landmarks = self.extract_landmarks(frame)
                
                if landmarks is not None:
                    row = {'frame': frame_idx, 'timestamp': timestamp, **landmarks}
                else:
                    # Missing detection - all zeros
                    row = {'frame': frame_idx, 'timestamp': timestamp}
                    for key in ['nose_x', 'nose_y', 'nose_vis', 'left_shoulder_x', 'left_shoulder_y',
                               'right_shoulder_x', 'right_shoulder_y', 'left_elbow_x', 'left_elbow_y',
                               'right_elbow_x', 'right_elbow_y', 'left_wrist_x', 'left_wrist_y',
                               'right_wrist_x', 'right_wrist_y', 'left_hip_x', 'left_hip_y',
                               'right_hip_x', 'right_hip_y', 'left_knee_x', 'left_knee_y',
                               'right_knee_x', 'right_knee_y', 'center_x', 'center_y',
                               'shoulder_width', 'left_hand_dist', 'right_hand_dist',
                               'hands_extended', 'body_height', 'is_crouching', 'lean_forward']:
                        row[key] = 0.0
                
                rows.append(row)
                frame_idx += 1
                pbar.update(1)
        
        cap.release()
        
        # Create dataframe
        df = pd.DataFrame(rows)
        
        # Interpolate missing frames
        valid_before = (df != 0).any(axis=1).sum()
        df = self.interpolate_missing(df)
        valid_after = (df != 0).any(axis=1).sum()
        
        print(f"  Interpolation: {valid_before} → {valid_after} valid frames")
        
        # Add velocities
        df = self.add_velocity_features(df, fps)
        
        detection_rate = valid_before / len(df) * 100
        
        return df, fps, detection_rate

def main():
    
    print("="*80)
    print("STEP 1: MEDIAPIPE POSE EXTRACTION WITH INTERPOLATION")
    print("="*80)
    
    # Paths
    BASE_DIR = Path(__file__).resolve().parent.parent
    VIDEO_DIR = BASE_DIR / "og_videos"
    OUTPUT_DIR = BASE_DIR / "data/mediapipe_features"
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Find all videos
    video_files = list(VIDEO_DIR.glob("*.mp4")) + list(VIDEO_DIR.glob("*.MP4"))
    
    if not video_files:
        print(f"\n❌ No videos found in {VIDEO_DIR}")
        return
    
    print(f"\n✓ Found {len(video_files)} videos")
    
    extractor = MediaPipeExtractor()
    
    results = []
    
    for video_path in video_files:
        video_name = video_path.stem
        print(f"\n📹 {video_name}")
        
        try:
            df, fps, detection_rate = extractor.process_video(video_path)
            
            # Save
            output_file = OUTPUT_DIR / f"{video_name}_features.csv"
            df.to_csv(output_file, index=False)
            
            results.append({
                'video': video_name,
                'frames': len(df),
                'fps': fps,
                'detection_rate': detection_rate,
                'interpolated_frames': len(df) - int(len(df) * detection_rate / 100)
            })
            
            print(f"  ✓ {len(df)} frames @ {fps:.1f} fps")
            print(f"  ✓ Detection: {detection_rate:.1f}%")
            print(f"  ✓ Saved: {output_file.name}")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results.append({
                'video': video_name,
                'frames': 0,
                'fps': 0,
                'detection_rate': 0,
                'error': str(e)
            })
    
    # Save summary
    summary_file = OUTPUT_DIR / "extraction_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"✅ EXTRACTION COMPLETE")
    print(f"   Videos processed: {len([r for r in results if r.get('frames', 0) > 0])}/{len(video_files)}")
    print(f"   Output: {OUTPUT_DIR}")
    print(f"   Summary: {summary_file}")
    print("="*80)

if __name__ == "__main__":
    main()