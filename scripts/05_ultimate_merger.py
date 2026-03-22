import pandas as pd
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print(" ULTIMATE MERGER: YOLO + MEDIAPIPE + AP-10K ")
print("="*80)

BASE_DIR = Path(__file__).resolve().parent.parent
YOLO_MP_DIR = BASE_DIR / "data/yolo_mediapipe_combined"
AP10K_DIR = BASE_DIR / "data/horse_pose_features"
OUTPUT_DIR = BASE_DIR / "data/ultimate_combined"

OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================================
# LOAD YOLO + MEDIAPIPE
# ============================================================================

print("\n📂 Loading YOLO + MediaPipe data...")

yolo_mp_file = YOLO_MP_DIR / "all_features.csv"

if not yolo_mp_file.exists():
    print(f"   ❌ Not found: {yolo_mp_file}")
    print(f"   ⏳ Wait for YOLO+MediaPipe extraction to finish!")
    exit(1)

df_yolo_mp = pd.read_csv(yolo_mp_file)

print(f"   ✓ Loaded: {len(df_yolo_mp)} samples")
print(f"   ✓ Videos: {df_yolo_mp['video'].nunique()}")

# Check categories
if 'category' in df_yolo_mp.columns:
    cat_counts = df_yolo_mp['category'].value_counts()
    print(f"\n   📊 Categories:")
    for cat in ['affiliative', 'neutral', 'avoidant']:
        count = cat_counts.get(cat, 0)
        status = "✅" if count > 0 else "❌"
        print(f"      {cat}: {count:5d} {status}")
    
    avoidant_count = cat_counts.get('avoidant', 0)
    if avoidant_count == 0:
        print(f"\n   ⚠️  No avoidant in YOLO+MediaPipe data yet!")
        print(f"      Wait for extraction to finish.")

# Count features
yolo_cols = [c for c in df_yolo_mp.columns if 'yolo' in c.lower()]
mp_cols = [c for c in df_yolo_mp.columns if 'mp_' in c.lower()]

print(f"\n   Features:")
print(f"      YOLO: {len(yolo_cols)}")
print(f"      MediaPipe: {len(mp_cols)}")

# ============================================================================
# LOAD AP-10K
# ============================================================================

print(f"\n📂 Loading AP-10K horse pose data...")

ap10k_files = list(AP10K_DIR.glob("*_horse_pose.json"))

if not ap10k_files:
    print(f"   ❌ No AP-10K files found in {AP10K_DIR}")
    print(f"   Running without AP-10K (YOLO+MediaPipe only)")
    use_ap10k = False
    df_merged = df_yolo_mp.copy()
else:
    use_ap10k = True
    print(f"   ✓ Found {len(ap10k_files)} AP-10K files")
    
    # Load all AP-10K data
    ap10k_data = []
    
    for ap_file in tqdm(ap10k_files, desc="Loading AP-10K"):
        try:
            with open(ap_file) as f:
                data = json.load(f)
            
            video = ap_file.stem.replace('_horse_pose', '')
            features = data.get('features', [])
            
            for feat in features:
                sample = {
                    'video': video,
                    'timestamp': feat.get('timestamp', 0),
                }
                
                # Add horse pose features
                for key, val in feat.items():
                    if key not in ['frame', 'timestamp', 'boris_event_timestamp']:
                        sample[f'horse_{key}'] = val
                
                ap10k_data.append(sample)
        except Exception as e:
            tqdm.write(f"   ⚠️  Error loading {ap_file.name}: {e}")
    
    if not ap10k_data:
        print(f"\n   ❌ No AP-10K data extracted!")
        use_ap10k = False
        df_merged = df_yolo_mp.copy()
    else:
        df_ap10k = pd.DataFrame(ap10k_data)
        
        print(f"\n   ✓ AP-10K samples: {len(df_ap10k)}")
        print(f"   ✓ AP-10K videos: {df_ap10k['video'].nunique()}")
        
        horse_cols = [c for c in df_ap10k.columns if c.startswith('horse_')]
        print(f"   ✓ Horse features: {len(horse_cols)}")
        
        # ========================================================================
        # MERGE YOLO+MEDIAPIPE WITH AP-10K
        # ========================================================================
        
        print(f"\n{'='*80}")
        print("MERGING YOLO+MEDIAPIPE WITH AP-10K")
        print("="*80)
        
        merged_rows = []
        
        for video in tqdm(df_yolo_mp['video'].unique(), desc="Merging"):
            # Get YOLO+MediaPipe data for this video
            yolo_mp_video = df_yolo_mp[df_yolo_mp['video'] == video].copy()
            
            if len(yolo_mp_video) == 0:
                continue
            
            # Get AP-10K data for this video (fuzzy match)
            video_clean = video.lower().replace('_', '').replace('-', '')
            
            ap10k_video = None
            for v in df_ap10k['video'].unique():
                v_clean = v.lower().replace('_', '').replace('-', '')
                if video_clean in v_clean or v_clean in video_clean:
                    ap10k_video = df_ap10k[df_ap10k['video'] == v].copy()
                    break
            
            if ap10k_video is None or len(ap10k_video) == 0:
                # No AP-10K for this video - keep YOLO+MediaPipe only
                for _, row in yolo_mp_video.iterrows():
                    merged_rows.append(row.to_dict())
                continue
            
            # Merge on timestamp
            for _, yolo_mp_row in yolo_mp_video.iterrows():
                timestamp = yolo_mp_row['timestamp']
                
                # Find closest AP-10K timestamp (within 0.3 seconds)
                time_diffs = np.abs(ap10k_video['timestamp'] - timestamp)
                
                if len(time_diffs) == 0:
                    merged_rows.append(yolo_mp_row.to_dict())
                    continue
                
                closest_idx = time_diffs.idxmin()
                
                if time_diffs.loc[closest_idx] <= 0.3:
                    # Merge
                    ap10k_row = ap10k_video.loc[closest_idx]
                    
                    merged = yolo_mp_row.to_dict()
                    
                    # Add AP-10K features
                    for col in ap10k_row.index:
                        if col not in ['video', 'timestamp']:
                            merged[col] = ap10k_row[col]
                    
                    merged_rows.append(merged)
                else:
                    # No close match - keep YOLO+MediaPipe only
                    merged_rows.append(yolo_mp_row.to_dict())
        
        df_merged = pd.DataFrame(merged_rows)
        
        print(f"\n✓ Merged: {len(df_merged)} samples")

# ============================================================================
# SAVE MERGED DATA
# ============================================================================

print(f"\n{'='*80}")
print("SAVING ULTIMATE COMBINED DATA")
print("="*80)

output_file = OUTPUT_DIR / "all_features_ULTIMATE.csv"
df_merged.to_csv(output_file, index=False)

# Count features
label_cols = ['video', 'timestamp', 'frame', 'behavior', 'category', 
              'grouped_behavior', 'event_type', 'boris_event_timestamp']

feature_cols = [c for c in df_merged.columns if c not in label_cols]

yolo_feats = [c for c in feature_cols if 'yolo' in c.lower()]
mp_feats = [c for c in feature_cols if 'mp_' in c.lower()]
horse_feats = [c for c in feature_cols if c.startswith('horse_')]

print(f"\n✅ ULTIMATE COMBINED DATA")
print(f"   Samples: {len(df_merged)}")
print(f"   Features: {len(feature_cols)}")
print(f"      - YOLO: {len(yolo_feats)}")
print(f"      - MediaPipe: {len(mp_feats)}")
if use_ap10k:
    print(f"      - AP-10K Horse: {len(horse_feats)}")

if 'category' in df_merged.columns:
    print(f"\n   📊 Categories:")
    cat_counts = df_merged['category'].value_counts()
    for cat in ['affiliative', 'neutral', 'avoidant']:
        count = cat_counts.get(cat, 0)
        pct = 100 * count / len(df_merged) if count > 0 else 0
        status = "✅" if count > 0 else "❌"
        print(f"      {cat:12s}: {count:5d} ({pct:5.1f}%) {status}")

if 'grouped_behavior' in df_merged.columns:
    print(f"\n   📊 Grouped Behaviors:")
    for beh in sorted(df_merged['grouped_behavior'].unique()):
        count = len(df_merged[df_merged['grouped_behavior'] == beh])
        pct = 100 * count / len(df_merged)
        print(f"      {beh:20s}: {count:5d} ({pct:5.1f}%)")

# Save summary
summary = {
    'samples': len(df_merged),
    'features': {
        'yolo': len(yolo_feats),
        'mediapipe': len(mp_feats),
        'ap10k': len(horse_feats) if use_ap10k else 0,
        'total': len(feature_cols)
    },
    'categories': dict(cat_counts) if 'category' in df_merged.columns else {},
    'has_ap10k': use_ap10k
}

with open(OUTPUT_DIR / 'summary.json', 'w') as f:
    json.dump(summary, f, indent=2, default=lambda x: int(x) if hasattr(x, 'item') else str(x))

print(f"\n📁 Output: {output_file}")
print(f"📁 Summary: {OUTPUT_DIR / 'summary.json'}")

print(f"\n{'='*80}")
print("✅ MERGE COMPLETE!")
print("="*80)


   
    

