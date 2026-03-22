import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.linewidth': 1.2,
})

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {'yolo': '#3498db', 'mediapipe': '#9b59b6', 'ap10k': '#e67e22'}

features = [
    'yolo_horse_x', 'yolo_human_y', 'yolo_human_area', 'yolo_horse_area',
    'yolo_human_x', 'mp_shoulder_width', 'mp_body_height', 'yolo_distance',
    'yolo_horse_y', 'yolo_dx', 'horse_ear_angle', 'mp_speed',
    'yolo_dy', 'horse_head_angle', 'mp_nose_x'
]
importance = [6.55, 6.37, 6.21, 6.16, 6.10, 4.97, 4.46, 4.32,
              3.89, 3.67, 3.45, 3.21, 2.98, 2.76, 2.54]

colors_feat = []
for f in features:
    if 'yolo' in f: colors_feat.append(COLORS['yolo'])
    elif 'mp' in f: colors_feat.append(COLORS['mediapipe'])
    else: colors_feat.append(COLORS['ap10k'])

fig, ax = plt.subplots(figsize=(11, 9))
y_pos = np.arange(len(features))
bars = ax.barh(y_pos, importance, color=colors_feat, alpha=0.85, edgecolor='black', linewidth=1.5)
ax.set_yticks(y_pos)
ax.set_yticklabels(features, fontsize=10, fontweight='bold')
ax.set_xlabel('Feature Importance (Gain)', fontweight='bold', fontsize=12)
ax.set_title('Top 15 Features by Importance\n(Three-Category Parent Classification)',
            fontweight='bold', fontsize=13, pad=15)
ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.8)
ax.invert_yaxis()
ax.set_xlim(0, max(importance) * 1.15)

for bar, val in zip(bars, importance):
    ax.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height()/2,
           f'{val:.2f}', va='center', fontweight='bold', fontsize=9)

legend_elements = [
    Patch(facecolor=COLORS['yolo'], label='YOLO Spatial (8/15)', edgecolor='black', linewidth=1),
    Patch(facecolor=COLORS['mediapipe'], label='MediaPipe Human (4/15)', edgecolor='black', linewidth=1),
    Patch(facecolor=COLORS['ap10k'], label='AP-10K Equine (3/15)', edgecolor='black', linewidth=1)
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10, frameon=True, shadow=True, fancybox=True)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig8_feature_importance.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'fig8_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print(" Saved: fig8_feature_importance.pdf/.png")