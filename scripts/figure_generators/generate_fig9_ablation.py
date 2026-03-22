import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('Ablation Study: Multi-Modal Feature Contribution',
            fontweight='bold', fontsize=14, y=0.98)

modalities = ['YOLO\nonly', 'AP-10K\nonly', 'MediaPipe\nonly']
single_acc = [58.9, 55.7, 52.4]
colors_single = [COLORS['yolo'], COLORS['ap10k'], COLORS['mediapipe']]

bars1 = ax1.bar(modalities, single_acc, color=colors_single, alpha=0.85,
               edgecolor='black', linewidth=2, width=0.6)
ax1.axhline(y=73.2, color='red', linestyle='--', linewidth=2.5, label='Full Model: 73.2%', alpha=0.8)
ax1.axhline(y=33.3, color='gray', linestyle=':', linewidth=2, label='Random: 33.3%', alpha=0.6)
ax1.set_ylabel('Balanced Accuracy (%)', fontweight='bold', fontsize=11)
ax1.set_title('(a) Single Modality Performance', fontweight='bold', pad=12, fontsize=12)
ax1.set_ylim(0, 80)
ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
ax1.legend(loc='upper right', fontsize=10, frameon=True, shadow=True)
for bar, val in zip(bars1, single_acc):
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1.5,
            f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

configs = ['All\nFeatures\n(35)', 'Without\nAP-10K\n(27)', 'Without\nYOLO\n(23)', 'Without\nMediaPipe\n(20)']
ablation_acc = [73.2, 60.1, 65.2, 68.4]
delta = [0, -13.1, -8.0, -4.8]
colors_abl = ['#2ecc71', '#e74c3c', '#e67e22', '#f39c12']

bars2 = ax2.bar(configs, ablation_acc, color=colors_abl, alpha=0.85,
               edgecolor='black', linewidth=2, width=0.6)
ax2.set_ylabel('Balanced Accuracy (%)', fontweight='bold', fontsize=11)
ax2.set_title('(b) Ablation Analysis (Remove One Modality)', fontweight='bold', pad=12, fontsize=12)
ax2.set_ylim(0, 80)
ax2.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
for bar, val, d in zip(bars2, ablation_acc, delta):
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1.5,
            f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
    if d < 0:
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height()/2,
                f'{d:.1f}%', ha='center', va='center', fontweight='bold', fontsize=10, color='white',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='darkred', edgecolor='white', linewidth=1, alpha=0.9))

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(OUTPUT_DIR / 'fig9_ablation.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'fig9_ablation.png', dpi=300, bbox_inches='tight')
plt.close()
print(" Saved: fig9_ablation.pdf/.png")