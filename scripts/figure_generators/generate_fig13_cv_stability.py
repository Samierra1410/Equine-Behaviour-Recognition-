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

COLORS = {'affiliative': '#2ecc71', 'avoidant': '#e74c3c'}

folds = ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5']
cv_3cat = [73.5, 72.8, 73.7, 72.9, 73.2]
cv_6beh = [88.8, 88.2, 88.9, 88.1, 88.5]
x_pos = np.arange(len(folds))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('Cross-Validation Stability Analysis (5-Fold Stratified)',
            fontweight='bold', fontsize=14, y=0.98)

bars1 = ax1.bar(x_pos, cv_3cat, color=COLORS['avoidant'], alpha=0.85,
               edgecolor='black', linewidth=2, width=0.6)
ax1.axhline(y=73.2, color='darkred', linestyle='--', linewidth=2.5, label='Mean: 73.2%', alpha=0.8)
ax1.fill_between(x_pos, 72.6, 73.8, color='red', alpha=0.2, label='Std Dev: ±0.4%')
ax1.set_xticks(x_pos); ax1.set_xticklabels(folds, fontweight='bold')
ax1.set_ylabel('Balanced Accuracy (%)', fontweight='bold', fontsize=11)
ax1.set_title('(a) Three-Category Classification', fontweight='bold', pad=12)
ax1.set_ylim(70, 76)
ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
ax1.legend(fontsize=10, frameon=True, shadow=True, loc='lower right')
for bar, val in zip(bars1, cv_3cat):
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.15,
            f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)

bars2 = ax2.bar(x_pos, cv_6beh, color=COLORS['affiliative'], alpha=0.85,
               edgecolor='black', linewidth=2, width=0.6)
ax2.axhline(y=88.5, color='darkgreen', linestyle='--', linewidth=2.5, label='Mean: 88.5%', alpha=0.8)
ax2.fill_between(x_pos, 88.0, 89.0, color='green', alpha=0.2, label='Std Dev: ±0.3%')
ax2.set_xticks(x_pos); ax2.set_xticklabels(folds, fontweight='bold')
ax2.set_ylabel('Balanced Accuracy (%)', fontweight='bold', fontsize=11)
ax2.set_title('(b) Six-Behavior Hierarchical Classification', fontweight='bold', pad=12)
ax2.set_ylim(86, 91)
ax2.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
ax2.legend(fontsize=10, frameon=True, shadow=True, loc='lower right')
for bar, val in zip(bars2, cv_6beh):
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.12,
            f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(OUTPUT_DIR / 'fig13_cv_stability.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'fig13_cv_stability.png', dpi=300, bbox_inches='tight')
plt.close()
print(" Saved: fig13_cv_stability.pdf/.png")