import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 12,
    'axes.titleweight': 'bold', 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'legend.fontsize': 9, 'figure.dpi': 100, 'savefig.dpi': 300,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.linewidth': 1.2,
})

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOTAL_SAMPLES = 50270
COLORS = {
    'affiliative':       '#2ecc71',
    'neutral':           '#3498db',
    'avoidant':          '#e74c3c',
    'affiliative_dark':  '#27ae60',
    'affiliative_light': '#52c77a',
    'neutral_dark':      '#2980b9',
    'neutral_light':     '#5dade2',
    'avoidant_dark':     '#c0392b',
    'avoidant_light':    '#e74c3c',
}
PARENT_COUNTS = {'affiliative': 34897, 'neutral': 13463, 'avoidant': 1910}
FINE_COUNTS   = {
    'affiliative-active': 25993, 'affiliative-subtle': 8904,
    'neutral-horse':       8717, 'neutral-human':      4746,
    'avoidant-horse':      1754, 'avoidant-human':      156,
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Dataset Class Distribution', fontweight='bold', fontsize=14, y=0.98)

parent_labels = ['Affiliative', 'Neutral', 'Avoidant']
parent_counts = [PARENT_COUNTS['affiliative'], PARENT_COUNTS['neutral'], PARENT_COUNTS['avoidant']]
parent_colors = [COLORS['affiliative'], COLORS['neutral'], COLORS['avoidant']]

bars1 = ax1.bar(parent_labels, parent_counts, color=parent_colors,
                alpha=0.85, edgecolor='black', linewidth=1.5, width=0.65)
ax1.set_ylabel('Number of Samples', fontweight='bold', fontsize=11)
ax1.set_title('(a) Parent Categories', fontweight='bold', pad=12, fontsize=12)
ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
ax1.set_axisbelow(True)
ax1.set_ylim(0, max(parent_counts) * 1.15)
for bar, count in zip(bars1, parent_counts):
    pct = count / TOTAL_SAMPLES * 100
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1200,
             f'{count:,}\n({pct:.1f}%)', ha='center', va='bottom',
             fontweight='bold', fontsize=9)

fine_labels = ['Aff-\nActive', 'Aff-\nSubtle', 'Neu-\nHorse', 'Neu-\nHuman', 'Avo-\nHorse', 'Avo-\nHuman']
fine_counts = [
    FINE_COUNTS['affiliative-active'], FINE_COUNTS['affiliative-subtle'],
    FINE_COUNTS['neutral-horse'],      FINE_COUNTS['neutral-human'],
    FINE_COUNTS['avoidant-horse'],     FINE_COUNTS['avoidant-human'],
]
fine_colors = [
    COLORS['affiliative_dark'],  COLORS['affiliative_light'],
    COLORS['neutral_dark'],      COLORS['neutral_light'],
    COLORS['avoidant_dark'],     COLORS['avoidant_light'],
]

bars2 = ax2.bar(range(len(fine_labels)), fine_counts, color=fine_colors,
                alpha=0.85, edgecolor='black', linewidth=1.5, width=0.7)
ax2.set_xticks(range(len(fine_labels)))
ax2.set_xticklabels(fine_labels, fontsize=9)
ax2.set_ylabel('Number of Samples', fontweight='bold', fontsize=11)
ax2.set_title('(b) Six Fine-Grained Behaviors', fontweight='bold', pad=12, fontsize=12)
ax2.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
ax2.set_axisbelow(True)
ax2.set_ylim(0, max(fine_counts) * 1.18)
for bar, count in zip(bars2, fine_counts):
    pct = count / TOTAL_SAMPLES * 100
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(fine_counts)*0.035,
             f'{count:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=8, fontweight='bold')
ax2.text(0.98, 0.95, 'Max Imbalance:\n166.6:1', transform=ax2.transAxes, ha='right', va='top',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat', edgecolor='black', linewidth=1.5, alpha=0.9),
         fontsize=9, fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(OUTPUT_DIR / 'fig1_class_distribution.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'fig1_class_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: fig1_class_distribution.pdf / .png")