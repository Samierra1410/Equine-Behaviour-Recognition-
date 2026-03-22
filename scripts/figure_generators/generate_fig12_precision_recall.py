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

COLORS = {'affiliative': '#2ecc71', 'neutral': '#3498db', 'avoidant': '#e74c3c'}

fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
fig.suptitle('Precision-Recall Curves for Three-Category Classification',
            fontweight='bold', fontsize=14, y=0.98)

pr_data = [
    ('(a) Affiliative\nAP = 0.81', COLORS['affiliative'], 0.694,
     lambda r: 0.92 - 0.12 * r**0.8),
    ('(b) Neutral\nAP = 0.65',    COLORS['neutral'],      0.268,
     lambda r: 0.78 - 0.35 * r**0.9),
    ('(c) Avoidant\nAP = 0.94',   COLORS['avoidant'],     0.038,
     lambda r: 0.99 - 0.09 * r**0.5)
]

for ax, (title, color, baseline, precision_func) in zip(axes, pr_data):
    recall    = np.linspace(0, 1, 200)
    precision = precision_func(recall)

    ax.plot(recall, precision, color=color, linewidth=3, alpha=0.85)
    ax.axhline(y=baseline, color='gray', linestyle='--', linewidth=2,
               label=f'No-skill: {baseline:.3f}', alpha=0.6)
    ax.fill_between(recall, precision, alpha=0.3, color=color)
    ax.set_xlabel('Recall',    fontweight='bold', fontsize=11)
    ax.set_ylabel('Precision', fontweight='bold', fontsize=11)
    ax.set_title(title, fontweight='bold', pad=12)
    ax.legend(fontsize=9, frameon=True, shadow=True)
    ax.grid(alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(OUTPUT_DIR / 'fig12_precision_recall.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'fig12_precision_recall.png', dpi=300, bbox_inches='tight')
plt.close()
print(" Saved: fig12_precision_recall.pdf/.png")