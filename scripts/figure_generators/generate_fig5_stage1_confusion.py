import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

support_aff = 34897
support_neu = 13463
support_avo = 1910
total       = 50270

diag_aff = 20061
diag_neu = 10380
diag_avo = 1621

col_aff = 22290
col_avo = 5404
col_neu = total - col_aff - col_avo

avo_to_aff = 30
avo_to_neu = (support_avo - diag_avo) - avo_to_aff
neu_to_aff = col_aff - diag_aff - avo_to_aff
neu_to_avo = (support_neu - diag_neu) - neu_to_aff
aff_to_avo = col_avo - diag_avo - neu_to_avo
aff_to_neu = support_aff - diag_aff - aff_to_avo

STAGE1_CM = np.array([
    [diag_aff,   aff_to_neu, aff_to_avo],
    [neu_to_aff, diag_neu,   neu_to_avo],
    [avo_to_aff, avo_to_neu, diag_avo  ],
])

labels   = ['Affiliative', 'Neutral', 'Avoidant']
recalls  = [STAGE1_CM[i, i] / STAGE1_CM[i].sum() for i in range(3)]
bal_acc  = np.mean(recalls) * 100
reg_acc  = np.trace(STAGE1_CM) / STAGE1_CM.sum() * 100

fig, ax = plt.subplots(figsize=(9, 8))

im = ax.imshow(STAGE1_CM, cmap='Blues', aspect='auto', interpolation='nearest')
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Sample Count', rotation=270, labelpad=22, fontweight='bold', fontsize=11)

ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(labels)))
ax.set_xticklabels(labels, fontweight='bold', fontsize=11)
ax.set_yticklabels(labels, fontweight='bold', fontsize=11)

for i in range(len(labels)):
    row_total = STAGE1_CM[i].sum()
    for j in range(len(labels)):
        count      = STAGE1_CM[i, j]
        percentage = count / row_total * 100
        text_color = 'white' if count > STAGE1_CM.max() * 0.45 else 'black'
        pct_color  = 'darkred' if percentage > 40 else text_color
        ax.text(j, i - 0.15, f'{count:,}',
                ha='center', va='center', color=text_color, fontweight='bold', fontsize=12)
        ax.text(j, i + 0.25, f'({percentage:.1f}%)',
                ha='center', va='center', color=pct_color, fontsize=10, fontweight='bold')

ax.set_xlabel('Predicted Category', fontweight='bold', fontsize=12, labelpad=10)
ax.set_ylabel('True Category',      fontweight='bold', fontsize=12, labelpad=10)
ax.set_title(
    f'Stage 1: Parent Category Classification\nBalanced Accuracy: {bal_acc:.1f}%',
    fontweight='bold', fontsize=13, pad=15
)

ax.set_xticks(np.arange(len(labels)) + 0.5, minor=True)
ax.set_yticks(np.arange(len(labels)) + 0.5, minor=True)
ax.grid(which='minor', color='gray', linestyle='-', linewidth=1.5, alpha=0.5)
ax.tick_params(which='minor', size=0)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig5_stage1_confusion.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'fig5_stage1_confusion.png', format='png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: fig5_stage1_confusion.pdf / .png")

