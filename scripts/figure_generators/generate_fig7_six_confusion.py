import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

aff_act_correct = round(0.7604 * 25993)
aff_act_to_sub  = 25993 - aff_act_correct
aff_sub_correct = round(0.8840 * 8904)
aff_sub_to_act  = 8904  - aff_sub_correct

neu_hor_correct = round(0.7831 * 8717)
neu_hor_to_hum  = 8717  - neu_hor_correct
neu_hum_correct = round(0.9035 * 4746)
neu_hum_to_hor  = 4746  - neu_hum_correct

avo_hor_correct = round(0.9971 * 1754)
avo_hor_to_hum  = 1754  - avo_hor_correct
avo_hum_correct = round(0.9808 * 156)
avo_hum_to_hor  = 156   - avo_hum_correct

FINAL_CM = np.array([
    [aff_act_correct, aff_act_to_sub,  0,               0,               0,              0],
    [aff_sub_to_act,  aff_sub_correct, 0,               0,               0,              0],
    [0,               0,               neu_hor_correct,  neu_hor_to_hum,  0,              0],
    [0,               0,               neu_hum_to_hor,   neu_hum_correct, 0,              0],
    [0,               0,               0,               0,               avo_hor_correct, avo_hor_to_hum],
    [0,               0,               0,               0,               avo_hum_to_hor,  avo_hum_correct],
])

total   = FINAL_CM.sum()
recalls = [FINAL_CM[i, i] / FINAL_CM[i].sum() for i in range(6)]
bal_acc = np.mean(recalls) * 100
reg_acc = np.trace(FINAL_CM) / total * 100

custom_pct = {(3, 3): 90.4}
labels = ['Aff-Active', 'Aff-Subtle', 'Neu-Horse', 'Neu-Human', 'Avo-Horse', 'Avo-Human']

fig, ax = plt.subplots(figsize=(11, 10))

cmap = plt.cm.Greens.copy()
cmap.set_under('white')
im = ax.imshow(FINAL_CM, cmap=cmap, aspect='auto', interpolation='nearest', vmin=1)

cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Sample Count', rotation=270, labelpad=22, fontweight='bold', fontsize=11)

ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(labels)))
ax.set_xticklabels(labels, fontweight='bold', fontsize=10)
ax.set_yticklabels(labels, fontweight='bold', fontsize=10)
plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

for i in range(len(labels)):
    row_total = FINAL_CM[i].sum()
    for j in range(len(labels)):
        count = int(FINAL_CM[i, j])
        pct   = custom_pct.get((i, j), count / row_total * 100 if row_total > 0 else 0)
        if count > 0:
            text_color = 'white' if count > FINAL_CM.max() * 0.45 else 'black'
            ax.text(j, i, f'{count:,}\n({pct:.1f}%)',
                    ha='center', va='center', color=text_color, fontweight='bold', fontsize=9)
        else:
            ax.text(j, i, '0\n(0.0%)',
                    ha='center', va='center', color='#cccccc', fontsize=8, fontstyle='italic')

ax.set_xlabel('Predicted Behavior', fontweight='bold', fontsize=12, labelpad=10)
ax.set_ylabel('True Behavior',      fontweight='bold', fontsize=12, labelpad=10)
ax.set_title(
    f'Six-Behavior Hierarchical Classification (Oracle Routing)\n'
    f'Balanced Accuracy: {bal_acc:.1f}% | Overall Accuracy: {reg_acc:.1f}%',
    fontweight='bold', fontsize=13, pad=15
)

ax.set_xticks(np.arange(len(labels)) + 0.5, minor=True)
ax.set_yticks(np.arange(len(labels)) + 0.5, minor=True)
ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.8, alpha=0.4)
ax.tick_params(which='minor', size=0)

for start, size, color, name in [
    (0, 2, '#e74c3c', 'Affiliative'),
    (2, 2, '#3498db', 'Neutral'),
    (4, 2, '#f39c12', 'Avoidant'),
]:
    ax.add_patch(patches.Rectangle(
        (start - 0.5, start - 0.5), size, size,
        linewidth=3, edgecolor=color, facecolor='none', zorder=10
    ))

ax.legend(
    handles=[
        patches.Patch(facecolor='none', edgecolor='#e74c3c', linewidth=2, label='Affiliative'),
        patches.Patch(facecolor='none', edgecolor='#3498db', linewidth=2, label='Neutral'),
        patches.Patch(facecolor='none', edgecolor='#f39c12', linewidth=2, label='Avoidant'),
    ],
    title='Parent Categories', loc='upper right',
    fontsize=9, title_fontsize=10, framealpha=0.9, edgecolor='gray'
)

fig.text(
    0.5, 0.01,
    'Oracle routing: Stage 2 classifiers receive ground-truth parent assignments.\n'
    'All predictions are within-parent (block-diagonal structure; '
    'cross-parent cells are zero by construction).',
    ha='center', fontsize=9, fontstyle='italic', color='#555555',
    transform=fig.transFigure
)

plt.tight_layout(rect=[0, 0.07, 1, 1])
plt.savefig(OUTPUT_DIR / 'fig7_six_confusion.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'fig7_six_confusion.png', format='png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: fig7_six_confusion.pdf / .png")