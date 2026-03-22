import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FS = 2.0

fig, ax = plt.subplots(figsize=(58, 34))
ax.set_xlim(0, 58)
ax.set_ylim(-9.0, 28.0)
ax.axis('off')
fig.patch.set_facecolor('white')

ax.text(29, 27.0,
        'Behavioral Taxonomy: From Original Ethogram to Hierarchical Classification',
        ha='center', va='center', fontsize=int(34*FS), fontweight='bold')

R1_Y, R1_H = 19.0, 2.2
R2_Y, R2_H = 11.5, 4.0
R3_Y, R3_H =  3.5, 3.5
ARROW_PAD  = 0.18


def draw_box(cx, cy, w, h, fc, ec, lw):
    ax.add_patch(FancyBboxPatch(
        (cx - w/2, cy), w, h,
        boxstyle='round,pad=0.14',
        facecolor=fc, edgecolor=ec, linewidth=lw,
        zorder=3, clip_on=False
    ))
    return {'cx': cx, 'top': cy+h, 'bot': cy,
            'left': cx-w/2, 'right': cx+w/2, 'w': w, 'h': h}


def arrow(x1, y1, x2, y2, lw, color, rad=0.0):
    ax.annotate('',
        xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle='->', linewidth=lw, color=color,
            connectionstyle=f'arc3,rad={rad}',
            mutation_scale=28
        ), zorder=2)


for label, y_mid in [
    ('13 Behavior\nGroups:',        R1_Y + R1_H/2),
    ('6 Consolidated\nCategories:', R2_Y + R2_H/2),
    ('3 Parent\nCategories:',       R3_Y + R3_H/2),
]:
    ax.text(0.3, y_mid, label, ha='right', va='center',
            fontsize=int(22*FS), fontweight='bold')

original_behaviors = [
    'Approach', 'Touch', 'Mutual\nGrooming', 'Follow', 'Stand\nTogether',
    'Explore\nNear', 'Graze', 'Self-\nGroom', 'Stand\nStill', 'Wait',
    'Move\nAway', 'Ear Pin', 'Back\nAway',
]
x_orig    = np.linspace(2.5, 55.5, 13)
orig_boxes = []
for label, cx in zip(original_behaviors, x_orig):
    b = draw_box(cx, R1_Y, 3.5, R1_H, '#EBEBEB', '#555555', 2.0)
    ax.text(cx, R1_Y + R1_H/2, label,
            ha='center', va='center', fontsize=int(18*FS), fontweight='bold',
            zorder=5, linespacing=1.35)
    orig_boxes.append(b)

consol_info = [
    ('Affiliative-\nActive',  '#ADD8E6', '#1a6b8a', [0,1,2,3], 'n=25,993  (51.7%)'),
    ('Affiliative-\nSubtle',  '#87CEEB', '#1a6b8a', [4,5],      'n=8,904   (17.7%)'),
    ('Neutral-\nHorse',       '#90EE90', '#1a5c1a', [6,7,8],    'n=8,717   (17.3%)'),
    ('Neutral-\nHuman',       '#98FB98', '#1a5c1a', [9],         'n=4,746    (9.4%)'),
    ('Avoidant-\nHorse',      '#FFB6B6', '#8b0000', [10,11],     'n=1,754    (3.5%)'),
    ('Avoidant-\nHuman',      '#FFA07A', '#8b0000', [12],         'n=156      (0.3%)'),
]
cx2 = [(x_orig[srcs[0]] + x_orig[srcs[-1]]) / 2 for (_, _, _, srcs, _) in consol_info]

consol_boxes = []
for (label, fc, ec_color, srcs, count), cx in zip(consol_info, cx2):
    b = draw_box(cx, R2_Y, 6.5, R2_H, fc, ec_color, 2.8)
    ax.text(cx, R2_Y + R2_H*0.67, label,
            ha='center', va='center', fontsize=int(20*FS), fontweight='bold',
            zorder=5, linespacing=1.3)
    ax.plot([cx - b['w']/2 + 0.4, cx + b['w']/2 - 0.4],
            [R2_Y + R2_H*0.40, R2_Y + R2_H*0.40],
            color='#888888', lw=1.4, zorder=4)
    ax.text(cx, R2_Y + R2_H*0.17, count,
            ha='center', va='center', fontsize=int(17*FS),
            style='italic', color='#222222', zorder=5)
    consol_boxes.append(b)

for ci, (_, _, _, srcs, _) in enumerate(consol_info):
    for si in srcs:
        dx_frac = (consol_boxes[ci]['cx'] - orig_boxes[si]['cx']) / (55.5 - 2.5)
        rad = float(np.clip(dx_frac * 0.32, -0.20, 0.20))
        arrow(orig_boxes[si]['cx'], orig_boxes[si]['bot'] - ARROW_PAD,
              consol_boxes[ci]['cx'], consol_boxes[ci]['top'] + ARROW_PAD,
              lw=2.0, color='#444444', rad=rad)

parent_info = [
    ('AFFILIATIVE', '#1565C0', [0, 1]),
    ('NEUTRAL',     '#2E7D32', [2, 3]),
    ('AVOIDANT',    '#C62828', [4, 5]),
]
parent_counts = ['n=34,897  (69.4%)', 'n=13,463  (26.8%)', 'n=1,910   (3.8%)']

parent_boxes = []
for (label, color, children), cnt in zip(parent_info, parent_counts):
    left_edge  = consol_boxes[children[0]]['left']  - 0.8
    right_edge = consol_boxes[children[-1]]['right'] + 0.8
    w  = right_edge - left_edge
    cx = (left_edge + right_edge) / 2
    b  = draw_box(cx, R3_Y, w, R3_H, color, '#111111', 3.5)
    ax.text(cx, R3_Y + R3_H/2, label,
            ha='center', va='center', fontsize=int(30*FS), fontweight='bold',
            color='white', zorder=5)
    ax.text(cx, R3_Y - 0.42, cnt,
            ha='center', va='top', fontsize=int(20*FS), fontweight='bold',
            style='italic', color='#111111', zorder=5)
    parent_boxes.append(b)

for pi, (_, _, children) in enumerate(parent_info):
    for ci in children:
        dx_frac = (parent_boxes[pi]['cx'] - consol_boxes[ci]['cx']) / (55.5 - 2.5)
        rad = float(np.clip(dx_frac * 0.28, -0.20, 0.20))
        arrow(consol_boxes[ci]['cx'], consol_boxes[ci]['bot'] - ARROW_PAD,
              parent_boxes[pi]['cx'], parent_boxes[pi]['top'] + ARROW_PAD,
              lw=2.8, color='#111111', rad=rad)

bracket_y = R2_Y + R2_H + 1.1
ax.annotate('',
    xy=(consol_boxes[5]['cx'], bracket_y),
    xytext=(consol_boxes[0]['cx'], bracket_y),
    arrowprops=dict(arrowstyle='<->', linewidth=3.0, color='darkred'), zorder=6)
mid_x = (consol_boxes[0]['cx'] + consol_boxes[5]['cx']) / 2
ax.text(mid_x, bracket_y + 0.28, 'Max Imbalance: 166.6 : 1',
        ha='center', va='bottom', fontsize=int(22*FS), fontweight='bold',
        color='darkred', zorder=7,
        bbox=dict(boxstyle='round,pad=0.45', facecolor='#FFFACD',
                  edgecolor='darkred', linewidth=2.8))

SEP_Y = R3_Y - 2.0
ax.axhline(y=SEP_Y, xmin=0.005, xmax=0.995, color='#AAAAAA', lw=2.2, linestyle='--', zorder=1)

BOX_Y, BOX_H = -8.2, 5.2
ax.add_patch(FancyBboxPatch((1.0, BOX_Y), 26.0, BOX_H,
    boxstyle='round,pad=0.2', facecolor='#FFF8DC', edgecolor='#8B7355', linewidth=3.0, zorder=3))
ax.text(14.0, BOX_Y + BOX_H/2,
        'Consolidation Principles:\n1.  Maintain 3-level taxonomy\n'
        '2.  Distinguish initiating party\n3.  Active vs passive engagement',
        ha='center', va='center', fontsize=int(21*FS), linespacing=2.1, zorder=5)

ax.add_patch(FancyBboxPatch((30.0, BOX_Y), 26.0, BOX_H,
    boxstyle='round,pad=0.2', facecolor='#E3EEF9', edgecolor='#4682B4', linewidth=3.0, zorder=3))
ax.text(43.0, BOX_Y + BOX_H/2,
        'Hierarchical Classification:\nStage 1:  73.2%  Balanced Accuracy\n'
        'Stage 2A\u2013C:  82\u201399%  Balanced Accuracy\nOverall:  88.5%  Balanced Accuracy',
        ha='center', va='center', fontsize=int(21*FS), linespacing=2.1, zorder=5)

plt.tight_layout(rect=[0.01, 0, 0.99, 0.985])
plt.savefig(OUTPUT_DIR / 'fig_taxonomy_tree.pdf', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(OUTPUT_DIR / 'fig_taxonomy_tree.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: fig_taxonomy_tree.pdf / .png")