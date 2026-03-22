import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FIG_W, FIG_H = 58, 22
fig = plt.figure(figsize=(FIG_W, FIG_H))
ax  = fig.add_subplot(111)
ax.set_xlim(0, 58)
ax.set_ylim(0, 22)
ax.axis('off')
fig.patch.set_facecolor('white')

COLORS = {
    'yolo':        '#2980b9',
    'mediapipe':   '#8e44ad',
    'ap10k':       '#d35400',
    'fusion':      '#16a085',
    'stage1':      '#c0392b',
    'affiliative': '#27ae60',
    'neutral':     '#2574a9',
    'avoidant':    '#c0392b',
    'output':      '#e67e22',
}

FS     = 2.5
MID_Y  = 11.0
TALL_H = 9.5
TALL_Y0 = MID_Y - TALL_H / 2

SBH     = 2.9
SGAP    = 0.45
STACK_Y0 = MID_Y - (3*SBH + 2*SGAP) / 2

GAP = 3.2
C1_X, C1_W = 0.4, 2.8
C2_X = C1_X + C1_W + GAP;  C2_W = 6.4
C3_X = C2_X + C2_W + GAP;  C3_W = 5.2
C4_X = C3_X + C3_W + GAP;  C4_W = 5.2
C5_X = C4_X + C4_W + GAP;  C5_W = 7.2
C6_X = C5_X + C5_W + GAP;  C6_W = 6.0


def rbox(x0, y0, w, h, fc, ec='#222222', lw=2.8):
    ax.add_patch(FancyBboxPatch(
        (x0, y0), w, h,
        boxstyle='round,pad=0.13',
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3
    ))
    return {'cx': x0+w/2, 'cy': y0+h/2, 'top': y0+h,
            'bot': y0, 'left': x0, 'right': x0+w}


def arrow(x1, x2, y, lw=3.2, ms=32):
    ax.add_patch(FancyArrowPatch(
        (x1, y), (x2, y),
        arrowstyle='->', mutation_scale=ms,
        linewidth=lw, color='#111111', zorder=4
    ))


def t(x, y, s, fs, fw='normal', c='white', ls=1.3):
    ax.text(x, y, s, ha='center', va='center',
            fontsize=fs, fontweight=fw, color=c, zorder=5, linespacing=ls)


ax.text(29, 21.2, 'Multi-Modal Hierarchical Classification Framework',
        ha='center', va='center', fontsize=int(40*FS), fontweight='bold')

v = rbox(C1_X, TALL_Y0, C1_W, TALL_H, '#dce3ea', ec='#555555')
t(v['cx'], v['cy']+1.00, 'Video\nInput',  int(19*FS), 'bold', '#111111')
t(v['cx'], v['cy']-0.20, '1920\u00d71080', int(15*FS), c='#333333')
t(v['cx'], v['cy']-0.95, '25\u201330 fps',  int(15*FS), c='#333333')

pipe_defs = [
    ('YOLOv8-nano',    '12 Spatial Features', COLORS['yolo']),
    ('MediaPipe Pose', '15 Human Features',   COLORS['mediapipe']),
    ('AP-10K HRNet',   '8 Equine Features',   COLORS['ap10k']),
]
pipe_boxes = []
for i, (name, feat, color) in enumerate(pipe_defs):
    y0 = STACK_Y0 + (2-i) * (SBH + SGAP)
    b  = rbox(C2_X, y0, C2_W, SBH, color)
    t(b['cx'], b['cy']+0.33, name, int(17*FS), 'bold')
    t(b['cx'], b['cy']-0.33, feat, int(15*FS))
    pipe_boxes.append(b)

f = rbox(C3_X, TALL_Y0, C3_W, TALL_H, COLORS['fusion'])
t(f['cx'], f['cy']+1.55, 'Feature',       int(20*FS), 'bold')
t(f['cx'], f['cy']+0.65, 'Concatenation', int(20*FS), 'bold')
t(f['cx'], f['cy']-0.15, '35D Vector',    int(17*FS), c='#d0f5ed')
t(f['cx'], f['cy']-0.90, '(12+15+8)',     int(16*FS))

s1 = rbox(C4_X, TALL_Y0, C4_W, TALL_H, COLORS['stage1'])
t(s1['cx'], s1['cy']+1.55, 'Stage 1',      int(22*FS), 'bold')
t(s1['cx'], s1['cy']+0.65, 'CatBoost',     int(19*FS))
t(s1['cx'], s1['cy']-0.10, '3 Categories', int(18*FS))
t(s1['cx'], s1['cy']-0.90, '73.2%',        int(19*FS), 'bold')
t(s1['cx'], s1['cy']-1.55, 'Bal. Acc.',    int(17*FS))

s2_defs = [
    ('Affiliative\nSub-behaviors', '82.2%', COLORS['affiliative']),
    ('Neutral\nSub-behaviors',     '84.3%', COLORS['neutral']),
    ('Avoidant\nSub-behaviors',    '98.9%', COLORS['avoidant']),
]
s2_boxes = []
for i, (label, acc, color) in enumerate(s2_defs):
    y0 = STACK_Y0 + (2-i) * (SBH + SGAP)
    b  = rbox(C5_X, y0, C5_W, SBH, color)
    t(b['cx'], b['cy']+0.33, label,             int(16*FS), 'bold', ls=1.2)
    t(b['cx'], b['cy']-0.40, f'Bal.Acc: {acc}', int(15*FS), 'bold')
    s2_boxes.append(b)

out = rbox(C6_X, TALL_Y0, C6_W, TALL_H, COLORS['output'])
t(out['cx'], out['cy']+1.55, 'Final',          int(21*FS), 'bold')
t(out['cx'], out['cy']+0.65, 'Classification', int(21*FS), 'bold')
t(out['cx'], out['cy']-0.20, '6 Fine-Grained', int(17*FS))
t(out['cx'], out['cy']-0.90, 'Behaviors',      int(17*FS))
t(out['cx'], out['cy']-1.60, '88.5% Bal.Acc',  int(19*FS), 'bold')

arrow(v['right'],  C2_X,      MID_Y)
arrow(C2_X+C2_W,  C3_X,      MID_Y)
arrow(f['right'],  C4_X,      MID_Y)
arrow(s1['right'], C5_X,      MID_Y)
arrow(C5_X+C5_W,  C6_X,      MID_Y)

header_y = TALL_Y0 + TALL_H + 0.9
for label, cx in [
    ('Input',              v['cx']),
    ('Feature Extraction', C2_X + C2_W/2),
    ('Fusion',             f['cx']),
    ('Stage 1',            s1['cx']),
    ('Stage 2',            C5_X + C5_W/2),
    ('Output',             out['cx']),
]:
    ax.text(cx, header_y, label, ha='center', va='bottom',
            fontsize=int(35*FS), fontstyle='italic', color='#444444')

plt.tight_layout(pad=0.4)
plt.savefig(OUTPUT_DIR / 'fig2_system_overview.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'fig2_system_overview.png', format='png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: fig2_system_overview.pdf / .png")