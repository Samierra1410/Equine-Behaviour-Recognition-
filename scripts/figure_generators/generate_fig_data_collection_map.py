import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FS = 1.5

fig = plt.figure(figsize=(20, 8))

ax1 = plt.subplot(1, 2, 1)
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.axis('off')

sites = {
    'Norway':             {'pos': (1.5, 7),   'color': '#B3D9FF', 'label': 'Fjord horses\nPasture\nenvironment',       'width': 1.6, 'height': 1.3},
    'Iceland':            {'pos': (3.5, 7.5), 'color': '#90EE90', 'label': 'Icelandic horses\nRocky hillside',         'width': 1.6, 'height': 1.2},
    'Sweden-1 (Saxtorp)': {'pos': (6.5, 6.5), 'color': '#FFB3BA', 'label': 'Indoor arena\nTraining\ncenter',           'width': 2.2, 'height': 1.6},
    'Sweden-2 (TorHall)': {'pos': (8.2, 5.2), 'color': '#FFFFBA', 'label': 'Outdoor\npaddock\nEquestrian\nfacility',  'width': 2.2, 'height': 1.9},
}

for site, info in sites.items():
    w, h = info.get('width', 1.6), info.get('height', 1.2)
    box = FancyBboxPatch(
        (info['pos'][0] - w/2, info['pos'][1] - h/2), w, h,
        boxstyle="round,pad=0.05", edgecolor='black', facecolor=info['color'], linewidth=2
    )
    ax1.add_patch(box)
    ax1.text(info['pos'][0], info['pos'][1] + h/2 - 0.25, site,
             ha='center', va='center', fontsize=int(9*FS), fontweight='bold')
    ax1.text(info['pos'][0], info['pos'][1] - h/2 + 0.45, info['label'],
             ha='center', va='center', fontsize=int(7.5*FS), style='italic', multialignment='center')

dot_positions = {
    'Norway':   (1.5, 5.8),
    'Iceland':  (3.5, 6.3),
    'Sweden-1': (6.5, 5.0),
    'Sweden-2': (8.2, 3.6),
}
dot_colors = ['#000080', '#006400', '#8B0000', '#FF8C00']
for i, (site, pos) in enumerate(dot_positions.items()):
    ax1.add_patch(plt.Circle(pos, 0.2, color=dot_colors[i], zorder=10))

ax1.annotate('', xy=(9.5, 9), xytext=(9.5, 8),
             arrowprops=dict(arrowstyle='->', lw=2, color='black'))
ax1.text(9.5, 9.3, 'N', ha='center', va='center', fontsize=int(14*FS), fontweight='bold')
ax1.text(5, 1, '~600 km', ha='center', va='center', fontsize=int(10*FS))
ax1.text(5, 9.5, '(a) Geographic Distribution of Data Collection Sites',
         ha='center', va='center', fontsize=int(12*FS), fontweight='bold')

legend_items = [('Norway', '#B3D9FF'), ('Iceland', '#90EE90'), ('Sweden-1', '#FFB3BA'), ('Sweden-2', '#FFFFBA')]
for i, (name, color) in enumerate(legend_items):
    y_pos = 2.5 - i * 0.5
    ax1.add_patch(patches.Rectangle((0.5, y_pos - 0.15), 0.3, 0.3,
                                     facecolor=color, edgecolor='black', linewidth=1))
    ax1.text(1.0, y_pos, name, ha='left', va='center', fontsize=int(9*FS))

ax2 = plt.subplot(1, 2, 2)
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis('off')

table_data = [
    ['Norway',  'Overcast/\nNatural',   'Flat\ngrassland', 'Open\npasture'],
    ['Iceland', 'Bright\nsunlight',     'Rocky\nhillside', 'Mountain-\nous'],
    ['Saxtorp', 'Artificial\n(indoor)', 'Arena\nfloor',    'Indoor\nwalls'],
    ['TorHall', 'Variable\noutdoor',    'Paddock\nterrain', 'Fenced\nenclosure'],
]
headers       = ['Site', 'Lighting', 'Terrain', 'Background']
colors_table  = ['#B3D9FF', '#90EE90', '#FFB3BA', '#FFFFBA']
table_top     = 8.5
table_left    = 0.5
col_width     = 2.375
row_height    = 1.0
header_height = 0.7

for i, header in enumerate(headers):
    x = table_left + i * col_width
    ax2.add_patch(patches.Rectangle((x, table_top), col_width, header_height,
                                     facecolor='white', edgecolor='black', linewidth=2))
    ax2.text(x + col_width/2, table_top + header_height/2, header,
             ha='center', va='center', fontsize=int(10*FS), fontweight='bold')

for ri, row_data in enumerate(table_data):
    y = table_top - (ri + 1) * row_height
    for ci, cell in enumerate(row_data):
        x = table_left + ci * col_width
        ax2.add_patch(patches.Rectangle((x, y), col_width, row_height,
                                         facecolor=colors_table[ri], edgecolor='black', linewidth=1.5))
        ax2.text(x + col_width/2, y + row_height/2, cell,
                 ha='center', va='center', multialignment='center',
                 fontsize=int((9 if ci == 0 else 8) * FS),
                 fontweight='bold' if ci == 0 else 'normal')

summary_w = col_width * len(headers)
summary_y = table_top - len(table_data) * row_height - 1.3
ax2.add_patch(FancyBboxPatch((table_left, summary_y), summary_w, 1.0,
                              boxstyle="round,pad=0.1", edgecolor='blue', facecolor='white', linewidth=2))
ax2.text(table_left + summary_w/2, summary_y + 0.5,
         'Dataset Summary:\n28 videos  •  4 locations  •  5 horse breeds\n85 minutes footage  •  50,270 temporal samples',
         ha='center', va='center', fontsize=int(8.5*FS), fontfamily='monospace', multialignment='center')
ax2.text(table_left + summary_w/2, 9.5, '(b) Environmental Diversity Across Sites',
         ha='center', va='center', fontsize=int(12*FS), fontweight='bold')

fig.text(0.5, 0.02,
         'Multi-site data collection introduces environmental variability (lighting, terrain, background) '
         'supporting model generalization. Base map: \u00a9OpenStreetMap contributors.',
         ha='center', va='center', fontsize=int(8*FS), style='italic', wrap=True)

plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.savefig(OUTPUT_DIR / 'fig_data_collection_map.pdf', dpi=300, bbox_inches='tight', format='pdf')
plt.savefig(OUTPUT_DIR / 'fig_data_collection_map.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: fig_data_collection_map.pdf / .png")