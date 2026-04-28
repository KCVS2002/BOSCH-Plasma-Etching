import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


def rounded_box(ax, cx, cy, w, h, text, fc, ec, fs=30, fw='bold', tc='white'):
    x, y = cx - w/2, cy - h/2
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.15",
        facecolor=fc, edgecolor=ec, linewidth=3, zorder=2))
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fs,
            fontweight=fw, color=tc, linespacing=1.25, zorder=3)
    return cx, cy + h/2, cy - h/2


def big_arrow(ax, x1, y1, x2, y2, color='#90A4AE', lw=5):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                mutation_scale=30), zorder=1)


def curved_arrow(ax, x1, y1, x2, y2, rad=0.3, color='#90A4AE', lw=4):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                connectionstyle=f"arc3,rad={rad}",
                                mutation_scale=25), zorder=1)


C_INPUT  = '#FF7043'
C_ENCODE = '#42A5F5'
C_FUSION = '#AB47BC'
C_TEMP   = '#66BB6A'
C_OUTPUT = '#EF5350'
C_GRAY   = '#546E7A'


# ============================================================
# Figure 1: Architecture
# ============================================================
fig, ax = plt.subplots(figsize=(14, 22))
ax.set_xlim(0, 14)
ax.set_ylim(0, 22)
ax.axis('off')
fig.patch.set_facecolor('white')

M = 7

# Title
ax.text(M, 21.2, 'Cycle-Aware Multi-Modal\nDeep Learning Architecture',
        ha='center', va='center', fontsize=30, fontweight='bold', color='#263238',
        linespacing=1.25)

# ── STAGE 1 ──
y = 18.8
ax.text(0.3, y + 0.35, 'Stage 1', fontsize=20, fontweight='bold', color=C_INPUT)
ax.text(0.3, y - 0.2, 'Input', fontsize=14, color='#757575')

L, R = 5, 9.5
_, _, oes_b = rounded_box(ax, L, y, 4, 1.4,
    'OES\n3,648 ch', fc=C_INPUT, ec='#BF360C', fs=24)
_, _, proc_b = rounded_box(ax, R, y, 4, 1.4,
    'Process\n44 features', fc=C_INPUT, ec='#BF360C', fs=22)

# ── STAGE 2 ──
y = 16.3
ax.text(0.3, y + 0.35, 'Stage 2', fontsize=20, fontweight='bold', color=C_ENCODE)
ax.text(0.3, y - 0.2, 'Encoding', fontsize=14, color='#757575')

big_arrow(ax, L, oes_b, L, y + 0.75)
big_arrow(ax, R, proc_b, R, y + 0.75)

_, _, cnn_b = rounded_box(ax, L, y, 4, 1.4,
    '2D-CNN Encoder\n(time × wavelength)', fc=C_ENCODE, ec='#1565C0', fs=18)
_, _, mlp_b = rounded_box(ax, R, y, 4, 1.4,
    '2D-CNN Encoder\n(time × feature)', fc=C_ENCODE, ec='#1565C0', fs=18)

# Unified encoder family note (side, no overlap with arrows)
ax.text(12.3, 16.3, 'Unified\nEncoder\nFamily\n(separate\nweights)',
        ha='center', va='center', fontsize=12, color=C_ENCODE, style='italic',
        bbox=dict(boxstyle='round,pad=0.35', facecolor='#E3F2FD',
                  edgecolor='#90CAF9', lw=1.5))

# ── STAGE 3 ──
y = 13.3
ax.text(0.3, y + 0.35, 'Stage 3', fontsize=20, fontweight='bold', color=C_FUSION)
ax.text(0.3, y - 0.2, 'Fusion', fontsize=14, color='#757575')

curved_arrow(ax, L, cnn_b, M - 1.5, y + 0.75, rad=0.15, color=C_ENCODE)
curved_arrow(ax, R, mlp_b, M + 1.5, y + 0.75, rad=-0.15, color=C_ENCODE)

_, _, fuse_b = rounded_box(ax, M, y, 6, 1.4,
    'Concat + FC', fc=C_FUSION, ec='#6A1B9A', fs=28)

# Cycle boxes
y_cyc = 11.4
big_arrow(ax, M, fuse_b, M, y_cyc + 0.5)

labels = ['e1', 'e2', '...', 'e100']
positions = [4.2, 6.2, 8.2, 10.2]
for cx, lb in zip(positions, labels):
    rounded_box(ax, cx, y_cyc, 1.7, 0.9, lb,
                fc='#CE93D8', ec='#8E24AA', fs=20, fw='bold')

# x 100 cycles label - placed to the RIGHT of cycle boxes (not above arrow)
ax.text(12.3, y_cyc, 'x 100\ncycles',
        ha='center', va='center', fontsize=16, fontweight='bold', color=C_FUSION,
        bbox=dict(boxstyle='round,pad=0.35', facecolor='#F3E5F5',
                  edgecolor='#CE93D8', lw=1.5))

# ── STAGE 4 ──
y = 9.0
ax.text(0.3, y + 0.35, 'Stage 4', fontsize=20, fontweight='bold', color=C_TEMP)
ax.text(0.3, y - 0.2, 'Temporal', fontsize=14, color='#757575')

big_arrow(ax, M, y_cyc - 0.55, M, y + 0.75)

_, _, lstm_b = rounded_box(ax, M, y, 7.5, 1.4,
    'Bi-LSTM', fc=C_TEMP, ec='#2E7D32', fs=28)

# Physical basis side note
ax.text(12.3, 9.0, 'ARDE\nChamber\nDrift',
        ha='center', va='center', fontsize=13, color=C_TEMP,
        bbox=dict(boxstyle='round,pad=0.35', facecolor='#E8F5E9',
                  edgecolor='#A5D6A7', lw=1.5))

y_rep = 7.2
big_arrow(ax, M, lstm_b, M, y_rep + 0.5)
_, _, rep_b = rounded_box(ax, M, y_rep, 6.5, 1.0,
    'Wafer Representation', fc='#A5D6A7', ec='#2E7D32', fs=22, tc='#1B5E20')

ax.text(12.3, 7.2, '88\nwafers',
        ha='center', va='center', fontsize=13, color=C_TEMP,
        bbox=dict(boxstyle='round,pad=0.35', facecolor='#E8F5E9',
                  edgecolor='#A5D6A7', lw=1.5))

# ── STAGE 5 ──
y = 5.2
ax.text(0.3, y + 0.35, 'Stage 5', fontsize=20, fontweight='bold', color=C_OUTPUT)
ax.text(0.3, y - 0.2, 'Predict', fontsize=14, color='#757575')

big_arrow(ax, M, rep_b, M, y + 0.75)

# (X, Y) box on side
rounded_box(ax, 12.0, y + 1.3, 2.2, 0.9,
    '(X, Y)', fc='#FFAB91', ec='#BF360C', fs=20)
curved_arrow(ax, 12.0, y + 0.85, M + 1.8, y + 0.75, rad=-0.15, color=C_INPUT)

_, _, reg_b = rounded_box(ax, M, y, 6.5, 1.4,
    'Regression Head', fc=C_OUTPUT, ec='#C62828', fs=24)

y_out = 2.8
big_arrow(ax, M, reg_b, M, y_out + 0.6)
rounded_box(ax, M, y_out, 7, 1.2,
    'si_etch / oxide_etch', fc='#EF9A9A', ec='#C62828', fs=26, fw='bold', tc='#B71C1C')

ax.text(12.3, 2.8, '7,832\npoints',
        ha='center', va='center', fontsize=13, color=C_OUTPUT,
        bbox=dict(boxstyle='round,pad=0.35', facecolor='#FFEBEE',
                  edgecolor='#EF9A9A', lw=1.5))

plt.savefig(r'c:\Users\ljse2\Desktop\4-1\종합_설계_프로젝트\BOSCH Plasma-Etching\docs\architecture_diagram.png',
            dpi=200, bbox_inches='tight', facecolor='white', pad_inches=0.3)
plt.close()
print("Architecture diagram saved!")


# ============================================================
# Figure 2: Pipeline
# ============================================================
fig2, ax2 = plt.subplots(figsize=(16, 11))
ax2.set_xlim(0, 16)
ax2.set_ylim(0, 11)
ax2.axis('off')
fig2.patch.set_facecolor('white')

ML = 4.5
DL = 11.5
MID = 8

ax2.text(MID, 10.4, 'Experiment Pipeline',
         ha='center', fontsize=34, fontweight='bold', color='#263238')

_, _, raw_b = rounded_box(ax2, MID, 9.0, 11, 1.2,
    'Raw Data: OES + Process + Measurements',
    fc=C_GRAY, ec='#37474F', fs=20)

# Branch labels ABOVE the branches (not overlapping arrows)
ax2.text(ML, 7.7, 'ML Baseline\n(비교 기준선)',
         ha='center', fontsize=20, fontweight='bold', color=C_INPUT)
ax2.text(DL, 7.7, 'DL Proposed ★ 본 연구의 주력',
         ha='center', fontsize=20, fontweight='bold', color=C_ENCODE)

# Arrows from raw to each branch (skip the label area)
big_arrow(ax2, MID - 2, raw_b, ML + 0.7, 6.85)
big_arrow(ax2, MID + 2, raw_b, DL - 0.7, 6.85)

# ML column
_, _, fe_b = rounded_box(ax2, ML, 6.3, 5.5, 1.1,
    'Feature Engineering',
    fc=C_INPUT, ec='#BF360C', fs=20)

big_arrow(ax2, ML, fe_b, ML, 5.0)

_, _, xgb_b = rounded_box(ax2, ML, 4.4, 5, 1.1,
    'XGBoost + (X, Y)',
    fc='#FF8A65', ec='#BF360C', fs=22, fw='bold')

big_arrow(ax2, ML, xgb_b, ML, 3.15)

_, _, pml_b = rounded_box(ax2, ML, 2.55, 3.5, 0.9,
    'pred_ML',
    fc='#FFAB91', ec='#BF360C', fs=24, fw='bold', tc='#BF360C')

# DL column
_, _, ca_b = rounded_box(ax2, DL, 6.3, 5.5, 1.1,
    'Cycle-Aware DL',
    fc=C_ENCODE, ec='#1565C0', fs=22)

big_arrow(ax2, DL, ca_b, DL, 5.0)

_, _, ef_b = rounded_box(ax2, DL, 4.4, 5, 1.1,
    'Early Fusion + (X, Y)',
    fc='#64B5F6', ec='#1565C0', fs=22, fw='bold')

big_arrow(ax2, DL, ef_b, DL, 3.15)

_, _, pdl_b = rounded_box(ax2, DL, 2.55, 3.5, 0.9,
    'pred_DL',
    fc='#90CAF9', ec='#1565C0', fs=24, fw='bold', tc='#1565C0')

# Compare (between branches)
ax2.text(MID, 4.4, 'Compare\n→', ha='center', fontsize=30, fontweight='bold', color='#BDBDBD',
         linespacing=1.1)

# Comparison — arrows from both to the comparison box
ens_top_y = 1.35
curved_arrow(ax2, ML, pml_b, MID - 2.5, ens_top_y, rad=0.3, color=C_INPUT)
curved_arrow(ax2, DL, pdl_b, MID + 2.5, ens_top_y, rad=-0.3, color=C_ENCODE)

rounded_box(ax2, MID, 0.7, 11, 1.3,
    'Comparison: DL vs XGBoost  (Optional: Seed Ensemble of DL)',
    fc=C_FUSION, ec='#6A1B9A', fs=19, fw='bold')

plt.savefig(r'c:\Users\ljse2\Desktop\4-1\종합_설계_프로젝트\BOSCH Plasma-Etching\docs\pipeline_diagram.png',
            dpi=200, bbox_inches='tight', facecolor='white', pad_inches=0.3)
plt.close()
print("Pipeline diagram saved!")
