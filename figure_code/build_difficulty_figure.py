from pathlib import Path
import os
# Use a fixed PDF date so repeated runs produce the same files.
os.environ.setdefault('SOURCE_DATE_EPOCH', '1700000000')
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt

P = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(description="Redraw the task-cost comparison using the original single-Opus results.")
parser.add_argument("--baseline", type=Path, required=True,
                    help="JSON results for 250 single-Opus tasks, with per_task, per_task_cost, and splits")
baseline = parser.parse_args().baseline
single = json.loads(baseline.read_text())
comp = json.loads((P / "data/matrix_cells/n250_opus_opus.json").read_text())
if single["splits"] != comp["splits"]:
    parser.error("baseline tasks must follow the same split order as the all-Opus results")

s_task = np.array(single['per_task'])
s_cost = np.array(single['per_task_cost'])
c_task = np.array(comp['per_task'])
n = len(s_task)
assert n == 250 and len(c_task) == 250

# Group tasks by the single-Opus baseline cost.
idx = np.argsort(s_cost, kind="stable")
q_size = n // 4
quartiles = [idx[i*q_size:(i+1)*q_size] for i in range(3)]
quartiles.append(idx[3*q_size:])  # last quartile absorbs remainder

q_single_sr, q_comp_sr, q_cost = [], [], []
for qidx in quartiles:
    q_single_sr.append(s_task[qidx].mean())
    q_comp_sr.append(c_task[qidx].mean())
    q_cost.append(s_cost[qidx].mean())

expected_q = [(1.000, 1.000, 0.060), (1.000, 1.000, 0.094),
              (1.000, 1.000, 0.138), (0.469, 0.812, 0.626)]
for i, (s_exp, c_exp, cost_exp) in enumerate(expected_q):
    assert abs(q_single_sr[i] - s_exp) < 0.01, (i, q_single_sr[i], s_exp)
    assert abs(q_comp_sr[i] - c_exp) < 0.01, (i, q_comp_sr[i], c_exp)
    assert abs(q_cost[i] - cost_exp) < 0.01, (i, q_cost[i], cost_exp)

# Separate tasks won by the two-role agent from all other tasks.
comp_wins_mask = (c_task == 1) & (s_task == 0)
comp_loses_mask = (c_task == 0) & (s_task == 1)
tie_or_lose_mask = ~comp_wins_mask

comp_wins_cost = np.sort(s_cost[comp_wins_mask])
tie_or_lose_cost = np.sort(s_cost[tie_or_lose_mask])

assert comp_wins_mask.sum() == 23, comp_wins_mask.sum()
assert tie_or_lose_mask.sum() == 227, tie_or_lose_mask.sum()
assert comp_loses_mask.sum() == 1, comp_loses_mask.sum()
assert abs(comp_wins_cost.mean() - 0.864) < 0.01, comp_wins_cost.mean()
assert abs(tie_or_lose_cost.mean() - 0.169) < 0.01, tie_or_lose_cost.mean()

# Pool the first three cost groups and distinguish curves by color and line style.
C_SINGLE = '#0072B2'   # blue (Okabe-Ito)
C_COMP = '#E69F00'     # orange (Okabe-Ito)
C_WINS = '#E69F00'     # orange: comp WINS curve
C_TIE = '#0072B2'      # blue: tie/comp-loses curve

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})
fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), constrained_layout=True)

# ---- Panel A: Q1-Q3 pooled vs Q4, bar chart ----
ax = axes[0]
pooled_single = s_task[np.concatenate(quartiles[:3])].mean()
pooled_comp = c_task[np.concatenate(quartiles[:3])].mean()
groups = ['Q1\u2013Q3 (pooled,\nlowest baseline cost)', 'Q4 (highest\nbaseline cost)']
single_vals = [pooled_single, q_single_sr[3]]
comp_vals = [pooled_comp, q_comp_sr[3]]

x = np.arange(2)
w = 0.32
bars_s = ax.bar(x - w/2, single_vals, w, color=C_SINGLE, edgecolor='#333333',
                 label='Single Opus', zorder=3)
bars_c = ax.bar(x + w/2, comp_vals, w, color=C_COMP, edgecolor='#333333',
                 hatch='//', label='Opus planner + Opus executor', zorder=3)

for i in range(2):
    ax.text(x[i] - w/2, single_vals[i] + 0.02, f'{single_vals[i]:.3f}',
            ha='center', va='bottom', fontsize=8.5)
    ax.text(x[i] + w/2, comp_vals[i] + 0.02, f'{comp_vals[i]:.3f}',
            ha='center', va='bottom', fontsize=8.5)
gain = comp_vals[1] - single_vals[1]
ax.text(x[1], max(single_vals[1], comp_vals[1]) + 0.13, f'+{gain:.3f}',
        ha='center', va='bottom', fontsize=11, fontweight='bold', color='#333333',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFF3E0', edgecolor='#CC8800', alpha=0.95))

ax.set_xticks(x)
ax.set_xticklabels(groups)
ax.set_ylabel('Success rate')
ax.set_ylim(0, 1.18)
ax.set_title('Success by baseline task cost',
             fontsize=9.5, fontweight='bold')
ax.legend(loc='lower left', fontsize=8, framealpha=0.95, edgecolor='gray')
ax.set_axisbelow(True)
ax.grid(axis='y', alpha=0.3)

# Footnote: exact per-quartile breakdown preserved in text for full transparency
ax.text(0.5, -0.30,
        f'Both agents solve every task in Q1\u2013Q3\n'
        f'(mean baseline cost \\${q_cost[0]:.3f}/\\${q_cost[1]:.3f}/\\${q_cost[2]:.3f}); '
        f'Q4 mean cost \\${q_cost[3]:.3f}',
        transform=ax.transAxes, ha='center', va='top', fontsize=7, color='#555555')

# Show how task costs are distributed within the two groups.
ax = axes[1]
y_tie = np.arange(1, len(tie_or_lose_cost) + 1) / len(tie_or_lose_cost)
y_wins = np.arange(1, len(comp_wins_cost) + 1) / len(comp_wins_cost)

ax.plot(tie_or_lose_cost, y_tie, '-', color=C_TIE, lw=2.2,
        label=f'Same outcome or single Opus wins ({len(tie_or_lose_cost)} tasks)', zorder=3)
ax.plot(comp_wins_cost, y_wins, '--', color=C_WINS, lw=2.2,
        label=f'Two-role agent wins ({len(comp_wins_cost)} tasks)', zorder=4)

mean_tie = tie_or_lose_cost.mean()
mean_wins = comp_wins_cost.mean()
ax.axvline(mean_tie, color=C_TIE, lw=1.0, ls=':', alpha=0.7)
ax.axvline(mean_wins, color=C_WINS, lw=1.0, ls=':', alpha=0.7)
ax.text(mean_tie, 0.42, f'mean \\${mean_tie:.2f}', fontsize=7.5, color=C_TIE,
        ha='right', va='center', rotation=90,
        bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.85))
ax.text(mean_wins, 0.18, f'mean \\${mean_wins:.2f}', fontsize=7.5, color=C_WINS,
        ha='left', va='center', rotation=90,
        bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.85))

ax.set_xlabel('Single Opus cost per task (USD)')
ax.set_ylabel('Share of tasks at or below this cost')
ax.set_xlim(0, 1.25)
ax.set_ylim(0, 1.03)
ax.set_title("Costs of tasks won by the two-role agent",
             fontsize=9.5, fontweight='bold')
ax.legend(loc='lower right', fontsize=8, framealpha=0.95, edgecolor='gray')
ax.grid(alpha=0.3)
ax.set_axisbelow(True)

fig.savefig(P / 'figures/scaling_difficulty.pdf', bbox_inches='tight')
fig.savefig(P / 'figures/scaling_difficulty.png', dpi=240, bbox_inches='tight')
print('Saved the task-cost figures.')
print(f'First three task groups: single Opus={pooled_single:.3f} comp={pooled_comp:.3f}')
print(f'Highest-cost task group: single Opus={q_single_sr[3]:.3f} comp={q_comp_sr[3]:.3f} gain={gain:+.3f}')
print(f'Two-role agent wins: tasks={len(comp_wins_cost)} mean_cost=${mean_wins:.3f}')
print(f'Same outcome or single Opus wins: tasks={len(tie_or_lose_cost)} mean_cost=${mean_tie:.3f}')
