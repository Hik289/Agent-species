from pathlib import Path
import os
# Byte-reproducibility: pin SOURCE_DATE_EPOCH so matplotlib's PDF backend does not embed
# a wall-clock CreationDate, keeping repeated runs byte-identical (verified via md5).
os.environ.setdefault('SOURCE_DATE_EPOCH', '1700000000')
import json
import matplotlib.pyplot as plt
import matplotlib as mpl

P=Path(__file__).resolve().parents[1]
traj=json.loads((P/'data/evo_runs/convergence_traj.json').read_text())

# Reviewer3 W5 fix: the previous palette used two shades of green (#1a9850, #66bd63) for
# the two flagship-pool runs AND a red (#d73027) for one no-flagship run, which is the
# classic red-green CVD confusion plus a within-hue (green-vs-green) confusion. Also three
# of the five lines sit on/near the 0.75 plateau for most of their length and were only
# distinguished by color, making them inseparable under deuteranopia/protanopia and on a
# black-and-white printout.
#
# Fix: Okabe-Ito CVD-safe palette (distinct hues, verified safe under all common CVD types)
# + a DIFFERENT line style and marker for every series, so no two series share more than
# one of {color, linestyle, marker}. Flagship-pool runs get solid lines; no-flagship-pool
# runs get distinct dashed/dotted/dash-dot styles so the 0.75-plateau cluster remains
# separable by line texture even where the color channel is unavailable (print/CVD).
order=[('claudeonly',        'Pure Claude (gen100)',                  '#0072B2', '-',  'o', True),
       ('cross_opus_sol',    'Opus + GPT-sol (19 saved generations)','#009E73', '-',  's', True),
       ('gptonly_v3',        'GPT-sol only (gen86)',                      '#D55E00', '--', '^', False),
       ('mixed',             'Full-mixed 4-model (gen100)',           '#E69F00', ':',  'D', False),
       ('cross_haiku_sol',   'Haiku + GPT-sol (29 saved generations)', '#CC79A7', '-.', 'v', False)]

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9})
fig,ax=plt.subplots(figsize=(6.6,3.9),constrained_layout=True)

# Plot no-flagship-pool (plateau-crowded) lines FIRST and flagship-pool lines on top
# (zorder), since the flagship lines are the ones that separate from the crowd and should
# read as the primary signal; the crowded group needs linestyle/marker, not z-stacking,
# to stay legible but drawing them first avoids visual occlusion of their distinct dash
# patterns by the (anyway non-overlapping-for-long) flagship lines.
no_flagship = [o for o in order if not o[5]]
flagship = [o for o in order if o[5]]

for name,label,color,ls,mk,is_flagship in no_flagship + flagship:
    y=traj[name]; x=list(range(len(y)))
    # Stagger markevery phase per series so markers interleave rather than stack directly
    # on top of each other where multiple series share a y-value for many generations.
    phase = {'gptonly_v3':0, 'mixed':3, 'cross_haiku_sol':6, 'claudeonly':1, 'cross_opus_sol':4}[name]
    step = max(1,len(y)//12)
    mevery = (phase % step, step)
    ax.plot(x,y,ls,color=color,lw=2.0 if is_flagship else 1.6,
            marker=mk,markevery=mevery,markersize=5.5,label=label,
            markeredgecolor='white',markeredgewidth=0.4,
            alpha=0.95 if is_flagship else 0.88,
            zorder=5 if is_flagship else 3)

ax.set_xlabel('Generation')
ax.set_ylabel('Best dev-panel SR (ALFWorld $n=8$)')
ax.set_ylim(0.58,1.03)
# ALFWorld n=8 means SR is quantized in 1/8 steps; all achievable values are k/8 for
# k=0..8. Mark every 1/8 tick (not just the four values that happen to appear in this
# run) so the reader sees the full quantization grid, per Reviewer3 W5.
eighths = [i/8 for i in range(5,9)]  # 0.625, 0.75, 0.875, 1.0 -- the range actually spanned
ax.set_yticks(eighths)
ax.set_yticklabels([f'{v:.3f}' for v in eighths])
ax.grid(alpha=.3, which='major')
ax.minorticks_on()
ax.yaxis.set_minor_locator(mpl.ticker.FixedLocator([i/8 for i in range(5,9)]))
ax.text(1.0, 1.001, '(1/8 = one task)', transform=ax.get_yaxis_transform(),
        fontsize=6.8, color='#666666', ha='left', va='center')

ax.axhline(0.75,color='#555555',lw=1.0,ls=':', zorder=1)

ax.legend(fontsize=7.4,loc='lower right',framealpha=.95,edgecolor='gray')
ax.set_title('Evolutionary search signal by gene pool (search-phase dev metric, not held-out SR)',
             fontsize=9,weight='bold')
fig.savefig(P/'figures/evo_convergence.pdf',bbox_inches='tight')
fig.savefig(P/'figures/evo_convergence.png',dpi=240,bbox_inches='tight')
print('figure written')
