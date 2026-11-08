from pathlib import Path
import os
# Use a fixed PDF date so repeated runs produce the same files.
os.environ.setdefault('SOURCE_DATE_EPOCH', '1700000000')
import json
import matplotlib.pyplot as plt
import matplotlib as mpl

P=Path(__file__).resolve().parents[1]
traj=json.loads((P/'data/evo_runs/convergence_traj.json').read_text())

# Give each run a different color, line style, and marker.
order=[('claudeonly',        'Claude models only',                  '#0072B2', '-',  'o', True),
       ('cross_opus_sol',    'Opus + GPT-sol','#009E73', '-',  's', True),
       ('gptonly_v3',        'GPT-sol only',                      '#D55E00', '--', '^', False),
       ('mixed',             'All four models',           '#E69F00', ':',  'D', False),
       ('cross_haiku_sol',   'Haiku + GPT-sol', '#CC79A7', '-.', 'v', False)]

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9})
fig,ax=plt.subplots(figsize=(6.6,3.9),constrained_layout=True)

# Draw overlapping curves first, then the curves that rise above them.
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
ax.set_ylabel('Best success rate on 8 search tasks')
ax.set_ylim(0.58,1.03)
# One solved task changes the success rate by one eighth.
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
ax.set_title('Task success during search',
             fontsize=9,weight='bold')
fig.savefig(P/'figures/evo_convergence.pdf',bbox_inches='tight')
fig.savefig(P/'figures/evo_convergence.png',dpi=240,bbox_inches='tight')
print('Saved the search figures.')
