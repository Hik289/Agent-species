from pathlib import Path
import os
# Use a fixed PDF date so repeated runs produce the same files.
os.environ.setdefault('SOURCE_DATE_EPOCH', '1700000000')
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

P=Path(__file__).resolve().parents[1]
order=['opus','sonnet','haiku','sol']
labels=['Opus','Sonnet','Haiku','GPT-sol']
D={}
for f in (P/'data/matrix_cells').glob('*.json'):
    d=json.loads(f.read_text()); D[tuple(d['geno'].split('->'))]=d
reported=np.array([[D[p,e]['SR'] for e in order] for p in order])
seen=np.array([[D[p,e]['seen_SR'] for e in order] for p in order])
ns=np.array([[D[p,e]['n'] for e in order] for p in order])

# Explain the task sets and text markers inside the figure.
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5})
fig,axes=plt.subplots(1,2,figsize=(7.0,3.55),constrained_layout=True)
for ax,M,title in zip(axes,[reported,seen],["(a) Each run's task set", '(b) Same 135 tasks for every pair']):
    im=ax.imshow(M,vmin=.60,vmax=.96,cmap='YlGnBu',aspect='equal')
    for i in range(4):
        for j in range(4):
            text=f'{M[i,j]:.3f}'
            if ax is axes[0] and ns[i,j] == 135: text+='*'
            ax.text(j,i,text,ha='center',va='center',fontsize=8.2,
                    color='white' if M[i,j]>.85 else 'black',weight='bold' if M[i,j]>.9 else 'normal')
    ax.set_xticks(range(4),labels,rotation=0,ha='center'); ax.set_yticks(range(4),labels)
    ax.set_xlabel('Executor'); ax.set_ylabel('Planner'); ax.set_title(title,weight='bold')
    ex=M.mean(0); pl=M.mean(1)
    ax.text(.5,-.26,f'Gap between best and worst role averages\nExecutors: {np.ptp(ex):.3f}  |  Planners: {np.ptp(pl):.3f}',
            transform=ax.transAxes,ha='center',va='top',fontsize=8,linespacing=1.3)

# In-figure legend explaining the asterisk and bold-text markers (panel a only, since
# bold threshold and asterisk both apply there; panel b has no asterisks by construction)
legend_text=('* = 135 seen tasks; all other cells = 250 tasks\n'
             r'$\mathbf{bold}$ = success rate above 0.90')
axes[0].text(0.5,-0.50,legend_text,transform=axes[0].transAxes,ha='center',va='top',
             fontsize=7.6,color='#333333',linespacing=1.5,
             bbox=dict(boxstyle='round,pad=0.3',facecolor='white',edgecolor='gray',alpha=0.95))

cb=fig.colorbar(im,ax=axes,shrink=.78,pad=.03); cb.set_label('Success rate')
fig.savefig(P/'figures/executor_dominance_full.pdf',bbox_inches='tight')
fig.savefig(P/'figures/executor_dominance_full.png',dpi=240,bbox_inches='tight')
