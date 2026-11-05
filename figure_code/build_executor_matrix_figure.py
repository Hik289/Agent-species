from pathlib import Path
import os
# Byte-reproducibility: matplotlib's PDF backend embeds a CreationDate from wall-clock
# time by default, which makes PDF output non-byte-identical across runs even when all
# plotted data and code are unchanged. Pin SOURCE_DATE_EPOCH so repeated runs of this
# script produce byte-identical PDFs (verified: two consecutive runs -> identical md5).
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

# Reviewer3 W5 fix: in-figure legend for asterisk (n=135 cell) and bold (SR>0.9) markers
# that were previously explained only in the caption text, not in the figure itself.
# Colormap 'YlGnBu' (ColorBrewer sequential, perceptually uniform) is already CVD-safe
# since it is a single-hue sequential ramp, not a red-green diverging scale.
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5})
fig,axes=plt.subplots(1,2,figsize=(7.0,3.55),constrained_layout=True)
for ax,M,title in zip(axes,[reported,seen],['(a) Reported panels','(b) Matched seen slice (n=135 each)']):
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
    ax.text(.5,-.24,f'Executor range {np.ptp(ex):.3f}  |  Planner range {np.ptp(pl):.3f}',
            transform=ax.transAxes,ha='center',fontsize=8)

# In-figure legend explaining the asterisk and bold-text markers (panel a only, since
# bold threshold and asterisk both apply there; panel b has no asterisks by construction)
legend_text=('* = $n{=}135$ cell (seen-only panel; all others $n{=}250$)\n'
             r'$\mathbf{bold}$ = success rate $>0.90$')
axes[0].text(0.5,-0.42,legend_text,transform=axes[0].transAxes,ha='center',va='top',
             fontsize=7.6,color='#333333',linespacing=1.5,
             bbox=dict(boxstyle='round,pad=0.3',facecolor='white',edgecolor='gray',alpha=0.95))

cb=fig.colorbar(im,ax=axes,shrink=.78,pad=.03); cb.set_label('Success rate')
fig.savefig(P/'figures/executor_dominance_full.pdf',bbox_inches='tight')
fig.savefig(P/'figures/executor_dominance_full.png',dpi=240,bbox_inches='tight')
