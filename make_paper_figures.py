"""Figures 1-3 of the main text, rebuilt from the saved results.

Usage: python make_paper_figures.py [--results results/replay]
Writes figs/fig1_three_actions, figs/fig2_three_forms and
figs/fig3_dose_response (.pdf and .png). It does not run any analysis.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy import stats

ROOT=Path(__file__).resolve().parent
ap=argparse.ArgumentParser()
ap.add_argument('--results', type=Path, default=ROOT/'results'/'replay')
args=ap.parse_args()
OUT=ROOT/'figs'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
                    'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                    'pdf.fonttype':42,'savefig.dpi':240})
GREY='#8a8a8a'; GREEN='#28774b'

def save(fig,name):
    for ext in ['pdf','png']:
        fig.savefig(OUT/f'{name}.{ext}',bbox_inches='tight')
    plt.close(fig)

# Covariance ellipses have the same total variance (sum of squared axes).
# Rewriting changes orientation; writing makes the spectrum more balanced.
fig,axs=plt.subplots(1,3,figsize=(8.0,3.0))
for ax,title,angle,axes,summary in zip(axs,
        ['(a) Reference','(b) Different orientation','(c) Higher dimension'],
        [0,30,30],[(4.6,1.8),(4.6,1.8),(4.0,np.sqrt(4.6**2+1.8**2-4**2))],
        ['Variance concentrated\nin one direction',
         'Direction changes\nConcentration unchanged',
         'Variance shared\nmore evenly']):
    ax.add_patch(Ellipse((0,0),4.6,1.8,facecolor=GREY,edgecolor=GREY,alpha=.22,lw=1.7))
    ax.add_patch(Ellipse((0,0),*axes,angle=angle,fill=False,edgecolor=GREEN,lw=2,ls='--'))
    for ang,leng,col in [(0,2.3,GREY),(angle,axes[0]/2,GREEN)]:
        a=np.deg2rad(ang)
        ax.plot([0,leng*np.cos(a)],[0,leng*np.sin(a)],color=col,lw=1.5)
    ax.set(xlim=(-2.65,2.65),ylim=(-1.7,1.7),aspect='equal')
    ax.axis('off'); ax.set_title(title,fontsize=12,pad=8)
    ax.text(.5,-.03,summary,transform=ax.transAxes,ha='center',va='top',fontsize=11,
            linespacing=1.45,color=GREEN if title.startswith('(c)') else '#333333')
fig.subplots_adjust(bottom=.30,wspace=.16,top=.86)
fig.text(.5,.02,'Grey: reference   •   Green: comparison   •   Conceptual illustration; equal total variance',
         ha='center',fontsize=9,color='#444444')
save(fig,'fig1_three_actions')

conds=['Accepted','Mixed','Human']
models=[('st','MiniLM','#3a6ea5'),('openai','OpenAI','#c9622a')]
fig,axs=plt.subplots(1,3,figsize=(8.0,3.6))
summary=[]
for ax,measure,title,ylabel in zip(axs,['deff','orient','oor'],
    ['(a) Effective\ndimension','(b) Orientation\ndifference','(c) Out-of-reference\nrate'],
    ['Difference from control','Excess variance share','Excess sentence share']):
    for j,(model,label,color) in enumerate(models):
        if measure=='orient':
            d=pd.read_csv(args.results/f'spectral_null_{model}.csv')
            vals=[d.loc[d.cond==c,'dL_orient'].to_numpy() for c in conds]
        else:
            d=pd.read_csv(args.results/f'geometry_{model}.csv')
            p=d.pivot_table(index='prompt',columns='cond',values=measure)
            vals=[(p[c]-p['control']).to_numpy() for c in conds]
        means=[v.mean() for v in vals]
        cis=[stats.t.ppf(.975,len(v)-1)*stats.sem(v) for v in vals]
        x=np.arange(3)+(j-.5)*.30
        ax.bar(x,means,.27,color=color,alpha=.87,label=label,zorder=2)
        ax.errorbar(x,means,yerr=cis,fmt='none',ecolor='#333333',elinewidth=1,capsize=3,zorder=3)
        summary.extend((measure,model,c,float(m),float(ci),len(v)) for c,m,ci,v in zip(conds,means,cis,vals))
    ax.axhline(0,color='#444444',lw=1)
    ax.set_xticks(range(3),['Choose','Rewrite','Write'])
    ax.set_title(title,fontsize=12,pad=12); ax.set_ylabel(ylabel,fontsize=11)
    ax.tick_params(labelsize=10)
fig.subplots_adjust(bottom=.23,top=.79,wspace=.65,left=.08,right=.99)
handles,labels=axs[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False,fontsize=11)
save(fig,'fig2_three_forms')
pd.DataFrame(summary,columns=['measure','model','operation','mean','ci95_halfwidth','groups']).to_csv(ROOT/'compact_figure_values.csv',index=False)


# Authorship-bin figure uses the same saved MiniLM estimates as the original.
d=pd.read_csv(args.results/'geometry_dose_st.csv')
bins=['h<0.2 (LLM text)', 'h 0.2-0.5 (light edit)',
      'h 0.5-0.8 (heavy edit)', 'h>0.8 (human text)']
p=d.pivot_table(index='prompt',columns='cond',values='deff')
e=pd.DataFrame({c:p[c]-p['control'] for c in bins})
o=pd.read_csv(args.results/'dose_orient_st.csv').pivot_table(index=['genre','prompt'],columns='cond',values='dL_orient')
fig,axs=plt.subplots(1,2,figsize=(7.8,3.6))
for ax,values,title in zip(axs,[e,o],['(a) Effective dimension','(b) Orientation difference']):
    for _,row in values.iterrows():
        ax.plot(range(4),row[bins].to_numpy(),color=GREY,alpha=.40,lw=.7)
    means=[values[c].mean() for c in bins]
    cis=[stats.t.ppf(.975,len(values[c])-1)*stats.sem(values[c]) for c in bins]
    ax.errorbar(range(4),means,yerr=cis,color=GREEN,lw=2,marker='o',ms=4,capsize=3)
    ax.axhline(0,color='#444444',lw=1)
    ax.set_xticks(range(4),['< 0.2\nModel','0.2–0.5\nLight','0.5–0.8\nHeavy','≥ 0.8\nHuman'])
    ax.set_xlabel('Human-typed share',fontsize=11)
    ax.set_ylabel('Excess over control',fontsize=11)
    ax.set_title(title,fontsize=12,pad=12)
    ax.tick_params(labelsize=10)
fig.subplots_adjust(bottom=.24,top=.83,wspace=.40,left=.08,right=.99)
save(fig,'fig3_dose_response')
