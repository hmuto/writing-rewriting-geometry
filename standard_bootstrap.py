"""Standard two-way (writer x prompt) cluster bootstrap for dimension, and
direct contrasts from the saved geometry files.

Usage: python standard_bootstrap.py REPO OUTPUT [B=2000]
No API calls; existing results are never overwritten. Bootstrap sampling
matches writer_prompt_bootstrap.py (seed 42, five within-prompt draws); only
dimension is recomputed. Trace identities replace the SVD without changing
the estimator.
"""

import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from threadpoolctl import threadpool_limits

ROOT, OUT = map(Path, sys.argv[1:3])
B = int(sys.argv[3]) if len(sys.argv)>3 else 2000
OUT.mkdir(parents=True, exist_ok=True)
threadpool_limits(limits=1)
rows=[]
for backend in ['st','openai']:
    for frame in ['', '_lenmatch']:
        df=pd.read_csv(ROOT/'results/replay'/f'geometry_{backend}{frame}.csv')
        p=df.pivot(index=['genre','prompt','n'],columns='cond',values='deff')
        for other in ['Mixed','Accepted']:
            x=(p.Human-p[other]).dropna().to_numpy(); n=len(x)
            mean=float(x.mean()); se=stats.sem(x); ci=stats.t.interval(.95,n-1,loc=mean,scale=se)
            rows.append(dict(encoder=backend,frame=frame or 'main',contrast='Human-'+other,
                groups=n,mean=mean,ci_low=ci[0],ci_high=ci[1],p=stats.ttest_1samp(x,0).pvalue,
                positive=int((x>0).sum()),n_min=int(p.index.get_level_values('n').min()),
                n_max=int(p.index.get_level_values('n').max())))
pd.DataFrame(rows).to_csv(OUT/'direct_contrasts.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False),flush=True)

df=pd.read_csv(ROOT/'results/replay/sentences.csv')
meta=pd.read_csv(ROOT/'data/coauthor_metadata.csv')
story=df[df.genre=='creative'].copy()
story['writer']=story.session.map(dict(zip(meta.session_id,meta.worker_id)))
story=story.dropna(subset=['writer'])
writers=story.writer.unique(); prompts=story.prompt.unique()
groups={}
for prompt,sub in story.groupby('prompt'):
    groups[prompt]={c:(sub.index[sub.label==c].to_numpy(),sub.loc[sub.label==c,'writer'].to_numpy()) for c in ['Dismissed','Human']}

def deff(x):
    x=x-x.mean(0)
    gram=x@x.T
    return float(np.trace(gram)**2/np.sum(gram*gram))

summary={}
for backend in ['st','openai']:
    emb=np.load(ROOT/'results/replay'/f'emb_{backend}.npy',mmap_mode='r')
    # Check numerical equivalence of the trace and SVD estimators.
    x=emb[:200]; xc=x-x.mean(0); lam=np.linalg.svd(xc,compute_uv=False)**2
    assert np.isclose(deff(x),lam.sum()**2/(lam**2).sum(),rtol=2e-6)
    rng=np.random.default_rng(42); boots=[]; start=time.monotonic()
    for b in range(B):
        wd=rng.choice(writers,len(writers),replace=True)
        cnt=pd.Series(wd).value_counts()
        pd_draw=rng.choice(prompts,len(prompts),replace=True)
        gd=[]
        for prompt in pd_draw:
            gr=groups[prompt]
            D=np.repeat(gr['Dismissed'][0],[cnt.get(w,0) for w in gr['Dismissed'][1]])
            H=np.repeat(gr['Human'][0],[cnt.get(w,0) for w in gr['Human'][1]])
            if len(D)<160 or len(H)<40: continue
            n=min(200,len(D)//4,len(H)); vals=[]
            for _ in range(5):
                p=rng.permutation(len(D)); h1,h2=D[p[:len(D)//2]],D[p[len(D)//2:]]
                # Preserve even the unused reference draw to match the original RNG.
                rng.choice(len(h1),n,replace=False)
                C=emb[h2[rng.choice(len(h2),n,replace=False)]]
                X=emb[H[rng.choice(len(H),n,replace=False)]]
                vals.append(deff(X)-deff(C))
            gd.append(np.mean(vals))
        if gd: boots.append(np.mean(gd))
        if (b+1)%100==0:
            a=np.array(boots); np.save(OUT/f'boot_dimension_{backend}.npy',a)
            print(backend,b+1,'CI',np.percentile(a,[2.5,97.5]),'seconds',round(time.monotonic()-start),flush=True)
    a=np.array(boots)
    old=np.load(ROOT/'results/replay'/f'boot2_{backend}.npy')[:,0]
    delta=float(np.max(np.abs(a[:len(old)]-old)))
    assert delta<1e-3, ('Original bootstrap draws differ',backend,delta)
    checkpoints={str(n):dict(mean=float(a[:n].mean()),ci=np.percentile(a[:n],[2.5,97.5]).tolist(),fraction_nonpositive=float((a[:n]<=0).mean())) for n in [100,200,500,1000,2000] if n<=len(a)}
    summary[backend]=dict(B=len(a),max_difference_from_original=delta,checkpoints=checkpoints)
    (OUT/'bootstrap_summary.json').write_text(json.dumps(summary,indent=2))
