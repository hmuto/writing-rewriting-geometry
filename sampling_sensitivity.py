"""Sampling sensitivity analysis. No API calls or original-result writes.

Usage: python sampling_sensitivity.py REPO OUTPUT
Recreates the published 4-word-bin pool restriction and random draws exactly.
Compares unrestricted and restricted pools at identical prompts and n; adds
inverse-session-size weighted sampling as a descriptive sensitivity, not a
cluster-valid inferential procedure. All settings reported, seed 42, 20 draws.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy import stats
from threadpoolctl import threadpool_limits

ROOT, OUT = map(Path, sys.argv[1:3])
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
from geometry_forms import length_match
threadpool_limits(limits=1)
df = pd.read_csv(ROOT / 'results/replay/sentences.csv')
meta = pd.read_csv(ROOT / 'data/coauthor_metadata.csv')
writer_map = dict(zip(meta.session_id, meta.worker_id))
conds = ['Accepted', 'Mixed', 'Human']
ref = pd.read_csv(ROOT / 'results/replay/geometry_st_lenmatch.csv')
eligible = {(r.genre, r.prompt): int(r.n) for r in ref.itertuples()}

def deff(x):
    x = x - x.mean(0)
    gram = x @ x.T
    return float(np.trace(gram)**2 / np.sum(gram*gram))

def draws(sub, n, rng, balanced=False):
    pools = {c: sub.index[sub.label == c].to_numpy() for c in conds + ['Dismissed']}
    def pick(p):
        weights = None
        if balanced:
            sessions = df.loc[p, 'session']
            weights = 1 / sessions.map(sessions.value_counts()).to_numpy()
            weights = weights / weights.sum()
        return p[rng.choice(len(p), n, replace=False, p=weights)]
    for _ in range(20):
        d = pools['Dismissed']; perm = rng.permutation(len(d))
        # Consume reference-fit draw even though deff only needs the control.
        pick(d[perm[:len(d)//2]])
        sel = {'control': pick(d[perm[len(d)//2:]])}
        sel.update({c: pick(pools[c]) for c in conds})
        yield sel

selections = {}; diagnostics = []
for key, sub in df.groupby(['genre', 'prompt']):
    n_main = min(200, (sub.label == 'Dismissed').sum()//4,
                 *((sub.label == c).sum() for c in conds))
    if n_main < 60: continue
    for c in conds + ['Dismissed']:
        s = sub[sub.label == c]; counts = s.session.value_counts()
        ws = s.session.map(writer_map).dropna().value_counts() if key[0]=='creative' else pd.Series(dtype=float)
        diagnostics.append(dict(genre=key[0],prompt=key[1],condition=c,
            sentences=len(s),sessions=len(counts),max_session_share=counts.max()/len(s),
            writers=len(ws) if len(ws) else np.nan,
            max_writer_share=ws.max()/len(s) if len(ws) else np.nan))
    if key not in eligible: continue
    n = eligible[key]
    matched = length_match(sub, np.random.default_rng(42), conds)
    assert n == min(200, (matched.label=='Dismissed').sum()//4,
                    *((matched.label==c).sum() for c in conds))
    for setting in ['unrestricted', 'length_restricted', 'session_weighted', 'length_session_weighted']:
        rng = np.random.default_rng(42)
        source = sub
        if setting.startswith('length'):
            source = length_match(sub, rng, conds)
        selections[(key,setting)] = list(draws(source,n,rng,'weighted' in setting))

rows = []; draw_diag = []
for backend in ['st','openai']:
    emb = np.load(ROOT/'results/replay'/f'emb_{backend}.npy', mmap_mode='r')
    x = emb[:62]; xc=x-x.mean(0); lam=np.linalg.svd(xc,compute_uv=False)**2
    assert np.isclose(deff(x),lam.sum()**2/(lam**2).sum(),rtol=2e-6)
    for (key, setting), samples in selections.items():
        for c in ['control']+conds:
            vals = [deff(emb[s[c]]) for s in samples]
            rows.append(dict(encoder=backend,genre=key[0],prompt=key[1],setting=setting,
                             n=eligible[key],condition=c,deff=np.mean(vals)))
            if backend=='st':
                for j,s in enumerate(samples):
                    counts=df.loc[s[c],'session'].value_counts()
                    draw_diag.append(dict(genre=key[0],prompt=key[1],setting=setting,
                        condition=c,draw=j,sessions=len(counts),max_session_share=counts.max()/len(s[c])))
    print(backend, 'complete', flush=True)
res = pd.DataFrame(rows)
for backend in ['st','openai']:
    old=pd.read_csv(ROOT/'results/replay'/f'geometry_{backend}_lenmatch.csv')
    new=res[(res.encoder==backend)&(res.setting=='length_restricted')]
    merged=old.merge(new,left_on=['genre','prompt','cond'],right_on=['genre','prompt','condition'])
    error=np.max(np.abs(merged.deff_x-merged.deff_y))
    assert len(merged)==40 and error < 1e-3, (backend,error)
    print(backend,'published replication maximum error',error,flush=True)
summary=[]
for (backend,setting), sub in res.groupby(['encoder','setting']):
    p=sub.pivot(index=['genre','prompt'],columns='condition',values='deff')
    for c in ['Mixed','Accepted','control']:
        x=p.Human-p[c]; ci=stats.t.interval(.95,len(x)-1,loc=x.mean(),scale=stats.sem(x))
        summary.append(dict(encoder=backend,setting=setting,contrast='Human-'+c,groups=len(x),
            mean=x.mean(),ci_low=ci[0],ci_high=ci[1],positive=int((x>0).sum())))
res.to_csv(OUT/'sampling_per_prompt.csv',index=False)
pd.DataFrame(summary).to_csv(OUT/'sampling_summary.csv',index=False)
pd.DataFrame(diagnostics).to_csv(OUT/'pool_contributions.csv',index=False)
pd.DataFrame(draw_diag).to_csv(OUT/'sample_contributions.csv',index=False)
print(pd.DataFrame(summary).to_string(index=False),flush=True)
