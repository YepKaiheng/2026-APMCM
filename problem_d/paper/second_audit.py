"""Additional calibration diagnostics and whole-grid Kaiwu annealing attempt."""
from pathlib import Path
import json,time
import numpy as np
import pandas as pd
import kaiwu as kw
from figures import plt,save,SAGE,BLUE
from audit_sdk_precision import qubo
HERE=Path(__file__).resolve().parent
out=HERE/'artifacts'
full=pd.read_csv(out/'q23_full.csv'); hidden=pd.read_csv(out/'q1_holdout.csv')
public=pd.concat([pd.read_csv(next((HERE.parent/'date').glob(k+'_*.csv'))) for k in ['02','03']])
joined=full.merge(public[['id','V']],on='id',validate='one_to_one')
rows=[]
for name,f in [('All public Q3 families',joined),('Public NO2 near-targets',joined[(joined.X=='NO2')&(joined.V<1)]),('Public near-targets',joined[joined.V<1])]:
 rows.append(dict(group=name,n=len(f),true_min=f.V.min(),pred_min=f.V_hat.min(),pred_median=f.V_hat.median(),bias=(f.V_hat-f.V).mean(),mae=(f.V_hat-f.V).abs().mean()))
pd.DataFrame(rows).to_csv(out/'fit_residuals.csv',index=False)
top=hidden.sort_values(['V_hat','id']).head(20);top.to_csv(out/'q1_top20.csv',index=False)
fig,ax=plt.subplots(figsize=(6.5,2.7));ax.hist(hidden.V_hat,bins=35,color=SAGE);ax.set(xlabel='Uncalibrated predicted violation score',ylabel='Hidden compositions');save(fig,'q1_distribution')
lines=[r'\subsection{Score calibration and hidden-set output}',r'The mixture is an uncalibrated prediction of violation used for ranking. A high predicted score does not exclude a physically good composition. The following in-training diagnostic uses the frozen 02+03 refit on public rows; it is not a generalization estimate.',r'\begin{table}[htbp]\centering\small\caption{In-training residual diagnostics for the three-family score table. Bias is mean predicted minus observed violation.}\begin{tabular}{lrrrr}\toprule Group & $n$ & Min. prediction & Median prediction & Bias\\\midrule']
for r in rows:lines.append(f"{r['group']} & {r['n']} & {r['pred_min']:.3f} & {r['pred_median']:.3f} & {r['bias']:.3f}"+r'\\')
lines += [r'\bottomrule\end{tabular}\end{table}',f'The minimum of all 3,703 Q1 hidden scores is {hidden.V_hat.min():.3f}. The organizers report 19 hidden zero-violation targets, so score magnitude must not be interpreted as a calibrated physical measurement.',r'\begin{figure}[htbp]\centering\includegraphics[width=.85\linewidth]{figures/q1_distribution.pdf}\caption{Distribution of Q1 predictions from the 02-only model. Hidden physical labels are unavailable.}\end{figure}',r'\begin{table}[htbp]\centering\small\caption{First 20 hidden compositions ranked by the Q1 model. All entries are predictions.}\begin{tabular}{rrlr}\toprule Rank & ID & Formula & Score\\\midrule']
for i,r in enumerate(top.itertuples(),1):lines.append(f'{i} & {r.id} & {r.formula} & {r.V_hat:.4f}'+r'\\')
lines += [r'\bottomrule\end{tabular}\end{table}']
(HERE/'calibration.tex').write_text('\n'.join(lines)+'\n')
# Whole-grid precision and stochastic solver experiment; preserve failure evidence.
cost=full.groupby(['A','B']).V_hat.min().unstack().to_numpy();records=[]
for cap in [1.,.1,.01]:
 high=np.quantile(cost,cap);d=(np.minimum(cost,high)-cost.min())/(high-cost.min())
 raw,_=kw.core.qubo_matrix_to_ising_matrix(qubo(d,1.1));j=kw.preprocess.adjust_ising_matrix_precision(raw,bit_width=8)
 for seed in range(5):
  t=time.perf_counter()
  try:
   solver=kw.classical.SimulatedAnnealingOptimizer(initial_temperature=100,alpha=.95,cutoff_temperature=.01,iterations_per_t=10,size_limit=10,rand_seed=seed)
   s=solver.solve(j);bits=(s[:,:-1]*s[:,-1:]+1)//2;legal=(bits[:,:52].sum(1)==1)&(bits[:,52:].sum(1)==1)
   vals=[cost[a.argmax(),b.argmax()] for a,b in zip(bits[legal,:52],bits[legal,52:])]
   records.append(dict(cap_quantile=cap,seed=seed,seconds=time.perf_counter()-t,feasible=int(legal.sum()),best=min(vals) if vals else None,hit=bool(vals and np.isclose(min(vals),cost.min())),status='completed'))
  except Exception as e:
   records.append(dict(cap_quantile=cap,seed=seed,seconds=time.perf_counter()-t,status='unavailable',error=type(e).__name__+': '+str(e)))
   break
 if records[-1]['status']=='unavailable':break
pd.DataFrame(records).to_csv(out/'whole_grid_annealing.csv',index=False)
print(json.dumps(records,indent=2))

# Replay equal-weight rank ensembles on archived validation predictions.
ensemble_rows=[]
for seed in (26,42):
 a=pd.read_csv(HERE.parent/'results'/f'validation_seed{seed}.csv')
 b=pd.read_csv(HERE.parent/'results'/f'physical_seed{seed}.csv')[['id','physical4']]
 a=a.merge(b,on='id',validate='one_to_one')
 for task,mask in [('Q2',a.X.eq('NO2')),('Q3',a.X.isin(['O3','NO2','FO2']))]:
  f=a[mask].copy()
  f['rank_ensemble']=sum(f[c].rank(method='min') for c in ['direct','mixture','physical4'])/3
  for model in ['mixture','physical4','rank_ensemble']:
   z=f.sort_values([model,'id'])
   ensemble_rows.append(dict(seed=seed,task=task,model=model,first_id=int(z.iloc[0].id),first_V=z.iloc[0].V,hits50=int((z.head(50).V<1).sum())))
pd.DataFrame(ensemble_rows).to_csv(out/'rank_ensemble.csv',index=False)
