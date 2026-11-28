"""Kaiwu exact block verification with explicit blind-pool membership."""
import time
import numpy as np
import pandas as pd
import kaiwu as kw
from solve import exact_block,OUT
hidden=pd.read_csv(OUT/'q1_holdout.csv');full=pd.read_csv(OUT/'q23_full.csv')
frames={'Q2':pd.read_csv(OUT/'q2_blind_comparison.csv').rename(columns={'property_score':'score'}),'Q3':full[full.id.isin(hidden.id)].rename(columns={'V_hat':'score'})}
records=[]
for task,pool in frames.items():
 best=pool.sort_values(['score','id']).drop_duplicates(['A','B']);elements=sorted(full.A.unique())
 cost=best.pivot(index='A',columns='B',values='score').reindex(index=elements,columns=elements).to_numpy()
 ids=best.pivot(index='A',columns='B',values='id').reindex(index=elements,columns=elements).fillna(999999).to_numpy(int)
 for a in range(0,52,4):
  for b in range(0,52,4):
   c=cost[a:a+4,b:b+4];valid=np.isfinite(c)
   if not valid.any():continue
   bound=c[valid].max();scaled=np.full((4,4),15.);scaled[valid]=14*c[valid]/bound if bound else 0
   t=time.perf_counter();r=exact_block(scaled,kw.classical.BruteForceOptimizer(),ids[a:a+4,b:b+4]);i,j=r['i'],r['j'];assert valid[i,j]
   records.append(dict(task=task,block_a=a//4,block_b=b//4,id=int(ids[a+i,b+j]),score=c[i,j],seconds=time.perf_counter()-t))
 result=min([r for r in records if r['task']==task],key=lambda r:(r['score'],r['id']))
 expected=pool.sort_values(['score','id']).iloc[0];assert result['id']==expected.id
 print(task,result,flush=True)
pd.DataFrame(records).to_csv(OUT/'blind_block_runs.csv',index=False)
