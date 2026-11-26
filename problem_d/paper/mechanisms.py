"""Recalculate edge-consistency vetoes and score margins on the archived tables."""
import numpy as np
import pandas as pd

from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT.parent / "results"
OUT = ROOT / "artifacts"


def edge_check():
    records=[]
    for seed in [26,42]:
        pool=pd.read_csv(RESULTS/f"physical_seed{seed}.csv")
        pool=pool[pool.X.isin(["O3","NO2","FO2"])]
        mismatch=np.abs(pool.pred_gap-(pool.pred_vbm-pool.pred_cbm))
        for limit in [np.inf,.05,.25,.5,1.0]:
            eligible=pool[mismatch<=limit].sort_values(["physical4","id"])
            first=eligible.iloc[0]
            records.append(dict(seed=seed,limit=limit,eligible=len(eligible),
                                near=(eligible.V<1).sum(),first_id=first.id,first_V=first.V))
    pd.DataFrame(records).to_csv(OUT/"edge_consistency.csv",index=False)


def score_margins():
    scores = pd.read_csv(OUT/"q23_full.csv")
    summaries = []
    for prefix in ["07","11","08","12"]:
        ids = pd.read_csv(next((ROOT.parent/"date").glob(prefix+"_*"))).id
        ranked = scores[scores.id.isin(ids)].sort_values(["V_hat","id"])
        best, second = ranked.iloc[0], ranked.iloc[1]
        margin = second.V_hat-best.V_hat
        summaries.append(dict(pool=prefix, first_id=best.id, second_id=second.id,
                              runner_up=second.formula, margin=margin, half_margin=margin/2,
                              close=(ranked.V_hat<=best.V_hat+2).sum()))
    pd.DataFrame(summaries).to_csv(OUT/"score_margins.csv",index=False)


if __name__=="__main__":
    edge_check()
    score_margins()
