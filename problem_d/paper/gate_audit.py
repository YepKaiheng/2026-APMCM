"""Replay the selected surrogate to inspect its branches on unseen validation pairs."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

from screen import HERE, OUT, composition_features, fit_mixture, load_file
from compare_surrogates import METAL_BOUND


def run():
    train, val = load_file("02"), load_file("03")
    gate, gapped, metal = fit_mixture(train, composition_features(train), 26)
    features = composition_features(val)
    p = gate.predict_proba(features)[:, 1]
    vplus = np.maximum(0, gapped.predict(features))
    vzero = np.maximum(METAL_BOUND, metal.predict(features))
    y = val.gap.gt(0).to_numpy()
    prior = train.gap.gt(0).mean()
    table = val[["id", "formula", "A", "B", "X", "V"]].assign(
        positive_gap=y, p_gap=p, gapped_score=vplus, metal_score=vzero,
        soft=p*vplus+(1-p)*vzero, hard=np.where(p>=.5,vplus,vzero),
        prior=prior*vplus+(1-prior)*vzero, oracle=np.where(y,vplus,vzero))
    reference = pd.read_csv(HERE.parent/"results"/"validation_seed26.csv").set_index("id")
    np.testing.assert_allclose(table.soft, reference.loc[table.id,"mixture"], rtol=0, atol=1e-8)
    print("Gap event prevalence, average precision, Brier and constant-prior Brier",
          y.mean(), average_precision_score(y,p), brier_score_loss(y,p),
          brier_score_loss(y,np.full(len(y),prior)))
    print("Confusion: TP, FP, FN, TN",*(int(v.sum()) for v in
          [y&(p>=.5),~y&(p>=.5),y&(p<.5),~y&(p<.5)]))
    reports = []
    for pool, subset in [("All",table),("Q2",table[table.X.eq("NO2")]),
                         ("Q3",table[table.X.isin(["O3","NO2","FO2"])])]:
        for method in ["prior","hard","soft","oracle"]:
            ranked = subset.sort_values([method,"id"])
            reports.append(dict(pool=pool,method=method,hits50=(ranked.head(50).V<1).sum(),
                                first_id=ranked.id.iloc[0],first_V=ranked.V.iloc[0],
                                MAE=np.abs(ranked[method]-ranked.V).mean()))
    result = pd.DataFrame(reports)
    table.to_csv(OUT/"gate_components.csv",index=False)
    result.to_csv(OUT/"gate_ablation.csv",index=False)
    print(result.to_string(index=False))
    print(table.nsmallest(1,"soft").to_string(index=False))


if __name__=="__main__":
    run()
