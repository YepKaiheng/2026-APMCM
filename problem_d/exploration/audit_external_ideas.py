"""Audit article-inspired claims against the public data and saved score tables."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from scipy.stats import spearmanr

from compare_surrogates import DATA


def audit():
    results = DATA.parent / "results"
    labeled = pd.concat([pd.read_csv(DATA / name) for name in
                         ["02_castelli_train.csv", "03_castelli_val.csv"]], ignore_index=True)
    property_residual = labeled.vbm - labeled.cbm - labeled.gap
    print("public labeled n", len(labeled), "V zeros", int((labeled.V == 0).sum()),
          "gap zeros", int((labeled.gap == 0).sum()), "V median", labeled.V.median())
    print("correct band-edge relation max abs residual", property_residual.abs().max())
    print("stress-pool target prevalence", 29/2254, "stated full-data prevalence", 19/18928)

    validation = pd.read_csv(results / "validation_seed26.csv")
    for model in ["direct", "mixture"]:
        ranked = validation.sort_values([model, "id"])
        true_top = set(validation.nsmallest(50, "V").id)
        print("official-validation", model, "Spearman",
              spearmanr(validation.V, validation[model]).statistic,
              "V<1 recall@50", int((ranked.head(50).V < 1).sum()) / int((validation.V < 1).sum()),
              "true-top50 overlap@50", len(set(ranked.head(50).id) & true_top) / 50)

    rows = []
    predictions = pd.read_csv(results / "encoding_seed26.csv")
    for model in ["direct", "mixture"]:
        for x in ["O3", "NO2", "FO2"]:
            scores = predictions.loc[predictions.X.eq(x)].pivot(index="A", columns="B", values=model)
            original = scores.to_numpy()
            additive = original.mean(0, keepdims=True) + original.mean(1, keepdims=True) - original.mean()
            u, singular, vt = np.linalg.svd(original - additive, full_matrices=False)
            for rank in [1, 3, 5, 10]:
                distilled = additive + (u[:, :rank] * singular[:rank]) @ vt[:rank]
                chosen = np.unravel_index(distilled.argmin(), distilled.shape)
                original_top = set(np.argsort(original.ravel())[:20])
                distilled_top = set(np.argsort(distilled.ravel())[:20])
                row = dict(model=model, X=x, rank=rank,
                           R2=r2_score(original.ravel(), distilled.ravel()),
                           original_score_gap=original[chosen] - original.min(),
                           top20_overlap=len(original_top & distilled_top))
                rows.append(row)
                print(json.dumps(row))
    pd.DataFrame(rows).to_csv(results / "low_rank_audit.csv", index=False)


if __name__ == "__main__":
    audit()
