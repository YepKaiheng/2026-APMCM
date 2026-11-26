"""Produce the manuscript's frozen scores and grouped development diagnostics."""
import argparse
import hashlib
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "exploration"))
from compare_surrogates import DATA, composition_features, fit_mixture, fit_predict, predict_mixture

OUT = HERE / "artifacts"
OUT.mkdir(exist_ok=True)


def load_file(prefix):
    return pd.read_csv(next(DATA.glob(prefix + "_*.csv")))


def scores():
    train, validation, hidden, full = map(load_file, ["02", "03", "04", "08"])
    runs = []
    for run, labeled, test in [
        ("q1_holdout", train, hidden),
        ("q23_full", pd.concat([train, validation], ignore_index=True), full),
    ]:
        start = perf_counter()
        features = composition_features(labeled)
        test_features = composition_features(test)
        t1 = perf_counter()
        models = fit_mixture(labeled, features, 26)
        t2 = perf_counter()
        predicted = predict_mixture(models, test_features)
        t3 = perf_counter()
        result = test[["id", "formula", "A", "B", "X"]].assign(V_hat=predicted)
        assert result.id.is_unique and np.isfinite(predicted).all()
        path = OUT / (run + ".csv")
        result.to_csv(path, index=False, lineterminator="\n", float_format="%.15g")
        runs.append(dict(run=run, seed=26, train_rows=len(labeled), candidates=len(test),
                         feature_s=t1-start, fit_s=t2-t1, predict_s=t3-t2,
                         sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    pd.DataFrame(runs).to_csv(OUT / "score_runs.csv", index=False)


def evaluate(public, scores_by_model, fold):
    rows = []
    for model, predicted in scores_by_model.items():
        frame = public.assign(predicted=predicted)
        for pool, subset in [("all", frame), ("Q2", frame[frame.X.eq("NO2")]),
                             ("Q3", frame[frame.X.isin(["NO2", "O3", "FO2"])])]:
            ranked = subset.sort_values(["predicted", "id"])
            picked = ranked.iloc[0]
            rows.append(dict(fold=fold, model=model, pool=pool, rows=len(ranked),
                             MAE=(ranked.predicted-ranked.V).abs().mean(),
                             RMSE=np.sqrt(((ranked.predicted-ranked.V)**2).mean()),
                             first_id=picked.id, first_V=picked.V, minimum=ranked.V.min(),
                             regret=picked.V-ranked.V.min(), targets=(ranked.V<1).sum(),
                             hits50=(ranked.head(50).V<1).sum()))
    return rows


def development():
    train = load_file("02")
    pairs = train.groupby(["A", "B"], sort=True).ngroup()
    rows = []
    for fold, (a, b) in enumerate(GroupKFold(n_splits=5, shuffle=True, random_state=26)
                                   .split(train, groups=pairs), start=1):
        assert not set(pairs.iloc[a]).intersection(pairs.iloc[b])
        estimates = fit_predict(train.iloc[a], train.iloc[b], 26)
        rows.extend(evaluate(train.iloc[b], estimates, fold))
    pd.DataFrame(rows).to_csv(OUT / "group_folds.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=["scores", "development"])
    args = parser.parse_args()
    (scores if args.task == "scores" else development)()

