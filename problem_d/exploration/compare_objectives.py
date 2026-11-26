"""Compare three regression objectives using identical composition features."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error

from compare_surrogates import DATA, SITES, composition_features


COMPONENTS = ["v_form", "v_gap_low", "v_gap_high", "v_vbm", "v_cbm"]


def fit(train, validation, seed):
    features = composition_features(train)
    new_features = composition_features(validation)
    options = dict(iterations=220, depth=5, learning_rate=0.08, random_seed=seed,
                   thread_count=4, verbose=False, allow_writing_files=False, cat_features=SITES)

    def predict(target):
        return CatBoostRegressor(**options).fit(features, target).predict(new_features)

    direct = np.maximum(0, predict(train.V))
    log_score = np.maximum(0, np.expm1(predict(np.log1p(train.V))))
    penalties = [np.maximum(0, predict(train[column])) for column in COMPONENTS]
    return {"direct": direct, "log": log_score, "components": np.sum(penalties, axis=0)}


def report(validation, predictions):
    groups = {"all": np.ones(len(validation), dtype=bool),
              "q2": validation.X.eq("NO2"), "q3": validation.X.isin(["O3", "NO2", "FO2"])}
    for name, scores in predictions.items():
        print(name, "MAE", round(mean_absolute_error(validation.V, scores), 5))
        for group, mask in groups.items():
            ranked = validation.loc[mask].assign(score=scores[mask]).sort_values(["score", "id"])
            budgets = [1, 10, 20, 50, 100]
            minimum = [round(float(ranked.V.head(k).min()), 5) for k in budgets]
            hits = [int((ranked.V.head(k) < 1).sum()) for k in budgets[1:]]
            print(group, "min@1/10/20/50/100", minimum, "hits@10/20/50/100", hits)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=26)
    parser.add_argument("--predictions", type=Path)
    args = parser.parse_args()
    train = pd.read_csv(DATA / "02_castelli_train.csv")
    validation = pd.read_csv(DATA / "03_castelli_val.csv")
    predictions = fit(train, validation, args.seed)
    if args.predictions:
        output = validation[["id", "A", "B", "X", "V"]].assign(**predictions)
        output.to_csv(args.predictions, index=False)
    report(validation, predictions)
