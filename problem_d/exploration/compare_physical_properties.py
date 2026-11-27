"""Test a physical-property-first surrogate using the fixed official split."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr

from compare_surrogates import DATA, SITES, composition_features

PROPERTIES = ["e_form", "gap", "vbm", "cbm"]


def violation(properties):
    deviations = np.column_stack([
        properties["e_form"] - 0.21,
        1.4 - properties["gap"],
        properties["gap"] - 3.1,
        5.73 - properties["vbm"],
        properties["cbm"] - 4.50,
    ])
    return np.square(np.maximum(0, deviations / 0.5)).sum(axis=1)


def settings_for(seed):
    return dict(iterations=220, depth=5, learning_rate=0.08, random_seed=seed,
                    thread_count=4, verbose=False, allow_writing_files=False,
                    cat_features=SITES)


def predict_properties(train, features, new_features, settings):
    properties = {}
    for field in PROPERTIES:
        regressor = CatBoostRegressor(**settings).fit(features, train[field])
        properties[field] = regressor.predict(new_features)
        if field == "gap":
            properties[field] = np.maximum(0, properties[field])
    return properties


def run(seed, output):
    train = pd.read_csv(DATA / "02_castelli_train.csv")
    validation = pd.read_csv(DATA / "03_castelli_val.csv")
    features, new_features = composition_features(train), composition_features(validation)
    settings = settings_for(seed)
    properties = predict_properties(train, features, new_features, settings)
    for field in PROPERTIES:
        print(field, "MAE", mean_absolute_error(validation[field], properties[field]), flush=True)

    physical = violation(properties)
    derived = dict(properties)
    # This relation is an approximate convention in this dataset.
    derived["cbm"] = properties["vbm"] - properties["gap"]
    constrained = violation(derived)
    direct = np.maximum(0, CatBoostRegressor(**settings).fit(features, train.V).predict(new_features))
    predictions = {"direct": direct, "physical4": physical, "physical3_derived_cbm": constrained}
    table = validation[["id", "A", "B", "X", "V"]].assign(**predictions)
    for field in PROPERTIES:
        table["pred_" + field] = properties[field]
    if output:
        table.to_csv(output, index=False)
    pools = {"all": np.ones(len(table), bool), "q2": table.X.eq("NO2"),
             "q3": table.X.isin(["O3", "NO2", "FO2"])}
    for model in predictions:
        print(model, "MAE", mean_absolute_error(table.V, table[model]),
              "RMSE", mean_squared_error(table.V, table[model]) ** 0.5,
              "Spearman", spearmanr(table.V, table[model]).statistic, flush=True)
        for name, mask in pools.items():
            subset = table.loc[mask]
            ranked = subset.sort_values([model, "id"])
            true_top50 = set(subset.nsmallest(50, "V").id)
            print(name, "first V", ranked.V.iloc[0], "hits@20/50",
                  [int((ranked.head(k).V < 1).sum()) for k in [20, 50]],
                  "true-top50 overlap@50", len(set(ranked.head(50).id) & true_top50),
                  flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=26)
    parser.add_argument("--predictions", type=Path)
    arguments = parser.parse_args()
    run(arguments.seed, arguments.predictions)
