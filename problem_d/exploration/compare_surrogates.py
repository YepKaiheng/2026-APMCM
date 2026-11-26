"""Compare direct regression and a band-gap mixture on the official validation set."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from pymatgen.core import Element
from sklearn.metrics import mean_absolute_error


DATA = Path(__file__).resolve().parents[1] / "date"
SITES = ["A", "B", "X"]
# For the zero-gap convention VBM = CBM, the two edge penalties are
# minimized at their midpoint, and the low-gap penalty is fixed.
METAL_BOUND = (1.4 / 0.5) ** 2 + (5.73 - 4.50) ** 2 / (2 * 0.5**2)


def composition_features(frame):
    features = frame[SITES].copy()
    for site in ["A", "B"]:
        elements = {symbol: Element(symbol) for symbol in frame[site].unique()}
        for attribute in ["Z", "X", "row", "group"]:
            lookup = {symbol: float(getattr(element, attribute)) for symbol, element in elements.items()}
            features[f"{site}_{attribute}"] = frame[site].map(lookup)
    return features


def fit_predict(train, validation, seed):
    features = composition_features(train)
    new_features = composition_features(validation)
    options = dict(iterations=350, depth=5, learning_rate=0.08, random_seed=seed,
                   thread_count=4, verbose=False, allow_writing_files=False, cat_features=SITES)

    direct = CatBoostRegressor(**options).fit(features, train.V).predict(new_features)
    has_gap = train.gap > 0
    classifier = CatBoostClassifier(**options).fit(features, has_gap.astype(int))
    p_gap = classifier.predict_proba(new_features)[:, 1]
    gapped_model = CatBoostRegressor(**options).fit(features[has_gap], train.V[has_gap])
    metal_model = CatBoostRegressor(**options).fit(features[~has_gap], train.V[~has_gap])
    gapped = np.maximum(0, gapped_model.predict(new_features))
    metal = np.maximum(METAL_BOUND, metal_model.predict(new_features))
    return {"direct": direct, "mixture": p_gap * gapped + (1 - p_gap) * metal}


def main(seed, predictions_path=None):
    train = pd.read_csv(DATA / "02_castelli_train.csv")
    validation = pd.read_csv(DATA / "03_castelli_val.csv")
    for name, frame in [("train", train), ("validation", validation)]:
        metals = frame.loc[frame.gap == 0]
        print(name, "zero-gap rows", len(metals), "max absolute edge difference",
              (metals.vbm - metals.cbm).abs().max(), "minimum observed V", metals.V.min())
    groups = {"all": np.ones(len(validation), dtype=bool), "NO2": validation.X == "NO2",
              "O3/NO2/FO2": validation.X.isin(["O3", "NO2", "FO2"])}
    predictions = fit_predict(train, validation, seed)
    if predictions_path:
        output = validation[["id", "A", "B", "X", "V"]].assign(**predictions)
        output.to_csv(predictions_path, index=False)
    for name, scores in predictions.items():
        print(name, "validation MAE", mean_absolute_error(validation.V, scores))
        for group, mask in groups.items():
            ranked = validation.loc[mask].assign(score=scores[mask]).sort_values(["score", "id"])
            print(group, "pool size", len(ranked), "total with V < 1", (ranked.V < 1).sum())
            for budget in [1, 10, 20, 50, 100]:
                selected = ranked.head(budget)
                print(" budget", budget, "count with V < 1", (selected.V < 1).sum(),
                      "lowest V", selected.V.min())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=26)
    parser.add_argument("--predictions", type=Path)
    args = parser.parse_args()
    main(args.seed, args.predictions)
