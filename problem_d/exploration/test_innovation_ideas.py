"""Exploratory tests for the 2026 APMCM D team. Run with stress or encoding."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from compare_surrogates import DATA, fit_predict


def load_table(prefix):
    return pd.read_csv(next(DATA.glob(prefix + "_*.csv")))


def stress(seed, predictions_path=None):
    train = load_table("02")
    pair = list(zip(train.A, train.B))
    all_pairs = set(pair)
    low_pairs = set(zip(train.A[train.V < 0.25], train.B[train.V < 0.25]))
    def order(ab):
        return hashlib.sha256(f"{seed}|{ab[0]}|{ab[1]}".encode()).digest()
    n_test = round(len(all_pairs) * 0.20)
    background = sorted(all_pairs - low_pairs, key=order)[:n_test-len(low_pairs)]
    test_pairs = low_pairs.union(background)
    hidden = np.array([ab in test_pairs for ab in pair])
    seen, blind = train.loc[~hidden].copy(), train.loc[hidden].copy()
    assert seen.V.min() >= 0.25 and sum(blind.V < 0.25) == sum(train.V < 0.25)
    print("public-training-only stress", "low pairs", len(low_pairs), "blind pairs", len(test_pairs),
          "rows", len(seen), len(blind), flush=True)
    predictions = fit_predict(seen, blind, seed)
    if predictions_path:
        blind[["id", "A", "B", "X", "V"]].assign(**predictions).to_csv(predictions_path, index=False)
    for model, scores in predictions.items():
        ranked = blind.assign(score=scores).sort_values(["score", "id"])
        for label, pool in [("all", ranked), ("Q2", ranked.loc[ranked.X.eq("NO2")]),
                            ("Q3", ranked.loc[ranked.X.isin(["O3", "NO2", "FO2"])])]:
            print(model, label, len(pool), "targets", int((pool.V < 0.25).sum()),
                  "first V", pool.V.iloc[0], "best attainable", pool.V.min(),
                  "hits@20,@50", [int((pool.V.head(k)<0.25).sum()) for k in [20, 50]])


def precision_on_legal_pairs(cost, bits, penalty=1.1):
    """Uniform rounding of Ising h and J, as a software sensitivity experiment."""
    rows, cols = cost.shape
    # Upper-triangular pair coefficients and linear fields for s = 2*x - 1.
    # cost is scaled to [0,1]; each violated one-hot constraint costs >= 1.1.
    j = np.zeros((rows + cols, rows + cols))
    j[:rows, :rows] = np.triu(np.full((rows, rows), penalty / 2), 1)
    j[rows:, rows:] = np.triu(np.full((cols, cols), penalty / 2), 1)
    j[:rows, rows:] = cost / 4
    fields = np.r_[(rows-2)*penalty/2 + cost.sum(axis=1)/4,
                   (cols-2)*penalty/2 + cost.sum(axis=0)/4]
    step = max(abs(fields).max(), abs(j).max()) / (2**(bits-1)-1)
    rounded_j = np.rint(j / step).astype(int)
    rounded_h = np.rint(fields / step).astype(int)
    # Relative to the all-minus state, flip one row spin and one column spin.
    one_flip = 2 * (rounded_h - rounded_j.sum(0) - rounded_j.sum(1))
    return one_flip[:rows,None] + one_flip[None,rows:] + 4 * rounded_j[:rows,rows:]


def check_encoding(pool, scores):
    frame = pool.assign(score=scores).set_index(["A","B","X"]).sort_index()
    shape = tuple(len(level) for level in frame.index.levels)
    assert shape == (52, 52, 3) and len(frame) == np.prod(shape)
    score = frame.score.to_numpy().reshape(shape)
    best_x = score.argmin(axis=2)
    envelope = score.min(axis=2)
    assert np.allclose(np.take_along_axis(score, best_x[...,None], axis=2)[...,0], envelope)
    assert abs(envelope.min() - score.min()) < 1e-10
    print('X elimination: 104 bits, min', envelope.min(), 'first-to-second pair margin',
          np.diff(np.sort(envelope.ravel())[:2]).item())
    for cap in [1.0, 0.1]:
        limit = np.quantile(envelope, cap)
        lo = envelope.min()
        assert limit > lo
        normalized = (np.minimum(envelope, limit) - lo) / (limit - lo)
        for bits in [8,12,16]:
            energies = precision_on_legal_pairs(normalized, bits)
            winners = envelope[energies == energies.min()]
            print('cap quantile', cap, 'bits', bits, 'tied pairs', winners.size,
                  'contains original minimum', abs(winners.min()-lo) < 1e-9,
                  'worst original-score gap', winners.max()-lo)
    # Uniform-grid functional ANOVA through order two.
    two = (score.mean(0, keepdims=True) + score.mean(1, keepdims=True)
           + score.mean(2, keepdims=True) - score.mean((0,1), keepdims=True)
           - score.mean((0,2), keepdims=True) - score.mean((1,2), keepdims=True) + score.mean())
    residual = score - two
    lower = two + residual.min(axis=2, keepdims=True)
    correction = score - lower
    provisional = np.unravel_index(np.argmin(two), shape)
    incumbent = score[provisional]
    cutoff = incumbent + 1e-9
    keep_ax = lower.min(axis=1) <= cutoff
    certified = lower + np.where(keep_ax[:,None,:], correction, 0.)
    # Checks concern exactly-one configurations, before coefficient quantization.
    assert np.all(lower <= score + 1e-10)
    assert np.allclose(score.min(), certified.min(), atol=1e-9, rtol=0)
    expected = np.isclose(score, score.min(), atol=1e-9, rtol=0)
    observed = np.isclose(certified, certified.min(), atol=1e-9, rtol=0)
    assert np.array_equal(expected, observed)
    print("min score", score.min(), "two-factor choice true score", incumbent,
          "original gap at two-factor choice", incumbent - score.min(),
          "lower-bound contenders", int((lower<=cutoff).sum()), "AX blocks", int(keep_ax.sum()),
          "bits including AX auxiliaries", 107+int(keep_ax.sum()),
          "optimal sets equal", np.array_equal(expected, observed))


def encoding(seed, predictions_path=None):
    train = pd.concat([load_table("02"), load_table("03")], ignore_index=True)
    pool = load_table("08")
    scores = fit_predict(train, pool, seed)
    if predictions_path:
        pool[["id", "A", "B", "X"]].assign(**scores).to_csv(predictions_path, index=False)
    for model, estimates in scores.items():
        print(model)
        check_encoding(pool, estimates)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("experiment", choices=["stress", "encoding"])
    cli.add_argument("--seed", type=int, default=26)
    cli.add_argument("--predictions", type=Path)
    args = cli.parse_args()
    (stress if args.experiment == "stress" else encoding)(args.seed, args.predictions)
