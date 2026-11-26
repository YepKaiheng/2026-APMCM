"""Recompute ranking summaries from the saved row-level predictions."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


def summarize():
    result_dir = Path(__file__).resolve().parents[1] / "results"
    records = []
    for family in ["validation", "objectives", "stress"]:
        for file in sorted(result_dir.glob(f"{family}_seed*.csv")):
            table = pd.read_csv(file)
            seed = int(file.stem.rsplit("seed", 1)[1])
            models = [col for col in ["direct", "mixture", "log", "components"] if col in table]
            threshold = 0.25 if family == "stress" else 1.0
            pools = {"all": np.ones(len(table), dtype=bool), "q2": table.X == "NO2",
                     "q3": table.X.isin(["O3", "NO2", "FO2"])}
            for model in models:
                for pool, mask in pools.items():
                    subset = table.loc[mask]
                    ranked = subset.sort_values([model, "id"])
                    true_best = float(subset.V.min())
                    for budget in [1, 10, 20, 50, 100]:
                        chosen = ranked.head(budget)
                        records.append({
                            "experiment": family, "seed": seed, "model": model, "pool": pool,
                            "n": len(subset), "threshold": threshold, "budget": budget,
                            "targets": int((subset.V < threshold).sum()),
                            "hits": int((chosen.V < threshold).sum()),
                            "best_true_V": float(chosen.V.min()),
                            "true_pool_min_V": true_best,
                            "best_selection_regret": float(chosen.V.min()) - true_best,
                            "MAE": mean_absolute_error(subset.V, subset[model])})
    if not records:
        raise RuntimeError("Run the prediction experiments before summarizing.")
    result = pd.DataFrame(records)
    result.to_csv(result_dir / "summary.csv", index=False)
    print(result.loc[(result.pool == "all") & (result.budget == 50),
                     ["experiment", "seed", "model", "threshold", "hits", "best_true_V", "MAE"]]
          .to_string(index=False))


if __name__ == "__main__":
    summarize()
