"""Cross-rank the two Q2 recommendations on the same unlabeled file-11 pool."""
import numpy as np
import pandas as pd

from screen import OUT, composition_features, load_file
from compare_physical_properties import predict_properties, settings_for, violation


def run():
    train = pd.concat([load_file("02"), load_file("03")], ignore_index=True)
    pool = load_file("11")[["id", "formula", "A", "B", "X"]].copy()
    assert len(train) == 15225 and len(pool) == 529 and pool.X.eq("NO2").all()
    assert not set(train.id).intersection(pool.id)
    props = predict_properties(train, composition_features(train),
                               composition_features(pool), settings_for(26))
    for field, prediction in props.items():
        pool["pred_" + field] = prediction
    pool["property_score"] = violation(props)
    frozen = pd.read_csv(OUT / "q23_full.csv").rename(columns={"V_hat": "mixture_score"})
    ranks = pool.merge(frozen, on=["id", "formula", "A", "B", "X"], how="left",
                       validate="one_to_one")
    for model in ("mixture", "property"):
        score = model + "_score"
        assert np.isfinite(ranks[score]).all()
        order = ranks.sort_values([score, "id"]).index
        ranks.loc[order, model + "_rank"] = np.arange(1, len(ranks) + 1)
        ranks[model + "_rank"] = ranks[model + "_rank"].astype(int)
    ranks.sort_values("mixture_rank", inplace=True)
    ranks.to_csv(OUT / "q2_blind_comparison.csv", index=False, lineterminator="\n")
    print(ranks.query("mixture_rank == 1 or property_rank == 1").to_string(index=False))


if __name__ == "__main__":
    run()
