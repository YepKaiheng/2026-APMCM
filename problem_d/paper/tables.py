"""Write measured quantities into LaTeX from the archived experiment files."""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "artifacts"


def make():
    out = []
    def macro(key, value):
        out.append(r"\newcommand{" + "\\" + key + "}{" + str(value) + "}")

    folds = pd.read_csv(DATA / "group_folds.csv")
    foldrows = []
    for pool in ["all", "Q2", "Q3"]:
        for model in ["direct", "mixture"]:
            s = folds[(folds.pool == pool) & (folds.model == model)]
            assert len(s) == 5
            foldrows.append(f"{pool.title() if pool == 'all' else pool} & {model.title()}"
                            f" & {s.hits50.sum()}/{s.targets.sum()} & {s.regret.median():.3f}"
                            f" & {s.regret.min():.3f}--{s.regret.max():.3f}" + r" \\")
    macro("FoldRows", "\n".join(foldrows))
    costs = pd.read_csv(DATA / "costs.csv").set_index("problem")
    timings = pd.read_csv(DATA / "score_runs.csv").set_index("run")
    for task, suffix in [("q1_holdout", "One"), ("q23_full", "Full")]:
        r = timings.loc[task]
        for field, label in [("feature_s","Desc"),("fit_s","Fit"),("predict_s","Score")]:
            macro(label + suffix, f"{r[field]:.2f}" if label == "Scan" else f"{r[field]:.4f}")
    for task, suffix in [("Q2","Two"),("Q3","Three")]:
        r = costs.loc[task]
        for field, label in [("setup_s","Pool"),("prepare_s","Encode"),
                             ("solve_s","Solve"),("decode_check_s","Check"),
                             ("enum_median_us","Scan")]:
            macro(label + suffix, f"{r[field]:.2f}" if label == "Scan" else f"{r[field]:.4f}")
        for field, label in [("tied_blocks","Tied"),("finalists","Finalists"),
                             ("j_min","CoeffLow"),("j_max","CoeffHigh")]:
            macro(label + suffix, int(r[field]))
    (HERE / "numbers.tex").write_text("\n".join(out)+"\n", encoding="ascii")


if __name__ == "__main__":
    make()

