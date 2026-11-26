"""Draw the screening-budget and coefficient-precision figures from saved runs."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
NAVY, TEAL, ORANGE, GRAY = "#244463", "#00796B", "#B14C20", "#566372"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})


def budget():
    fig, axes = plt.subplots(1, 3, figsize=(8.25, 3.1), layout="constrained")
    for ax, pool, title in zip(axes, ["all", "q2", "q3"],
                               ["All seven families", "NO2 only", "O3, NO2, FO2"]):
        for seed in [26, 42]:
            t = pd.read_csv(RESULTS / f"validation_seed{seed}.csv")
            if pool == "q2":
                t = t[t.X.eq("NO2")]
            elif pool == "q3":
                t = t[t.X.isin(["NO2", "O3", "FO2"])]
            for model, color in [("direct", NAVY), ("mixture", TEAL)]:
                hits = t.sort_values([model, "id"]).V.iloc[:100].lt(1).cumsum()
                ax.step(np.arange(101), np.r_[0, hits], where="post", color=color,
                        linewidth=1.75, linestyle="-" if seed == 26 else "--")
        ax.plot([0, 100], [0, 100*t.V.lt(1).mean()], color=ORANGE, linewidth=1.25)
        ax.set(title=title, xlim=(0, 100), ylim=(0, None), xlabel="Candidates checked")
        ax.set_xticks([0, 20, 50, 100])
        ax.yaxis.get_major_locator().set_params(integer=True)
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Cumulative hits ($V<1$)")
    fig.savefig(FIG / "screening.pdf")
    plt.close(fig)


def precision():
    frame = pd.read_csv(RESULTS / "sdk_precision_audit.csv")
    frame = frame[frame.model.eq("mixture")]
    fig, ax = plt.subplots(figsize=(7.4, 2.85), layout="constrained")
    ticks, names = [], []
    for i, r in enumerate(frame.itertuples()):
        y = 6-i
        ax.plot([r.best_original_gap, r.worst_original_gap], [y,y], color=GRAY, lw=2)
        ax.scatter(r.worst_original_gap, y, facecolor="white", edgecolor=ORANGE, zorder=3)
        ax.scatter(r.best_original_gap, y, color=NAVY, s=23, zorder=4)
        ticks.append(y)
        names.append(("Full range" if r.cap_quantile == 1 else "10th percentile cap")
                     + f", P = {r.penalty:g}")
    row = pd.read_csv(HERE / "artifacts" / "selections.csv").query("problem=='Q3'").iloc[0]
    ax.scatter(row.enum_gap, 0, marker="D", color=TEAL, s=38, zorder=4)
    ax.set_yticks(ticks+[0], names+["4 by 4 blocks, tie resolution"])
    ax.set_xlabel("Excess score on the original Q3 surrogate (lower is better)")
    ax.set_xlim(-.6,20)
    ax.grid(axis="x", alpha=.2)
    ax.legend(handles=[Line2D([],[],marker="o",lw=0,color=NAVY,label="Best tied state"),
                       Line2D([],[],marker="o",lw=0,color=ORANGE,markerfacecolor="white",
                              label="Worst tied state")],frameon=False,loc="lower right")
    fig.savefig(FIG / "precision.pdf")
    plt.close(fig)


if __name__ == "__main__":
    budget()
    precision()

