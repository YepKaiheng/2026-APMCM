"""Shared manuscript style and plots derived from the saved screening runs."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
INK, PAPER, GRID = "#3F4543", "#FFFFFF", "#EEECE7"
SAGE, MIST, BLUE = "#567367", "#AABDB1", "#5D7187"
GRAY, CLAY, OCHRE, PLUM = "#787169", "#9C6963", "#9A8164", "#84738B"
MODELS = (("mixture", "Mixture", SAGE, "o"), ("direct", "Direct", BLUE, "s"))
FAMILIES = ["O3", "NO2", "FO2"]
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "semibold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": GRAY,
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": .8,
    "legend.frameon": False, "pdf.fonttype": 42, "mathtext.fontset": "dejavusans",
    "axes.facecolor": PAPER, "figure.facecolor": PAPER,
})


def save(fig, name):
    fig.savefig(FIG / (name + ".pdf"), metadata={"CreationDate": None})
    plt.close(fig)


def workflow():
    fig, ax = plt.subplots(figsize=(6.7, 3.55))
    fig.subplots_adjust(left=.012, right=.988, top=.985, bottom=.015)
    ax.set(xlim=(0, 10), ylim=(0, 5.2))
    ax.axis("off")

    def box(x, y, width, heading, detail, color):
        ax.add_patch(FancyBboxPatch((x, y), width, .84, boxstyle="round,pad=.025",
                                   facecolor="#F7F8F6", edgecolor=color, lw=1))
        ax.text(x+.12, y+.55, heading, fontsize=9.6, weight="bold", color=color)
        ax.text(x+.12, y+.19, detail, fontsize=8.8)

    def arrow(start, end):
        ax.annotate("", end, start, arrowprops=dict(arrowstyle="-|>", color=GRAY,
                    lw=1.15, shrinkA=2, shrinkB=2))

    ax.text(.04, 4.93, "Q1   Learn to rank unseen compositions", weight="bold", color=BLUE)
    box(.05, 3.70, 2.56, "02   Train", "Composition + labels", BLUE)
    box(3.05, 3.70, 3.05, "03   Select model", "Hits and rank-one regret", BLUE)
    box(6.55, 3.70, 3.35, "04   Predict hidden scores", "3,703 IDs · fit from 02 only", BLUE)
    arrow((2.63, 4.12), (3.03, 4.12))
    arrow((6.12, 4.12), (6.53, 4.12))
    arrow((4.58, 3.69), (4.58, 2.83))
    ax.text(4.78, 3.2, "Carry forward the selected model design", va="center", fontsize=8.8)
    ax.text(.04, 3.08, "Q2 / Q3   Select one composition", weight="bold", color=SAGE)
    box(.05, 1.95, 6.05, "02 + 03   Refit on all public labels",
        "Freeze one score per eligible composition", SAGE)
    ax.plot([3.08, 3.08], [1.94, 1.6], color=GRAY, lw=1.15)
    ax.plot([2.38, 7.48], [1.6, 1.6], color=GRAY, lw=1.15)
    for x in (2.38, 7.48):
        arrow((x, 1.62), (x, 1.21))
    box(.05, .34, 4.65, "Full pools 07 / 08", "Kaiwu block QUBOs → verify by scan", SAGE)
    box(5.12, .34, 4.77, "Blind subsets 11 / 12", "Scan eligible scores → blind nominee", SAGE)
    ax.text(.06, .04, "Q2 fixes NO\u2082. Q3 compares O\u2083, NO\u2082 and FO\u2082.", fontsize=8.8)
    save(fig, "workflow")


def budget():
    """Supplementary curves for every pool and both training seeds."""
    fig, axes = plt.subplots(1, 3, figsize=(8, 2.95), layout="constrained")
    for seed in (26, 42):
        records = pd.read_csv(RESULTS / f"validation_seed{seed}.csv")
        pools = (records, records[records.X.eq("NO2")], records[records.X.isin(FAMILIES)])
        for ax, pool, title in zip(axes, pools, ("All families", "Q2: fixed family", "Q3: three families")):
            for key, name, color, marker in MODELS:
                hits = pool.sort_values([key, "id"]).V.iloc[:100].lt(1).cumsum()
                ax.step(np.arange(101), np.r_[0, hits], where="post", color=color,
                        lw=1.5, ls="-" if seed == 26 else "--", marker=marker,
                        markevery=[20, 50, 100], ms=3)
            if seed == 26:
                ax.plot([0, 100], [0, 100 * pool.V.lt(1).mean()], color=GRAY, ls=":")
            ax.set(title=title, xlim=(0, 100), xticks=[0, 20, 50, 100], xlabel="Checks")
            ax.yaxis.get_major_locator().set_params(integer=True)
            ax.grid(axis="y")
    for ax in axes:
        ax.set_ylim(bottom=0)
    handles = [Line2D([], [], color=color, marker=marker, label=name)
               for _, name, color, marker in MODELS]
    handles.extend(Line2D([], [], color=GRAY, ls=style, label=label)
                   for style, label in (("-", "Seed 26"), ("--", "Seed 42"), (":", "Random")))
    fig.legend(handles=handles, loc="outside lower center", ncols=5, fontsize=8.5,
               handlelength=1.6, columnspacing=1.25)
    axes[0].set_ylabel("Cumulative hits ($V<1$)")
    save(fig, "screening")


def single_or_shortlist():
    capacity = pd.read_csv(HERE / "artifacts" / "capacity_control.csv")
    q3 = capacity[capacity.pool.eq("Q3")].set_index(["seed", "model", "trees"])
    models = [("mixture", 1050, "Mixture (3 × 350)"),
              ("direct", 1050, "Direct (1,050)"), ("direct", 350, "Direct (350)")]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.65), sharey=True, layout="constrained")
    for ax, metric, title, label, limit in zip(
        axes, ("hits50", "regret"), ("a   Check 50 candidates", "b   Check one candidate"),
        ("Near-targets found (out of 11)", "Excess violation at rank one"), (7.6, 57)):
        for y, (model, trees, _) in enumerate(models):
            for seed, offset, color, marker in ((26, -.16, BLUE, "o"), (42, .16, CLAY, "s")):
                value = q3.loc[(seed, model, trees), metric]
                ax.plot(value, y + offset, marker, color=color, ms=5)
                label_value = f"{value:.0f}" if metric == "hits50" else f"{value:.2f}"
                ax.annotate(label_value, (value, y + offset), (5, 0),
                            textcoords="offset points", va="center", fontsize=8)
        ax.set(title=title, xlabel=label, xlim=(0, limit), ylim=(2.5, -.5),
               yticks=range(3), yticklabels=[name for _, _, name in models])
        ax.grid(axis="x")
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_xticks(range(0, 7, 2))
    handles = [Line2D([], [], ls="none", marker=m, color=c, label=f"Seed {s}")
               for s, c, m in ((26, BLUE, "o"), (42, CLAY, "s"))]
    fig.legend(handles=handles, ncols=2, loc="outside lower center", fontsize=8.5)
    save(fig, "q3_budget")


def precision():
    frame = pd.read_csv(RESULTS / "sdk_precision_audit.csv").query("model == 'mixture'")
    fig, ax = plt.subplots(figsize=(6.4, 2.9), layout="constrained")
    ticks, names = [], []
    for y, row in zip(range(len(frame), 0, -1), frame.itertuples()):
        ax.plot([row.best_original_gap, row.worst_original_gap], [y, y], color=GRAY, lw=2)
        ax.scatter(row.worst_original_gap, y, marker="s", facecolors=PAPER, edgecolors=CLAY, zorder=3)
        ax.scatter(row.best_original_gap, y, color=BLUE, s=20, zorder=4)
        ticks.append(y)
        names.append(("Full range" if row.cap_quantile == 1 else "10th pct. cap") + f", P = {row.penalty:g}")
    winner = pd.read_csv(HERE / "artifacts" / "selections.csv").query("problem == 'Q3'").iloc[0]
    ax.scatter(winner.enum_gap, 0, marker="D", color=SAGE, s=36)
    ax.set(yticks=ticks+[0], yticklabels=names+["Blocks, ties resolved"], xlim=(-.6, 20),
           xlabel="Excess predicted violation on the original Q3 table")
    ax.grid(axis="x")
    ax.legend(handles=[Line2D([], [], marker="o", lw=0, color=BLUE, label="Best tied state"),
                       Line2D([], [], marker="s", lw=0, color=CLAY, markerfacecolor=PAPER,
                              label="Worst tied state")], loc="lower right", fontsize=9)
    save(fig, "precision")


if __name__ == "__main__":
    workflow()
    budget()
    single_or_shortlist()
    precision()
