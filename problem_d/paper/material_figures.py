"""Draw crystallographic, validation and QUBO diagnostics for Problem D."""
from itertools import combinations, product

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import pandas as pd

from figures import FIG, GRAY, HERE, NAVY, ORANGE, RESULTS, TEAL

DATA = HERE.parent / "date"
ART = HERE / "artifacts"
FAMILIES = ["O3", "NO2", "FO2"]
FAMILY_LABELS = [r"$\mathrm{O_3}$", r"$\mathrm{NO_2}$", r"$\mathrm{FO_2}$"]


def crystal(public):
    fig = plt.figure(figsize=(7.6, 3.8), layout="constrained")
    ax = fig.add_subplot(121, projection="3d", proj_type="ortho")
    corners = np.array(list(product([0, 1], repeat=3)))
    center = np.full(3, 0.5)
    offsets = 0.5 * np.eye(3)
    faces = np.vstack([center + offsets, center - offsets])
    for a, b in combinations(corners, 2):
        if np.abs(a-b).sum() == 1:
            ax.plot(*np.vstack([a,b]).T, color="#8d99a3", lw=1.2)
    triangles = [center + offsets * np.array(signs)[:, None]
                 for signs in product([-1, 1], repeat=3)]
    ax.add_collection3d(Poly3DCollection(triangles, facecolor="#8cd1c8",
                                         edgecolor="#578886", alpha=.13, linewidth=.6))
    for point in faces:
        ax.plot(*np.vstack([center, point]).T, color="#4a6868", lw=1)
    markers = [(corners, NAVY, 85, "A  corner"), ([center], ORANGE, 145, "B  center"),
               (faces, TEAL, 60, "X  face")]
    for points, color, size, _ in markers:
        ax.scatter(*np.array(points).T, s=size, color=color, edgecolor="white",
                   linewidth=.7, depthshade=False)
    ax.set_box_aspect((1,1,1), zoom=1.04)
    ax.view_init(elev=19, azim=-61)
    ax.set_axis_off()
    ax.set_title("Ideal cubic cell", loc="left", fontweight="bold")
    fig.legend(handles=[Line2D([], [], ls="", marker="o", color=c, label=s)
                        for _, c, _, s in markers], loc="lower left",
               bbox_to_anchor=(.025,.01), ncol=3, frameon=False, columnspacing=.65,
               handletextpad=.2, fontsize=9)
    bx = fig.add_subplot(122)
    swapped = public[["A","B","X","V"]].rename(columns={"A":"B","B":"A","V":"reversed"})
    pairs = public.merge(swapped, on=["A","B","X"], validate="one_to_one").query("A < B")
    diffs = np.sort(np.abs(pairs.V - pairs.reversed))
    bx.step(diffs, np.arange(1, len(diffs)+1)/len(diffs), where="post", color=NAVY, lw=2)
    median = np.median(diffs)
    bx.plot([0,median,median],[.5,.5,0], "--", color=ORANGE, lw=1)
    bx.scatter(median,.5,color=ORANGE,s=29,zorder=3)
    bx.annotate(f"Median {median:.2f}",(median,.5), xytext=(15,.41), textcoords="data")
    bx.set(title="Observed effect of exchanging A and B", xlabel="Absolute change in V",
           ylabel="Fraction of matched pairs",xlim=(0,np.ceil(diffs.max())),ylim=(0,1.01))
    bx.grid(axis="y",alpha=.2)
    fig.savefig(FIG/"crystal_sites.pdf")
    fig.savefig(FIG/"crystal_sites.png",dpi=180)
    plt.close(fig)


def physical(public):
    fig,(ax,bx)=plt.subplots(1,2,figsize=(7.5,3.5),layout="constrained")
    metallic=public[public.gap.eq(0)]
    assert (metallic.vbm == metallic.cbm).all()
    grid=np.linspace(3.5,6.8,300)
    bound=7.84+(np.maximum(5.73-grid,0)/.5)**2+(np.maximum(grid-4.50,0)/.5)**2
    ax.scatter(metallic.vbm,metallic.V,s=3,alpha=.16,color=NAVY,rasterized=True)
    ax.plot(grid,bound,color=ORANGE,lw=2,label="Zero formation penalty")
    ax.scatter([5.115],[10.8658],color=ORANGE,s=28)
    ax.set(xlim=(grid.min(),grid.max()),ylim=(6,60),xlabel="Coincident band edge (eV)",
           ylabel="Observed violation V",title="Metallic records")
    ax.legend(frameon=False,fontsize=8)
    counts=public.groupby("X").agg(gapped=("gap",lambda x:(x>0).sum()),
                                    near=("V",lambda x:(x<1).sum())).sort_values("gapped")
    y=np.arange(len(counts))
    bx.barh(y,counts.gapped,color="#a8bac8",label="Positive gap")
    bx.barh(y,counts.near,color=TEAL,label="Also V < 1")
    for i,row in enumerate(counts.itertuples()):
        bx.text(row.gapped+2,i,f"{row.near}/{row.gapped}",va="center",fontsize=8)
    bx.set(yticks=y,yticklabels=counts.index,xlim=(0,175),xlabel="Public records per family",
           title="Near-targets among gapped records")
    bx.legend(frameon=False,fontsize=8,loc="lower right")
    fig.savefig(FIG/"physical_regimes.pdf")
    plt.close(fig)


def stability():
    # Paired cluster bootstrap, conditional on the two fitted seed-26 models.
    base=pd.read_csv(RESULTS/"validation_seed26.csv")
    second=pd.read_csv(RESULTS/"validation_seed42.csv")
    frame=base.merge(second[["id","mixture"]].rename(columns={"mixture":"other_seed"}),
                     on="id",validate="one_to_one")
    groups=frame.groupby(["A","B"],sort=True).ngroup()
    frame=frame.assign(group=groups)
    n=groups.nunique()
    counts=np.random.default_rng(2601).multinomial(n,np.full(n,1/n),size=2000)
    fig,(ax,bx)=plt.subplots(1,2,figsize=(7.4,3.25),layout="constrained")
    records=[]
    for i,(name,pool) in enumerate([("All",frame),("Q2",frame[frame.X.eq("NO2")]),
                                     ("Q3",frame[frame.X.isin(FAMILIES)])]):
        scores={}
        for model in ["direct","mixture"]:
            ranked=pool.sort_values([model,"id"])
            # Keep multiplicities when spending a fixed budget on a bootstrap sample.
            weights=counts[:,ranked.group.to_numpy()]
            remaining=50-(np.cumsum(weights,axis=1)-weights)
            used=np.minimum(weights,np.maximum(remaining,0))
            scores[model]=used @ (ranked.V.to_numpy()<1)
        delta=scores["mixture"]-scores["direct"]
        lo,hi=np.percentile(delta,[2.5,97.5],method="nearest")
        observed=sum(pool.nsmallest(50,"mixture").V<1)-sum(pool.nsmallest(50,"direct").V<1)
        ax.plot([lo,hi],[i,i],color=NAVY,lw=2)
        ax.scatter(observed,i,s=42,color=TEAL,zorder=3)
        ax.text(hi+.35,i,f"[{lo}, {hi}]",va="center",fontsize=8)
        ranks=[pool.sort_values([s,"id"]).id for s in ["mixture","other_seed"]]
        budgets=[1,10,20,50]
        overlap=[len(set(ranks[0].head(k)) & set(ranks[1].head(k))) for k in budgets]
        bx.plot(budgets,np.array(overlap)/budgets,marker=["o","s","^"][i],
                color=[NAVY,ORANGE,TEAL][i],label=name)
        records.append(dict(pool=name,delta_hits50=observed,p025=lo,p975=hi,
                            **{f"overlap{k}":v for k,v in zip(budgets,overlap)}))
    ax.axvline(0,color=GRAY,lw=.9,ls="--")
    ax.set(xlim=(-9,22),ylim=(-.6,2.6),yticks=[0,1,2],yticklabels=["All","Q2","Q3"],
           xlabel="Additional hits in 50 checks",title="Pair-resampling sensitivity")
    bx.set(xlabel="Shortlist size",ylabel="Shared fraction",title="Agreement across seeds",
           xticks=budgets,ylim=(-.05,1.05))
    bx.legend(frameon=False,loc="lower right",fontsize=8)
    fig.savefig(FIG/"validation_stability.pdf")
    plt.close(fig)
    pd.DataFrame(records).to_csv(ART/"shortlist_sensitivity.csv",index=False)


def landscapes(scores):
    fig,axes=plt.subplots(1,3,figsize=(8.3,3.25),layout="constrained")
    lo,hi=scores.V_hat.min(),scores.V_hat.max()
    for ax,x,title in zip(axes,FAMILIES,FAMILY_LABELS):
        part=scores[scores.X.eq(x)]
        matrix=part.pivot(index="A",columns="B",values="V_hat").sort_index().sort_index(axis=1)
        assert matrix.shape==(52,52) and matrix.notna().all().all()
        im=ax.imshow(matrix,cmap="viridis",vmin=lo,vmax=hi,interpolation="nearest")
        best=part.nsmallest(1,"V_hat").iloc[0]
        i,j=matrix.index.get_loc(best.A),matrix.columns.get_loc(best.B)
        ax.scatter(j,i,marker="o",facecolors="none",edgecolors="white",s=65,linewidth=1.3)
        ticks=[0,13,26,39,51]
        ax.set(xticks=ticks,xticklabels=matrix.columns[ticks],yticks=ticks,
               yticklabels=matrix.index[ticks],xlabel="B site",title=title)
        ax.set_xlabel(f"B site\nMinimum: {best.A}, {best.B} ({best.V_hat:.2f})")
    axes[0].set_ylabel("A site")
    fig.colorbar(im,ax=axes,shrink=.68,label="Predicted V")
    fig.savefig(FIG/"family_landscapes.pdf")
    plt.close(fig)


def tied_block(scores):
    ledger=pd.read_csv(ART/"block_runs.csv").query("problem == 'Q3'").copy()
    ledger["span"]=ledger.worst_tied-ledger.score
    record=ledger.loc[ledger.span.idxmax()]
    envelope=scores.groupby(["A","B"]).V_hat.min().unstack()
    a,b=int(record.block_a)*4,int(record.block_b)*4
    tile=envelope.iloc[a:a+4,b:b+4]
    values=tile.to_numpy()
    bins=np.rint(15*values/values.max()).astype(int)
    mask=bins==bins.min()
    np.testing.assert_allclose([values[mask].min(),values[mask].max()],
                               [record.score,record.worst_tied])
    assert mask.sum()==record.ties
    fig,axes=plt.subplots(1,2,figsize=(6.8,3.4),layout="constrained")
    for ax,table,title,decimals in zip(axes,[values,bins],["Original pair minima","Integer costs"],[2,0]):
        ax.imshow(mask,cmap=ListedColormap(["#eaf0f4","#a4dad2"]),vmin=0,vmax=1)
        for i,j in np.ndindex((4,4)):
            ax.text(j,i,f"{table[i,j]:.{decimals}f}",ha="center",va="center",fontsize=10)
        i,j=np.unravel_index(values.argmin(),values.shape)
        ax.add_patch(plt.Rectangle((j-.47,i-.47),.94,.94,fill=False,color=ORANGE,lw=2))
        ax.set(xticks=range(4),xticklabels=tile.columns,yticks=range(4),
               yticklabels=tile.index,ylabel="A site",xlabel="B site",title=title)
    fig.savefig(FIG/"tied_block.pdf")
    plt.close(fig)


if __name__=="__main__":
    public=pd.concat([pd.read_csv(next(DATA.glob(prefix+"_*.csv"))) for prefix in ["02","03"]])
    scores=pd.read_csv(ART/"q23_full.csv")
    crystal(public)
    physical(public)
    stability()
    landscapes(scores)
    tied_block(scores)

