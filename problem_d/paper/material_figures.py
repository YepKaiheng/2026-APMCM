"""Draw crystallographic, validation and QUBO diagnostics for Problem D."""

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd

from figures import FIG, GRAY, HERE, NAVY, ORANGE, RESULTS, TEAL

DATA = HERE.parent / "date"
ART = HERE / "artifacts"
FAMILIES = ["O3", "NO2", "FO2"]
FAMILY_LABELS = [r"$\mathrm{O_3}$", r"$\mathrm{NO_2}$", r"$\mathrm{FO_2}$"]


def site_pair_audit(public):
    swapped = public[["A","B","X","V"]].rename(columns={"A":"B","B":"A","V":"reversed"})
    pairs = public.merge(swapped, on=["A","B","X"], validate="one_to_one").query("A < B")
    diff = np.abs(pairs.V - pairs.reversed)
    print("Site-exchange pairs", len(pairs), "quartiles", diff.quantile([.25,.5,.75]).to_list(),
          "site-invariant MAE floor", diff.mean()/2)


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
    site_pair_audit(public)
    physical(public)
    stability()
    landscapes(scores)
    tied_block(scores)

