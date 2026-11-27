"""Use ideal fractional coordinates for geometry and archived records for properties."""
from itertools import combinations, product

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle, Polygon
import numpy as np
import pandas as pd

from figures import (BLUE, CLAY, GRAY, GRID, HERE, INK, MIST, OCHRE, PAPER,
                     PLUM, SAGE, save)

DATA = HERE.parent / "date"
ELEMENT = {"Sn": BLUE, "Ti": OCHRE, "O": CLAY, "N": PLUM, "F": SAGE}


def draw_site(ax, x, y, radius, element, depth):
    # Nested vector disks shade the same spherical glyph from the upper left.
    color = np.array(to_rgb(ELEMENT[element]))
    ax.add_patch(Circle((x, y), radius, fc=color*.65, ec=INK, lw=.55, zorder=depth))
    for i in range(1, 15):
        t = i / 15
        shade = color * (.65 + .35*t) + (1-color) * .72*t*t
        shift = .36 * radius * t
        ax.add_patch(Circle((x-shift, y+shift), radius*(1-t), fc=shade,
                            ec="none", zorder=depth+.001*t))


def project(points):
    yaw, elevation = np.radians([32, 19])
    right = np.array([np.cos(yaw), -np.sin(yaw), 0])
    forward = np.array([np.sin(yaw), np.cos(yaw), 0])
    up = np.array([0, 0, 1])
    basis = [right, up*np.cos(elevation)-forward*np.sin(elevation),
             forward*np.cos(elevation)+up*np.sin(elevation)]
    return np.asarray(points) @ np.array(basis).T


def cell(ax, a_site, b_site):
    corners = np.array(list(product([-.5,.5], repeat=3)))
    faces = np.vstack([np.eye(3), -np.eye(3)])*.5
    center = np.zeros(3)
    for a, b in combinations(corners, 2):
        if np.isclose(np.linalg.norm(a-b), 1):
            pts = project([a,b])
            ax.plot(*pts[:,:2].T, color=GRAY, lw=.9, zorder=5+pts[:,2].mean())
    for signs in product([-1,1], repeat=3):
        pts = project(np.diag(signs)*.5)
        ax.add_patch(Polygon(pts[:,:2], fc=MIST, ec=GRAY, lw=.4, alpha=.25,
                             zorder=5+pts[:,2].mean()))
    for p in faces:
        pts = project([center,p])
        ax.plot(*pts[:,:2].T, color=GRAY, lw=.85, zorder=5+pts[:,2].mean())
    for sites, el, radius in ((corners,a_site,.080),(faces,"O",.058),([center],b_site,.105)):
        for x, y, depth in project(sites):
            draw_site(ax, x, y, radius, el, 7 + depth)
    ax.set(xlim=(-.93,.93),ylim=(-.89,.89),aspect="equal")
    ax.axis("off")


def crystal_and_exchange(public):
    fig = plt.figure(figsize=(6.45, 2.85))
    for left, formula in ((.025, "SnTiO3"), (.57, "TiSnO3")):
        row = public.loc[public.formula.eq(formula) & public.X.eq("O3")].squeeze()
        assert isinstance(row, pd.Series)
        ax = fig.add_axes([left, .21, .405, .62])
        cell(ax, row.A, row.B)
        cx = left + .2025
        fig.text(cx, .925, formula.replace("3", "\u2083"), ha="center", weight="bold", fontsize=14)
        fig.text(cx, .85, f"Corners: {row.A}    Center: {row.B}", ha="center", fontsize=10)
        fig.text(cx, .115, f"Gap  {row.gap:.2f} eV", ha="center", fontsize=11)
        fig.text(cx, .035, f"Violation  {row.V:.6f}", ha="center", fontsize=11)
    mid = fig.add_axes([.40, .32, .20, .36])
    mid.set(xlim=(0, 1), ylim=(0, 1))
    mid.axis("off")
    mid.text(.5, .72, "Exchange A / B", ha="center", fontsize=10)
    mid.annotate("", (.88, .51), (.12, .51),
                 arrowprops=dict(arrowstyle="<->", color=GRAY, lw=1.4))
    mid.text(.5, .27, "O at face centers", ha="center", fontsize=9)
    save(fig, "crystal_sites")


def electronic(public):
    sn, ca = (public.loc[public.id.eq(i)].iloc[0] for i in (8760,8811))
    pred = pd.read_csv(HERE.parent/"results"/"physical_seed26.csv").query("id == 8811").iloc[0]
    fig, (ax,bx) = plt.subplots(1,2, figsize=(6.55,2.75), layout="constrained")
    ax.set(title="a   Band edges",xlim=(-.5,2.5),ylim=(9,1),ylabel="Reported edge (eV)",
           xticks=[0,1,2],xticklabels=["SnTiO\u2083\nDFT","CaZrO\u2083\nmodel","CaZrO\u2083\nDFT"])
    ax.grid(axis="y")
    for val,color,marker,label in ((4.50,SAGE,"o","CBM"),(5.73,PLUM,"s","VBM")):
        ax.axhline(val,ls=":",color=color,lw=1.1)
        levels = [sn.cbm,pred.pred_cbm,ca.cbm] if label=="CBM" else [sn.vbm,pred.pred_vbm,ca.vbm]
        ax.plot(range(3),levels,ls="none",marker=marker,color=color,label=label,ms=6)
    ax.legend(ncol=2,loc="upper center",fontsize=8.5,handletextpad=.1,columnspacing=.9)
    bx.set(title="b   CaZrO\u2083 band gap",xlim=(0,8.2),ylim=(-.62,1.65),
           yticks=[1,0],yticklabels=["Model","DFT"],xlabel="GLLB-SC gap (eV)")
    bx.axvspan(1.4,3.1,color=GRID)
    bx.axvline(3.1,color=GRAY,lw=.9,ls=":")
    bx.text(3.2,1.41,"3.10 eV limit",ha="left",va="center",fontsize=8.5)
    for y, value, color, marker in ((1,pred.pred_gap,BLUE,"s"),(0,ca.gap,CLAY,"o")):
        bx.hlines(y,0,value,color=color,lw=1.8)
        bx.plot(value,y,marker=marker,color=color,ms=6)
        bx.annotate(f"{value:.2f}",(value,y),(5,0),textcoords="offset points",va="center",fontsize=9)
    save(fig,"band_edges")


def anions(scores):
    pair = scores[scores.A.eq("La") & scores.B.eq("Ti")].set_index("X")
    fig, (ax,bx) = plt.subplots(1,2,figsize=(6.15,2.0),layout="constrained")
    families = [("NO2","LaTiO\u2082N","N"),("FO2","LaTiO\u2082F","F"),("O3","LaTiO\u2083","O")]
    ax.set(title="a   Anion counts",xlim=(-.2,2.7),ylim=(-.6,2.6),aspect="equal")
    bx.set(title="b   Frozen prediction",xlim=(0,14.6),ylim=(-.6,2.6),xlabel="Predicted violation")
    for y,(key,formula,last) in zip([2,1,0],families):
        for x,el in enumerate(("O","O",last)):
            ax.add_patch(Circle((x*.65+.4,y),.23,fc=ELEMENT[el],ec=INK,lw=.6))
            ax.text(x*.65+.4,y,el,ha="center",va="center",color=PAPER,fontsize=8,weight="bold")
        val = pair.loc[key,"V_hat"]
        bx.hlines(y,0,val,color=ELEMENT[last],lw=2)
        bx.plot(val,y,"o",color=ELEMENT[last])
        bx.annotate(f"{val:.4f}",(val,y),(4,0),textcoords="offset points",va="center",fontsize=9)
    ax.set_yticks([2,1,0], [f[1] for f in families])
    ax.set_xticks([])
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    bx.set_yticks([])
    bx.grid(axis="x")
    save(fig,"anion_substitution")


if __name__ == "__main__":
    public = pd.concat(pd.read_csv(next(DATA.glob(p+"_*.csv"))) for p in ("02","03"))
    crystal_and_exchange(public)
    electronic(public)
    anions(pd.read_csv(HERE/"artifacts"/"q23_full.csv"))
