"""Draw the ideal cubic prototype, its sections and the two electronic/composition plates."""
from itertools import combinations, product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle, Polygon, Rectangle
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
DATA = ROOT.parent / "date"
INK, SOFT, RULE = "#172f3d", "#526977", "#b6c5c9"
BLUE, GOLD, RED = "#326d91", "#b37324", "#bd534f"
VIOLET, GREEN = "#745384", "#428376"
plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10, "text.color":INK,
                     "axes.labelcolor":INK, "pdf.fonttype":42, "mathtext.fontset":"dejavusans"})


def board(height):
    fig, ax = plt.subplots(figsize=(8, height))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set(xlim=(0,8), ylim=(0,height), aspect="equal")
    ax.axis("off")
    return fig, ax


def save(fig, name):
    for fmt in ["pdf","png"]:
        fig.savefig(FIG / f"{name}.{fmt}", dpi=220, facecolor="white")
    plt.close(fig)


def label(ax, x, y, s, size=10, color=INK, weight="normal", align="left"):
    ax.text(x,y,s,ha=align,va="center",fontsize=size,color=color,fontweight=weight,zorder=20)


def sphere(ax, x, y, radius, color, depth=1):
    # Nested tangent disks approximate a lit sphere in vector output.
    rgb=np.asarray(to_rgb(color))
    ax.add_patch(Circle((x,y),radius,color=rgb*.56,lw=0,zorder=depth))
    for k in range(1,31):
        f=k/31
        tint=np.minimum(1,rgb*(.65+.55*np.sin(np.pi*f/2))+.42*f**2)
        ax.add_patch(Circle((x-radius*.29*f,y+radius*.37*f),radius*(1-f),
                            facecolor=tint,lw=0,zorder=depth+.001*k))


def framework(ax, sites, links, origin, scale, facets=()):
    # Orthographic projection. Rotate the cubic axes before removing camera depth.
    yaw, elev=np.radians([32,19])
    right=np.array([np.cos(yaw),-np.sin(yaw),0])
    forward=np.array([np.sin(yaw),np.cos(yaw),0])
    up=np.array([0,0,1])
    transform=np.array([right,up*np.cos(elev)-forward*np.sin(elev),
                        forward*np.cos(elev)+up*np.sin(elev)])
    def project(pos):
        v=transform@np.array(pos)
        return np.r_[np.array(origin)+scale*v[:2],v[2]]
    for vertices in facets:
        q=np.array([project(v) for v in vertices])
        ax.add_patch(Polygon(q[:,:2],facecolor="#99c6ca",edgecolor="#8aafb3",lw=.38,
                             alpha=.26,zorder=5+q[:,2].mean()))
    for a,b,kind in links:
        q=np.array([project(a),project(b)])
        ax.plot(q[:,0],q[:,1],color=("#799099" if kind=="cell" else "#577780"),
                lw=(.95 if kind=="cell" else 1.55),solid_capstyle="round",
                zorder=5+q[:,2].mean())
    for xyz,r,c in sites:
        x,y,d=project(xyz)
        sphere(ax,x,y,scale*r,c,5+d+.055)


def crystal():
    fig,ax=board(3.65)
    corners=np.array(list(product([-.5,.5],repeat=3)))
    faces=np.vstack([np.eye(3),-np.eye(3)])*.5
    center=np.zeros(3)
    edges=[(a,b,"cell") for a,b in combinations(corners,2) if np.linalg.norm(a-b)==1]
    bonds=[(center,f,"bond") for f in faces]
    facets=[np.diag(sign)*.5 for sign in product([-1,1],repeat=3)]
    sites=[(p,.071,BLUE) for p in corners]+[(p,.059,RED) for p in faces]+[(center,.094,GOLD)]
    framework(ax,sites,edges+bonds,(1.8,1.78),1.82,facets)
    label(ax,.16,3.38,"Cubic repeat unit",14,weight="bold")
    ax.plot([3.51,3.51],[.47,3.37],lw=.65,color=RULE)
    cage=np.array([p for p in product([-.5,0,.5],repeat=3) if np.count_nonzero(p)==2])
    assert len(cage)==12
    cage_edges=[(a,b,"cell") for a,b in combinations(cage,2)
                if np.isclose(np.linalg.norm(a-b),np.sqrt(.5))]
    framework(ax,[(p,.058,RED) for p in cage]+[(center,.12,BLUE)],
              cage_edges,(4.45,2.53),.88)
    framework(ax,[(p,.068,RED) for p in faces]+[(center,.12,GOLD)],bonds,
              (4.45,1.03),1.16,facets)
    label(ax,5.42,2.83,"A site",14,BLUE,"bold")
    label(ax,5.42,2.49,"12 anion neighbors",10.8)
    label(ax,5.42,2.20,"Corner shared by eight cells",8.7,SOFT)
    label(ax,5.42,1.30,"B site",14,GOLD,"bold")
    label(ax,5.42,.96,"6 anion neighbors",10.8)
    label(ax,5.42,.67,"Center of one octahedron",8.7,SOFT)
    for x,c,s in [(.40,BLUE,"A"),(1.24,GOLD,"B"),(2.07,RED,"Anion site")]:
        sphere(ax,x,.20,.075,c)
        label(ax,x+.15,.20,s,9)
    save(fig,"crystal_sites")


def site_exchange(public):
    fig,ax=board(3.6)
    colors={"Sn":BLUE,"Ti":GOLD,"O":RED}
    label(ax,.14,3.38,"(001) planes",11,weight="bold")
    for j,key in enumerate(["SnTiO3","TiSnO3"]):
        r=public.loc[public.formula.eq(key) & public.X.eq("O3")].squeeze()
        assert isinstance(r,pd.Series)
        x=2.60+3.37*j
        label(ax,x,3.38,key.replace("3","\u2083"),14,weight="bold",align="center")
        label(ax,x,3.05,f"A = {r.A}      B = {r.B}",9,SOFT,align="center")
        for mid,y in [(False,2.29),(True,.98)]:
            side=.91
            ax.add_patch(Rectangle((x-side/2,y-side/2),side,side,
                         facecolor="#f5f8f8",edgecolor=RULE,lw=1))
            coords=([(0,0,r.B),(.5,0,"O"),(-.5,0,"O"),(0,.5,"O"),(0,-.5,"O")]
                    if mid else [(i,k,r.A) for i,k in product([-.5,.5],repeat=2)]+[(0,0,"O")])
            for u,v,el in coords:
                sphere(ax,x+side*u,y+side*v,.095 if el=="O" else .135,colors[el])
        label(ax,x,.20,f"Gap {r.gap:.2f} eV    \u00b7    V {r.V:.4f}",9.5,align="center")
    for y,z,layer in [(2.29,"z = 0","AO plane"),(.98,"z = \u00bd","BO\u2082 plane")]:
        label(ax,.14,y+.11,z,12,weight="bold")
        label(ax,.14,y-.16,layer,9,SOFT)
    for x,c,s in [(3.98,BLUE,"Sn"),(3.98,GOLD,"Ti"),(3.98,RED,"O")]:
        y={"Sn":2.24,"Ti":1.88,"O":1.52}[s]
        sphere(ax,x,y,.069,c)
        label(ax,x+.13,y,s,8.5)
    save(fig,"site_exchange")


def electronic(public):
    fig, ax = plt.subplots(figsize=(8.0, 4.15))
    fig.subplots_adjust(left=.115, right=.78, top=.92, bottom=.31)
    sn = public.query("id==8760").iloc[0]
    ca = public.query("id==8811").iloc[0]
    pred = pd.read_csv(ROOT.parent/"results"/"physical_seed26.csv").query("id==8811").iloc[0]
    columns = [(r"$\mathrm{SnTiO_3}$","Public DFT",sn.cbm,sn.vbm,sn.gap),
               (r"$\mathrm{CaZrO_3}$","Property model",pred.pred_cbm,pred.pred_vbm,pred.pred_gap),
               (r"$\mathrm{CaZrO_3}$","Public DFT",ca.cbm,ca.vbm,ca.gap)]
    ax.set(xlim=(-.48,2.48), ylim=(9,1), yticks=range(1,10),
           ylabel="Reported band-edge value (eV)")
    ax.set_xticks(range(3), [title+"\n"+source for title,source,*_ in columns])
    ax.set_xlabel("Composition and data source",labelpad=28)
    ax.spines[["top","right"]].set_visible(False)
    ax.spines[["left","bottom"]].set_color(SOFT)
    ax.tick_params(axis="both", colors=INK, labelsize=9)
    ax.grid(axis="y",lw=.45,color="#e0e7e9")
    for threshold,color,name in [(4.50,GREEN,"CBM cutoff"),(5.73,VIOLET,"VBM cutoff")]:
        ax.axhline(threshold,color=color,lw=1,ls=(0,(4,3)))
        ax.text(1.02,threshold,f"{name}\n{threshold:.2f} eV",color=color,
                transform=ax.get_yaxis_transform(),va="center",fontsize=9)
    for x,(_,_,cb,vb,gap) in enumerate(columns):
        for e,c,offset in [(cb,GREEN,-.19),(vb,VIOLET,.24)]:
            ax.hlines(e,x-.28,x+.28,color=c,lw=3.2)
            ax.text(x,e+offset,f"{e:.3f}",color=c,ha="center",
                    va="bottom" if offset<0 else "top",fontsize=9)
        ax.annotate("",xy=(x,cb),xytext=(x,vb),
                    arrowprops=dict(arrowstyle="<->",lw=.9,color=SOFT,mutation_scale=7))
        ax.text(x,-.24,f"Gap  {gap:.2f} eV",transform=ax.get_xaxis_transform(),
                ha="center",fontsize=9,fontweight="bold")
    save(fig,"band_edges")


def anions(scores):
    fig,ax=board(3.40)
    o,n,f=np.array([.65,.57]),np.array([1.96,2.83]),np.array([3.27,.57])
    label(ax,.10,3.20,"Anion fractions",13,weight="bold")
    ax.add_patch(Polygon([o,n,f],facecolor="#f5f7f7",edgecolor=RULE,lw=1))
    for i in [1/3,2/3]:
        for v,a,b in [(o,n,f),(n,f,o),(f,o,n)]:
            u,w=i*v+(1-i)*a,i*v+(1-i)*b
            ax.plot([u[0],w[0]],[u[1],w[1]],color="#ccd7d8",lw=.65)
    for point,s,dy in [(o,"O",-.23),(n,"N",.18),(f,"F",-.23)]:
        label(ax,point[0],point[1]+dy,s,11,weight="bold",align="center")
    label(ax,4.0,3.20,"Fixed La\u2013Ti pair",13,weight="bold")
    label(ax,4.03,2.79,"Composition",8.7,SOFT)
    label(ax,7.80,2.79,"Predicted V",8.7,SOFT,align="right")
    configs=[("NO2","LaTiO\u2082N",["O","O","N"],VIOLET),
             ("FO2","LaTiO\u2082F",["O","O","F"],GREEN),
             ("O3","LaTiO\u2083",["O","O","O"],RED)]
    pair=scores[scores.A.eq("La") & scores.B.eq("Ti")].set_index("X")
    for j,(family,name,atoms,col) in enumerate(configs):
        yrow=2.30-.68*j
        point=np.mean([{"O":o,"N":n,"F":f}[el] for el in atoms],axis=0)
        ax.plot(*point,marker="o",ms=8,color=col,mec="white",mew=1,zorder=5)
        dx,dy={"NO2":(-.46,.21),"FO2":(.27,-.25),"O3":(-.42,.13)}[family]
        label(ax,point[0]+dx,point[1]+dy,family.translate(str.maketrans("23","\u2082\u2083")),9,col)
        label(ax,4.03,yrow,name,11.5)
        for k,el in enumerate(atoms):
            sphere(ax,5.56+k*.33,yrow,.132,{"O":RED,"N":VIOLET,"F":GREEN}[el])
            label(ax,5.56+k*.33,yrow,el,7.2,"white","bold","center")
        label(ax,7.80,yrow,f"{pair.loc[family,'V_hat']:.4f}",12,col,"bold","right")
        ax.plot([4,7.8],[yrow-.30,yrow-.30],lw=.55,color=RULE)
    save(fig,"anion_substitution")


if __name__=="__main__":
    public=pd.concat(pd.read_csv(next(DATA.glob(prefix+"_*.csv"))) for prefix in ["02","03"])
    scores=pd.read_csv(ROOT/"artifacts"/"q23_full.csv")
    crystal()
    site_exchange(public)
    electronic(public)
    anions(scores)

