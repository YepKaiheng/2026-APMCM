"""Orthographic, depth-tested rendering of three idealized perovskite scenes."""
from itertools import combinations, product

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

from perovskite_plates import DATA, ROOT, FIG, INK, SOFT
OUT = ROOT / "artifacts"

A = "#467b99"
B = "#ce9954"
O = "#c46157"
BOND = "#719499"


class Scene:
    """Camera-space analytic surfaces and a soft projected floor shadow."""

    def __init__(self, origin, scale, yaw=30, elev=19):
        self.width, self.height = 1350, 930
        self.origin = np.array(origin)
        self.scale = scale
        a, e = np.radians([yaw,elev])
        forward = np.array([np.sin(a),np.cos(a),0])
        up = np.array([0,0,1])
        self.camera = np.array([[np.cos(a),-np.sin(a),0],
                                forward*np.sin(e)-up*np.cos(e),
                                forward*np.cos(e)+up*np.sin(e)])
        self.rgb = np.zeros((self.height,self.width,3),np.uint8)
        self.depth = np.full((self.height,self.width),-np.inf)
        self.shadow = Image.new("L",(self.width,self.height))
        self.light = np.array([-.46,-.61,.645])
        self.light /= np.linalg.norm(self.light)
        self.half = self.light+[0,0,1]
        self.half /= np.linalg.norm(self.half)

    def project(self, xyz):
        p = self.scale*(self.camera @ xyz)
        p[:2] += self.origin
        return p

    def window(self, a, b, radius):
        lo = np.maximum([0,0],np.floor(np.minimum(a[:2],b[:2])-radius-1)).astype(int)
        hi = np.minimum([self.width,self.height],np.ceil(np.maximum(a[:2],b[:2])+radius+1)).astype(int)
        ys,xs = slice(lo[1],hi[1]),slice(lo[0],hi[0])
        y,x = np.mgrid[ys,xs]
        return ys,xs,x,y

    def paint(self, ys, xs, valid, z, normals, color):
        front = valid & (z>self.depth[ys,xs])
        normals = normals[front]
        diffuse = np.maximum(normals @ self.light,0)
        gloss = np.maximum(normals @ self.half,0)**24
        rgb = np.asarray(to_rgb(color))*(.51+.49*diffuse[:,None])+.26*gloss[:,None]
        self.rgb[ys,xs][front] = np.uint8(255*np.clip(rgb,0,1))
        self.depth[ys,xs][front] = z[front]

    def ball(self, xyz, radius, color):
        point = self.project(xyz)
        r = self.scale*radius
        ys,xs,x,y = self.window(point,point,r)
        nx,ny = (x-point[0])/r,(y-point[1])/r
        rr = 1-nx**2-ny**2
        nz = np.sqrt(np.maximum(rr,0))
        self.paint(ys,xs,rr>=0,point[2]+r*nz,np.stack([nx,ny,nz],axis=-1),color)
        pos = np.asarray(xyz)
        u,v,_ = self.project([pos[0]+.16*pos[2], pos[1]+.12*pos[2],-.15])
        ImageDraw.Draw(self.shadow).ellipse((u-r*1.3,v-r*.55,u+r*1.3,v+r*.55),fill=95)

    def rod(self, a, b, radius, color):
        p,q = self.project(a),self.project(b)
        delta = q-p
        length = np.linalg.norm(delta[:2])
        if length < .1:
            return
        t = delta[:2]/length
        side = np.r_[-t[1],t[0],0.]
        axis = delta/np.linalg.norm(delta)
        facing = np.array([0,0,1.])-axis*axis[2]
        facing /= np.linalg.norm(facing)
        r = radius*self.scale
        ys,xs,x,y = self.window(p,q,r)
        across = ((x-p[0])*side[0]+(y-p[1])*side[1])/r
        arc = np.sqrt(np.maximum(0,1-across**2))
        normal = across[...,None]*side+arc[...,None]*facing
        along = ((x-p[0]-r*normal[...,0])*t[0]+(y-p[1]-r*normal[...,1])*t[1])/length
        z = p[2]+along*delta[2]+r*normal[...,2]
        self.paint(ys,xs,(np.abs(across)<=1)&(along>=0)&(along<=1),z,normal,color)

    def image(self):
        alpha = np.asarray(self.shadow.filter(ImageFilter.GaussianBlur(14)))/255
        floor = np.uint8(255*(1-.45*alpha))
        return np.where(np.isfinite(self.depth)[...,None],self.rgb,floor[...,None])


def oxide(scene, repeat, omit=()):
    atoms = {}
    for cell in product(*(range(n) for n in repeat)):
        center = np.array(cell)+.5
        atoms[tuple(center)] = (.15,B)
        for sign in [-1,1]:
            for direction in np.eye(3):
                face = center+sign*.5*direction
                atoms[tuple(face)] = (.092,O)
                scene.rod(center,face,.023,BOND)
        for corner in product([0,1],repeat=3):
            atoms[tuple(np.array(cell)+corner)] = (.135,A)
    # Wire outlines of the bounding repeat volume.
    corners = list(product(*[(0,n) for n in repeat]))
    for a,b in combinations(corners,2):
        if np.count_nonzero(np.subtract(a,b))==1:
            scene.rod(a,b,.010,"#a8bdc2")
    for point,(rad,color) in atoms.items():
        if point not in omit:
            scene.ball(point,rad,color)


def plate(scene, title, subtitle, notes, name):
    fig = plt.figure(figsize=(8,3.8),facecolor="white")
    ax = fig.add_axes([.005,.025,.62,.84])
    ax.imshow(scene.image(),interpolation="lanczos")
    ax.axis("off")
    fig.text(.025,.925,title,fontsize=17,fontfamily="DejaVu Serif",color=INK)
    fig.text(.028,.852,subtitle,fontsize=9,color=SOFT)
    for y,heading,detail,color in notes:
        fig.text(.62,y,heading,fontsize=13.2,color=color,fontweight="bold")
        fig.text(.62,y-.074,detail,fontsize=9.2,color=INK,linespacing=1.38,va="top")
    for fmt in ["pdf","png"]:
        fig.savefig(FIG/f"{name}.{fmt}",dpi=240,facecolor="white")
    plt.close(fig)


def gapped_scene():
    table = pd.read_csv(OUT/"gate_components.csv")
    r = table[table.X.isin(["O3","NO2","FO2"])].nsmallest(1,"soft").iloc[0]
    assert r.formula=="InTiO3" and not r.positive_gap
    s = Scene((445,565),380,yaw=28,elev=18)
    oxide(s,(2,1,1))
    plate(s,r"$\mathrm{InTiO_3}$  |  Corner-sharing octahedra",
          "Ideal cubic motif extended along [100]",
          [(.69,"In at the corners","A-site identity is an input.",A),
           (.46,"Ti inside each cage","Two Ti sites share one O atom.",B),
           (.23,"Gap unknown at screening",f"Predicted P(g > 0) = {r.p_gap:.3f}\nObserved later: g = 0 eV",INK)],
          "gate_mechanism")


def photon_scene():
    r = pd.read_csv(ROOT.parent/"results"/"physical_seed26.csv").query("id==8811").iloc[0]
    actual = pd.read_csv(next(DATA.glob("03_*"))).query("id==8811").iloc[0]
    s = Scene((600,540),290,yaw=26,elev=22)
    oxide(s,(2,2,1))
    # Incoming photon symbol terminates at the oxide surface.
    u = np.linspace(0,1,130)
    beam = np.column_stack([-1.45+1.45*u,.5+.13*np.sin(8*np.pi*u),1.72-1.02*u])
    for a,b in zip(beam,beam[1:]):
        s.rod(a,b,.025,"#73569c")
    plate(s,r"$\mathrm{CaZrO_3}$  |  A photon at the gap cutoff",
          "Ideal oxide slab with an incident 3.10 eV photon",
          [(.69,"3.10 eV incident", "The top of the contest gap window.","#73569c"),
           (.46,f"{r.pred_gap:.2f} eV model gap","Photon energy exceeds this estimate.",B),
           (.23,f"{actual.gap:.2f} eV DFT gap","Photon energy falls below this gap.",A)],
          "property_tests")


def decoded_scene():
    winner = pd.read_csv(OUT/"selections.csv").query("problem=='Q3'").iloc[0]
    assert winner.formula=="SnTiO3" and winner.bin_min==1
    s = Scene((565,650),343,yaw=32,elev=22)
    # Separate adjacent ideal (001) planes along z for visibility.
    for z,mid in [(0,False),(.79,True),(1.58,False)]:
        ring = [np.array([i,j,z]) for i,j in [(0,0),(1,0),(1,1),(0,1)]]
        for a,b in zip(ring,ring[1:]+ring[:1]):
            s.rod(a,b,.014,"#90acb2")
        center = np.array([.5,.5,z])
        if mid:
            for a,b in zip(ring,ring[1:]+ring[:1]):
                face = (a+b)/2
                s.rod(center,face,.024,BOND)
                s.ball(face,.089,O)
            s.ball(center,.15,B)
        else:
            for corner in ring:
                s.ball(corner,.128,A)
            s.ball(center,.09,O)
    plate(s,r"$\mathrm{SnTiO_3}$  |  Reading out the selected sites",
          "Exploded (001) planes, with spacing enlarged for visibility",
          [(.69,"Sn  |  A site","0100 selects Sn from Si, Sn, Sr, Ta.",A),
           (.46,"Ti  |  B site","0100 selects Ti from Te, Ti, Tl, V.",B),
           (.23,"O  |  face sites",f"Saved family: O3\nRecovered row {int(winner.id)}, score {winner.V_hat:.4f}",O)],
          "qubo_decode")


def substitution():
    """Illustrate changing one site in the periodic anion basis from O to N."""
    nitrogen = "#835ba2"
    pair = pd.read_csv(OUT/"q23_full.csv").query("A=='La' and B=='Ti'").set_index("X")
    assert pair.loc["O3","formula"]=="LaTiO3" and pair.loc["NO2","formula"]=="LaTiO2N"
    faces = [(0.5,0.5,1.0),(0.5,0.5,0.0)]
    fig = plt.figure(figsize=(9.2,3.72),facecolor="white")
    fig.text(.022,.92,"One anion substitution in the La-Ti lattice",
             fontsize=16,color=INK,fontfamily="DejaVu Serif")
    fig.text(.748,.92,"O out",fontsize=10,color=O,fontweight="bold")
    fig.text(.850,.92,"N in",fontsize=10,color=nitrogen,fontweight="bold")
    stages = [("Parent oxide",r"$\mathrm{LaTiO_3}$"),
              ("Exchange one periodic site","Two shared faces"),
              ("Oxynitride",r"$\mathrm{LaTiO_2N}$")]
    for i,(title,formula) in enumerate(stages):
        left=.018+.335*i
        fig.text(left,.787,title,fontsize=10.7,fontweight="bold",color=INK)
        ax=fig.add_axes([left,.205,.295,.545])
        scene=Scene((558,571),435,yaw=31,elev=20)
        oxide(scene,(1,1,1),omit=faces)
        arrows=[]
        for j,pos in enumerate(faces):
            site=np.array(pos)
            if i==1:
                # Move symbols in the image plane, leaving the ideal site marked.
                shift=1 if j==0 else -1
                leave=np.array([-184,-95*shift,105])/scene.scale
                arrive=np.array([207,-71*shift,105])/scene.scale
                scene.ball(site+scene.camera.T@leave,.094,O)
                scene.ball(site+scene.camera.T@arrive,.096,nitrogen)
                u,v,_=scene.project(site)
                arrows.append((u,v,shift))
            else:
                scene.ball(site,.107,O if i==0 else nitrogen)
        ax.imshow(scene.image())
        for u,v,shift in arrows:
            ax.add_patch(plt.Circle((u,v),23,fill=False,ls=":",lw=1,color=SOFT))
            for start,end,col,bend in [
                    ((u-23,v-8*shift),(u-147,v-73*shift),O,.15*shift),
                    ((u+168,v-60*shift),(u+26,v-8*shift),nitrogen,-.12*shift)]:
                ax.annotate("",xy=end,xytext=start,arrowprops=dict(
                    arrowstyle="-|>",lw=1.65,color=col,mutation_scale=11,
                    connectionstyle=f"arc3,rad={bend}"))
        ax.set(xlim=(170,1090),ylim=(860,65))
        ax.axis("off")
        fig.text(left+.14,.154,formula,ha="center",fontsize=11.6,color=INK)
        footer = [f"Predicted V   {pair.loc['O3','V_hat']:.4f}",
                  r"$\frac{1}{2}+\frac{1}{2}=1$ anion per formula",
                  f"Predicted V   {pair.loc['NO2','V_hat']:.4f}"][i]
        fig.text(left+.14,.075,footer,ha="center",fontsize=9,color=SOFT)
    overlay=fig.add_axes([0,0,1,1],frameon=False)
    overlay.set(xlim=(0,1),ylim=(0,1))
    overlay.axis("off")
    for x in [.316,.650]:
        overlay.annotate("",xy=(x+.032,.47),xytext=(x-.007,.47),
                         arrowprops=dict(arrowstyle="-|>",color=SOFT,lw=1.55,mutation_scale=12))
    for fmt in ["pdf","png"]:
        fig.savefig(FIG/f"anion_exchange_scene.{fmt}",dpi=240,facecolor="white")
    plt.close(fig)



if __name__=="__main__":
    gapped_scene()
    photon_scene()
    decoded_scene()
    substitution()
