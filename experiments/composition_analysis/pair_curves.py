# SPDX-License-Identifier: MIT
# MIT License
#
# Copyright (c) 2026 Fixed-query experiment contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Report pair-level composition and audit unrestricted versus domain roots."""
from collections import defaultdict
from itertools import groupby

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from figures import STYLE
from replay import HERE, draw_effect, read, root, table
from summarize import csv_rows

KEYS = ("stage", "train", "baseline", "evaluation")
LABELS = {"chair":"椅子","bottle":"瓶子","diningtable":"餐桌","person":"人","pottedplant":"盆栽",
          "sofa":"沙发","tvmonitor":"电视／显示器","aggregate":"三目标平均"}
ENGLISH = {"chair":"Chair","bottle":"Bottle","diningtable":"Dining table","person":"Person",
           "pottedplant":"Potted plant","sofa":"Sofa","tvmonitor":"TV / monitor"}


def audit_roots():
    rows = []
    for stage in ("S0","A","B","VOC"):
        record = read(HERE/f"{stage}_REPLAY.json")
        with np.load(HERE/f"{stage}_REPLAY.npz") as file:
            effects, indices = file["effects"],file["indices"]
        for e,(evaluation,pairs) in enumerate(record["design"].items()):
            for train,baseline in record["contrasts"]:
                t,b = [record["arms"].index(a) for a in (train,baseline)]
                coefficients = 100*(effects[:,:,:,t,e,:,2:]-effects[:,:,:,b,e,:,2:]).mean(axis=2)
                groups = [("aggregate","",coefficients.mean(axis=2))]
                groups.extend((record["classes"][target],record["classes"][partner],coefficients[:,:,j])
                              for j,(target,partner,_) in enumerate(pairs))
                for target,partner,value in groups:
                    a,c = draw_effect(value,indices).T
                    roots = root(a,c)
                    finite = roots[np.isfinite(roots)]
                    in_domain = finite[(finite>=0)&(finite<=1)]
                    lo_all,hi_all = np.quantile(finite,[.025,.975])
                    lo_domain,hi_domain = np.quantile(in_domain,[.025,.975])
                    rows.append(dict(stage=stage,train=train,baseline=baseline,evaluation=evaluation,
                                     target=target,partner=partner,replicates=len(roots),below_0=int((finite<0).sum()),
                                     in_domain=len(in_domain),above_1=int((finite>1).sum()),
                                     undefined=len(roots)-len(finite),equal_0_or_1=int(((finite==0)|(finite==1)).sum()),
                                     in_domain_percent=100*len(in_domain)/len(roots),
                                     unconstrained_lo_percent=100*lo_all,unconstrained_hi_percent=100*hi_all,
                                     conditional_lo_percent=100*lo_domain,conditional_hi_percent=100*hi_domain))
    table("Z2_root_domain_audit.csv",rows)
    return rows


def title(key):
    stage,train,base,evaluation = key
    arm = train.removesuffix("_TRUE")
    return f"{arm} train / {evaluation} eval" if stage in ("B","VOC") else f"TRUE vs {'LOW' if base=='LOW' else 'PLC'}"


def figures(roots):
    data = defaultdict(list)
    for row in csv_rows(HERE/"Z2_curve.csv"):
        if row["target"]!="aggregate":
            data[tuple(row[k] for k in (*KEYS,"target"))].append(
                [float(row[k]) for k in ("r","point_pp","low_pp","high_pp")])
    with plt.rc_context(STYLE), PdfPages(HERE/"Z2_pair_curves.pdf") as pdf:
        for stage in ("S0","A","B","VOC"):
            current = [r for r in roots if r["stage"]==stage]
            grouped = [(key,list(items)) for key,items in groupby(current,key=lambda r:tuple(r[k] for k in KEYS))]
            values = np.concatenate([data[tuple(row[k] for k in (*KEYS,"target"))] for row in current])
            ylo,yhi = min(0,values[:,2].min()),max(0,values[:,3].max())
            padding = .12*(yhi-ylo)
            fig,axes = plt.subplots(len(grouped),3,figsize=(11.3,2.33*len(grouped)+1.28),squeeze=False,sharey=True)
            fig.subplots_adjust(left=.08,right=.98,bottom=.53/fig.get_figheight(),top=1-.70/fig.get_figheight(),
                                hspace=.61,wspace=.15)
            for axrow,(key,rows) in zip(axes,grouped,strict=True):
                for ax,row in zip(axrow,rows,strict=True):
                    x,y,lo,hi = np.asarray(data[(*key,row["target"])]).T
                    ax.fill_between(x*100,lo,hi,color="#d0e2ed",linewidth=0)
                    ax.plot(x*100,y,color="#156184",lw=1.7)
                    ax.axhline(0,color="#65747d",ls="--",lw=.8)
                    natural=float(row["r_natural"])*100
                    ax.axvline(natural,color="#9b430f",ls=":",lw=1.0)
                    ax.scatter(natural,float(row["natural_at_pstar_pp"]),color="#9b430f",s=17,zorder=4)
                    if row["root_in_domain"]:
                        ax.scatter(float(row["root_in_domain"])*100,0,color="#152633",s=18,marker="D",zorder=4)
                    target,partner = [ENGLISH[row[k]] for k in ("target","partner")]
                    ax.set_title(f"{target} / {partner}",loc="left",fontsize=8.5,pad=5)
                    ax.set(xlim=(0,100),ylim=(ylo-padding,yhi+padding),xticks=(0,25,50,75,100))
                    ax.tick_params(labelsize=8)
                    if ax is axrow[0]:
                        ax.set_ylabel(f"{title(key)}\nAUC difference (pp)",fontsize=8)
            fig.suptitle(f"{stage}  ·  Composition curves by target and partner",x=.08,ha="left",fontweight="bold",
                         fontsize=14,y=1-.12/fig.get_figheight())
            fig.text(.08,1-.38/fig.get_figheight(),
                     "Blue band: pointwise 95% CI   •   Orange: natural r on the fixed-p* curve   •   Diamond: zero crossing",
                     fontsize=8.5)
            fig.supxlabel("Partner-bearing share among target negatives, r (%)",y=.12/fig.get_figheight(),fontsize=9)
            pdf.savefig(fig)
            fig.savefig(HERE/f"Z2_pair_curves_{stage}.png",dpi=300)
            plt.close(fig)


def report(roots,audit):
    lines = [
        "# B 池自然构成、零交点区间及逐对曲线",
        "",
        "## B 池自然构成完整 AUC",
        "",
        "单位为百分点（pp）。每格是相对共同 PLC 的差值及双层双侧 95% 区间，5,000 次有效重采样。",
        "",
        "| 目标 | S0 训练／S0 或 N1 评价 | N1 训练／S0 或 N1 评价 |",
        "| --- | ---: | ---: |",
    ]
    natural=[r for r in csv_rows(HERE/"Z1_natural_auc.csv") if r["stage"]=="B" and r["evaluation"]=="S0"]
    lookup={(r["target"],r["train"]):r for r in natural}
    for target in ("aggregate","chair","diningtable","pottedplant"):
        cells=[]
        for arm in ("S0_TRUE","N1_TRUE"):
            r=lookup[target,arm]
            d,lo,hi=[float(r[k]) for k in ("point_pp","low_pp","high_pp")]
            cells.append(f"{d:+.5f} [{lo:+.5f}, {hi:+.5f}]")
        lines.append(f"| {LABELS[target]} | {' | '.join(cells)} |")
    lines += [
        "",
        "同一训练臂在 B 的两列对同一批图像、同一目标分数做自然完整排序，改变评价伙伴后结果相同。",
        "",
        "## 零交点区间的域内筛选",
        "",
        r"每次双层重采样用 $r^*=-a/b$ 求交点。域内区间仅对 $0\le r^*\le1$ 的样本取 2.5% 和 97.5% 分位。越界交点单独计数。5% 反事实评价档独立于这个交点计算。",
        "",
        "| 池 | 对比 | 域内交点／5,000 | 占比 |",
        "| --- | --- | ---: | ---: |",
    ]
    for r in audit:
        if r["target"]=="aggregate":
            k=tuple(r[x] for x in KEYS)
            lines.append(f"| {k[0]} | {title(k)} | {r['in_domain']:,} | {r['in_domain_percent']:.2f}% |")
    lines += [
        "",
        "[全量域内及越界计数](Z2_root_domain_audit.csv)",
        "",
        "## 逐目标—伙伴构成曲线",
        "",
        "[四页曲线 PDF](Z2_pair_curves.pdf) · [S0](Z2_pair_curves_S0.png) · [A](Z2_pair_curves_A.png) · [B](Z2_pair_curves_B.png) · [VOC](Z2_pair_curves_VOC.png) · [完整曲线及区间](Z2_curve.csv)",
        "",
        r"每格均固定原协议的正例构成 $p^*$。图中橙点标记实际自然负例比例 $r_{\rm nat}$ 在该曲线上的位置。真实自然 AUC 同时采用实际正例构成和负例构成，见 [自然 AUC 表](Z1_natural_auc.csv)。",
        "",
        "| 池 | 对比 | 目标／伙伴 | 自然 r | 点估计零交点 | 自然 r 的位置 | 固定 p* 时的差值（pp） |",
        "| --- | --- | --- | ---: | ---: | --- | ---: |",
    ]
    for r in roots:
        key=tuple(r[x] for x in KEYS)
        natural=float(r["r_natural"])
        delta=float(r["natural_at_pstar_pp"])
        crossing=r["root_in_domain"]
        sign="正" if delta>0 else "负" if delta<0 else "零"
        if crossing:
            rr=float(crossing)
            direction=("左侧" if natural<rr else "右侧" if natural>rr else "交点")+f"（{sign}）"
            cross=f"{100*rr:.2f}%"
        else:
            direction=f"域内恒{sign}"
            cross="—"
        pair=f"{LABELS[r['target']]}／{LABELS[r['partner']]}"
        lines.append(f"| {r['stage']} | {title(key)} | {pair} | {100*natural:.2f}% | {cross} | {direction} | {delta:+.3f} |")
    (HERE/"composition_curve_audit.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def main():
    roots=csv_rows(HERE/"Z2_roots.csv")
    audit=audit_roots()
    for original,replay in zip(roots,audit,strict=True):
        assert tuple(original[k] for k in (*KEYS,"target","partner"))==tuple(replay[k] for k in (*KEYS,"target","partner"))
        assert int(original["boot_roots"])==replay["in_domain"]
        np.testing.assert_allclose(float(original["root_ci_low_conditional"])*100,replay["conditional_lo_percent"],atol=1e-12)
    pair_rows=[r for r in roots if r["target"]!="aggregate"]
    assert len(pair_rows)==33
    figures(pair_rows)
    report(pair_rows,audit)


if __name__=="__main__":
    main()


