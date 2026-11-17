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

"""Calculate development interactions and a necessary mask-variance precision bound."""
from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, t


HERE = Path(__file__).resolve().parent
OLD = HERE.parent/"coco_reference_labels"


def auc(scores,pos,neg):
    v=np.r_[scores[pos],scores[neg]]
    n,m=int(pos.sum()),int(neg.sum())
    assert min(n,m)>0 and np.isfinite(v).all()
    return (rankdata(v)[:n].sum()-n*(n+1)/2)/(n*m)


def main(epochs,seeds):
    mask=json.loads((HERE/"PANOPTIC_MASK_SCREEN.json").read_text(encoding="utf-8"))
    with (np.load(OLD/"DEV80.npz",allow_pickle=False) as a,
          np.load(HERE/"PANOPTIC_OFFICIAL_LABELS.npz",allow_pickle=False) as b):
        np.testing.assert_array_equal(a["ids"],b["dev_ids"])
        cats = np.r_[a["category_ids"],b["category_ids"]].tolist()
        y = np.column_stack((a["truth"],b["dev"]))
        index={v:j for j,v in enumerate(a["ids"].tolist())}
    slots = [(cats.index(row["target_id"]),[cats.index(k) for k in row["partners"]]) for row in mask["chosen"]]
    metrics=[]
    for draw in mask["dev_seeds"]:
        file=HERE/f"LINEAR_DEV_{draw}_{epochs}E_{seeds}R.npz"
        if not file.exists():
            continue
        with np.load(file,allow_pickle=False) as f:
            assert f["categories"].tolist()==cats
            assert f["arms"].tolist()==["DIR1","DIR2","PLC"]*seeds
            truth=y[[index[i] for i in f["ids"]]]
            values = f["scores"]
        output=np.zeros((seeds,len(slots),2))
        matched=np.zeros_like(output)
        for j,(target,partners) in enumerate(slots):
            for col,p in enumerate(partners):
                pos=(truth[:,target]==1)&(truth[:,p]==0)
                neg=(truth[:,target]==0)&(truth[:,p]==1)
                u=np.array([auc(s[:,target],pos,neg) for s in values]).reshape(seeds,3)
                output[:,j,col]=u[:,col]-u[:,1-col]
                matched[:,j,col]=u[:,col]-u[:,2]
        metrics.append(dict(draw=draw,directional_advantage_pp=(100*output).tolist(),
                        matched_vs_placebo_pp=(100*matched).tolist(),avg_I_pp=float(output.mean()*100)))
    assert metrics
    bounds=[]
    if len(metrics)>1:
        u=np.array([r["directional_advantage_pp"] for r in metrics]).mean((2,3))
        means=u.mean(1)
        within=float(u.var(1,ddof=1).mean()) if seeds>1 else 0.
        mask_var=max(0.,float(means.var(ddof=1))-within/seeds)
        for n in mask["eligible_formal_prefixes"]:
            for r in (4,6,8):
                assumed=(mask_var+within/r)/n
                half=float(t.ppf(.975,n-1)*np.sqrt(assumed))
                bounds.append(dict(masks=n,inits=r,training_only_95pct_halfwidth_pp=half,training_only_stress_halfwidth_pp=float(t.ppf(.975,n-1)*np.sqrt((2*mask_var+within/r)/n))))
    result=dict(status="ENGINEERING_PILOT" if epochs<50 else "DEVELOPMENT_ONLY_VARIANCE_SCREEN",endpoints=len(metrics)*seeds*3,
         class_order=cats,development_images=5000,formal_score_reads=0,rows=metrics,training_only_matrix=bounds,
         mean_development_I_pp=float(np.mean([r["avg_I_pp"] for r in metrics])),
         by_target_pp=np.mean([r["directional_advantage_pp"] for r in metrics],axis=(0,1,3)).tolist(),
         by_direction_pp=np.mean([r["directional_advantage_pp"] for r in metrics],axis=(0,1,2)).tolist(),
         all_development_draws=len(metrics)==len(mask["dev_seeds"]),
         scope="Variance calculations use development predictions and omit finite-population image sampling. Candidate projections are not full P2 power gates.")
    (HERE/f"LINEAR_PRECISION_{epochs}E_{seeds}R.json").write_text(json.dumps(result,indent=2)+"\n")
    print({k:v for k,v in result.items() if k not in ("rows","class_order","training_only_matrix")},flush=True)


if __name__=="__main__":
    parser=ArgumentParser()
    parser.add_argument("--epochs",type=int,default=50)
    parser.add_argument("--seeds",type=int,default=6)
    main(**vars(parser.parse_args()))
