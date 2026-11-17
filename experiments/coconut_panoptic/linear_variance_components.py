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

"""Estimate the crossed training and validation-image variance of the linear pilot."""
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm, t


HERE=Path(__file__).resolve().parent
V9=HERE.parent


def one_score_influence(score,y,negative,n):
    a,b=int(y.sum()),int(negative.sum())
    sp,sn=np.sort(score[y]),np.sort(score[negative])
    lp=np.searchsorted(sn,score[y],"left")
    rp=np.searchsorted(sn,score[y],"right")
    ln=np.searchsorted(sp,score[negative],"left")
    rn=np.searchsorted(sp,score[negative],"right")
    u=.5*(lp+rp)/b
    v=1.-.5*(ln+rn)/a
    auc=u.mean()
    out=np.zeros(n)
    out[y]=(u-auc)/a
    out[negative]=(v-auc)/b
    return out


if __name__=="__main__":
    mask=json.loads((HERE/"PANOPTIC_MASK_SCREEN.json").read_text(encoding="utf-8"))
    ref=json.loads((HERE/"LINEAR_PRECISION_50E_6R.json").read_text(encoding="utf-8"))
    assert ref["endpoints"]==144 and ref["all_development_draws"]
    orig=np.load(V9/"coco_reference_labels/DEV80.npz",allow_pickle=False)
    extra=np.load(HERE/"PANOPTIC_OFFICIAL_LABELS.npz",allow_pickle=False)
    np.testing.assert_array_equal(orig["ids"],extra["dev_ids"])
    idx={key:i for i,key in enumerate(orig["ids"].tolist())}
    truth=np.column_stack((orig["truth"],extra["dev"]))
    classes=np.r_[orig["category_ids"],extra["category_ids"]].tolist()
    from itertools import product
    spots=[(classes.index(row["target_id"]),classes.index(p)) for row in mask["chosen"] for p in row["partners"]]
    image=np.zeros((8,5000))
    for d,row in enumerate(ref["rows"]):
        path=HERE/f"LINEAR_DEV_{row['draw']}_50E_6R.npz"
        with np.load(path,allow_pickle=False) as v:
            f=v["scores"].reshape(6,3,5000,133)
            y=truth[[idx[k] for k in v["ids"]]]
        for j,(target,partner) in enumerate(spots):
            pos=(y[:,target]==1)&(y[:,partner]==0)
            neg=(y[:,target]==0)&(y[:,partner]==1)
            k=j%2
            image[d]+=sum(one_score_influence(f[i,k,:,target],pos,neg,len(y))-
                          one_score_influence(f[i,1-k,:,target],pos,neg,len(y)) for i in range(6))/(6*len(spots))
    image_variance_pp2=float(np.sum(image.mean(0)**2)*100**2)
    values=np.array([r["directional_advantage_pp"] for r in ref["rows"]]).mean((2,3))
    dy,ni=values.shape
    x=values.mean()
    means=values.mean(1)
    init=values.mean(0)
    residual=values-means[:,None]-init[None,:]+x
    res=float(np.sum(residual**2)/((dy-1)*(ni-1)))
    mm=max(0.,float(means.var(ddof=1)-res/ni))
    ii=max(0.,float(init.var(ddof=1)-res/dy))
    rows=[]
    for m,r,n,stress in product(mask["eligible_formal_prefixes"],[4,6,8],[10000,12000,16000,20000,30000],[1,2]):
        training=stress*mm/m+ii/r+res/(m*r)
        scaled_image=image_variance_pp2*5000/n
        se=(training+scaled_image)**.5
        power=1.-norm.cdf((max(.30,norm.ppf(.95)*se)-.60)/se)
        rows.append(dict(masks=m,inits=r,n_nominal=n,mask_variance_stress=stress,halfwidth_pp=float(t.ppf(.975,m-1)*se),
                         indicative_normal_power=float(power)))
    out=dict(status="DEVELOPMENT_PROJECTION_ONLY",draws=dy,inits=ni,groups=len(spots),mask_variance_pp2=mm,
             init_variance_pp2=ii,residual_variance_pp2=res,dev_image_variance_pp2=image_variance_pp2,
             projected_rows=rows,scope="DEV influence and size scaling; original stratified confirmation design and full joint gate still require calibration",
             confirmation_scores_read=0)
    (HERE/"LINEAR_CROSSED_VARIANCE_QA.json").write_text(json.dumps(out,indent=2)+"\n")
    print({k:v for k,v in out.items() if k!="projected_rows"},flush=True)
