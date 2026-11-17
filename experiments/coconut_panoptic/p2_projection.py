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

"""Estimate crossed and image covariance for every frozen precision candidate."""
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from itertools import product
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.stats import beta, norm, t

import verify_joint_dev_replicas as a1
from joint_replica import Ranking


HERE=Path(__file__).resolve().parent
SEED=93120261115


def read(name):
    return json.loads((HERE/name).read_text(encoding="utf-8"))


def dev_gold_path():
    return Path(os.environ.get("FQ_COCO_DEV_LABELS",HERE.parent/"coco_reference_labels/DEV80.npz"))


def positive_covariance(matrix):
    values,basis=np.linalg.eigh((matrix+matrix.T)*.5)
    return (basis*values.clip(min=0))@basis.T


def compute_components(heads,truth,pairs,stratum,Nh):
    m,r,_,n,_=heads.shape
    counts=np.bincount(stratum,minlength=len(Nh))
    assert counts.shape==Nh.shape and (counts>=2).all()
    weight=(Nh/counts)[stratum]
    values=np.zeros((m,r,len(pairs)))
    image=np.zeros((n,len(pairs)))
    for j,(target,partner) in enumerate(pairs):
        pos=(truth[:,target]==1)&(truth[:,partner]==0)
        neg=(truth[:,target]==0)&(truth[:,partner]==1)
        for mask,init in product(range(m),range(r)):
            for arm,sign in ((j%2,1),(1-j%2,-1)):
                rank=Ranking(heads[mask,init,arm,:,target],pos,neg)
                mean,gradient=rank.influence(weight)
                values[mask,init,j]+=sign*mean*100
                image[:,j]+=sign*gradient*100/(m*r)
    mask_means=values.mean(1)
    init_means=values.mean(0)
    residual=values-mask_means[:,None]-init_means[None,:]+values.mean((0,1))
    flat=residual.reshape(m*r,len(pairs))
    residual_cov=flat.T@flat/((m-1)*(r-1))
    raw_mask=np.cov(mask_means,rowvar=False)-residual_cov/r
    raw_init=np.cov(init_means,rowvar=False)-residual_cov/m
    image_cov=np.asarray([np.cov(image[stratum==h],rowvar=False) for h in range(len(Nh))])
    return values,(positive_covariance(raw_mask),positive_covariance(raw_init),residual_cov),image_cov,counts


def passed_rows(point,by_pair,lower,lower_t,prompt):
    good=(point>=.30)&(lower>0)&(lower_t>0)
    if prompt:
        target=by_pair.reshape(len(point),-1,2).mean(2)
        dirs=by_pair.reshape(len(point),-1,2).mean(1)
        good&=((target>0).sum(1)>=int(np.ceil(2*target.shape[1]/3)))
        good&=(dirs>0).all(1)
        good&=(target.sum(1,keepdims=True)-target>0).all(1)
    return good


def project(heads,truth,pairs,stratum,designs,model,reps):
    Nh=np.asarray(designs[0]["strata_N"],dtype=np.int64)
    assert all(d["strata_N"]==Nh.tolist() for d in designs)
    values,training,image,dev_counts=compute_components(heads,truth,pairs,stratum,Nh)
    mask_cov,init_cov,interaction_cov=training
    image_for_design=[]
    for d in designs:
        nh=np.asarray(d["sample_counts"])
        image_for_design.append(positive_covariance(np.einsum("h,hij->ij",Nh**2*(1-nh/Nh)/nh,image)))
    rng=np.random.default_rng(np.random.SeedSequence([SEED,30,832 if model=="dualcoop" else 831]))
    contrast=values.mean((0,1))
    heterogeneous=contrast-contrast.mean()
    if np.max(np.abs(heterogeneous)):
        heterogeneous*=.25/np.max(np.abs(heterogeneous))
    output=[]
    for m,r,stress in product((12,16,24,32,48),(4,6,8),(1,2)):
        per_mask=rng.multivariate_normal(np.zeros(len(pairs)),mask_cov*stress+interaction_cov/r,size=(reps,m))
        per_init=rng.multivariate_normal(np.zeros(len(pairs)),init_cov/r,size=reps)
        train_center=per_mask.mean(1)+per_init
        mask_error=t.ppf(.95,m-1)*per_mask.mean(2).std(1,ddof=1)/np.sqrt(m)
        var_training=(mask_cov*stress/m+init_cov/r+interaction_cov/(m*r)).mean()
        for d,img_cov in zip(designs,image_for_design,strict=True):
            img=rng.multivariate_normal(np.zeros(len(pairs)),img_cov,size=reps)
            ci_se=np.sqrt(max(0.,var_training+img_cov.mean()))
            halfwidth=norm.ppf(.975)*ci_se
            for label,effect in (("null",0.),("homogeneous",.60),("heterogeneous",.60+heterogeneous)):
                by_pair=train_center+img+effect
                point=by_pair.mean(1)
                hits=int(passed_rows(point,by_pair,point-norm.ppf(.95)*ci_se,point-mask_error,
                                     model=="dualcoop").sum())
                upper=1. if hits==reps else float(beta.ppf(.95,hits+1,reps-hits))
                output.append(dict(masks=m,inits=r,n=d["total"],quota=d["sample_counts"],stress=stress,
                                   effect=label,gate_probability=hits/reps,gate_upper95=upper,
                                   halfwidth_pp=float(halfwidth)))
    return dict(status="DEVELOPMENT_COVARIANCE_SCREEN",model=model,outer_projections_per_scenario=reps,
                design_model="Gaussian crossed effects and A1 design-weighted rank influence by stratum",
                dev_stratum_counts=dev_counts.tolist(),score_matrix=list(heads.shape[:2]),
                mean_weighted_development_pp=float(contrast.mean()),effect_direction_pp=contrast.tolist(),
                mask_covariance_pp2=mask_cov.tolist(),init_covariance_pp2=init_cov.tolist(),
                residual_covariance_pp2=interaction_cov.tolist(),formal_scores_read=0,
                rows=output)


def data(model):
    if model=="linear":
        a1.load()
        heads,y,stratum,_,pairs,_=a1.DEVELOPMENT_DATA
        return heads,y,pairs,stratum
    design=read("PANOPTIC_MASK_SCREEN.json")
    with (np.load(dev_gold_path(),allow_pickle=False) as old,
          np.load(HERE/"PANOPTIC_OFFICIAL_LABELS.npz",allow_pickle=False) as extra):
        np.testing.assert_array_equal(old["ids"],extra["dev_ids"])
        lookup={v:i for i,v in enumerate(old["ids"].tolist())}
        all_labels=np.c_[old["truth"],extra["dev"]]
        schema=np.r_[old["category_ids"],extra["category_ids"]].tolist()
    selected=design["chosen"]
    columns=sorted({x for row in selected for x in [row["target_id"],*row["partners"]]})
    positions=[schema.index(c) for c in columns]
    pairs=[(columns.index(row["target_id"]),columns.index(partner)) for row in selected for partner in row["partners"]]
    IDs=None
    arrays=[]
    for draw in design["dev_seeds"]:
        with np.load(HERE/f"PROMPT_DEV_{draw}.npz",allow_pickle=False) as file:
            assert file["categories"].tolist()==schema and file["arms"].tolist()==["DIR1","DIR2","PLC"]*4
            rows=file["ids"][:5000]
            if IDs is not None:
                np.testing.assert_array_equal(IDs,rows)
            IDs=rows
            arrays.append(file["scores"][:,:5000,positions].reshape(4,3,len(rows),len(positions)))
    y=all_labels[np.ix_([lookup[i] for i in IDs],positions)]
    strata=np.logical_or.reduce([(y[:,target]==1)&(y[:,partner]==1) for target,partner in pairs]).astype(np.int64)
    return np.asarray(arrays),y,pairs,strata


if __name__=="__main__":
    parser=ArgumentParser()
    parser.add_argument("model",choices=["linear","dualcoop"])
    parser.add_argument("--reps",type=int,default=2500)
    parser.add_argument("--projection-only",action="store_true")
    parser.add_argument("--replica-workers",type=int,default=4)
    args=parser.parse_args()
    designs=[row for row in read("PANOPTIC_SAMPLE_QA.json")["designs"] if row["passed"]]
    report=project(*data(args.model),designs,args.model,args.reps)
    output=HERE/f"P2_{args.model.upper()}_CANDIDATE_PROJECTION.json"
    stage=output.with_suffix(".partial")
    stage.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    stage.replace(output)
    print(report["status"],report["model"],len(report["rows"]),"scenario rows",flush=True)
    if args.model=="dualcoop" and not args.projection_only:
        with ThreadPoolExecutor(max_workers=2) as pool:
            parity=pool.submit(subprocess.run,[sys.executable,str(HERE/"verify_joint_dev_replicas.py"),"--model","dualcoop",
                                              "--workers",str(args.replica_workers)],check=True)
            coverage=pool.submit(subprocess.run,[sys.executable,str(HERE/"p2_dev_coverage.py"),"dualcoop"],check=True)
            parity.result()
            coverage.result()
