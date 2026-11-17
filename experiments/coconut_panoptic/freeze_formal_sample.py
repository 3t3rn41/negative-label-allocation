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

"""Freeze the one formal sample and derive its pixel request after the precision lock."""
from argparse import ArgumentParser
import csv
import json
from pathlib import Path

import numpy as np

from panoptic_sampling import ORDER, SEED, ess_gate, groups


HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/"coconut_inputs/preparation/inputs/UNLABELED2017_SOURCE_ROWS.csv"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path,obj):
    stage=path.with_suffix(".partial")
    stage.write_text(json.dumps(obj,indent=2)+"\n",encoding="utf-8")
    stage.replace(path)


def fixed_sample(strata,N,quota,seed_index):
    rng=np.random.default_rng(np.random.SeedSequence([SEED,3,seed_index]))
    assert np.array_equal(np.bincount(strata,minlength=len(N)),N)
    promoted=[]
    for h,nh in enumerate(quota):
        candidates=np.flatnonzero(strata==h)
        assert 2<=nh<=len(candidates) or nh==len(candidates)
        promoted.append(rng.choice(candidates,nh,replace=False))
    return np.concatenate(promoted)


def pair_support(signed,by_stratum,strata_total,quota,source_p):
    counts=np.zeros((len(quota),len(source_p),4),dtype=int)
    for h in range(len(quota)):
        counts[h]=np.asarray([np.bincount(value,minlength=4)[ORDER] for value in signed[by_stratum==h].T])
    weights=np.divide(strata_total,quota,out=np.zeros(len(quota)),where=quota>0)
    weighted=np.sum(counts*weights[:,None,None],axis=0)
    squares=np.sum(counts*weights[:,None,None]**2,axis=0)
    with np.errstate(divide="ignore",invalid="ignore"):
        fractions=squares/weighted**2
        pos=(weighted[:,0]+weighted[:,2])**2/(squares[:,0]+squares[:,2])
        neg=(weighted[:,1]+weighted[:,3])**2/(squares[:,1]+squares[:,3])
        fixed_pos=1/((1-source_p)**2*fractions[:,0]+source_p**2*fractions[:,2])
        fixed_neg=np.stack([1/(r**2*fractions[:,1]+(1-r)**2*fractions[:,3]) for r in (.05,.50)],axis=1)
    passed=bool(ess_gate(counts[:,None],weights,source_p)[0])
    finite=lambda a: np.where(np.isfinite(a),a,None).tolist()
    return dict(passed=passed,four_cell_order=["10","01","11","00"],counts_by_stratum=counts.tolist(),
                weights_by_stratum=weights.tolist(),natural_positive_ess=finite(pos),natural_negative_ess=finite(neg),
                fixed_positive_ess=finite(fixed_pos),fixed_negative_ess_005_050=finite(fixed_neg))


def main(precision_lock,lineage_path=SOURCE):
    workspace=precision_lock.resolve().parent
    lock=read(precision_lock)
    assert lock["status"]=="READY_TO_DRAW" and lock["confirmation_scores_read"]==0
    seed=lock["seed_index"]
    assert isinstance(seed,int) and seed>=0
    chosen=lock["sample_design"]
    quota=np.asarray(chosen["sample_counts"],dtype=np.int64)
    prospect=read(workspace/"PANOPTIC_SAMPLE_QA.json")
    matches=[x for x in prospect["designs"] if x["passed"] and x["sample_counts"]==quota.tolist()]
    assert len(matches)>=1
    panels=read(workspace/"TARGET_REPAIR_STATUS.json")["selected"]
    assert panels==prospect["panel"] and len(panels)>=4
    with np.load(workspace/"PANOPTIC_POPULATION133.npz",allow_pickle=False) as population:
        ids=population["ids"]
        truth=population["truth"]
        categories=population["categories"].tolist()
    assert len(np.unique(ids))==len(ids) and ((truth==0)|(truth==1)).all()
    definition=[(t["target_id"],p) for t in panels for p in t["partners"]]
    cols=[(categories.index(t),categories.index(p)) for t,p in definition]
    code=groups(truth,cols)
    strata=(code==3).any(1).astype(np.int64)
    N=np.bincount(strata,minlength=2)
    assert N.tolist()==matches[0]["strata_N"]
    with np.load(workspace/"PANOPTIC_BRIDGE_LABELS.npz",allow_pickle=False) as bridge:
        assert bridge["categories"].tolist()==categories
        source=bridge["truth"]
    source_p=np.array([((source[:,a]==1)&(source[:,b]==1)).sum()/((source[:,a]==1)&(source[:,b]>=0)).sum()
                       for a,b in cols])
    saved=workspace/"SAMPLING_LOCK.json"
    signature=dict(seed_sequence=[SEED,3,seed],population_by_stratum=N.tolist(),quota=quota.tolist(),pairs=definition)
    if saved.exists():
        record=read(saved)
        assert record["signature"]==json.loads(json.dumps(signature))
        order={v:i for i,v in enumerate(ids.tolist())}
        drawn=np.asarray([order[v] for v in record["sample_ids"]],dtype=np.int64)
    else:
        drawn=fixed_sample(strata,N,quota,seed)
        record=dict(signature=signature,population_ids=ids.tolist(),population_strata=strata.tolist(),
                    sample_ids=ids[drawn].tolist(),sample_strata=strata[drawn].tolist(),
                    inclusion_probability_by_stratum=(quota/N).tolist(),
                    sampling_weight_by_stratum=(N/quota).tolist(),algorithm="stratified SRSWOR using PCG64",
                    confirmation_model_scores_read=0,pixel_verification_pending=True)
        write(saved,record)
    assert len(np.unique(drawn))==quota.sum() and np.array_equal(np.bincount(strata[drawn],minlength=2),quota)
    pair=pair_support(code[drawn],strata[drawn],N,quota,source_p)
    positives=truth[drawn].sum(0)
    negatives=len(drawn)-positives
    sample_weight=(N/quota)[strata[drawn]]
    ps=(truth[drawn]*sample_weight[:,None]).sum(0)
    ns=(1-truth[drawn])*sample_weight[:,None]
    divide=lambda a,b: np.divide(a,b,out=np.zeros_like(a,dtype=float),where=b>0)
    p_ess=divide(ps**2,(truth[drawn]*sample_weight[:,None]**2).sum(0))
    n_ess=divide(ns.sum(0)**2,(ns**2).sum(0))
    passed=pair["passed"] and (positives>0).all() and (negatives>0).all()
    qa=dict(status="PASS_LABEL_SUPPORT" if passed else "HOLD_SAMPLE_SUPPORT",sample_images=len(drawn),
            activated_pairs=definition,paired_support=pair,categories=categories,positive_counts=positives.tolist(),
            negative_counts=negatives.tolist(),positive_ess=p_ess.tolist(),negative_ess=n_ess.tolist(),
            confirmation_scores_read=0)
    write(workspace/"SAMPLE_SUPPORT_QA.json",qa)
    if not passed:
        raise RuntimeError("HOLD_SAMPLE_SUPPORT")
    sample_labels=workspace/"CONF_M_LABELS.npz"
    temp=sample_labels.with_suffix(".partial")
    with temp.open("wb") as file:
        np.savez_compressed(file,ids=ids[drawn],truth=truth[drawn],categories=categories,
                            strata=strata[drawn],weights=sample_weight,totals=N,quotas=quota)
    temp.replace(sample_labels)
    with lineage_path.open(encoding="utf-8-sig",newline="") as f:
        lineage={int(row["id"]):row for row in csv.DictReader(f)}
    assert all(i in lineage for i in record["sample_ids"])
    table=workspace/"CONF_M_PIXEL_REQUEST.csv"
    stage=table.with_suffix(".partial")
    with stage.open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=["id","width","height","flickr_id","zip_member"])
        writer.writeheader()
        for i in record["sample_ids"]:
            row={key:lineage[i][key] for key in writer.fieldnames[:-1]}
            writer.writerow(dict(row,zip_member=f"unlabeled2017/{i:012d}.jpg"))
    stage.replace(table)
    print(qa["status"],len(drawn),flush=True)


if __name__=="__main__":
    parser=ArgumentParser()
    parser.add_argument("precision_lock",type=Path)
    parser.add_argument("--lineage",type=Path,default=SOURCE)
    args=parser.parse_args()
    main(args.precision_lock,args.lineage)
