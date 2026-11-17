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

"""Analyze the supplement with the original paired bootstrap and frozen pools."""
import argparse
from datetime import datetime
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

HERE=Path(__file__).resolve().parent
V9=HERE.parent
CPU=V9/"composition_analysis"
sys.path.insert(0,str(CPU))
from replay import (AB, N_BOOT, P0, ROLE, S0, VOC, Metric, bootstrap_for, design_for,
                    draw_effect, group_codes, read, replay_auc, set_num_threads)
from numba import njit, prange

STAGE_INPUTS={"B":AB/"B_GOLD.npz","VOC":ROLE/"TEST_GOLD.npz","S0":P0/"A1/CONFIRM_GOLD.npz"}
STAGE_FOR={"T1":"B","T2":"B","T3":"S0","T4":"VOC","T5":"B","T7":"B"}


@njit(parallel=True,cache=True)
def ap_kernel(weights,positive,negative,thresholds,neg_limits,voc):
    result=np.empty(len(weights),dtype=np.float64)
    for replicate in prange(len(weights)):
        precision=np.empty(len(thresholds),dtype=np.float64)
        recall_counts=np.empty(len(thresholds),dtype=np.int32)
        pi=ni=tp=fp=0
        w=weights[replicate]
        for j in range(len(thresholds)):
            while pi<thresholds[j]:
                tp+=int(w[positive[pi]])
                pi+=1
            while ni<neg_limits[j]:
                fp+=int(w[negative[ni]])
                ni+=1
            precision[j]=tp/(tp+fp) if tp+fp else 0.0
            recall_counts[j]=tp
        if not tp:
            result[replicate]=np.nan
        elif voc:
            score=0.
            for q in range(11):
                best=0.
                for j in range(len(precision)):
                    if 10*recall_counts[j]>=q*tp:
                        best=max(best,precision[j])
                score+=best
            result[replicate]=score/11.
        else:
            score=best=0.
            for j in range(len(precision)-1,-1,-1):
                best=max(best,precision[j])
                before=recall_counts[j-1] if j>0 else 0
                score+=(recall_counts[j]-before)*best
            result[replicate]=score/tp
    return result


def fast_ap(score,label,weights,voc):
    positive=np.flatnonzero(label==1)
    negative=np.flatnonzero(label==0)
    assert len(positive) and len(negative)
    positive=positive[np.argsort(-score[positive],kind="stable")]
    negative=negative[np.argsort(-score[negative],kind="stable")]
    thresholds=np.r_[np.flatnonzero(np.diff(score[positive])!=0)+1,len(positive)]
    neg_limits=np.searchsorted(-score[negative],-score[positive[thresholds-1]],side="right")
    return ap_kernel(weights,positive,negative,thresholds,neg_limits,voc)


def toy():
    rng=np.random.default_rng(924620)
    label=rng.integers(-1,2,size=42)
    score=rng.integers(0,12,size=42).astype(float)
    w=rng.integers(0,4,size=(11,len(label)),dtype=np.uint16)
    for voc in (False,True):
        oracle=Metric(score,label).compute_ap(w,rule="VOC11" if voc else "ENV")
        np.testing.assert_allclose(fast_ap(score,label,w,voc),oracle,atol=1e-12,rtol=0)


def sealed_archive(folder,stem=""):
    receipt=read(folder/(stem+"SCORE_RECEIPT.json"))
    protocol=read(folder/("T4_PROTOCOL.json" if stem else "RUN_PROTOCOL.json"))
    assert protocol["status"]=="LOCKED" and receipt["status"]=="SCORES_SEALED"
    assert receipt["truth_loaded"] is False and receipt["sealed_unix"]>datetime.fromisoformat(protocol["locked_at"]).timestamp()
    return folder/(stem+"SCORES.npz"),receipt["shape"]


class Analysis:
    def __init__(self,stage,limit):
        self.stage=stage
        self.voc=stage=="VOC"
        with np.load(STAGE_INPUTS[stage]) as data:
            self.ids,self.classes,self.truth=data["ids"],data["classes"],data["truth"]
        self.names=self.classes.tolist()
        self.weights,streams=bootstrap_for(stage,len(self.ids),limit)
        self.indices=streams[0]
        self.seeds=list(range(90901201,90901213))
        self.draws=[90921101,90921102] if self.voc else [90901101,90901102]
        self.subset=np.unique(np.linspace(0,limit,min(7,limit+1),dtype=int))
        self.pool={}
        self.spec=design_for(stage,self.names)
        self.max_auc_error=self.max_ap_error=0.
        self.map_cache={}

    def archive(self,path,shape=None,rename=None,keep=None):
        with np.load(path) as z:
            np.testing.assert_array_equal(self.ids,z["ids"])
            np.testing.assert_array_equal(self.classes,z["classes"])
            keys=list(zip(z["draws"].tolist(),z["seeds"].tolist(),z["arms"].tolist(),strict=True))
            vals=z["scores"]
        assert len(set(keys))==len(keys) and vals.shape==(len(keys),len(self.ids),20)
        if shape is not None:
            assert tuple(shape)==vals.shape
        assert vals.dtype==np.float32 and np.isfinite(vals).all() and ((vals>=0)&(vals<=1)).all()
        for (draw,seed,arm),value in zip(keys,vals,strict=True):
            if keep is not None and arm not in keep:
                continue
            a=rename.get(arm,arm) if rename else arm
            k=draw,seed,a
            assert k not in self.pool
            self.pool[k]=value

    def auc(self,arms,target,partner,pstar):
        group=group_codes(self.truth,target,partner,self.voc)
        count=np.array([self.weights[:,group==i].sum(axis=1,dtype=np.int32) for i in range(4)])
        assert (count>0).all()
        n00,n01,n10,n11=count
        p=n11/(n10+n11)
        out={}
        for arm in arms:
            a=np.empty((len(self.weights),2,12,5))
            for di,d in enumerate(self.draws):
                for si,s in enumerate(self.seeds):
                    score=self.pool[d,s,arm][:,target]
                    z=replay_auc(score,group,self.weights)
                    c0,c1,c2,c3,natural=z.T
                    background=(1-p)*c0+p*c2
                    bg_fixed=(1-pstar)*c0+pstar*c2
                    partner_fixed=(1-pstar)*c1+pstar*c3
                    a[:,di,si]=np.stack((c1,background,natural,.95*bg_fixed+.05*partner_fixed,
                                         .5*(bg_fixed+partner_fixed)),axis=1)
                    for y,pos,neg in ((c1,group==2,group==1),(background,group>=2,group==0),
                                      (natural,group>=2,(group>=0)&(group<2))):
                        labels=np.where(pos,1,np.where(neg,0,-1))
                        reference=Metric(score,labels).compute_auc(self.weights[self.subset])
                        np.testing.assert_allclose(y[self.subset],reference,atol=1e-12,rtol=0)
                        self.max_auc_error=max(self.max_auc_error,float(np.max(np.abs(y[self.subset]-reference))))
            out[arm]=a.mean(axis=1)
        return out

    def ap(self,arm):
        if arm in self.map_cache:
            return self.map_cache[arm]
        out=np.empty((len(self.weights),2,12))
        for di,d in enumerate(self.draws):
            for si,s in enumerate(self.seeds):
                score=self.pool[d,s,arm]
                values=[]
                for j,label in enumerate(self.truth.T):
                    label=np.where(label==0,-1,np.where(label==1,1,0)) if self.voc else label
                    got=fast_ap(score[:,j],label,self.weights,self.voc)
                    ref=Metric(score[:,j],label).compute_ap(self.weights[self.subset],rule="VOC11" if self.voc else "ENV")
                    np.testing.assert_allclose(got[self.subset],ref,atol=1e-12,rtol=0)
                    self.max_ap_error=max(self.max_ap_error,float(np.max(np.abs(got[self.subset]-ref))))
                    values.append(got)
                out[:,di,si]=np.mean(values,axis=0)
        self.map_cache[arm]=out.mean(axis=1)
        print("AP checked",arm,flush=True)
        return self.map_cache[arm]

    def compare(self,values):
        v=100*values
        sample=draw_effect(v,self.indices)
        finite=sample[np.isfinite(sample)]
        assert len(finite)==len(self.indices)
        lo,lower,upper,hi=np.quantile(finite,[.025,.05,.95,.975])
        return dict(point_pp=float(v[0].mean()),lower95_one_sided_pp=float(lower),upper95_one_sided_pp=float(upper),
                    ci95_pp=[float(lo),float(hi)],valid=len(finite),by_seed_pp=v[0].tolist())


METRICS=("conditional","background","natural","r05","r50")


def metrics_dict(a,values):
    return {m:a.compare(values[...,i]) for i,m in enumerate(METRICS)}


def summarize_pairs(a,pair_values):
    out=[]
    for target,partner,values in pair_values:
        out.append(dict(target=target,partner=partner,metrics=metrics_dict(a,values)))
    return dict(aggregate=metrics_dict(a,np.mean([p[2] for p in pair_values],axis=0)),per_pair=out)


def t1(a):
    design=read(HERE/"COCO_DESIGN.json")
    rows=[]
    effects=[]
    m0=[]
    m1=[]
    diagonals=[]
    for group in design["T1"]:
        A,B,C=group["arms"]
        for row in group["pairs"]:
            t=a.names.index(row["target"])
            p=a.names.index(row["anchor"])
            q=a.names.index(row["alternate"])
            v=a.auc((A,B,C),t,p,row["anchor_pstar"])
            w=a.auc((A,B,C),t,q,row["alternate_pstar"])
            di=v[A]-v[C]
            dj=w[B]-w[C]
            x=(v[A]-v[B])[...,0]
            y=(w[B]-w[A])[...,0]
            i=(x+y)/2
            effects.append(i)
            m0.append(x)
            m1.append(y)
            diagonals.append(di[...,0])
            rows.append({**row,"group":group["group"],"anchor_TRUE_minus_PLC":metrics_dict(a,di),
                         "alternate_TRUE_minus_PLC":metrics_dict(a,dj),
                         "anchor_match_advantage":a.compare(x),"alternate_match_advantage":a.compare(y),
                         "interaction":a.compare(i)})
    aggregate=a.compare(np.mean(effects,axis=0))
    assert len(rows)==7
    positive=sum(r["anchor_TRUE_minus_PLC"]["conditional"]["point_pp"]>0 for r in rows)
    passed=aggregate["point_pp"]>=.30 and aggregate["lower95_one_sided_pp"]>0 and positive>=5
    return dict(status="PASS" if passed else "GATE_NOT_MET",interaction=aggregate,
                anchor_column_advantage=a.compare(np.mean(m0,axis=0)),alternate_column_advantage=a.compare(np.mean(m1,axis=0)),
                anchor_TRUE_minus_PLC=a.compare(np.mean(diagonals,axis=0)),anchor_positive=positive,per_pair=rows)


def t2(a):
    baselines=("LOW","T2_RANDOM","T2_HARD","S0_PLC","T2_PBCE")
    arms=("S0_TRUE",*baselines)
    maps={arm:a.ap(arm) for arm in arms}
    by_pair=[(a.names[t],a.names[p],a.auc(arms,t,p,pstar)) for t,p,pstar in a.spec["S0"]]
    rows=[]
    for arm in baselines:
        result=summarize_pairs(a,[(t,p,vals["S0_TRUE"]-vals[arm]) for t,p,vals in by_pair])
        m=a.compare(maps["S0_TRUE"]-maps[arm])
        lo,hi=m["ci95_pp"]
        gate=result["aggregate"]["conditional"]["lower95_one_sided_pp"]>0 and lo>=-.5 and hi<=.5
        rows.append(dict(baseline=arm,comparison=result,map_difference=m,passed=bool(gate)))
    simple=all(r["passed"] for r in rows if r["baseline"] in ("LOW","T2_RANDOM","T2_HARD"))
    full=all(r["passed"] for r in rows)
    return dict(status="PASS" if full else "GATE_NOT_MET",simple_pass=simple,full_pass=full,
                comparisons=rows,map_by_arm={arm:a.compare(v) for arm,v in maps.items()})


def t3(a):
    arms=("S0_TRUE","S0_PLC","LOW")
    by_pair=[(a.names[t],a.names[p],a.auc(arms,t,p,pstar)) for t,p,pstar in a.spec["S0"]]
    maps={arm:a.ap(arm) for arm in arms}
    results={}
    for baseline in arms[1:]:
        value=summarize_pairs(a,[(t,p,v["S0_TRUE"]-v[baseline]) for t,p,v in by_pair])
        results[baseline]=dict(auc=value,map_difference=a.compare(maps["S0_TRUE"]-maps[baseline]))
    h1=results["S0_PLC"]["auc"]["aggregate"]["conditional"]
    return dict(status="PASS" if h1["point_pp"]>=1 and h1["lower95_one_sided_pp"]>0 else "GATE_NOT_MET",
                comparisons=results,map_by_arm={arm:a.compare(v) for arm,v in maps.items()})


def t4(a):
    arms=("P1_TRUE","P2_TRUE","PLC","LOW")
    maps={arm:a.ap(arm) for arm in arms}
    rows=[]
    for evaluation,pairs in a.spec.items():
        vals=[(a.names[t],a.names[p],a.auc(arms,t,p,pstar)) for t,p,pstar in pairs]
        for arm in arms[:-1]:
            rows.append(dict(train=arm,evaluation=evaluation,baseline="LOW",
                  **summarize_pairs(a,[(t,p,v[arm]-v["LOW"]) for t,p,v in vals])))
    diff={arm:a.compare(maps[arm]-maps["LOW"]) for arm in arms[:-1]}
    return dict(status="COMPLETE",auc=rows,map_vs_low=diff,map_by_arm={arm:a.compare(v) for arm,v in maps.items()})


def crossed(a,arms):
    rows=[]
    advantages=[]
    for ei,(eval_name,pairs) in enumerate(a.spec.items()):
        values=[(a.names[t],a.names[p],a.auc(arms,t,p,pstar)) for t,p,pstar in pairs]
        col=np.mean([(v[arms[ei]]-v[arms[1-ei]])[...,0] for _,_,v in values],axis=0)
        advantages.append(col)
        for arm in arms[:2]:
            val=[(t,p,v[arm]-v[arms[2]]) for t,p,v in values]
            rows.append(dict(train=arm,evaluation=eval_name,**summarize_pairs(a,val)))
    interaction=(advantages[0]+advantages[1])/2
    return dict(cross=rows,matched_advantages=[a.compare(x) for x in advantages],
                interaction=a.compare(interaction)),interaction


def t5(a):
    result,interaction=crossed(a,("S0_TRUE","N1_TRUE","S0_PLC"))
    b=np.load(AB/"B_analysis/B_ENDPOINT_BOOTSTRAP.npy",mmap_mode="r")[:len(a.weights)]
    # replicate, seed, draw, arm, evaluation, target, metric
    r50=((b[:,:,:,0,0,:,0]-b[:,:,:,1,0,:,0])+(b[:,:,:,1,1,:,0]-b[:,:,:,0,1,:,0])).mean(axis=(2,3))/2
    old=a.compare(r50)
    if len(a.indices)==N_BOOT:
        archived=read(AB/"B_RESULT.json")["matched_effect_pp"]
        np.testing.assert_allclose(old["point_pp"],archived,rtol=0,atol=1e-12)
    i=result["interaction"]
    status="PASS" if i["point_pp"]>=.30 and i["lower95_one_sided_pp"]>0 else "GATE_NOT_MET"
    return dict(status=status,**result,rn50_comparator=old,vit_minus_rn50_interaction=a.compare(interaction-r50))


def t7(a):
    results=[]
    values=[]
    for d,dose in enumerate(read(HERE/"COCO_DESIGN.json")["T7"],1):
        result,x=crossed(a,tuple(f"T7D{d}_{arm}" for arm in ("S0_TRUE","N1_TRUE","PLC")))
        results.append(dict(dose=dose,**result))
        values.append(x)
    adjacent=[a.compare(y-x) for x,y in zip(values[:-1],values[1:],strict=True)]
    return dict(status="COMPLETE_EXPLORATORY",dose_results=results,adjacent_I=adjacent,
                nondecreasing_points=all(x["point_pp"]>=0 for x in adjacent),
                both_positive_lower=all(x["lower95_one_sided_pp"]>0 for x in adjacent))


def main(job,limit):
    toy()
    start=perf_counter()
    a=Analysis(STAGE_FOR[job],limit)
    folder=HERE/(job if job in ("T3","T4","T5") else "COCO")
    if job=="T4":
        a.archive(*sealed_archive(folder,"T4_"))
        a.archive(VOC/"SCORES.npz")
    elif job=="T2":
        a.archive(*sealed_archive(folder),keep={"LOW","T2_RANDOM","T2_HARD","T2_PBCE"})
        a.archive(AB/"B_SCORES.npz",keep={"S0_TRUE","S0_PLC"})
    elif job in ("T1","T7"):
        design=read(HERE/"COCO_DESIGN.json")
        a.archive(*sealed_archive(folder),keep={x for x in design["arms"] if x.startswith(job)})
    else:
        a.archive(*sealed_archive(folder))
    output=globals()[job.lower()](a)
    print(job,output["status"],"analysis seconds",round(perf_counter()-start,1),flush=True)
    if limit==N_BOOT:
        out=dict(job=job,stage=a.stage,bootstrap=limit,original_weights_and_indices=True,
                 auc_reference_max_abs=a.max_auc_error,ap_reference_max_abs=a.max_ap_error,
                 elapsed_seconds=perf_counter()-start,**output)
        path=folder/(job+"_RESULT.json")
        tmp=path.with_suffix(".tmp")
        tmp.write_text(__import__("json").dumps(out,indent=2,allow_nan=False),encoding="utf8")
        tmp.replace(path)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("job",choices=list(STAGE_FOR)+["toy"])
    parser.add_argument("--replicates",type=int,choices=(8,N_BOOT),default=N_BOOT)
    parser.add_argument("--threads",type=int,default=6)
    args=parser.parse_args()
    set_num_threads(args.threads)
    if args.job=="toy":
        toy()
        print("weighted AP tie and exclusion checks PASS",flush=True)
    else:
        main(args.job,args.replicates)
