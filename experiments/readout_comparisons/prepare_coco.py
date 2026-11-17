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

"""Prepare COCO fit-only constrained masks for T1, T2 and T7."""
import csv
import itertools
import json
from pathlib import Path
import sys
from datetime import datetime, timedelta, timezone

import numpy as np

HERE=Path(__file__).resolve().parent
V9=HERE.parent
ROOT=V9.parent
CPU=V9/"composition_analysis"
sys.path.insert(0,str(CPU))
from dose_feasibility import load_module
from replay import AB,G1,P0,VOC,read
from eligibility import counts, support

COCO=load_module("t1_coco_mask",P0/"build_masks.py")
SOLVER=load_module("t1_voc_solver",VOC/"solve_masks.py")
RULES=read(VOC/"RULES.json")
SEEDS=list(range(90901201,90901213))
DRAWS=COCO.RULES["draw_seeds"]


def save(file,obj):
    file.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding="utf8")


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def strata_parts(ids,truth,cosine,low,pairs,doses,draw,tag):
    rows=list(SOLVER.stratify(ids,truth,cosine,low,pairs))
    quotas=[SOLVER.largest_remainder(x,d) for x,d in zip(rows,doses,strict=True)]
    if any(q is None for q in quotas):
        return None,"CAPACITY",None
    for attempt,offset in enumerate(RULES["plc_rank_offsets"]):
        tie,rng=map(np.random.default_rng,np.random.SeedSequence([draw,tag,attempt]).spawn(2))
        masks={key:low.copy() for key in SOLVER.ARMS}
        groups={}
        for (t,a,b),layer,q,dose in zip(pairs,rows,quotas,doses,strict=True):
            plc=SOLVER.neutral_placebo(layer,q,cosine[:,t],ids,tie,offset)
            members,_=SOLVER.directed_coordinates(layer,q,dose,t,a,b,truth,cosine,plc,rng,RULES)
            if members is None:
                break
            groups[t]=members
            for _,_,(remove,*adds) in members:
                for m,add in zip(masks.values(),adds,strict=True):
                    m[remove,t]=-1
                    m[add,t]=0
        if len(groups)==len(pairs):
            fail,metrics,density=SOLVER.audit(truth,cosine,low,pairs,doses,masks,groups)
            if not fail:
                return masks,"PASS",dict(offset=offset,metrics=metrics,density=density)
    return None,"NO_PASSING_CONSTRUCTION",None


def t1(ids,y,classes,cosine,low):
    truth=(2*y-1).astype(np.int8)
    with (CPU/"T1_label_screen.csv").open(encoding="utf-8-sig",newline="") as stream:
        qualified=[r for r in csv.DictReader(stream) if r["passes"]=="True"]
    first=qualified[:7]
    assert len(first)==7 and all(r["top_seven"]=="True" for r in first)
    groups=[]
    for r in first:
        candidates=[g for g in groups if len(g)<3 and r["target"] not in {x["target"] for x in g}]
        if not candidates:
            groups.append([])
            candidates=[groups[-1]]
        candidates[0].append(r)
    anchors={(r["target"],r["partner"]) for r in first}
    codes={str(c):j for j,c in enumerate(classes)}
    with np.load(G1/"raw/data/COCO_CONFIRM_GOLD.npz") as d, np.load(AB/"B_GOLD.npz") as b:
        dev,evaluation=d["truth"],b["truth"]
    historical=[]
    for setting in ("S0","N1"):
        for x in COCO.PAIR_MAP[setting]:
            t,p=x["target"],x["partner"]
            row=support(counts(dev,t,p),counts(evaluation,t,p),(50,50,30,200))
            if row["passes"]:
                historical.append(dict(target=x["target_name"],partner=x["partner_name"],pstar=row["pstar"]))

    out={draw:{} for draw in DRAWS}
    design=[]
    for group_no,group in enumerate(groups,1):
        alternate_lists=[[r for r in (*qualified,*historical) if r["target"]==a["target"] and
                          (r["target"],r["partner"]) not in anchors] for a in group]
        assert all(alternate_lists)
        combinations=sorted(itertools.product(*(range(len(a)) for a in alternate_lists)),
                            key=lambda x:(sum(x),x))
        log=[]
        found=False
        for alternate_choice in combinations[:60]:
            alts=[lst[index] for lst,index in zip(alternate_lists,alternate_choice,strict=True)]
            pairs=[(codes[a["target"]],codes[a["partner"]],codes[b["partner"]]) for a,b in zip(group,alts,strict=True)]
            for dose in (30,20,15):
                doses=[dose]*len(pairs)
                tested=[strata_parts(ids,truth,cosine,low[draw],pairs,doses,draw,100+group_no) for draw in DRAWS]
                log.append(dict(alternates=[a["partner"] for a in alts],dose=dose,
                                status=[x[1] for x in tested]))
                if all(x[1]=="PASS" for x in tested):
                    found=True
                    break
            if found:
                break
        assert found,(group_no,log)
        arms=[f"T1G{group_no}_{suffix}" for suffix in ("ANCHOR","ALT","PLC")]
        for draw,(mask,status,qa) in zip(DRAWS,tested,strict=True):
            out[draw].update(dict(zip(arms,mask.values(),strict=True)))
        design.append(dict(group=group_no,arms=arms,doses=doses,
                          pairs=[dict(target=a["target"],anchor=a["partner"],alternate=b["partner"],
                                      anchor_pstar=float(a["pstar"]),alternate_pstar=float(b["pstar"]))
                                 for a,b in zip(group,alts,strict=True)],attempts=log,
                          qa=[x[2] for x in tested]))
        print("T1",group_no,design[-1]["pairs"],doses,"attempts",len(log),flush=True)
    return out,design


def density(a,b):
    a,b=[(x>=0).sum(axis=1) for x in (a,b)]
    return int(abs(np.count_nonzero(a)-np.count_nonzero(b))),float(
        np.abs(np.bincount(a,minlength=21)-np.bincount(b,minlength=21)).sum()/(2*len(a)))


def t2(ids,y,classes,cosine,low):
    coords={}
    with (P0/"MASK_COORDINATES.csv").open(encoding="utf-8-sig",newline="") as stream:
        for row in csv.DictReader(stream):
            if row["setting"]=="S0" and row["role"]=="B":
                key=tuple(int(row[x]) for x in ("draw","target","difficulty","density"))
                coords.setdefault(key,[]).append(int(row["coco_id"]))
    positions={i:j for j,i in enumerate(ids.tolist())}
    targets=[p["target"] for p in COCO.PAIR_MAP["S0"]]
    all_masks={}
    qa=[]
    for draw in DRAWS:
        with np.load(P0/f"MASK_{draw}.npz") as old:
            np.testing.assert_array_equal(old["ids"],ids)
            true,plc=old["S0_TRUE"],old["S0_PLC"]
        masks={}
        for name in ("T2_RANDOM","T2_HARD"):
            for attempt in range(20):
                matrix=low[draw].copy()
                rng=np.random.default_rng(np.random.SeedSequence([draw,200,attempt]))
                checks=[]
                for t in targets:
                    subset=[x for x in COCO.strata(ids,y,cosine,low[draw]) if x["target"]==t]
                    sizes=[len(coords.get((draw,t,r["difficulty"],r["density"]),[])) for r in subset]
                    removed=np.array([positions[i] for r in subset for i in coords.get(
                        (draw,t,r["difficulty"],r["density"]),[])])
                    assert set(removed)==set(np.flatnonzero((low[draw][:,t]==0)&(true[:,t]==-1)))
                    cells=[r["hidden"] for r,n in zip(subset,sizes,strict=True) if n]
                    n=[n for n in sizes if n]
                    included=np.concatenate(cells)
                    std=cosine[y[:,t]==0,t].std()
                    z=(cosine[:,t].astype(float)-cosine[removed,t].mean(dtype=float))/std
                    bound=.10*len(removed)-1e-5
                    problem=SOLVER.Constraints(len(included))
                    start=0
                    for size,ix in zip(n,cells,strict=True):
                        problem.add(np.arange(start,start+len(ix)),np.ones(len(ix)),size,size)
                        start+=len(ix)
                    problem.add(np.arange(len(included)),z[included],-bound,bound)
                    cost=rng.uniform(0,1,len(included)) if name=="T2_RANDOM" else -z[included]+1e-5*rng.uniform(0,1,len(included))
                    ans=problem.solve(cost)
                    assert ans.x is not None and np.max(np.abs(ans.x-np.rint(ans.x)))<1e-5,(name,draw,t,ans.message)
                    added=included[np.rint(ans.x).astype(bool)]
                    assert len(added)==len(removed)
                    matrix[removed,t]=-1
                    matrix[added,t]=0
                    smd=float(z[added].mean())
                    assert abs(smd)<=.10
                    checks.append(dict(target=str(classes[t]),dose=len(added),smd=smd,
                                      optimal=bool(ans.success),solver_message=str(ans.message)))
                d=[density(matrix,x) for x in (low[draw],true,plc,*masks.values())]
                if all(a<=20 and b<=.01 for a,b in d):
                    break
            assert all(a<=20 and b<=.01 for a,b in d)
            observed=matrix>=0
            np.testing.assert_array_equal(matrix[observed],y[observed])
            np.testing.assert_array_equal(matrix==1,low[draw]==1)
            assert int(observed.sum())==int((low[draw]>=0).sum())
            masks[name]=matrix
            qa.append(dict(draw=draw,arm=name,attempt=attempt+1,density=d,difficulty=checks))
        all_masks[draw]=masks
    return all_masks,qa


def main():
    ids,y,classes,cosine=COCO.load_training()
    low={}
    for draw in DRAWS:
        with np.load(P0/f"MASK_{draw}.npz") as z:
            np.testing.assert_array_equal(z['ids'],ids)
            low[draw]=z['LOW']
    lock=HERE/"COCO_MASK_RULES.json"
    if not lock.is_file():
        save(lock,dict(status="LOCKED",locked_at=now(),
             T1="Seven registered anchors from CPU label-only screen; first fit into groups of up to three distinct targets. Alternates use the same development ranking among eligible non-anchor partners. Enumerate increasing sum of alternate ranks and test doses 30, 20, 15 for two-mask fit-only feasibility. Require original 0.10 SMD, 0.01 TV and 0.20 direction gap; aggregate I >= .30 pp and one-sided 95% lower > 0. Prevalence gate: at least 5/7 anchor TRUE-PLC conditional AUC effects positive. B reused for frozen evaluation.",
             T2="Use S0 original removals and strata. Same-budget random feasible swaps via uniform random linear priorities, hardest feasible swaps by maximum cosine under the same quotas and absolute SMD <= .10. Global density TV <= .01, including LOW, TRUE and PLC references. Require conditional AUC improvement over every baseline with one-sided 95% lower >0 and paired two-sided 95% mAP interval fully within +/- .5 pp.",
             T7="Use existing qualified three-dose crossed mask set; paired adjacent matched I contrasts and point monotonicity are exploratory."))
    amendment=HERE/"T1_ALTERNATE_AMENDMENT.json"
    if not amendment.is_file():
        save(amendment,dict(status="LOCKED_BEFORE_MASK_SELECTION_AND_SCORING",locked_at=now(),
             trigger="No eligible non-anchor new alternate for chair because all four eligible new chair pairs are anchors.",
             rule="After eligible non-anchor novel alternates, append eligible historical S0 then N1 partners using the same development and B four-cell and ESS gate. Alternate only, seven anchors unchanged."))
    a,design=t1(ids,y,classes,cosine,low)
    b,qa=t2(ids,y,classes,cosine,low)
    for draw in DRAWS:
        new={**a[draw],**b[draw]}
        for d in (1,2,3):
            with np.load(CPU/f"T7_dose{d}_mask{draw}.npz") as z:
                np.testing.assert_array_equal(ids,z["ids"])
                np.testing.assert_array_equal(low[draw],z["LOW"])
                for arm in ("S0_TRUE","N1_TRUE","PLC"):
                    new[f"T7D{d}_{arm}"]=z[arm]
        np.savez_compressed(HERE/f"COCO_MASK_{draw}.npz",ids=ids,classes=classes,LOW=low[draw],**new)
    with np.load(HERE/f"COCO_MASK_{DRAWS[0]}.npz") as z:
        arms=[a for a in z.files if a.startswith('T')]
    save(HERE/"COCO_DESIGN.json",dict(status="MASKS_QUALIFIED",locked_at=now(),draws=DRAWS,seeds=SEEDS,
         arms=arms,T1=design,T2=qa,T7=["20/10/30","40/20/60","60/30/90"]))
    print("mask design ready",len(arms),flush=True)


if __name__=="__main__":
    main()

