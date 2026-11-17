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

"""Compare complete and first-order joint intervals on scored A1 development images."""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from itertools import product
import json
import os
from pathlib import Path
from time import perf_counter

os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
import numpy as np
from scipy.stats import t

from joint_replica import paired_replicas


HERE=Path(__file__).resolve().parent
SEED=93120261115
DEVELOPMENT_DATA=None


def load():
    global DEVELOPMENT_DATA
    mask=json.loads((HERE/"PANOPTIC_MASK_SCREEN.json").read_text(encoding="utf-8"))
    summary=json.loads((HERE/"LINEAR_PRECISION_50E_6R.json").read_text(encoding="utf-8"))
    dev_gold=Path(os.environ.get("FQ_COCO_DEV_LABELS",HERE.parent/"coco_reference_labels/DEV80.npz"))
    with (np.load(dev_gold,allow_pickle=False) as old,
          np.load(HERE/"PANOPTIC_OFFICIAL_LABELS.npz",allow_pickle=False) as new):
        np.testing.assert_array_equal(old["ids"],new["dev_ids"])
        lookup={i:j for j,i in enumerate(old["ids"].tolist())}
        names=np.r_[old["category_ids"],new["category_ids"]].tolist()
        truth=np.c_[old["truth"],new["dev"]]
    selected=mask["chosen"]
    target=[q["target_id"] for q in selected]
    categories=target+sorted({p for q in selected for p in q["partners"]})
    positions=[names.index(x) for x in categories]
    pairs=[(categories.index(q["target_id"]),categories.index(p)) for q in selected for p in q["partners"]]
    ids=None
    heads=[]
    for key in mask["dev_seeds"]:
        with np.load(HERE/f"LINEAR_DEV_{key}_50E_6R.npz",allow_pickle=False) as f:
            previous=ids
            ids=f["ids"]
            if previous is not None:
                np.testing.assert_array_equal(previous,ids)
            assert f["categories"].tolist()==names and f["arms"].tolist()==["DIR1","DIR2","PLC"]*6
            heads.append(f["scores"][:,:,positions].reshape(6,3,len(ids),len(positions)))
    assert len(ids)==5000
    y=truth[np.ix_([lookup[x] for x in ids],positions)]
    stratum=np.logical_or.reduce([(y[:,a]==1)&(y[:,b]==1) for a,b in pairs]).astype(np.int64)
    groups=[np.flatnonzero(stratum==h) for h in (0,1)]
    DEVELOPMENT_DATA=np.asarray(heads),y,stratum,groups,pairs,summary["mean_development_I_pp"]


def one(task):
    index,fractions,reps,model=task
    heads,y,stratum,groups,pairs,full=DEVELOPMENT_DATA
    rng=np.random.default_rng(np.random.SeedSequence([SEED,30,731,index]))
    chosen=np.concatenate([rng.choice(g,round(len(g)*f),replace=False) for g,f in zip(groups,fractions,strict=True)])
    totals=np.array([len(g) for g in groups])
    estimate,exact,approx=paired_replicas(heads[:,:,:,chosen],y[chosen],pairs,stratum[chosen],totals,reps,rng)
    point=estimate.mean()*100
    by_mask=estimate.mean((1,2))*100
    lower_t=point-t.ppf(.95,len(by_mask)-1)*by_mask.std(ddof=1)/np.sqrt(len(by_mask))
    offsets=np.array([0.,.15,.30,.60])-full
    available=(point+offsets>=.30)&(lower_t+offsets>0)
    if model=="dualcoop":
        shifts=estimate.mean((0,1)).reshape(1,-1,2)*100+offsets[:,None,None]
        targets=shifts.mean(2)
        available&=((targets>0).sum(1)>=int(np.ceil(2*targets.shape[1]/3)))
        available&=(shifts.mean(1)>0).all(1)
        available&=(targets.sum(1,keepdims=True)-targets>0).all(1)
    exact_pass=available&(exact[1]*100+offsets>0)
    approx_pass=available&(approx[1]*100+offsets>0)
    return dict(index=index,fractions=fractions,quota=np.bincount(stratum[chosen]).tolist(),point_pp=point,
                interval_full_pp=(exact*100).tolist(),interval_first_order_pp=(approx*100).tolist(),
                error_pp=(abs(exact-approx)*100).tolist(),population_I_pp=full,gates_full=exact_pass.tolist(),
                gates_first_order=approx_pass.tolist())


def load_prompt():
    from p2_projection import data
    global DEVELOPMENT_DATA
    heads,y,pairs,stratum=data("dualcoop")
    qa=json.loads((HERE/"PROMPT_PILOT_INDEPENDENT_QA.json").read_text(encoding="utf-8"))
    assert qa["status"]=="PASS" and qa["endpoints"]==96 and qa["confirmation_scores_read"]==0
    groups=[np.flatnonzero(stratum==h) for h in (0,1)]
    DEVELOPMENT_DATA=heads,y,stratum,groups,pairs,qa["mean_I_pp"]


if __name__=="__main__":
    parser=ArgumentParser()
    parser.add_argument("--model",choices=("linear","dualcoop"),default="linear")
    parser.add_argument("--rounds",type=int,default=25)
    parser.add_argument("--replicates",type=int,default=5000)
    parser.add_argument("--workers",type=int,default=4)
    args=parser.parse_args()
    designs=[(.20,.45),(.35,.70),(.55,.75),(.75,1.)]
    tasks=[(index,design,args.replicates,args.model) for index,(_,design) in enumerate(product(range(args.rounds),designs))]
    started=perf_counter()
    rows=[]
    with ProcessPoolExecutor(max_workers=args.workers,initializer=load if args.model=="linear" else load_prompt) as pool:
        for row in pool.map(one,tasks,chunksize=1):
            rows.append(row)
            if len(rows)%10==0 or len(rows)==len(tasks):
                print(len(rows),"of",len(tasks),"seconds",round(perf_counter()-started,1),flush=True)
    dist=np.array([r["error_pp"] for r in rows])
    complete=np.array([r["gates_full"] for r in rows])
    approximate=np.array([r["gates_first_order"] for r in rows])
    receipt=dict(status="DEVELOPMENT_REPLAY_ONLY",model=args.model,scope="Fixed A1 validation population with development masks and starts",
        outer_samples=len(rows),bootstrap_replicates=args.replicates,interval_names=["two_sided_lower","one_sided_lower","two_sided_upper"],
        shifted_effects_pp=[0.,.15,.30,.60],max_abs_error_pp=float(dist.max()),endpoint_q95_pp=float(np.quantile(dist,.95)),
        gate_agreement=float((complete==approximate).mean()),
        exact_rejected_but_first_order_passed=int((~complete&approximate).sum()),
        exact_passed_but_first_order_rejected=int((complete&~approximate).sum()),
        confirmation_scores_read=0,wall_seconds=perf_counter()-started,rows=rows)
    suffix="DEVELOPMENT_QA" if args.rounds==25 and args.replicates==5000 else "REPLAY_TRIAL"
    name=f"{args.model.upper()}_JOINT_REPLICA_{suffix}.json"
    temp=(HERE/name).with_suffix(".partial")
    temp.write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf-8")
    temp.replace(HERE/name)
    print({k:v for k,v in receipt.items() if k!="rows"},flush=True)
