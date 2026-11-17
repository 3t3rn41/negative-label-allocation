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

"""Evaluate M1 and M2 on the same sealed, design-weighted confirmation sample."""
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.stats import t

from formal_index_lock import generator, load_index
from joint_replica import paired_replicas
from statistics_primitives import pairwise_design_auc, weighted_ap, weighted_auc


ARMS=("DIR1","DIR2","PLC")
COCO20=(1,2,3,4,5,6,7,9,16,17,18,19,20,21,44,62,63,64,67,72)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def seal(path,data):
    temp=path.with_suffix(".partial")
    temp.write_text(json.dumps(data,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    temp.replace(path)


def check_receipt(work,model,phase,lock,ids):
    receipt=read(work/f"{model}_{phase.upper()}_SCORE_RECEIPT.json")
    matrix=lock[model]
    assert receipt["status"]=="SCORES_SEALED" and receipt["truth_loaded"] is False
    assert receipt["ids"]==ids.tolist() and receipt["draws"]==matrix["draws"]
    assert receipt["endpoints"]==len(matrix["draws"])*len(matrix["seeds"])*len(ARMS)


def score_rows(work,model,phase,lock,ids):
    check_receipt(work,model,phase,lock,ids)
    matrix=lock[model]
    seeds=matrix["seeds"]
    categories=lock["categories"]
    for draw in matrix["draws"]:
        with np.load(work/f"{model}_{phase}_{draw}.npz",allow_pickle=False) as item:
            assert item["categories"].tolist()==categories and item["arms"].tolist()==list(ARMS)*len(seeds)
            np.testing.assert_array_equal(item["ids"],ids)
            np.testing.assert_array_equal(item["seeds"],np.repeat(np.asarray(seeds,dtype=np.uint64),len(ARMS)))
            inputs=item["scores"].reshape(len(seeds),len(ARMS),len(ids),len(categories))
        assert np.isfinite(inputs).all() and ((inputs>=0)&(inputs<=1)).all()
        yield draw,inputs


def pair_columns(lock):
    classes=lock["categories"]
    return [(classes.index(row["target_id"]),classes.index(partner)) for row in lock["targets"] for partner in row["partners"]]


def subsets(lock,pairs):
    classes=lock["categories"]
    return dict(mAP133=list(range(len(classes))),mAP80=list(range(80)),
                mAP20_bridge=[classes.index(i) for i in COCO20],mAP_K=sorted({a for a,_ in pairs}))


def ancillary_metrics(scores,truth,weights,strata,totals,pairs,groups):
    signed=[]
    metrics=[]
    for init in scores:
        armed=[]
        for arm in init:
            ap=np.array([weighted_ap(arm[:,j],truth[:,j]==1,truth[:,j]==0,weights) for j in range(arm.shape[1])])*100
            conditional=[]
            background=[]
            complete=[]
            sensitivity=[]
            for target,partner in pairs:
                positive=truth[:,target]==1
                negative=truth[:,target]==0
                present=truth[:,partner]==1
                absent=truth[:,partner]==0
                risk_pos=positive&absent
                risk_neg=negative&present
                value=arm[:,target]
                conditional.append(weighted_auc(value,risk_pos,risk_neg,weights)*100)
                background.append(weighted_auc(value,risk_pos,negative&absent,weights)*100)
                complete.append(weighted_auc(value,positive,negative,weights)*100)
                sensitivity.append(pairwise_design_auc(value,risk_pos,risk_neg,strata,totals)*100)
            arm_report={key:float(ap[columns].mean()) for key,columns in groups.items()}
            arm_report.update(conditional_auc_pp=conditional,background_auc_pp=background,
                              natural_auc_pp=complete,pairwise_design_auc_pp=sensitivity)
            armed.append(arm_report)
        a=np.asarray([row["conditional_auc_pp"] for row in armed])
        switch=np.arange(len(pairs))%2
        signed.append(a[switch,np.arange(len(pairs))]-a[1-switch,np.arange(len(pairs))])
        metrics.append(armed)
    return np.asarray(signed),metrics


def evaluate_model(work,model,lock,ids,truth,strata,weights,totals,reps):
    start=perf_counter()
    pairs=pair_columns(lock)
    by_class=subsets(lock,pairs)
    used=sorted({column for pair in pairs for column in pair})
    small_pairs=[(used.index(a),used.index(b)) for a,b in pairs]
    focused=[]
    oracle=[]
    rows=[]
    for draw,score in score_rows(work,model,"confirm",lock,ids):
        ref,description=ancillary_metrics(score,truth,weights,strata,totals,pairs,by_class)
        oracle.append(ref)
        focused.append(score[...,used])
        rows.append(dict(draw=draw,seeds=lock[model]["seeds"],arms=list(ARMS),metrics=description))
    replay=np.asarray(focused)
    image_index=load_index(work,model,lock,ids,strata,totals,reps)
    effects,interval,approx=paired_replicas(replay,truth[:,used],small_pairs,strata,totals,reps,
                                          generator(model),image_index=image_index)
    discrepancy=float(np.max(np.abs(100*effects-np.asarray(oracle))))
    assert discrepancy<=1e-4
    point=100*effects.mean()
    by_mask=effects.mean((1,2))*100
    t_bound=point-t.ppf(.95,len(rows)-1)*by_mask.std(ddof=1)/np.sqrt(len(rows))
    directional=effects.mean((0,1)).reshape(-1,2)*100
    targets=directional.mean(1)
    leave_one=((targets.sum()-targets)/(len(targets)-1)).tolist()
    out=dict(status="SCORES_EVALUATED",run_id=lock["run_id"],model=model,
             endpoints=len(rows)*replay.shape[1]*len(ARMS),bootstrap_replicates=reps,
             I_pp=float(point),I_by_target_pp=targets.tolist(),I_by_mask_pp=by_mask.tolist(),loo_pp=leave_one,
             directional_by_target_pp=directional.tolist(),directional_pp=directional.mean(0).tolist(),
             exact_bounds_pp=(100*interval).tolist(),approximate_bounds_pp=(100*approx).tolist(),mask_t_lower_pp=float(t_bound),
             interval_order=["two_sided_lower","one_sided_lower","two_sided_upper"],input_images=len(ids),
             independent_point_max_error_pp=discrepancy,wall_seconds=perf_counter()-start,per_draw=rows)
    return out,replay,truth[:,used],small_pairs


def commit_model(work,result,opened):
    model=result["model"]
    gates=dict(point=result["I_pp"]>=.30,bootstrap_lower=result["exact_bounds_pp"][1]>0,
               mask_t_lower=result["mask_t_lower_pp"]>0)
    if model=="M1":
        targets=np.asarray(result["I_by_target_pp"])
        gates.update(targets_positive=bool((targets>0).sum()>=np.ceil(2*len(targets)/3)),
                     directions_positive=all(v>0 for v in result["directional_pp"]),
                     leave_one_positive=all(v>0 for v in result["loo_pp"]))
    result.update(passed=all(gates.values()),gates=gates,confirmatory_test_open=opened)
    seal(work/f"{model}_RESULT.json",result)


def shared_difference(a,b,lock,truth,pairs,strata,totals,reps,image_index):
    draws=[d for d in lock["M1"]["draws"] if d in lock["M2"]["draws"]]
    seeds=[s for s in lock["M1"]["seeds"] if s in lock["M2"]["seeds"]]
    assert draws and seeds
    points=[]
    draws_on_same_units=[]
    for model,grid in (("M1",a),("M2",b)):
        m=lock[model]
        picked=grid[[m["draws"].index(d) for d in draws]][:,[m["seeds"].index(s) for s in seeds]]
        point,_,_,samples=paired_replicas(picked,truth,pairs,strata,totals,reps,generator("paired_comparison"),
                                         return_replicates=True,image_index=image_index)
        points.append(point.mean())
        draws_on_same_units.append(samples)
    observed=points[0]-points[1]
    bootstrap=draws_on_same_units[0]-draws_on_same_units[1]
    interval=observed-np.quantile(bootstrap-observed,[.975,.025])
    return dict(paired_I_difference_pp=float(observed*100),two_sided_95_pp=(100*interval).tolist(),
                matched_masks=draws,matched_initializations=seeds,replicates=reps)


def main(work,reps):
    from formal_m3 import commit_result, evaluate_components

    lock=read(work/"PROTOCOL_LOCK.json")
    assert lock["status"]=="READY" and lock["version"]=="v9.3.1-coco133"
    with np.load(work/"CONF_M_LABELS.npz",allow_pickle=False) as file:
        ids,truth,strata,totals,weights=(file[name] for name in ("ids","truth","strata","totals","weights"))
        assert file["categories"].tolist()==lock["categories"] and ids.tolist()==lock["confirm_ids"]
    assert ((truth==0)|(truth==1)).all()
    np.testing.assert_allclose(weights,(totals/np.bincount(strata))[strata],rtol=0,atol=0)
    for model in ("M1","M2"):
        check_receipt(work,model,"confirm",lock,ids)
    print("Started parallel M1, M2 and M3 confirmation estimates",reps,flush=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        first=pool.submit(evaluate_model,work,"M1",lock,ids,truth,strata,weights,totals,reps)
        second=pool.submit(evaluate_model,work,"M2",lock,ids,truth,strata,weights,totals,reps)
        third=pool.submit(evaluate_components,work,lock,reps)
        m1,a,small,pairs=first.result()
        print("M1 estimates ready",flush=True)
        m2,b,_,_=second.result()
        print("M2 estimates ready",flush=True)
        m3=third.result()
        print("M3 estimates ready",flush=True)
    index=load_index(work,"paired_comparison",lock,ids,strata,totals,reps)
    comparison=shared_difference(a,b,lock,small,pairs,strata,totals,reps,index)
    commit_model(work,m1,True)
    commit_model(work,m2,m1["passed"])
    commit_result(work,lock,m3)
    comparison.update(run_id=lock["run_id"],M1_passed=m1["passed"],M2_passed=m2["passed"],
                      M2_confirmatory_test_open=m1["passed"],parallel_estimations=["M1","M2","M3"],
                      confirmatory_decision_order=["M1","M2","M3"])
    seal(work/"M1_M2_COMPARISON.json",comparison)
    print(json.dumps(comparison),flush=True)


if __name__=="__main__":
    cli=ArgumentParser()
    cli.add_argument("work",type=Path)
    args=cli.parse_args()
    main(args.work,5000)
