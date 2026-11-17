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

"""Seal source composition predictions and evaluate them after confirmation scoring."""
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from formal_index_lock import RECEIPT, SEEDS, generator, load_index
from formal_metrics import pair_columns, read, score_rows, seal
from joint_replica import Ranking, image_sample_weights
from statistics_primitives import weighted_auc


SCENARIOS=("population_natural","p_source_r_005","p_source_r_050")


def cell_masks(truth,target,partner):
    y=truth[:,target]
    p=truth[:,partner]
    return [(y==a)&(p==b) for a,b in ((1,0),(1,1),(0,0),(0,1))]


def proportions(cells,weight):
    counts=np.stack([weight[:,cell].sum(1) for cell in cells],axis=-1)
    assert (counts>0).all()
    return counts[:,1]/(counts[:,0]+counts[:,1]),counts[:,3]/(counts[:,2]+counts[:,3])


def mix(p,r):
    return np.stack(((1-p)*(1-r),(1-p)*r,p*(1-r),p*r),axis=-1)


def ranks_for(scores,cells):
    return [Ranking(scores,a,b) for a in cells[:2] for b in cells[2:]]


def ranked_curves(matched,control,cells,weights):
    a=ranks_for(matched,cells)
    b=ranks_for(control,cells)
    return np.column_stack([u.calculate(weights)-v.calculate(weights) for u,v in zip(a,b,strict=True)])


def scenario_weights(cells,base,mixture):
    joint=mixture.reshape(-1,2,2)
    desired=np.column_stack((joint.sum(2),joint.sum(1)))
    weighted=np.zeros((len(mixture),len(base)))
    effective=[]
    for col,cell in enumerate(cells):
        original=base[cell]
        weighted[:,cell]=desired[:,col,None]*(original/original.sum())
        effective.append(float(original.sum()**2/(original@original)))
    positive=cells[0]|cells[1]
    negative=cells[2]|cells[3]
    group_ess=np.column_stack([1/np.square(weighted[:,group]).sum(1) for group in (positive,negative)])
    return weighted,positive,negative,dict(positive_negative=group_ess.tolist(),four_cells=effective)


def point_discrepancy(matched,control,positive,negative,scenario_weight,rebuilt):
    direct=np.array([weighted_auc(matched,positive,negative,w)-weighted_auc(control,positive,negative,w)
                     for w in scenario_weight])
    assert np.isfinite(direct).all()
    return float(np.max(np.abs(direct-rebuilt)))


def source_curves(work,lock):
    with np.load(work/"PANOPTIC_BRIDGE_LABELS.npz",allow_pickle=False) as file:
        ids=file["ids"]
        assert ids.tolist()==lock["bridge_ids"] and file["categories"].tolist()==lock["categories"]
        truth={mode:file[mode] for mode in ("truth","original_coco")}
    pairs=pair_columns(lock)
    weights=np.ones((1,len(ids)))
    cells={name:[cell_masks(y,t,p) for t,p in pairs] for name,y in truth.items()}
    totals={name:np.array([np.r_[proportions(v,weights)] for v in pools]) for name,pools in cells.items()}
    natural={mode:mix(p[:,0],p[:,1]) for mode,p in totals.items()}
    output={name:[] for name in truth}
    discrepancy=0.
    for _,scores in score_rows(work,"M1","bridge",lock,ids):
        for mode in truth:
            matrix=np.empty((len(scores),len(pairs),4))
            for i,fit in enumerate(scores):
                for j,(target,_) in enumerate(pairs):
                    matched,control=fit[j%2,:,target],fit[2,:,target]
                    groups=cells[mode][j]
                    matrix[i,j]=ranked_curves(matched,control,groups,weights)[0]
                    discrepancy=max(discrepancy,point_discrepancy(matched,control,groups[0]|groups[1],
                        groups[2]|groups[3],weights,natural[mode][j]@matrix[i,j]))
            output[mode].append(matrix)
    assert discrepancy<=1e-6
    return {mode:np.asarray(values) for mode,values in output.items()},totals,discrepancy


def frozen_masses(work,lock,source):
    sample_path=work.parent/"PANOPTIC_POPULATION133.npz"
    with np.load(sample_path,allow_pickle=False) as file:
        assert file["categories"].tolist()==lock["categories"]
        y=file["truth"]
    full=np.ones((1,len(y)))
    pairs=pair_columns(lock)
    loc=np.array([np.r_[proportions(cell_masks(y,t,p),full)] for t,p in pairs])
    return np.stack([mix(loc[:,0],loc[:,1]),mix(source[:,0],np.full(len(pairs),.05)),
                     mix(source[:,0],np.full(len(pairs),.50))],axis=1)


def leave_target_curves(curves):
    targets=curves.shape[2]//2
    targets_only=curves.reshape(*curves.shape[:2],targets,2,4).mean(3)
    without=(targets_only.sum(2,keepdims=True)-targets_only)/(targets-1)
    return np.repeat(without,2,axis=2)


def training_replicas(m,r,reps,rng):
    mask=rng.multinomial(m,np.full(m,1./m),size=reps)
    init=rng.multinomial(r,np.full(r,1./r),size=reps)
    return (mask[:,:,None]*init[:,None,:])/(m*r)


def predicted(curves,source,masses):
    main=np.einsum("mrpc,psc->mrps",curves,masses)
    original_natural=np.einsum("mrpc,pc->mrp",curves,mix(source[:,0],source[:,1]))
    b1=np.repeat(original_natural[...,None],len(SCENARIOS),axis=-1)
    b2=np.einsum("mrpc,psc->mrps",leave_target_curves(curves),masses)
    return dict(four_cells=main,B0=np.zeros_like(main),B1=b1,B2=b2)


def source_prediction(work,lock):
    index=read(work/RECEIPT)
    assert index["run_id"]==lock["run_id"] and index["confirmation_scores_read"]==0
    assert index["status"]=="FROZEN_BEFORE_CONFIRMATION_SCORING"
    assert index["seed_sequences"]["M3_source"]==list(SEEDS["M3_source"])
    destination=work/"M3_PREDICTIONS_BEFORE_SCORING.json"
    sensitivity=work/"M3_ORIGINAL_COCO_PREDICTIONS_BEFORE_SCORING.json"
    if destination.exists():
        previous=read(destination)
        assert previous["run_id"]==lock["run_id"] and previous["status"]=="SEALED_SOURCE_PREDICTIONS"
        assert sensitivity.exists() and read(sensitivity)["run_id"]==lock["run_id"]
        return previous
    assert not any(work.glob("M1_confirm_*.npz")) and not any(work.glob("M2_confirm_*.npz"))
    curves,source,discrepancy=source_curves(work,lock)
    masses=frozen_masses(work,lock,source["truth"])
    main=predicted(curves["truth"],source["truth"],masses)
    old=predicted(curves["original_coco"],source["original_coco"],masses)
    observed=main["four_cells"].mean((0,1))
    rng=generator("M3_source")
    weight=training_replicas(*curves["truth"].shape[:2],5000,rng)
    reps=np.einsum("bmr,mrps->bps",weight,main["four_cells"])
    q=np.quantile(reps-observed,[.975,.025],axis=0)
    lower,upper=observed-q
    allowed=(np.abs(observed)>.001)&((lower>0)|(upper<0))
    data=dict(status="SEALED_SOURCE_PREDICTIONS",run_id=lock["run_id"],scenario_order=list(SCENARIOS),
              fixed_source_images=len(lock["bridge_ids"]),confirmation_scores_read=0,
              scope="COCONut bridge source, fixed validation images and paired training repetitions",
              target_mixture_weights=masses.tolist(),source_pr=source["truth"].tolist(),
              source_recomposition_max_abs_error=discrepancy,
              directional_eligible=allowed.tolist(),source_interval_95_pp=(100*np.stack((lower,upper),-1)).tolist(),
              prediction_pp={k:(v.mean((0,1))*100).tolist() for k,v in main.items()},
              original_coco_sensitivity_pp={k:(v.mean((0,1))*100).tolist() for k,v in old.items()})
    seal(sensitivity,dict(status="SEALED_SOURCE_PREDICTIONS",run_id=lock["run_id"],source_labels="original COCO",
         target_labels="COCONut",confirmation_scores_read=0,scenario_order=list(SCENARIOS),
         fixed_source_images=len(lock["bridge_ids"]),source_pr=source["original_coco"].tolist(),
         target_mixture_weights=masses.tolist(),prediction_pp=data["original_coco_sensitivity_pp"]))
    seal(destination,data)
    return data


def confirmation_replicas(work,lock,ids,y,pairs,strata,totals,source_p,population_natural,reps,rng,image_index):
    base,weights=image_sample_weights(strata,totals,reps,rng,image_index)
    m,r=len(lock["M1"]["draws"]),len(lock["M1"]["seeds"])
    training=training_replicas(m,r,reps,rng)
    points=np.zeros((len(pairs),len(SCENARIOS)))
    samples=np.zeros((reps,len(pairs),len(SCENARIOS)))
    cells=[cell_masks(y,t,p) for t,p in pairs]
    allocation=[]
    structure=[]
    readback=[]
    audit=[]
    discrepancy=0.
    for j,pool in enumerate(cells):
        p0,r0=proportions(pool,base[None])
        pb,rb=proportions(pool,weights)
        fixed=mix(np.full(2,source_p[j]),np.array([.05,.50]))
        structure.append(np.vstack((mix(p0,r0),fixed)))
        allocation.append(np.concatenate((mix(pb,rb)[:,None],np.broadcast_to(fixed,(reps,2,4))),axis=1))
        scenario_weight,positive,negative,ess=scenario_weights(pool,base,structure[-1])
        readback.append((scenario_weight,positive,negative))
        natural=structure[-1][0].reshape(2,2)
        population=population_natural[j].reshape(2,2)
        pr=np.array([natural[1].sum(),natural[:,1].sum()])
        census=np.array([population[1].sum(),population[:,1].sum()])
        audit.append(dict(ess=ess,sample_natural_pr=pr.tolist(),population_natural_pr=census.tolist(),
                          sample_minus_population_pr=(pr-census).tolist()))
    for draw_index,(_,draw) in enumerate(score_rows(work,"M1","confirm",lock,ids)):
        for init_index,fit in enumerate(draw):
            active=np.flatnonzero(training[:,draw_index,init_index])
            replication_weights=weights[active]
            for j,(target,_) in enumerate(pairs):
                matched,control=fit[j%2,:,target],fit[2,:,target]
                a=ranks_for(matched,cells[j])
                b=ranks_for(control,cells[j])
                value=np.array([(u.calculate(base[None])-v.calculate(base[None]))[0] for u,v in zip(a,b,strict=True)])
                rebuilt=structure[j]@value
                scenario_weight,positive,negative=readback[j]
                discrepancy=max(discrepancy,point_discrepancy(matched,control,positive,negative,scenario_weight,rebuilt))
                points[j]+=rebuilt/(m*r)
                for first in range(0,len(active),128):
                    subset=active[first:first+128]
                    w=replication_weights[first:first+128]
                    estimate=np.column_stack([u.calculate(w)-v.calculate(w) for u,v in zip(a,b,strict=True)])
                    reweighted=np.einsum("bsc,bc->bs",allocation[j][subset],estimate)
                    samples[subset,j]+=training[subset,draw_index,init_index,None]*reweighted
    assert discrepancy<=1e-6 and np.isfinite(samples).all()
    return points,samples,training,dict(pair_support=audit,independent_recomposition_max_abs_error=discrepancy)


def evaluate_components(work,lock,reps):
    sealed=read(work/"M3_PREDICTIONS_BEFORE_SCORING.json")
    assert sealed["run_id"]==lock["run_id"] and sealed["status"]=="SEALED_SOURCE_PREDICTIONS"
    with np.load(work/"CONF_M_LABELS.npz",allow_pickle=False) as file:
        ids,y,strata,totals=(file[key] for key in ("ids","truth","strata","totals"))
        assert ids.tolist()==lock["confirm_ids"] and file["categories"].tolist()==lock["categories"]
    curves,source,_=source_curves(work,lock)
    masses=np.asarray(sealed["target_mixture_weights"])
    predicts=predicted(curves["truth"],source["truth"],masses)
    rng=generator("M3")
    image_index=load_index(work,"M3",lock,ids,strata,totals,reps)
    actual,actual_replicas,training,audit=confirmation_replicas(work,lock,ids,y,pair_columns(lock),strata,totals,
                                                              source["truth"][:,0],masses[:,0],reps,rng,image_index)
    errors={}
    bootstrap={}
    for name,grid in predicts.items():
        estimate=grid.mean((0,1))
        np.testing.assert_allclose(estimate*100,sealed["prediction_pp"][name],atol=1e-10,rtol=0)
        errors[name]=float(np.abs(estimate-actual).mean()*100)
        sample=np.einsum("bmr,mrps->bps",training,grid)
        bootstrap[name]=np.mean(np.abs(sample-actual_replicas),axis=(1,2))*100
    gains={}
    for name in ("B0","B1","B2"):
        difference=errors[name]-errors["four_cells"]
        samples=bootstrap[name]-bootstrap["four_cells"]
        gains[name]=dict(point_pp=difference,lower95_pp=float(difference-np.quantile(samples-difference,.95)),
                         two_sided_95_pp=(difference-np.quantile(samples-difference,[.975,.025])).tolist())
    mae_intervals={name:(point-np.quantile(bootstrap[name]-point,[.975,.025])).tolist()
                   for name,point in errors.items()}
    usable=np.asarray(sealed["directional_eligible"],dtype=bool)
    predicted_sign=np.sign(np.asarray(sealed["prediction_pp"]["four_cells"]))
    hit=float((predicted_sign[usable]==np.sign(actual[usable])).mean()) if usable.any() else 0.
    coverage=float(usable.mean())
    old=np.asarray(sealed["original_coco_sensitivity_pp"]["four_cells"])
    return dict(status="SCORES_EVALUATED",run_id=lock["run_id"],replicates=reps,
                mae_pp=errors,mae_reduction_by_baseline=gains,coverage=coverage,directional_hit_rate=hit,
                mae_two_sided_95_pp=mae_intervals,design_weighted_audit=audit,
                source_resampling_scope="conditional on the frozen DEV-BRIDGE images",
                exceeds_all_baseline_lower_bounds=all(row["lower95_pp"]>0 for row in gains.values()),
                observed_pp=(actual*100).tolist(),original_coco_sensitivity_mae_pp=float(np.abs(old-actual*100).mean()))


def commit_result(work,lock,result):
    first=read(work/"M1_RESULT.json")
    second=read(work/"M2_RESULT.json")
    assert first["run_id"]==second["run_id"]==result["run_id"]==lock["run_id"]
    passes=dict(mae=result["mae_pp"]["four_cells"]<=.25,
                beats_b1_lower=result["mae_reduction_by_baseline"]["B1"]["lower95_pp"]>0,
                direction_coverage=result["coverage"]>=.50,direction_hit_rate=result["directional_hit_rate"]>=.80)
    result.update(passed=all(passes.values()),gates=passes,confirmatory_test_open=first["passed"] and second["passed"])
    seal(work/"M3_RESULT.json",result)


def evaluate(work,lock,reps):
    path=work/"M3_RESULT.json"
    if path.exists():
        saved=read(path)
        assert saved["run_id"]==lock["run_id"] and saved["replicates"]==reps
        return
    commit_result(work,lock,evaluate_components(work,lock,reps))


if __name__=="__main__":
    parser=ArgumentParser()
    parser.add_argument("work",type=Path)
    parser.add_argument("phase",choices=("predict","evaluate"))
    args=parser.parse_args()
    spec=read(args.work/"PROTOCOL_LOCK.json")
    assert spec["status"]=="READY" and spec["version"]=="v9.3.1-coco133"
    if args.phase=="predict":
        source_prediction(args.work,spec)
    else:
        evaluate(args.work,spec,5000)
