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

"""Screen statistical designs, then validate the selected matrix before a formal draw."""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from itertools import product
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.stats import beta, norm, t

from p2_projection import compute_components, data, passed_rows
from p2_stratified_panel import atom_histogram, image_inner, image_outer


HERE=Path(__file__).resolve().parent
SEED=93120261115
EFFECTS=("null","homogeneous","heterogeneous")
MODEL_ID={"linear":1,"dualcoop":2}
_INNER_MODEL=None


def load(name):
    return json.loads((HERE/name).read_text(encoding="utf-8"))


def save(name,value):
    target=HERE/name
    stage=target.with_suffix(".partial")
    stage.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    stage.replace(target)


def seed(*tokens):
    return np.random.default_rng(np.random.SeedSequence([SEED,30,*tokens]))


def input_model(name,bins):
    heads,y,pairs,h=data(name)
    panel=load("PANOPTIC_MASK_SCREEN.json")["chosen"]
    target_ids=[item["target_id"] for item in panel for _ in item["partners"]]
    partner_ids=[v for item in panel for v in item["partners"]]
    codes={}
    for (a,b),target,partner in zip(pairs,target_ids,partner_ids,strict=True):
        assert codes.setdefault(a,target)==target and codes.setdefault(b,partner)==partner
    with np.load(HERE/"PANOPTIC_POPULATION133.npz",allow_pickle=False) as archive:
        all_names=archive["categories"].tolist()
        truth=archive["truth"][:,[all_names.index(codes[i]) for i in range(y.shape[1])]]
    strata=np.logical_or.reduce([(truth[:,a]==1)&(truth[:,b]==1) for a,b in pairs]).astype(np.int64)
    population=atom_histogram(y,heads.mean((0,1)),h,truth,strata,pairs,bins,seed(1201,MODEL_ID[name],bins))
    values,cov,_,_=compute_components(heads,y,pairs,h,population.totals)
    hetero=values.mean((0,1))
    hetero-=hetero.mean()
    amplitude=np.max(np.abs(hetero))
    hetero=hetero*.25/amplitude if amplitude else hetero
    return population,cov,hetero


def training_grid(cov,m,r,reps,stress,rng):
    q=cov[0].shape[0]
    mask=rng.multivariate_normal(np.zeros(q),stress*cov[0],size=(reps,m))
    init=rng.multivariate_normal(np.zeros(q),cov[1],size=(reps,r))
    noise=rng.multivariate_normal(np.zeros(q),cov[2],size=(reps,m,r))
    noise+=mask[:,:,None,:]
    noise+=init[:,None,:,:]
    return noise


def boot_variance_terms(grid):
    scalar=grid.mean(3)
    grand=scalar.mean((1,2))
    rm=scalar.mean(2)-grand[:,None]
    cm=scalar.mean(1)-grand[:,None]
    dif=scalar-grand[:,None,None]-rm[:,:,None]-cm[:,None,:]
    m,r=scalar.shape[1:]
    bootstrap_var=np.square(rm).sum(1)/m**2+np.square(cm).sum(1)/r**2+np.square(dif).sum((1,2))/(m*r)**2
    cut=t.ppf(.95,m-1)*scalar.mean(2).std(1,ddof=1)/np.sqrt(m)
    row=rm[:,:,None]
    col=cm[:,None,:]
    interaction_square=np.square(dif)
    third=(np.power(rm,3).sum(1)/m**3+np.power(cm,3).sum(1)/r**3
           +6*(row*dif*col).sum((1,2))/(m*r)**2
           +3*(row*interaction_square).sum((1,2))/(m**3*r**2)
           +3*(col*interaction_square).sum((1,2))/(m**2*r**3)
           +np.power(dif,3).sum((1,2))/(m*r)**3)
    return bootstrap_var,cut,third


def ci_factors():
    return norm.ppf([.975,.95,.025])


def approximate_quantiles(variance,third=None):
    factors=ci_factors()
    sigma=np.sqrt(variance)
    result=sigma[...,None]*factors
    if third is not None:
        correction=np.divide(third,variance,out=np.zeros_like(variance),where=variance>0)/6
        result+=correction[...,None]*(factors**2-1)
    return result


def evaluate_outer(grid,errors,variances,hetero,is_prompt,corrected=False):
    tv,mask_cut,third=boot_variance_terms(grid)
    q=approximate_quantiles(tv+variances,third if corrected else None)
    upper_error=q[:,1]
    standard=(q[:,0]-q[:,2])/2
    outcome=grid.mean((1,2))+errors
    err=outcome.mean(1)
    cov=float(((err>=q[:,2])&(err<=q[:,0])).mean())
    count=len(err)
    cov_hits=int(round(cov*count))
    lower=0. if cov_hits==0 else float(beta.ppf(.025,cov_hits,count-cov_hits+1))
    upper=1. if cov_hits==count else float(beta.ppf(.975,cov_hits+1,count-cov_hits))
    bundle=dict(coverage=cov,coverage_mc95=[lower,upper],bias_pp=float(errors.mean()),
                mean_halfwidth_pp=float(standard.mean()),
                mask_t_cut_pp=[float(v) for v in np.quantile(mask_cut,[.05,.5,.95])])
    for label,shift in zip(EFFECTS,(0.,.60,.60+hetero),strict=True):
        direction=outcome+shift
        point=direction.mean(1)
        gate=passed_rows(point,direction,point-upper_error,point-mask_cut,is_prompt)
        hits=int(gate.sum())
        u=1. if hits==count else float(beta.ppf(.95,hits+1,count-hits))
        bundle[label]=dict(gate_rate=hits/count,rate_upper95=u)
    return bundle


def hard_pass(scenario,model):
    return (scenario["coverage"]>=.93 and abs(scenario["bias_pp"])<=.05
            and scenario["homogeneous"]["gate_rate"]>=.80 and scenario["null"]["gate_rate"]<=.05
            and scenario["null"]["rate_upper95"]<=.075
            and (model!="dualcoop" or scenario["mean_halfwidth_pp"]<=.30))


def screen(name,prepared,designs,reps,bins):
    population,cov,hetero=prepared
    image=[]
    for i,d in enumerate(designs):
        quota=np.asarray(d["sample_counts"])
        err,var,_,_=image_outer(population,quota,reps,seed(1205,MODEL_ID[name],bins,i))
        image.append((err,var))
    rows=[]
    for m,r,inflation in product((12,16,24,32,48),(4,6,8),(1,2)):
        grid=training_grid(cov,m,r,reps,inflation,seed(1209,MODEL_ID[name],bins,m,r,inflation))
        for i,(err,var) in enumerate(image):
            summary=evaluate_outer(grid,err,var,hetero,name=="dualcoop")
            rows.append(dict(design=i,masks=m,initializations=r,mask_variance_multiplier=inflation,
                             numerical_planning_gate=hard_pass(summary,name),**summary))
    return dict(status="SYNTHETIC_FINITE_POPULATION_DESIGN_SCREEN",model=name,score_bins=bins,
                simulated_outer_per_case=reps,confirmation_scores_read=0,
                donor_binding=population.donor_match_counts,stratum_totals=population.totals.tolist(),rows=rows)


def setup_inner(population):
    global _INNER_MODEL
    _INNER_MODEL=population


def one_inner(task):
    j,counts,quota,scalar,variance,third,pair_effects,half_t,hetero,is_prompt=task
    rng=seed(1307,j,int(is_prompt))
    full_image,linear_image=image_inner(_INNER_MODEL,counts,quota,5000,rng)
    m,r=scalar.shape
    mask=rng.multinomial(m,np.full(m,1/m),size=5000)
    init=rng.multinomial(r,np.full(r,1/r),size=5000)
    ew=(mask[:,:,None]*init[:,None,:])/(m*r)
    train=(ew*scalar).sum((1,2))-scalar.mean()
    exact=np.quantile(train+full_image,[.975,.95,.025])
    linear=np.quantile(train+linear_image,[.975,.95,.025])
    normal=approximate_quantiles(np.asarray(variance))
    corrected=approximate_quantiles(np.asarray(variance),np.asarray(third))
    reached={}
    for key,errors in (("exact",exact),("normal",normal),("skew_corrected",corrected),("first_order",linear)):
        hits=[]
        for size in (0.,.60,.60+hetero):
            direction=pair_effects+size
            pt=float(direction.mean())
            keep=passed_rows(np.array([pt]),direction[None],np.array([pt-errors[1]]),
                             np.array([pt-half_t]),is_prompt)[0]
            hits.append(bool(keep))
        reached[key]=hits
    return dict(index=j,exact_quantile_pp=exact.tolist(),normal_pp=normal.tolist(),
                skew_corrected_pp=corrected.tolist(),first_order_pp=linear.tolist(),
                exact_vs_normal_error_pp=np.abs(exact-normal).tolist(),
                exact_vs_skew_corrected_error_pp=np.abs(exact-corrected).tolist(),
                exact_vs_first_order_error_pp=np.abs(exact-linear).tolist(),gates=reached)


def selected(name,prepared,design,m,r,workers):
    pop,cov,hetero=prepared
    quota=np.asarray(design["sample_counts"])
    final_seed=seed(1300,MODEL_ID[name],m,r,*quota.tolist())
    chosen=sorted(final_seed.choice(5000,size=100,replace=False).tolist())
    errors,var,_,saved=image_outer(pop,quota,5000,final_seed,chosen)
    grid=training_grid(cov,m,r,5000,1,seed(1301,MODEL_ID[name],m,r))
    stress=training_grid(cov,m,r,5000,2,seed(1302,MODEL_ID[name],m,r))
    tv,half_t,third=boot_variance_terms(grid)
    points=grid.mean((1,2))+errors
    jobs=[(i,saved[i],quota,grid[i].mean(2),tv[i]+var[i],third[i],points[i],half_t[i],hetero,name=="dualcoop") for i in chosen]
    with ProcessPoolExecutor(max_workers=workers,initializer=setup_inner,initargs=(pop,)) as pool:
        parity=list(pool.map(one_inner,jobs,chunksize=1))
    first_distances=np.array([row["exact_vs_first_order_error_pp"] for row in parity])
    exact=np.array([row["gates"]["exact"] for row in parity])
    comparisons={}
    passing=False
    for method in ("normal","skew_corrected"):
        candidate=evaluate_outer(grid,errors,var,hetero,name=="dualcoop",corrected=method=="skew_corrected")
        discovery=evaluate_outer(stress,errors,var,hetero,name=="dualcoop",corrected=method=="skew_corrected")
        distance=np.array([row[f"exact_vs_{method}_error_pp"] for row in parity])
        decisions=np.array([row["gates"][method] for row in parity])
        checks=dict(outer_scientific_gate=hard_pass(candidate,name),parity_q95=float(np.quantile(distance,.95)),
                    gate_agreement=float((exact==decisions).mean()),first_order_parity_q95=float(np.quantile(first_distances,.95)),
                    accelerated_false_accepts=int((~exact&decisions).sum()),accelerated_false_rejects=int((exact&~decisions).sum()))
        passing=checks["outer_scientific_gate"] and checks["parity_q95"]<=.03 and checks["gate_agreement"]>=.99
        comparisons[method]=dict(scientific_scenario=candidate,checks=checks)
        if passing:
            break
    return dict(status="PASS" if passing else "HOLD_STATISTICS_OR_PRECISION",model=name,mask_count=m,initializations=r,
                sample_design=design,outer_replicates=5000,complete_outer_cases=100,complete_inner_replicates=5000,
                scenario=candidate,double_mask_variance=discovery,checks=checks,parity=parity,methods_audited=comparisons,
                approximate_method=method,
                confirmation_scores_read=0)


def options_for(model,result,designs,timing,linear_seconds):
    result_rows=[]
    for candidate in result["rows"]:
        if candidate["mask_variance_multiplier"]!=1 or not candidate["numerical_planning_gate"]:
            continue
        d=designs[candidate["design"]]
        m,r=candidate["masks"],candidate["initializations"]
        if model=="dualcoop":
            visual,group,one_score=timing
            per_epoch=visual+(group-visual)*r/4
            hours=(50*m*per_epoch+m*one_score*(d["total"]+5000)/10000*max(1.,r/4))/3600+.5
            charge=hours/.8
            if charge>60:
                continue
            value=dict(design=candidate["design"],masks=m,initializations=r,estimated_hours=charge,
                       formal_gpu_endpoints=3*m*r)
        else:
            value=dict(design=candidate["design"],masks=m,initializations=r,
                       estimated_hours=m*linear_seconds*r/6/3600/.8)
            if value["estimated_hours"]>32:
                continue
        result_rows.append(value)
    return result_rows


def run(which,workers):
    started=perf_counter()
    design=[row for row in load("PANOPTIC_SAMPLE_QA.json")["designs"] if row["passed"]]
    models=("linear","dualcoop") if which=="both" else (which,)
    prepared={}
    all_results={}
    for model in models:
        prepared[model]=input_model(model,128)
        name=f"P2_{model.upper()}_SYNTHETIC_SCREEN.json"
        if (HERE/name).exists():
            result=load(name)
            assert result["model"]==model and result["simulated_outer_per_case"]==2500
            assert result["stratum_totals"]==prepared[model][0].totals.tolist() and result["confirmation_scores_read"]==0
        else:
            result=screen(model,prepared[model],design,2500,128)
            save(name,result)
        all_results[model]=result
        print("P2 screen",model,len(result["rows"]),"rows",flush=True)
    if which!="both":
        return
    p1=load("PROMPT_DEVELOPMENT_QA.json")
    measured=load("P1_MEASURED_FORECAST.json")
    smoke=load("PROMPT_SMOKE_QA.json")
    source=load("PROMPT_PILOT_INDEPENDENT_QA.json")
    linear=load("LINEAR_BRIDGE_QA_50E_6R.json")
    assert p1["status"]=="COMPLETE" and p1["development_endpoints"]==source["endpoints"]==96 and source["status"]=="PASS"
    assert linear["status"]=="COMPLETE" and linear["endpoints"]==144 and source["source_bridge_images"]==5000
    from m3_dev_preflight import run as assessment
    if not (HERE/"M3_DEV_PREFLIGHT_QA.json").exists():
        assessment(HERE)
    assert load("M3_DEV_PREFLIGHT_QA.json")["status"]=="PASS"
    timing=(smoke["image_and_head_timings"][-1]["visual_seconds"],measured["steady_12_endpoint_epoch_seconds"],
            float(np.median([r["score_and_seal_seconds"] for r in p1["runs"]])))
    candidates={name:options_for(name,result,design,timing,linear["wall_seconds"]/8) for name,result in all_results.items()}
    joins=[(m1,m2) for m1 in candidates["dualcoop"] for m2 in candidates["linear"] if m1["design"]==m2["design"]]
    joins.sort(key=lambda x:(x[0]["estimated_hours"]+x[1]["estimated_hours"],x[0]["formal_gpu_endpoints"],
                             design[x[0]["design"]]["total"],-x[0]["masks"],-x[1]["masks"]))
    attempted={}
    for m1,m2 in joins:
        choice=design[m1["design"]]
        cycle=[]
        for name,option in (("dualcoop",m1),("linear",m2)):
            key=(name,option["design"],option["masks"],option["initializations"])
            if key not in attempted:
                filename=f"P2_FINAL_TEST_{name}_{key[1]}_{key[2]}_{key[3]}.json"
                if (HERE/filename).exists():
                    report=load(filename)
                    assert report["model"]==name and report["sample_design"]==choice
                    assert report["mask_count"]==key[2] and report["initializations"]==key[3]
                    assert report["outer_replicates"]==report["complete_inner_replicates"]==5000
                    assert report["complete_outer_cases"]==100 and report["confirmation_scores_read"]==0
                else:
                    report=selected(name,prepared[name],choice,option["masks"],option["initializations"],workers)
                    save(filename,report)
                attempted[key]=report["status"]
            cycle.append(attempted[key])
            if cycle[-1]!="PASS":
                break
        if cycle==["PASS","PASS"]:
            save("P2_FINAL_DESIGN_LOCK.json",dict(status="READY_TO_DRAW",sample_design=choice,M1=m1,M2=m2,
                 seed_index=0,confirmation_scores_read=0,source_bridge_images=5000,wall_seconds=perf_counter()-started))
            return
        if perf_counter()-started>40*.8*3600:
            break
    save("P2_FINAL_DESIGN_LOCK.json",dict(status="HOLD_PRECISION" if not joins else "HOLD_STATISTICS_OR_PRECISION",
         confirmation_scores_read=0,eligible_joint_screen_candidates=len(joins),final_tests=len(attempted),
         wall_seconds=perf_counter()-started))


if __name__=="__main__":
    parser=ArgumentParser()
    parser.add_argument("model",choices=("linear","dualcoop","both"))
    parser.add_argument("--workers",type=int,default=8)
    parser.add_argument("--candidate",nargs=3,type=int,metavar=("MASKS","INITIALIZATIONS","DESIGN_INDEX"))
    args=parser.parse_args()
    if args.candidate is None:
        run(args.model,args.workers)
    else:
        assert args.model!="both"
        m,r,i=args.candidate
        assert m in (12,16,24,32,48) and r in (4,6,8)
        choices=[row for row in load("PANOPTIC_SAMPLE_QA.json")["designs"] if row["passed"]]
        receipt=selected(args.model,input_model(args.model,128),choices[i],m,r,args.workers)
        save(f"P2_FINAL_TEST_{args.model}_{i}_{m}_{r}.json",receipt)
        print(receipt["status"],receipt["checks"],flush=True)
