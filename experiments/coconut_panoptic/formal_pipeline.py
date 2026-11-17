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

"""Run v9.3.1 formal phases in the registered score-reveal order on the original host."""
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime
from pathlib import Path
import os
import shutil
import subprocess
import sys
from time import perf_counter, time

import numpy as np

import freeze_formal_sample
from formal_metrics import read, seal


SEED=93120261115


def stage(work,name):
    seal(work/"FORMAL_RUN_STATE.json",dict(stage=name,updated=datetime.now().astimezone().isoformat(),
         gpu_policy="RETAIN_UNTIL_EXPERIMENT_COMPLETE"))


def call(folder,script,*arguments,priority=None):
    argv=[sys.executable,"-u",str(Path(__file__).resolve().parent / "formal" / script),*map(str,arguments)]
    if priority is not None:
        argv=["nice","-n",str(priority),*argv]
    role=str(arguments[1]) if len(arguments)>1 and arguments[1] in ("fit","bridge","predict","confirm","evaluate") else "run"
    log=folder/f"{Path(script).stem}_{role}.log"
    env=os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",
               PYTHONPATH=str(folder)+os.pathsep+env.get("PYTHONPATH",""))
    with log.open("a",encoding="utf-8",buffering=1) as stream:
        stream.write(f"{datetime.now().astimezone().isoformat()} {script} {list(map(str,arguments))}\n")
        subprocess.run(argv,cwd=folder,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)


def validation_lock(root,formal,design):
    targets=read(root/"TARGET_REPAIR_STATUS.json")["selected"]
    panel=read(root/"PANOPTIC_MASK_SCREEN.json")
    independent=read(root/"PANOPTIC_MASK_INDEPENDENT_QA.json")
    fit=read(root/"MODEL_INPUT_QA.json")
    assert targets==panel["chosen"] and len(targets)>=4 and fit["image_roles_disjoint"]
    assert independent["status"]=="PASS" and independent["rebuilt_masks"]>=56
    source=read(root/"PANOPTIC_BRIDGE_PAIR_QA.json")
    prompt=read(root/"PROMPT_PILOT_INDEPENDENT_QA.json")
    assert source["status"]=="PASS_LABEL_SUPPORT" and source["validated_pairs"]==2*len(targets)
    assert prompt["status"]=="PASS" and prompt["endpoints"]==96 and prompt["source_bridge_images"]==5000
    linear=read(root/"LINEAR_BRIDGE_INDEPENDENT_QA.json")
    assert linear["status"]=="PASS" and linear["endpoints"]==144 and linear["confirmation_scores_read"]==0
    labels=read(root/"SAMPLE_SUPPORT_QA.json")
    sample=read(root/"SAMPLING_LOCK.json")
    pixels=read(formal/"CONF_M_PIXEL_QA.json")
    assert labels["status"]=="PASS_LABEL_SUPPORT" and labels["sample_images"]==len(sample["sample_ids"])
    assert pixels["status"]=="PASS" and pixels["sample_ids"]==sample["sample_ids"]
    assert pixels["decoded"]==len(sample["sample_ids"]) and pixels["score_exposure_count"]==0
    classes=read(root/"PANOPTIC_CLASS_SPACE_LOCK.json")["categories"]
    categories=[row["id"] for row in classes]
    with np.load(root/"PANOPTIC_BRIDGE_LABELS.npz",allow_pickle=False) as bridge:
        source_ids=bridge["ids"].tolist()
        assert bridge["categories"].tolist()==categories
    with np.load(root/f"PANOPTIC_MASK_{panel['formal_seeds'][0]}.npz",allow_pickle=False) as mask:
        fit_ids=mask["ids"].tolist()
    assert len(set(fit_ids+source_ids+sample["sample_ids"]))==len(fit_ids)+len(source_ids)+len(sample["sample_ids"])
    result=dict(status="READY",version="v9.3.1-coco133",run_id=f"v931-133-n{len(sample['sample_ids'])}-seed{design['seed_index']}",
                targets=targets,categories=categories,fit_ids=fit_ids,bridge_ids=source_ids,
                confirm_ids=sample["sample_ids"],confirm_scores_read=0)
    for stage_name,model in (("M1","dualcoop"),("M2","linear")):
        care=design[stage_name]
        m,r=care["masks"],care["initializations"]
        assert m in panel["eligible_formal_prefixes"] and r in (4,6,8)
        audit=read(root/f"P2_FINAL_TEST_{model}_{care['design']}_{m}_{r}.json")
        assert audit["status"]=="PASS" and audit["sample_design"]["sample_counts"]==design["sample_design"]["sample_counts"]
        assert audit["outer_replicates"]==audit["complete_inner_replicates"]==5000 and audit["complete_outer_cases"]==100
        draws=panel["formal_seeds"][:m]
        assert set(draws).isdisjoint(panel["dev_seeds"])
        assert set(draws)<={row["seed"] for row in independent["per_mask"]}
        values=[int(np.random.SeedSequence([SEED,21,i]).generate_state(1,np.uint64)[0]) for i in range(1,r+1)]
        result[stage_name]=dict(draws=draws,seeds=values,epochs=50,updates_per_endpoint=6300)
    return result


def allocation_record(root,formal,lock,design):
    boot=read(root/"P1_GPU_BOOT.json")
    price=read(root/"P1_GPU_BUDGET.json")["listed_rate_yuan_per_hour"]
    acquired=datetime.fromisoformat(boot["timestamp"]).timestamp()
    now=time()
    allocated=(now-acquired)/3600
    p3=design["M1"]["estimated_hours"]
    p4=design["M2"]["estimated_hours"]
    budget_path=formal/"BUDGET_LOCK.json"
    if budget_path.exists():
        prior=read(budget_path)
        if prior["status"]=="READY":
            assert prior["P3_gpu_hours_with_recovery"]==p3 and prior["P4_linear_cpu_hours_with_recovery"]==p4
            assert prior["rate_yuan_per_gpu_hour"]==price
            started=acquired+prior["already_allocated_gpu_hours"]*3600
            deadline=started+p3*3600
            if now+300>=deadline and not (formal/"M1_CONFIRM_SCORE_RECEIPT.json").is_file():
                raise RuntimeError("HOLD_BUDGET")
            return deadline
    n=len(lock["confirm_ids"])+len(lock["bridge_ids"])
    m1=lock["M1"]
    m2=lock["M2"]
    n1=3*len(m1["draws"])*len(m1["seeds"])
    n2=3*len(m2["draws"])*len(m2["seeds"])
    bytes_score=(n1+n2)*n*len(lock["categories"])*4
    bytes_transient=3*max(len(m1["seeds"]),len(m2["seeds"]))*len(lock["confirm_ids"])*len(lock["categories"])*4
    bytes_features=len(lock["confirm_ids"])*1024*4
    bytes_states=(n1*2*2*16*512+n2*1025*len(lock["categories"]))*4
    expected_bytes=bytes_score+bytes_transient+bytes_features+bytes_states+25_000_000
    free=shutil.disk_usage(formal).free
    projected=allocated+p3+p4
    okay_storage=free>=1.25*expected_bytes
    okay_budget=p3<=60 and p4<=32
    state="READY" if okay_storage and okay_budget else ("HOLD_STORAGE" if not okay_storage else "HOLD_BUDGET")
    seal(budget_path,dict(status=state,applicable_stage="P3/P4",rate_yuan_per_gpu_hour=price,
         already_allocated_gpu_hours=allocated,P3_gpu_hours_with_recovery=p3,P4_linear_cpu_hours_with_recovery=p4,
         estimated_p4_linear_is_concurrent=True,physical_upper_envelope_hours=projected,
         suggested_total_gpu_hours=84,estimated_physical_envelope_exceeds_suggestion=projected>84,
         estimated_peak_additional_bytes=expected_bytes,minimum_free_bytes=int(np.ceil(1.25*expected_bytes)),free_bytes=free))
    if state!="READY":
        raise RuntimeError(state)
    return now+p3*.8*3600


def seal_endpoint_manifest(formal,lock):
    file=formal/"ENDPOINT_MANIFEST.csv"
    tmp=file.with_suffix(".partial")
    columns=["model","mask_seed","init_seed","arm","epochs","updates","checkpoint","bridge_scores","confirm_scores"]
    count=0
    with tmp.open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=columns)
        writer.writeheader()
        for model in ("M1","M2"):
            matrix=lock[model]
            extension="pt" if model=="M1" else "npz"
            for draw in matrix["draws"]:
                checkpoint=f"{model}_DRAW_{draw}.{extension}"
                source=f"{model}_bridge_{draw}.npz"
                confirmed=f"{model}_confirm_{draw}.npz"
                assert all((formal/name).stat().st_size>0 for name in (checkpoint,source,confirmed))
                for init in matrix["seeds"]:
                    for arm in ("DIR1","DIR2","PLC"):
                        writer.writerow(dict(model=model,mask_seed=draw,init_seed=init,arm=arm,epochs=50,updates=6300,
                                             checkpoint=checkpoint,bridge_scores=source,confirm_scores=confirmed))
                        count+=1
    tmp.replace(file)
    return count


def run(root,lineage,fit_manifest,fit_cache):
    formal=root/"formal"
    formal.mkdir(exist_ok=True)
    started=perf_counter()
    design=read(root/"P2_FINAL_DESIGN_LOCK.json")
    assert design["status"]=="READY_TO_DRAW" and design["confirmation_scores_read"]==0
    path=formal/"PROTOCOL_LOCK.json"
    if not path.exists():
        stage(root,"FINAL_SAMPLE_AND_PIXELS")
        freeze_formal_sample.main(root/"P2_FINAL_DESIGN_LOCK.json",lineage)
        for name in ("CONF_M_LABELS.npz","PANOPTIC_BRIDGE_LABELS.npz","PANOPTIC_CLASS_SPACE_LOCK.json"):
            shutil.copyfile(root/name,formal/name)
        call(formal,"formal_pixels.py",formal,root/"CONF_M_PIXEL_REQUEST.csv",root/"SAMPLING_LOCK.json")
    lock=validation_lock(root,formal,design)
    if path.exists():
        assert read(path)==lock
    deadline=allocation_record(root,formal,lock,design)
    if not path.exists():
        seal(path,lock)
    for draw in sorted(set(lock["M1"]["draws"])|set(lock["M2"]["draws"])):
        name=f"PANOPTIC_MASK_{draw}.npz"
        if not (formal/name).exists():
            tmp=(formal/name).with_suffix(".partial")
            shutil.copyfile(root/name,tmp)
            tmp.replace(formal/name)
    stage(root,"P3_M1_GPU_AND_M2_CPU_FIT")
    with ThreadPoolExecutor(max_workers=2) as pool:
        linear=pool.submit(call,formal,"formal_linear.py",formal,"fit","--fit-cache",fit_cache,priority=8)
        if not (formal/"M1_TRAIN_QA.json").exists():
            call(formal,"formal_prompt.py",formal,"fit",deadline,"--fit-manifest",fit_manifest)
        stage(root,"FORMAL_BRIDGE_SCORING")
        if not (formal/"M1_BRIDGE_SCORE_RECEIPT.json").exists():
            call(formal,"formal_prompt.py",formal,"bridge",deadline)
        linear.result()
        if not (formal/"M2_BRIDGE_SCORE_RECEIPT.json").exists():
            call(formal,"formal_linear.py",formal,"bridge",priority=8)
    stage(root,"M3_SOURCE_PREDICTION_SEAL")
    call(formal,"formal_m3.py",formal,"predict")
    stage(root,"FORMAL_CONFIRMATION_SCORING")
    if not (formal/"M1_CONFIRM_SCORE_RECEIPT.json").exists():
        call(formal,"formal_prompt.py",formal,"confirm",deadline)
    if not (formal/"M2_CONFIRM_SCORE_RECEIPT.json").exists():
        call(formal,"formal_linear.py",formal,"confirm",priority=8)
    stage(root,"FORMAL_INFERENCE_AND_INDEPENDENT_RECOMPUTATION")
    if not (formal/"M1_M2_COMPARISON.json").exists():
        call(formal,"formal_metrics.py",formal)
    assert read(formal/"M1_M2_COMPARISON.json")["run_id"]==lock["run_id"]
    if not (formal/"M3_RESULT.json").exists():
        call(formal,"formal_m3.py",formal,"evaluate")
    models={name:read(formal/f"{name}_RESULT.json") for name in ("M1","M2","M3")}
    assert all(result["run_id"]==lock["run_id"] for result in models.values())
    passed={name:result["passed"] for name,result in models.items()}
    status="COMPLETE_PASS" if all(passed.values()) else "COMPLETE_GATE_NOT_MET"
    entries=seal_endpoint_manifest(formal,lock)
    boot=read(root/"P1_GPU_BOOT.json")
    held=(time()-datetime.fromisoformat(boot["timestamp"]).timestamp())/3600
    budget=read(formal/"BUDGET_LOCK.json")
    record=dict(status=status,models=passed,run_id=lock["run_id"],confirmed_images=len(lock["confirm_ids"]),
                fit_targets=len(lock["targets"]),wall_seconds_since_formal_start=perf_counter()-started,
                M1_dualcoop_endpoints=3*len(lock["M1"]["draws"])*len(lock["M1"]["seeds"]),
                M2_linear_endpoints=3*len(lock["M2"]["draws"])*len(lock["M2"]["seeds"]),
                endpoint_manifest_rows=entries,
                metric_summary={name:{"I_pp":models[name]["I_pp"],"bounds_pp":models[name]["exact_bounds_pp"]}
                                for name in ("M1","M2")},M3_mae_pp=models["M3"]["mae_pp"],
                M3_direction_coverage=models["M3"]["coverage"],M3_direction_hit=models["M3"]["directional_hit_rate"],
                physical_gpu_wall_hours_at_seal=held,listed_rate_yuan_per_hour=budget["rate_yuan_per_gpu_hour"],
                estimated_cost_yuan_at_seal=held*budget["rate_yuan_per_gpu_hour"],
                score_reveal_order="bridge, source predictions, confirmation, evaluation")
    seal(formal/"FORMAL_COMPLETE.json",record)
    stage(root,status)


if __name__=="__main__":
    parser=ArgumentParser()
    for name in ("root","lineage","fit_manifest","fit_cache"):
        parser.add_argument(name,type=Path)
    opts=parser.parse_args()
    run(**{name: path.resolve() for name, path in vars(opts).items()})
