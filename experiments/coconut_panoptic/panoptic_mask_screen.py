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

"""Preflight deterministic candidate masks on the fixed FIT IDs and frozen difficulty."""
from collections import Counter
import csv
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations, product
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np


HERE = Path(__file__).resolve().parent
V9 = HERE.parent
sys.path[:0] = [str(V9/"coconut_inputs/preparation"),str(V9/"voc_crossed")]
from mask_gate import evaluate_draw, materialize

SEED = 93120261115
ARMS = ("LOW","DIR1","DIR2","PLC")
PREFIXES = (12,16,24,32,48)


def create_seed(stream,index):
    rng = np.random.SeedSequence([SEED,stream,index])
    return int(rng.generate_state(1,dtype=np.uint64)[0])


def qa(masks, fit):
    low = masks["LOW"]
    bins = [np.bincount((masks[a]>=0).sum(1),minlength=fit.shape[1]+1) for a in ARMS]
    active = [len(fit)-b[0] for b in bins]
    tv = max(float(abs(a-b).sum()/(2*len(fit))) for a,b in combinations(bins,2))
    difference = int(max(active)-min(active))
    for arm in ARMS:
        obs = masks[arm] >=0
        assert np.array_equal(masks[arm][obs],fit[obs])
        assert np.array_equal(masks[arm]==1,low==1) and np.array_equal((masks[arm]==0).sum(0),(low==0).sum(0))
    return tv, difference


def screen_one(spec):
    fit,score,ids,c,rows,categories,dev_seeds,formal_seeds = spec
    lookup = {int(category):i for i,category in enumerate(categories)}
    lows = {seed:np.where(np.random.default_rng(seed).random(fit.shape)<.10,fit,-1).astype(np.int8)
            for seed in (*dev_seeds,*formal_seeds[:12])}
    attempts = []
    for index,row in enumerate(rows):
        pair = tuple(lookup[i] for i in [c,*json.loads(row["partner_ids"])])
        for dose in (40,20):
            for seed,low in lows.items():
                selection,reason = evaluate_draw(fit,score,low,ids,[pair],[dose],seed)
                if selection is None:
                    break
            else:
                return c,(row,pair,dose,index+1),attempts
            attempts.append(dict(target=row["target"],partner_ids=json.loads(row["partner_ids"]),dose=dose,seed=seed,reason=reason))
    return c,None,attempts


def main():
    started = perf_counter()
    base = np.load(V9/"coco_reference_labels/FIT80.npz",allow_pickle=False)
    official = np.load(HERE/"PANOPTIC_OFFICIAL_LABELS.npz",allow_pickle=False)
    difficulty = np.load(HERE/"PANOPTIC_FIT_DIFFICULTY.npz",allow_pickle=False)
    np.testing.assert_array_equal(base["ids"],official["fit_ids"])
    np.testing.assert_array_equal(official["fit_ids"],difficulty["ids"])
    np.testing.assert_array_equal(official["category_ids"],difficulty["categories"])
    # Keep all 80 original outputs and add the entire native 53-class stuff ontology.
    fit = np.column_stack((base["truth"],official["fit"]))
    score = np.column_stack((np.zeros_like(base["truth"],dtype=float),difficulty["scores"]))
    ids = base["ids"]
    category = np.r_[base["category_ids"],official["category_ids"]].tolist()
    assert fit.shape == (4008,133)
    with (HERE/"PANOPTIC_CANDIDATES.csv").open(encoding="utf-8-sig",newline="") as f:
        candidates = list(csv.DictReader(f))
    grouped = {}
    history = set([122,189,196])
    for row in candidates:
        c = int(row["target_id"])
        if c not in history and row["source_support_pass"] == "True":
            grouped.setdefault(c,[]).append(row)
    dev_seeds = [create_seed(10,j+1) for j in range(8)]
    formal_seeds = [create_seed(20,j+1) for j in range(48)]
    lows = {seed:np.where(np.random.default_rng(seed).random(fit.shape)<.10,fit,-1).astype(np.int8)
            for seed in (*dev_seeds,*formal_seeds)}
    attempts, qualified = [], []
    limits = {}
    resume = HERE/"PANOPTIC_MASK_SINGLE_CHECKPOINT.json"
    existing = json.loads(resume.read_text(encoding="utf-8")) if resume.exists() else {}
    for c, rows in grouped.items():
        if str(c) in existing:
            row,pair,dose,n = existing[str(c)]
            qualified.append((row,tuple(pair),dose,n))
    pending = [(fit,score,ids,c,rows,category,dev_seeds,formal_seeds) for c,rows in grouped.items() if str(c) not in existing]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for c,result,trials in pool.map(screen_one,pending):
            attempts.extend(trials)
            if result is None:
                limits[grouped[c][0]["target"]] = len(grouped[c])
            else:
                qualified.append(result)
                existing[str(c)] = result
                resume.write_text(json.dumps(existing,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
                print("qualified",result[0]["target"],result[0]["partners"],result[2],result[3],flush=True)

    dev = np.load(HERE/"PANOPTIC_OFFICIAL_LABELS.npz",allow_pickle=False)["dev"]
    qualified.sort(key=lambda r:(int(dev[:,r[1][0]-80].sum()),r[1][0]))
    layers = np.array_split(np.arange(len(qualified)),3)
    rng = np.random.default_rng(np.random.SeedSequence([SEED,1,0]))
    groups = [rng.permutation(g).tolist() for g in layers]
    count = min(6,len(qualified))
    front = [i for layer in groups for i in layer[:2]]
    next_ids = [groups[k][i] for i in range(2,max(map(len,groups),default=0)) for k in (1,2,0) if i<len(groups[k])]
    order = front + next_ids
    combined = []

    def assess(members,doses,must_pass):
        panel = [qualified[i] for i in members]
        pairs = [row[1] for row in panel]
        state = []
        for j,seed in enumerate((*dev_seeds,*formal_seeds)):
            selections,reason = evaluate_draw(fit,score,lows[seed],ids,pairs,doses,seed)
            if selections is None:
                state.append(dict(seed=seed,status="FAILED",reason=reason))
            else:
                masks = materialize(lows[seed],selections)
                tv,active = qa(masks,fit)
                state.append(dict(seed=seed,status="PASS" if tv<=.01 and active<=20 else "FAILED", density_tv=tv,active_difference=active))
            if j<must_pass and state[-1]["status"]!="PASS":
                return state
        return state

    locked = None
    if count>=4:
        for size in range(count,3,-1):
            for members in [order[:size], *[order[:size-1]+[candidate] for candidate in order[size:size+2]]]:
                maximum = [qualified[i][2] for i in members]
                vectors = sorted(product(*[(40,20) if d==40 else (20,) for d in maximum]),
                                 key=lambda x:(-sum(x),tuple(-v for _,v in sorted(zip([qualified[i][1][0] for i in members],x)))))
                for doses in vectors:
                    state = assess(members,doses,20)
                    combined.append(dict(targets=[qualified[i][0]["target"] for i in members],doses=doses,results=state))
                    if len(state)>=20 and all(row["status"]=="PASS" for row in state[:20]):
                        locked = (members,doses,state)
                        break
                if locked:
                    break
            if locked:
                break
    if locked:
        members,doses,state = locked
        panel = [qualified[i] for i in members]
        eligible_prefixes = [n for n in PREFIXES if all(row["status"]=="PASS" for row in state[:8+n])]
        selection_info = [dict(target=x[0]["target"],target_id=int(x[0]["target_id"]),
                               partners=json.loads(x[0]["partner_ids"]), dose=dose) for x,dose in zip(panel,doses,strict=True)]
        for record in state:
            if record["status"] != "PASS":
                continue
            seed = record["seed"]
            choices,_ = evaluate_draw(fit,score,lows[seed],ids,[p[1] for p in panel],doses,seed)
            realized = materialize(lows[seed],choices)
            np.savez_compressed(HERE/f"PANOPTIC_MASK_{seed}.npz",ids=ids,categories=category,**realized)
        status = "MASK_CANDIDATE_PASS"
    else:
        status,selection_info,eligible_prefixes = "HOLD_MASK",[],[]
    result = dict(status=status, minimum_new_targets=4, candidates_with_source_support=len(grouped),
                  individually_eligible=[dict(target=row["target"],target_id=int(row["target_id"]),partners=json.loads(row["partner_ids"]),
                      dose=dose,ranking_tries=n) for row,_,dose,n in qualified], failures_by_target=limits,
                  independent_rng_observation="LOW regenerated by seed; full panel independent reconstruction pending",
                  chosen=selection_info, eligible_formal_prefixes=eligible_prefixes, dev_seeds=dev_seeds, formal_seeds=formal_seeds,
                  attempts_by_reason=Counter(row["reason"] for row in attempts), combined=combined,
                  reserve=[qualified[i][0]["target"] for i in order if not locked or i not in locked[0]],
                  comparison_model_score_reads=0, elapsed_seconds=perf_counter()-started)
    (HERE/"PANOPTIC_MASK_SCREEN.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(status,selection_info,eligible_prefixes,"elapsed",result["elapsed_seconds"],flush=True)


if __name__=="__main__":
    main()
