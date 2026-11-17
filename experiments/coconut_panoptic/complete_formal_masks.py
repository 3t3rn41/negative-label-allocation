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

"""Retry failed formal suffix masks with deterministic solver order at fixed seeds/dose."""
from datetime import datetime
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from panoptic_mask_screen import HERE,V9,SEED,PREFIXES,evaluate_draw,materialize,qa


def main():
    started = perf_counter()
    path = HERE/"PANOPTIC_MASK_SCREEN.json"
    original = HERE/"PANOPTIC_MASK_SCREEN_BEFORE_FORMAL_RETRY.json"
    if not original.exists():
        original.write_bytes(path.read_bytes())
    design = json.loads(original.read_text(encoding="utf-8"))
    assert design["status"]=="MASK_CANDIDATE_PASS"
    base = np.load(V9/"coco_reference_labels/FIT80.npz",allow_pickle=False)
    official = np.load(HERE/"PANOPTIC_OFFICIAL_LABELS.npz",allow_pickle=False)
    difficulty = np.load(HERE/"PANOPTIC_FIT_DIFFICULTY.npz",allow_pickle=False)
    fit = np.column_stack((base["truth"],official["fit"]))
    score = np.column_stack((np.zeros_like(base["truth"],dtype=float),difficulty["scores"]))
    ids = base["ids"]
    cats = np.r_[base["category_ids"],official["category_ids"]].tolist()
    pairs = [tuple(cats.index(j) for j in [row["target_id"],*row["partners"]]) for row in design["chosen"]]
    doses = [row["dose"] for row in design["chosen"]]
    existing = design["combined"][-1]["results"]
    attempts=[]
    for i,seed in enumerate(design["formal_seeds"],1):
        entry=existing[7+i]
        assert entry["seed"]==seed
        if entry["status"]=="PASS":
            continue
        low=np.where(np.random.default_rng(seed).random(fit.shape)<.10,fit,-1).astype(np.int8)
        for retry in range(1,25):
            alternate=int(np.random.SeedSequence([SEED,20,i,retry]).generate_state(1,np.uint64)[0])
            coords,reason=evaluate_draw(fit,score,low,ids,pairs,doses,alternate)
            trial=dict(seed=seed,formal_index=i,solver_order_seed=alternate,retry=retry,reason=reason,status="FAILED")
            attempts.append(trial)
            if coords is None:
                continue
            masks=materialize(low,coords)
            tv,active=qa(masks,fit)
            trial.update(density_tv=tv,active_difference=active)
            if tv<=.01 and active<=20:
                trial["status"]="PASS"
                entry.update(trial)
                np.savez_compressed(HERE/f"PANOPTIC_MASK_{seed}.npz",ids=ids,categories=cats,**masks)
                print(i,retry,tv,flush=True)
                break
    design["eligible_formal_prefixes"]=[n for n in PREFIXES if all(row["status"]=="PASS" for row in existing[:8+n])]
    design["formal_retry"] = dict(as_of=datetime.now().astimezone().isoformat(),attempts=attempts,
        wall_seconds=perf_counter()-started,policy="Only failed formal suffix masks. LOW, image IDs, panel, doses and all numerical gates are fixed.",
        developer_or_formal_outcomes_read=False)
    path.write_text(json.dumps(design,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("formal prefixes",design["eligible_formal_prefixes"],flush=True)


if __name__ == "__main__":
    main()
