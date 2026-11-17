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

"""Reconstruct delivered masks without invoking the mask solver."""
from collections import Counter
from itertools import combinations
import json
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp


HERE = Path(__file__).resolve().parent
ARMS = ["DIR1","DIR2","PLC"]


def main():
    plan = json.loads((HERE/"PANOPTIC_MASK_SCREEN.json").read_text(encoding="utf-8"))
    base = np.load(HERE.parent/"coco_reference_labels/FIT80.npz",allow_pickle=False)
    official = np.load(HERE/"PANOPTIC_OFFICIAL_LABELS.npz",allow_pickle=False)
    diff = np.load(HERE/"PANOPTIC_FIT_DIFFICULTY.npz",allow_pickle=False)
    fit = np.column_stack((base["truth"],official["fit"]))
    ids = base["ids"]
    cats = np.r_[base["category_ids"],official["category_ids"]].tolist()
    pairs = [(cats.index(x["target_id"]),[cats.index(p) for p in x["partners"]],x["dose"]) for x in plan["chosen"]]
    targets = {t for t,_,_ in pairs}
    kept = [j for j in range(fit.shape[1]) if j not in targets]
    results = []
    for seed in [*plan["dev_seeds"],*plan["formal_seeds"]]:
        p = HERE/f"PANOPTIC_MASK_{seed}.npz"
        assert p.is_file(),p
        with np.load(p,allow_pickle=False) as d:
            assert d["categories"].tolist() == cats
            np.testing.assert_array_equal(d["ids"],ids)
            low = np.where(np.random.default_rng(seed).random(fit.shape)<.10,fit,-1)
            np.testing.assert_array_equal(d["LOW"],low)
            a = [d[arm] for arm in ARMS]
            density = np.minimum(2,(low[:,kept]>=0).sum(1))
            details = []
            for arm in a:
                np.testing.assert_array_equal(arm[:,kept],low[:,kept])
                np.testing.assert_array_equal(arm==1,low==1)
                np.testing.assert_array_equal(arm[arm>=0],fit[arm>=0])
            for t,(p1,p2),dose in pairs:
                rm = [np.flatnonzero((v[:,t]<0)&(low[:,t]==0)) for v in a]
                assert all(np.array_equal(rm[0],v) for v in rm[1:])
                added = [np.flatnonzero((v[:,t]==0)&(low[:,t]<0)) for v in a]
                assert all(len(v)==dose for v in [rm[0],*added])
                assert not fit[rm[0]][:,[p1,p2]].any()
                assert fit[added[0],p1].all() and fit[added[1],p2].all()
                x = diff["scores"][:,t-80]
                negatives = np.flatnonzero(fit[:,t]==0)
                decile = np.empty(len(ids),dtype=np.int8)
                for b,r in enumerate(np.array_split(negatives[np.lexsort((ids[negatives],x[negatives]))],10)):
                    decile[r] = b
                for new in added:
                    assert Counter(zip(decile[new],density[new])) == Counter(zip(decile[rm[0]],density[rm[0]]))
                full = [rm[0],*added]
                smd = float(max(abs(x[z].mean()-x[w].mean()) for z,w in combinations(full,2))/x[negatives].std())
                gap = float(min(1-fit[w,q].mean() for w,q in ((added[1],p1),(added[2],p1),(added[0],p2),(added[2],p2))))
                tail = (x>np.quantile(x[negatives],.9))
                assert len({int(tail[w].sum()) for w in added})==1 and gap>=.2-1e-12 and smd<=.1
                details.append(dict(target_id=cats[t],positive_fractions=[[float(fit[w,p].mean()) for w in added] for p in (p1,p2)],
                    dose=dose,smd=smd,partner_gap=gap,upper_tail=int(tail[added[0]].sum()),
                    ks=float(max(ks_2samp(x[z],x[w]).statistic for z,w in combinations(full,2))),
                    max_coordinate_overlap=max(len(set(z)&set(w)) for z,w in combinations(full,2))))
            doses = [np.bincount((v>=0).sum(1),minlength=fit.shape[1]+1) for v in [low,*a]]
            tv = float(max(abs(x-y).sum() for x,y in combinations(doses,2))/(2*len(fit)))
            active = int(max(x[0] for x in doses)-min(x[0] for x in doses))
            assert tv<=.01 and active<=20
            results.append(dict(seed=seed,density_tv=tv,active_difference=active,targets=details))
    assert len(results)==56
    result = dict(status="PASS", rebuilt_masks=len(results), max_density_tv=max(r["density_tv"] for r in results),
                  max_active=max(r["active_difference"] for r in results),
                  max_smd=max(p["smd"] for r in results for p in r["targets"]),
                  min_partner_gap=min(p["partner_gap"] for r in results for p in r["targets"]), per_mask=results)
    (HERE/"PANOPTIC_MASK_INDEPENDENT_QA.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print({k:v for k,v in result.items() if k!="per_mask"},flush=True)


if __name__=="__main__":
    main()
