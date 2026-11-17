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

"""Reconcile the paired annotation transition matrices for the selected targets."""
import csv
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


if __name__=="__main__":
    panel=json.loads((HERE/"TARGET_REPAIR_STATUS.json").read_text(encoding="utf-8"))["selected"]
    bridge=np.load(HERE/"PANOPTIC_BRIDGE_LABELS.npz",allow_pickle=False)
    classes=bridge["categories"].tolist()
    result=[]
    for row in panel:
        for partner in row["partners"]:
            cols=[classes.index(i) for i in (row["target_id"],partner)]
            before,after=(bridge[name][:,cols] for name in ("original_coco","truth"))
            known=(after>=0).all(1)
            left=2*before[known,0]+before[known,1]
            right=2*after[known,0]+after[known,1]
            positions=[2,1,3,0]
            trans=np.bincount(left*4+right,minlength=16).reshape(4,4)[np.ix_(positions,positions)]
            cells=trans.sum(0)
            n10,n01,n11,n00=cells
            npos=float(n10+n11)
            neg_ess=min(1/((1-r)**2/n00+r*r/n01) for r in (.05,.5))
            p = n11/npos
            proportions=[(t[2]/(t[0]+t[2]),t[1]/(t[1]+t[3])) for t in (trans.sum(1),cells)]
            assert (cells>=np.array([50,50,30,200])).all() and npos>=80 and neg_ess>=250
            result.append(dict(target=row["target"],target_id=row["target_id"],partner_id=partner,
                known_images=int(known.sum()),unknown_images=int((~known).sum()),
                nonempty_bridge_cells=cells.tolist(),transition_10_01_11_00=trans.tolist(),
                source_p=p,source_fixed_negative_ess=neg_ess,composition_original=list(proportions[0]),composition_coconut=list(proportions[1])))
    with (HERE/"PANOPTIC_BRIDGE_PAIR_QA.csv").open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(result[0]));writer.writeheader();writer.writerows(result)
    report=dict(status="PASS_LABEL_SUPPORT",validated_pairs=len(result),both_annotated_images=len(bridge["ids"]),
        source_model_scores=0,minimum_pss=min(r["nonempty_bridge_cells"][0]+r["nonempty_bridge_cells"][2] for r in result),
        minimum_negative_ess=min(r["source_fixed_negative_ess"] for r in result))
    (HERE/"PANOPTIC_BRIDGE_PAIR_QA.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(report,flush=True)
