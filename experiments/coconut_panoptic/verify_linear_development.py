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

"""Independently recompute development interactions by score-threshold integration."""
import json
from pathlib import Path

import numpy as np
from statistics_primitives import weighted_auc


HERE=Path(__file__).resolve().parent
V9=HERE.parent


if __name__=="__main__":
    summary=json.loads((HERE/"LINEAR_PRECISION_50E_6R.json").read_text())
    with (np.load(V9/"coco_reference_labels/DEV80.npz",allow_pickle=False) as a,
          np.load(HERE/"PANOPTIC_OFFICIAL_LABELS.npz",allow_pickle=False) as b):
        np.testing.assert_array_equal(a["ids"],b["dev_ids"])
        lookup={i:r for r,i in enumerate(a["ids"].tolist())}
        truth=np.column_stack((a["truth"],b["dev"]))
        columns=np.r_[a["category_ids"],b["category_ids"]].tolist()
    spec=json.loads((HERE/"TARGET_REPAIR_STATUS.json").read_text(encoding="utf-8"))["selected"]
    largest=0.
    endpoints=0
    for group in summary["rows"]:
        path=HERE/f"LINEAR_DEV_{group['draw']}_50E_6R.npz"
        with np.load(path,allow_pickle=False) as f:
            assert f["categories"].tolist()==columns
            y=truth[[lookup[i] for i in f["ids"]]]
            x=f["scores"].reshape(6,3,len(y),len(columns))
        expected=np.array(group["directional_advantage_pp"])
        for j,row in enumerate(spec):
            t=columns.index(row["target_id"])
            for d,p in enumerate(row["partners"]):
                p=columns.index(p)
                positive=(y[:,t]==1)&(y[:,p]==0)
                negative=(y[:,t]==0)&(y[:,p]==1)
                weight=np.ones(len(y))
                for r in range(6):
                    u=weighted_auc(x[r,d,:,t],positive,negative,weight)
                    v=weighted_auc(x[r,1-d,:,t],positive,negative,weight)
                    largest=max(largest,abs((u-v)*100-expected[r,j,d]))
        endpoints+=x.shape[0]*x.shape[1]
    assert largest<1e-10
    result=dict(status="PASS",endpoints=endpoints,model_scores_role="DEV subset only",max_interaction_abs_error_pp=largest,
                confirmation_scores_read=0)
    (HERE/"LINEAR_DEV_INDEPENDENT_QA.json").write_text(json.dumps(result,indent=2)+"\n")
    print(result,flush=True)
