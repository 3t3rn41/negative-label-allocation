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

"""Verify fixed class order, target coordinates and development/confirmation separation."""
import json
from pathlib import Path

import numpy as np


HERE=Path(__file__).resolve().parent
V9=HERE.parent


if __name__=="__main__":
    old=V9/"coco_reference_labels"
    projection=np.load(HERE/"PANOPTIC_OFFICIAL_LABELS.npz",allow_pickle=False)
    bridge=np.load(HERE/"PANOPTIC_BRIDGE_LABELS.npz",allow_pickle=False)
    population=np.load(HERE/"PANOPTIC_POPULATION133.npz",allow_pickle=False)
    lock=json.loads((HERE/"PANOPTIC_CLASS_SPACE_LOCK.json").read_text())
    ids=[row["id"] for row in lock["categories"]]
    assert len(ids)==133 and sorted(ids)==ids
    assert population["categories"].tolist()==bridge["categories"].tolist()==ids
    prior_fit=np.load(old/"FIT80.npz",allow_pickle=False)
    prior_dev=np.load(old/"DEV80.npz",allow_pickle=False)
    np.testing.assert_array_equal(prior_fit["ids"],projection["fit_ids"])
    np.testing.assert_array_equal(prior_dev["ids"],projection["dev_ids"])
    np.testing.assert_array_equal(projection["bridge_ids"],bridge["ids"])
    groups=[set(x.tolist()) for x in (prior_fit["ids"],prior_dev["ids"],bridge["ids"],population["ids"])]
    from itertools import combinations
    for a,b in combinations(groups,2):
        assert not a.intersection(b)
    masks=list(HERE.glob("PANOPTIC_MASK_*.npz"))
    assert len(masks)==56
    for path in masks:
        with np.load(path,allow_pickle=False) as f:
            assert f["categories"].tolist()==ids
            np.testing.assert_array_equal(f["ids"],prior_fit["ids"])
    result=dict(status="PASS",class_order=ids,fit_images=len(groups[0]),dev_images=len(groups[1]),
                bridge_images=len(groups[2]),population_images=len(groups[3]),masks=len(masks),image_roles_disjoint=True)
    (HERE/"MODEL_INPUT_QA.json").write_text(json.dumps(result,indent=2)+"\n")
    print({k:v for k,v in result.items() if k!="class_order"},flush=True)
