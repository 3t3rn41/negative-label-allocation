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

"""Project 53 stuff labels on the previously lineage-qualified COCONut population."""
from argparse import ArgumentParser
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from panoptic_screen import HERE, PREVIOUS, coconut


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("parquet",type=Path)
    path = parser.parse_args().parquet
    started = perf_counter()
    with np.load(PREVIOUS/"POPULATION80_VALIDATED.npz",allow_pickle=False) as x:
        ids,y,c = x["ids"],x["truth"],x["category_ids"]
    new = np.load(HERE/"PANOPTIC_OFFICIAL_LABELS.npz",allow_pickle=False)["category_ids"]
    paths = sorted(path.glob("train-*-of-00004.parquet"))
    assert len(paths) == 4
    stuff = coconut(paths,ids,new.tolist())
    valid = (stuff>=0).all(1)
    y = np.column_stack((y[valid],stuff[valid]))
    assert y.shape[1] == 133 and np.isin(y,(0,1)).all()
    counts = y.sum(0)
    assert ((counts>0)&(counts<len(y))).all()
    np.savez_compressed(HERE/"PANOPTIC_POPULATION133.npz",ids=ids[valid],truth=y,categories=np.r_[c,new])
    result = dict(status="PASS_BASE_POPULATION", input_population=len(ids), complete_population=len(y),
                  excluded_unknown=int((~valid).sum()), excluded_ids=ids[~valid].tolist(),
                  ap_supported_categories=133, confirmation_model_scores_read=0,
                  base_lineage="experiments/coco_reference_labels/POPULATION_LOCK.json", wall_seconds=perf_counter()-started)
    (HERE/"PANOPTIC_POPULATION_QA.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print({k:v for k,v in result.items() if k!="excluded_ids"},flush=True)
