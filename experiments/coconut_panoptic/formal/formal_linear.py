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

"""Use the frozen RN50 and observable-label linear recipe for formal M2."""
from argparse import ArgumentParser
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from linear_development import ARMS, WIDTH, calculate_scores, features, fit_weights, smoke


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path,receipt):
    temp=path.with_suffix(".partial")
    temp.write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf-8")
    temp.replace(path)


def seal_arrays(path,**arrays):
    temp=path.with_suffix(".partial")
    with temp.open("wb") as stream:
        np.savez(stream,**arrays)
    temp.replace(path)


def load_weights(path,categories,seeds):
    with np.load(path,allow_pickle=False) as file:
        assert file["categories"].tolist()==categories and file["seeds"].tolist()==seeds
        assert int(file["updates"])==6300
        w,b=(torch.from_numpy(file[name]) for name in ("w","b"))
    columns=len(ARMS)*len(seeds)*WIDTH
    assert w.shape==(1024,columns) and b.shape==(columns,)
    assert torch.isfinite(w).all() and torch.isfinite(b).all()
    return w,b


def train(work,fit_cache,lock):
    m2=lock["M2"]
    draws,seeds=m2["draws"],m2["seeds"]
    categories=lock["categories"]
    fit_ids,x=features(fit_cache)
    assert set(fit_ids.tolist())==set(lock["fit_ids"]) and len(fit_ids)==4008
    positions={img:i for i,img in enumerate(fit_ids.tolist())}
    started=perf_counter()
    for draw in draws:
        path=work/f"M2_DRAW_{draw}.npz"
        if path.is_file():
            load_weights(path,categories,seeds)
            continue
        with np.load(work/f"PANOPTIC_MASK_{draw}.npz",allow_pickle=False) as mask:
            assert mask["categories"].tolist()==categories and set(mask["ids"].tolist())==set(positions)
            shuffled=x[[positions[i] for i in mask["ids"].tolist()]]
            labels=torch.from_numpy(np.tile(np.stack([mask[a] for a in ARMS],axis=1),(1,len(seeds),1))).float()
        w,b=fit_weights(shuffled,labels,seeds,draw,50)
        seal_arrays(path,draw=np.asarray(draw,dtype=np.uint64),seeds=np.asarray(seeds,dtype=np.uint64),
                    categories=categories,w=w.numpy(),b=b.numpy(),updates=np.asarray(6300))
    write_json(work/"M2_TRAIN_QA.json",dict(status="COMPLETE",fit_role="FIT",endpoints=3*len(draws)*len(seeds),
              wall_seconds=perf_counter()-started,draws=draws,gradient_oracle="PASS",updates_per_endpoint=6300))


def score(work,phase,lock):
    m2=lock["M2"]
    draws,seeds=m2["draws"],m2["seeds"]
    categories=lock["categories"]
    assert load_json(work/"M2_TRAIN_QA.json")["status"]=="COMPLETE"
    if phase=="confirm":
        bound=load_json(work/"M3_PREDICTIONS_BEFORE_SCORING.json")
        assert bound["status"]=="SEALED_SOURCE_PREDICTIONS" and bound["run_id"]==lock["run_id"]
        cache=work/"CONF_M_FEATURES.npz"
        expected=lock["confirm_ids"]
    else:
        cache=work.parent/"PANOPTIC_BRIDGE_FEATURES.npz"
        expected=lock["bridge_ids"]
    ids,x=features(cache)
    assert ids.tolist()==expected and not set(expected)&set(lock["fit_ids"])
    started=perf_counter()
    order=np.repeat(np.asarray(seeds,dtype=np.uint64),3)
    arms=np.tile(ARMS,len(seeds))
    receipt_path=work/f"M2_{phase.upper()}_SCORE_RECEIPT.json"
    sealed=receipt_path.is_file()
    if sealed:
        prior=load_json(receipt_path)
        assert prior["status"]=="SCORES_SEALED" and prior["phase"]==phase and prior["truth_loaded"] is False
        assert prior["ids"]==expected and prior["draws"]==draws and prior["endpoints"]==len(draws)*len(arms)
    for draw in draws:
        path=work/f"M2_{phase}_{draw}.npz"
        if path.exists():
            with np.load(path,allow_pickle=False) as existing:
                np.testing.assert_array_equal(existing["ids"],ids)
                np.testing.assert_array_equal(existing["seeds"],order)
                np.testing.assert_array_equal(existing["arms"],arms)
                assert existing["categories"].tolist()==categories
                assert existing["scores"].shape==(len(arms),len(ids),WIDTH)
            continue
        assert not sealed,"Sealed M2 score file is missing"
        w,b=load_weights(work/f"M2_DRAW_{draw}.npz",categories,seeds)
        values=calculate_scores(x,w,b,len(seeds))
        assert np.isfinite(values).all()
        seal_arrays(path,ids=ids,scores=values,categories=categories,seeds=order,arms=arms)
    if not sealed:
        write_json(receipt_path,dict(status="SCORES_SEALED",phase=phase,ids=expected,
                   draws=draws,endpoints=len(draws)*len(arms),wall_seconds=perf_counter()-started,truth_loaded=False))


if __name__=="__main__":
    parser=ArgumentParser()
    parser.add_argument("work",type=Path)
    parser.add_argument("phase",choices=("fit","bridge","confirm"))
    parser.add_argument("--fit-cache",type=Path)
    args=parser.parse_args()
    lock=load_json(args.work/"PROTOCOL_LOCK.json")
    assert lock["status"]=="READY" and lock["version"]=="v9.3.1-coco133" and len(lock["targets"])>=4
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.phase=="fit":
        assert args.fit_cache
        smoke()
        train(args.work,args.fit_cache,lock)
    else:
        score(args.work,args.phase,lock)
