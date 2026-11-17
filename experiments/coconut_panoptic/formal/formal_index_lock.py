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

"""Prepare the registered confirmation image indices and random streams before scoring."""
from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np


SEEDS={"M1":(93120261115,31,31),"M2":(93120261115,31,32),
       "paired_comparison":(93120261115,31,33),"M3_source":(93120261115,31,34),
       "M3":(93120261115,31,35)}
IMAGE_ROLES=("M1","M2","paired_comparison","M3")
RECEIPT="FORMAL_INDEX_SEED_LOCK.json"
INDICES="FORMAL_IMAGE_REPLICA_INDICES.npz"


def generator(role):
    return np.random.default_rng(np.random.SeedSequence(SEEDS[role]))


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def baseline_sample(signature,population_strata):
    rng=np.random.default_rng(np.random.SeedSequence(signature["seed_sequence"]))
    return np.concatenate([rng.choice(np.flatnonzero(population_strata==h),nh,replace=False)
                           for h,nh in enumerate(signature["quota"])])


def prepare(work,reps):
    protocol=read(work/"PROTOCOL_LOCK.json")
    sample=read(work.parent/"SAMPLING_LOCK.json")
    signature=sample["signature"]
    assert not list(work.glob("M?_confirm_*.npz"))
    assert protocol["status"]=="READY" and protocol["confirm_scores_read"]==0
    assert len(protocol["targets"])==4 and len(protocol["M1"]["draws"])==16 and len(protocol["M1"]["seeds"])==6
    with np.load(work/"CONF_M_LABELS.npz",allow_pickle=False) as labels:
        ids,strata,totals,quota=(labels[name] for name in ("ids","strata","totals","quotas"))
        assert labels["categories"].tolist()==protocol["categories"]
    assert ids.tolist()==protocol["confirm_ids"]==sample["sample_ids"]
    assert strata.tolist()==sample["sample_strata"] and totals.tolist()==signature["population_by_stratum"]
    assert quota.tolist()==signature["quota"]
    population_strata=np.asarray(sample["population_strata"])
    np.testing.assert_array_equal(np.bincount(population_strata),totals)
    selection=baseline_sample(signature,population_strata)
    np.testing.assert_array_equal(np.asarray(sample["population_ids"])[selection],ids)
    np.testing.assert_array_equal(population_strata[selection],strata)
    np.testing.assert_array_equal(np.bincount(strata),quota)

    index_dtype=np.uint16 if len(ids)<=np.iinfo(np.uint16).max+1 else np.uint32
    rows=dict(ids=ids,strata=strata,totals=totals,quota=quota,population_positions=selection)
    states={}
    for role in IMAGE_ROLES:
        rng=generator(role)
        for h,(N,n) in enumerate(zip(totals,quota,strict=True)):
            group=np.flatnonzero(strata==h)
            if N==n:
                continue
            assert 2<=n<N
            positions=np.empty((reps,n//2),dtype=index_dtype)
            for row in positions:
                row[:]=rng.choice(group,n//2,replace=False,shuffle=False)
            rows[f"{role}_{h}"]=positions
        states[role]=rng.bit_generator.state
    path=work/INDICES
    temp=path.with_suffix(".partial")
    with temp.open("wb") as stream:
        np.savez(stream,**rows)
    temp.replace(path)
    receipt=dict(status="FROZEN_BEFORE_CONFIRMATION_SCORING",run_id=protocol["run_id"],replicates=reps,
                 image_ids=ids.tolist(),stratum_population=totals.tolist(),stratum_sample=quota.tolist(),
                 sample_seed_sequence=signature["seed_sequence"],seed_sequences={k:list(v) for k,v in SEEDS.items()},
                 model_training_seeds={name:protocol[name]["seeds"] for name in ("M1","M2")},
                 post_image_rng_states=states,index_bytes=path.stat().st_size,confirmation_scores_read=0)
    dest=work/RECEIPT
    stage=dest.with_suffix(".partial")
    stage.write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf-8")
    stage.replace(dest)
    print(receipt["status"],len(ids),reps,receipt["index_bytes"],flush=True)


def load_index(work,role,lock,ids,strata,totals,reps):
    receipt=read(work/RECEIPT)
    assert receipt["status"]=="FROZEN_BEFORE_CONFIRMATION_SCORING" and receipt["confirmation_scores_read"]==0
    assert receipt["run_id"]==lock["run_id"] and receipt["replicates"]==reps
    assert receipt["image_ids"]==ids.tolist() and receipt["seed_sequences"][role]==list(SEEDS[role])
    assert receipt["model_training_seeds"]=={name:lock[name]["seeds"] for name in ("M1","M2")}
    assert (work/INDICES).stat().st_size==receipt["index_bytes"]
    with np.load(work/INDICES,allow_pickle=False) as data:
        for name,actual in (("ids",ids),("strata",strata),("totals",totals)):
            np.testing.assert_array_equal(data[name],actual)
        positions={h:data[f"{role}_{h}"] for h in range(len(totals)) if f"{role}_{h}" in data}
    return positions,receipt["post_image_rng_states"][role]


if __name__=="__main__":
    cli=ArgumentParser()
    cli.add_argument("work",type=Path)
    prepare(cli.parse_args().work,5000)
