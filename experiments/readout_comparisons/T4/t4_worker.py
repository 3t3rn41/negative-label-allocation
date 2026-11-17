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

"""Train and seal a LOW-only VOC replication using the frozen v9.2.2 runner."""
import json
from pathlib import Path
import random
import signal
import sys
import time

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'voc_training/worker'))
import pilot as P
import formal_train_shared as F
import worker as W

HERE = Path(__file__).resolve().parent
TORCH = P.torch


def main(mode):
    protocol = json.loads((HERE/"T4_PROTOCOL.json").read_text())
    assert protocol["arm"] == "LOW" and tuple(P.ARMS) == ("LOW",)
    assert protocol["draws"] == P.RULES["draws"] and protocol["seeds"] == P.RULES["initializations"]
    role = W.read("DATA_ROLE_LOCK.json")
    assert role["classes"] == P.D.CLASSES and role["status"] == "PASS_DATA_QUALIFICATION"
    assert role["role_ids"]["test"] == protocol["evaluation_ids"]
    assert not (HERE/"TEST_GOLD.npz").exists()
    signal.signal(signal.SIGTERM,F.request_stop)
    random.seed(90921100)
    np.random.seed(90921100)
    TORCH.manual_seed(90921100)
    TORCH.cuda.manual_seed_all(90921100)
    P.D.setup()
    torch_device=TORCH.cuda.get_device_name()
    print("DEVICE",torch_device,flush=True)
    deadline = time.time() + protocol["worker_time_limit_seconds"]
    model=P.Native().cuda().eval()
    fit, paths=W.inputs("fit",role)
    assert len(fit)==4008 and all(Path(x).is_file() for x in paths.values())
    assert not set(fit) & set(map(int,protocol["evaluation_ids"]))
    peak={}
    if mode=="pilot":
        W.parity(model,role)
        TORCH.cuda.reset_peak_memory_stats()
        draw=protocol["draws"][0]
        assert F.train_draw(model,fit,paths,HERE/f"MASK_{draw}.npz",draw,protocol["seeds"],3,deadline,
                            arms=("LOW",),limit=1)
        state=read_checkpoint(draw,1,protocol)
        peak["train_bytes"]=TORCH.cuda.max_memory_allocated()
        dev,dev_paths=W.inputs("dev",role)
        jobs=restored_jobs(model,[state],protocol)
        score,perf=W.evaluate(model,dev[:192],dev_paths,jobs,deadline)
        assert score.shape==(12,192,20)
        estimate=99*state["history"][0]["seconds"]+5*perf["seconds"]+300
        F.atomic_json(HERE/"T4_PILOT.json",dict(status="PASS",gpu=torch_device,
             epoch_seconds=state["history"][0]["seconds"],remaining_seconds_estimate=estimate,
             scoring=perf,peak=peak,formal_images_scored=0))
        return
    assert mode=="formal" and W.read("T4_PILOT.json")["status"]=="PASS"
    if (HERE/"T4_SCORE_RECEIPT.json").is_file():
        assert (HERE/"T4_SCORES.npz").is_file()
        return
    for draw in protocol["draws"]:
        assert F.train_draw(model,fit,paths,HERE/f"MASK_{draw}.npz",draw,protocol["seeds"],3,deadline,
                            arms=("LOW",),limit=50)
    states=[read_checkpoint(d,50,protocol) for d in protocol["draws"]]
    F.atomic_json(HERE/"T4_TRAIN_QA.json",dict(status="COMPLETE",device=torch_device,
        endpoints=24,updates=151200,draws=[dict(draw=s["draw"],history=s["history"]) for s in states]))
    ids, paths=W.inputs("test",role)
    jobs=restored_jobs(model,states,protocol)
    score,perf=W.evaluate(model,ids,paths,jobs,deadline)
    draws,seeds,arms=zip(*[(d,s,a) for d,s,a,_ in jobs],strict=True)
    assert score.shape==(24,4952,20)
    W.save_npz("T4_SCORES.npz",ids=np.array(protocol["evaluation_ids"]),classes=np.array(P.D.CLASSES),
               scores=score,draws=draws,seeds=seeds,arms=arms)
    F.atomic_json(HERE/"T4_SCORE_RECEIPT.json",dict(status="SCORES_SEALED",truth_loaded=False,
          sealed_unix=time.time(),shape=list(score.shape),**perf))


def read_checkpoint(draw,epoch,protocol):
    state=TORCH.load(HERE/f"draw_{draw}"/"CHECKPOINT.pt",map_location="cpu",weights_only=False)
    assert state["draw"]==draw and state["seeds"]==protocol["seeds"] and state["arms"]==["LOW"]
    assert state["epoch"]==epoch and state["steps"]==126*epoch and len(state["heads"])==12
    return state


def restored_jobs(model,states,protocol):
    jobs=[]
    for state in states:
        for (seed,arm,head,_),saved in zip(F.make_jobs(model,protocol["seeds"],("LOW",)),state["heads"],strict=True):
            F.restore_head(head,saved["parameters"])
            jobs.append((state["draw"],seed,arm,head))
    return jobs


if __name__=="__main__":
    main(sys.argv[1])
