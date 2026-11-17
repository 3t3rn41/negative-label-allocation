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

"""Shared RN50 or ViT training, fixed-pool scoring, and RN50 linear-feature extraction."""
import csv
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time
from zipfile import ZipFile

import numpy as np
sys.path[:0] = [str(Path(__file__).resolve().parents[1]/subdir) for subdir in ('coco_fixed_partner','coco_crossed')]
import pilot as P
import formal_train_shared as F

HERE=Path(__file__).resolve().parent
TORCH=P.torch
S0=Path(os.environ.get('FQ_COCO_S0_WORK', Path(__file__).resolve().parents[1] / 'coco_fixed_partner'))
B=Path(os.environ.get('FQ_COCO_B_WORK', Path(__file__).resolve().parents[1] / 'coco_crossed'))
ARCHIVE=Path(os.environ.get('FQ_COCO2017_ROOT',Path(__file__).resolve().parents[2] / "inputs/coco2017"))/'train2017.zip'
CACHE=Path(__file__).resolve().parent / "pixel_cache"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def train_inputs(role):
    paths={}
    manifest=S0/"PILOT_PIXEL_MANIFEST.csv"
    with manifest.open(encoding="utf-8-sig",newline="") as stream:
        for row in csv.DictReader(stream):
            if row["role"]==role:
                path=P.RECOVERY/"datasets/coco"/row["existing_subdirectory"]/row["file_name"]
                assert path.is_file()
                paths[row["id"]]=str(path)
    return np.array(list(map(int,paths))),paths


def evaluation_inputs(pool,expected):
    path=B/"B_PIXEL_BINDING.csv" if pool=="B" else S0/"CONFIRM_PIXEL_BINDING.csv"
    with path.open(encoding="utf-8-sig",newline="") as stream:
        rows=list(csv.DictReader(stream))
    ids=np.array([int(row["id"]) for row in rows])
    assert ids.tolist()==expected
    scratch=CACHE/pool
    scratch.mkdir(parents=True,exist_ok=True)
    missing=[row for row in rows if not (scratch/f'{int(row["id"]):012d}.jpg').is_file()]
    if missing:
        assert shutil.disk_usage(scratch).free>sum(int(r["bytes"]) for r in missing)+100_000_000
        with ZipFile(ARCHIVE) as archive:
            for row in sorted(missing,key=lambda x:archive.getinfo(x["member"]).header_offset):
                payload=archive.read(row["member"])
                assert len(payload)==int(row["bytes"])
                path=scratch/f'{int(row["id"]):012d}.jpg'
                tmp=path.with_suffix(".tmp")
                tmp.write_bytes(payload)
                tmp.replace(path)
    paths={row["id"]:str(scratch/f'{int(row["id"]):012d}.jpg') for row in rows}
    for row in rows:
        assert Path(paths[row["id"]]).stat().st_size==int(row["bytes"])
    return ids,paths


def score(model,ids,paths,jobs,deadline,backend):
    out=np.empty((len(jobs),len(ids),20),dtype=np.float32)
    seen=[]
    started=time.monotonic()
    with TORCH.inference_mode():
        heads=[job[3] for job in jobs]
        if backend=="ViT-B/16":
            vectors=[head.text() for head in heads]
        else:
            vectors=[]
            for h in heads:
                prompts,tokens=h.learner()
                vec=h.encoder(prompts,tokens)
                vectors.append(vec/vec.norm(dim=-1,keepdim=True))
        for batch,(pixels,_,ii) in enumerate(F.epoch_loader(ids,paths,workers=6)):
            assert time.time()<deadline and not F.STOP
            first=len(seen)
            seen.extend(ii.tolist())
            feature,_=model.net.image_encoder(pixels.cuda(non_blocking=True))
            if backend=="RN50":
                image=feature/feature.norm(dim=1,keepdim=True)
            for k,(head,text) in enumerate(zip(heads,vectors,strict=True)):
                if backend=="ViT-B/16":
                    logits=P.projected_logits(feature,text)
                else:
                    raw=20*TORCH.nn.functional.conv1d(image,text[:,:,None])
                    n,c,_=raw.shape
                    w=raw[:,c//2:].softmax(-1)
                    logits=(5*(raw*TORCH.cat([w,w],1)).sum(-1)).reshape(n,2,c//2)
                if batch==0 or len(seen)==len(ids):
                    TORCH.testing.assert_close(logits,head(feature),rtol=0,atol=0)
                out[k,first:len(seen)]=logits.softmax(1)[:,1].cpu().numpy()
            if batch%40==0:
                print("SCORE",len(seen),"/",len(ids),"heads",len(jobs),flush=True)
    assert seen==ids.tolist() and np.isfinite(out).all() and ((out>=0)&(out<=1)).all()
    return out,dict(rows=len(ids),heads=len(jobs),seconds=time.monotonic()-started,parity="exact_first_and_tail")


def load_state(path,draw,arms,seeds,epoch):
    value=TORCH.load(path/f"draw_{draw}"/"CHECKPOINT.pt",map_location="cpu",weights_only=False)
    assert (value["draw"],value["arms"],value["seeds"],value["epoch"],value["steps"])==(draw,arms,seeds,epoch,126*epoch)
    assert len(value["heads"])==len(arms)*len(seeds) and len(value["history"])==epoch
    return value


def heads(model,states,old_low):
    jobs=[]
    for s in states:
        for (seed,arm,head,_),params in zip(F.make_jobs(model,s["seeds"],s["arms"]),s["heads"],strict=True):
            F.restore_head(head,params["parameters"])
            jobs.append((s["draw"],seed,arm,head))
        if old_low:
            old=load_state(S0,s["draw"],list(P.ARMS),s["seeds"],50)
            k=old["arms"].index("LOW")
            for (seed,arm,head,_),i in zip(F.make_jobs(model,s["seeds"],["LOW"]),range(len(s["seeds"])),strict=True):
                F.restore_head(head,old["heads"][i*len(old["arms"])+k]["parameters"])
                jobs.append((s["draw"],seed,arm,head))
    return jobs


def atomic_npz(path,**arrays):
    tmp=path.with_suffix(".tmp")
    with tmp.open("wb") as stream:
        np.savez(stream,**arrays)
    tmp.replace(path)


def configure_bce():
    from bce_loss import observed_bce
    make=F.make_jobs
    step=P.step
    def tagged(model,seeds,arms):
        jobs=make(model,seeds,arms)
        for _,arm,head,_ in jobs:
            head.use_observed_bce=arm=="T2_PBCE"
        return jobs
    def mixed_step(head,optimizer,features,target,known):
        if not head.use_observed_bce:
            return step(head,optimizer,features,target,known)
        optimizer.zero_grad(set_to_none=True)
        loss=observed_bce(head(features),target,known)
        loss.backward()
        optimizer.step()
        return TORCH.stack([loss.detach(),*(p.grad.detach().abs().amax() for p in P.params(head))])
    F.make_jobs=tagged
    P.step=mixed_step


def pilot_or_formal(mode,protocol,model,deadline):
    arms,seeds,draws=protocol["arms"],protocol["seeds"],protocol["draws"]
    assert not (HERE/"B_GOLD.npz").is_file()
    train,paths=train_inputs("TRAIN")
    assert len(train)==4008 and not set(train)&set(protocol["b_ids"])
    prefix=protocol["job"]
    pilot=mode=="pilot"
    assert pilot or read(HERE/"PILOT_QA.json")["status"]=="PASS"
    if not pilot and (HERE/"SCORE_RECEIPT.json").is_file():
        return
    TORCH.cuda.reset_peak_memory_stats()
    for draw in draws[:1] if pilot else draws:
        assert F.train_draw(model,train,paths,HERE/f"MASK_{draw}.npz",draw,seeds,3,deadline,
                           arms=arms,limit=1 if pilot else 50)
    states=[load_state(HERE,draw,arms,seeds,1 if pilot else 50) for draw in (draws[:1] if pilot else draws)]
    peak=TORCH.cuda.max_memory_allocated()
    jobs=heads(model,states,protocol["score_original_low"] and not pilot)
    if pilot:
        ids,paths=train_inputs("DEVELOPMENT")
        assert not set(ids)&set(protocol["b_ids"])
        dev=ids[:160]
        score_array,perf=score(model,dev,paths,jobs,deadline,protocol["backend"])
        assert score_array.shape==(len(arms)*12,160,20)
        secs=states[0]["history"][0]["seconds"]
        F.atomic_json(HERE/"PILOT_QA.json",dict(status="PASS",device=TORCH.cuda.get_device_name(),
            job=prefix,epoch_seconds=secs,remaining_seconds_estimate=99*secs+100*perf["seconds"]+600,
            peak_bytes=peak,formal_images_scored=0,dev_scoring=perf))
        return
    F.atomic_json(HERE/"TRAIN_QA.json",dict(status="COMPLETE",job=prefix,device=TORCH.cuda.get_device_name(),
        endpoints=len(arms)*24,updates=len(arms)*24*6300,peak_bytes=peak,
        histories=[dict(draw=s["draw"],history=s["history"]) for s in states]))
    ids,paths=evaluation_inputs("B",protocol["b_ids"])
    vals,perf=score(model,ids,paths,jobs,deadline,protocol["backend"])
    d,i,a=zip(*[(d,s,a) for d,s,a,_ in jobs],strict=True)
    atomic_npz(HERE/"SCORES.npz",ids=ids,classes=np.array(P.D.CLASSES),scores=vals,draws=d,seeds=i,arms=a)
    F.atomic_json(HERE/"SCORE_RECEIPT.json",dict(status="SCORES_SEALED",job=prefix,truth_loaded=False,
        sealed_unix=time.time(),shape=list(vals.shape),**perf))


def extract(protocol,model,deadline):
    assert protocol["backend"]=="RN50" and not (HERE/"A1_GOLD.npz").is_file()
    ids_fit,p_fit=train_inputs("TRAIN")
    assert len(ids_fit)==4008 and not set(ids_fit)&set(protocol["a1_ids"])
    for label,ids,pixels in [("fit",ids_fit,p_fit),("a1",*evaluation_inputs("A1",protocol["a1_ids"]))]:
        destination=HERE/f"T3_{label}_features.npz"
        if destination.is_file():
            with np.load(destination) as d:
                assert np.array_equal(d["ids"],ids) and d["features"].shape==(len(ids),1024)
            continue
        out=[]
        seen=[]
        with TORCH.inference_mode():
            for index,(batch,_,ii) in enumerate(F.epoch_loader(ids,pixels,workers=8)):
                assert time.time()<deadline and not F.STOP
                image=batch.cuda(non_blocking=True)
                value=model.encode_global(image)
                if index==0:
                    torch_repeat=model.encode_global(image)
                    TORCH.testing.assert_close(value,torch_repeat,rtol=0,atol=0)
                out.append(value.cpu().numpy())
                seen.extend(ii.tolist())
        assert seen==ids.tolist()
        f=np.concatenate(out)
        assert f.shape==(len(ids),1024) and np.isfinite(f).all()
        atomic_npz(destination,ids=ids,classes=np.array(P.D.CLASSES),features=f)
    F.atomic_json(HERE/"T3_FEATURE_RECEIPT.json",dict(status="FEATURES_SEALED",truth_loaded=False,
        fit_rows=len(ids_fit),a1_rows=len(protocol["a1_ids"]),features=1024,
        backend="RN50",same_transform=True,repeat="exact_first_batch",sealed_unix=time.time()))


def main():
    mode=sys.argv[1]
    protocol=read(HERE/"RUN_PROTOCOL.json")
    signal.signal(signal.SIGTERM,F.request_stop)
    assert protocol["status"]=="LOCKED"
    P.D.setup()
    if protocol["backend"]=="RN50":
        P.seed_all()
        model=P.D.build_dc(init_seed=protocol["seeds"][0])
        if "T2_PBCE" in protocol["arms"]:
            configure_bce()
    else:
        TORCH.manual_seed(protocol["seeds"][0])
        model=P.Native().cuda().eval()
    deadline=time.time()+protocol["worker_time_limit_seconds"]
    if mode=="features":
        extract(protocol,model,deadline)
    else:
        assert mode in ("pilot","formal")
        pilot_or_formal(mode,protocol,model,deadline)


if __name__=="__main__":
    main()
