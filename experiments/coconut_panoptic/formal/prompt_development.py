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

"""Run the complete paired 133-output DualCoOp development matrix."""
from argparse import ArgumentParser
import copy
import csv
import json
import os
from pathlib import Path
import shutil
from statistics import median
from time import perf_counter,time
from zipfile import ZipFile

import numpy as np
import torch
from prompt_smoke import (ARMS,D,RECOVERY,SEED,WIDTH,complete,loader,normalized,
                          order_for,projected_logits,text_vectors)


def safe_torch(path,data):
    temp=path.with_suffix(".partial")
    torch.save(data,temp)
    temp.replace(path)


def pixel_inputs(work,fixed,a1,bridge,draws):
    training={}
    with fixed.open(encoding="utf-8-sig",newline="") as f:
        for r in csv.DictReader(f):
            if r["role"]=="TRAIN":
                training[r["id"]]=str(RECOVERY/"datasets/coco"/r["existing_subdirectory"]/r["file_name"])
    dev_ids=np.load(a1,allow_pickle=False)["ids"]
    bridge_ids=np.load(bridge,allow_pickle=False)["ids"]
    train_ids=np.array(list(map(int,training)))
    assert len(train_ids)==4008 and len(dev_ids)==len(bridge_ids)==5000
    assert len(set(np.r_[train_ids,dev_ids,bridge_ids]))==14008
    paths={}
    for j in dev_ids.tolist():
        matches=[RECOVERY/"datasets/coco"/p/f"{j:012d}.jpg" for p in ("train2017","val2017")]
        path=next((r for r in matches if r.is_file()),None)
        if path is not None:
            paths[str(j)]=str(path)
    extraction=[]
    pixels_to_write=0
    for ids,current in [(dev_ids,paths),(bridge_ids,{})]:
        scratch=work/("a1_pixels" if ids is dev_ids else "bridge_pixels")
        scratch.mkdir(exist_ok=True)
        missing=[int(i) for i in ids if str(int(i)) not in current]
        for split in ("train2017","val2017") if ids is dev_ids else ("val2017",):
            if not missing:
                break
            archive=Path(os.environ.get("FQ_COCO2017_ROOT", Path(__file__).resolve().parents[3] / "inputs/coco2017"))/f"{split}.zip"
            with ZipFile(archive) as images:
                candidates={i:images.NameToInfo.get(f"{split}/{i:012d}.jpg") for i in missing}
                for i,z in sorted(((i,z) for i,z in candidates.items() if z is not None),key=lambda x:x[1].header_offset):
                    path=scratch/f"{i:012d}.jpg"
                    if not path.exists() or path.stat().st_size!=z.file_size:
                        extraction.append((archive,z,path))
                        pixels_to_write+=z.file_size
                    current[str(i)]=str(path)
            missing=[i for i in missing if str(i) not in current]
        assert not missing
        paths.update(current)
    score_file_bytes=4*12*(len(dev_ids)+len(bridge_ids))*WIDTH
    # Score outputs and an extra scoring retry, prompt tensors and optimizer moments, and the bridge cache.
    resident_score_files=sum(1 for draw in draws if not (work/f"PROMPT_DEV_{draw}.npz").exists())
    checkpoint_bytes=(len(draws)+1)*12*4*2*2*16*512
    bridge_bytes=4*len(bridge_ids)*1024 if not (work/"PANOPTIC_BRIDGE_FEATURES.npz").exists() else 0
    estimated_peak=pixels_to_write+(resident_score_files+1)*(score_file_bytes+65536)+checkpoint_bytes+bridge_bytes
    free=shutil.disk_usage(work).free
    storage_ok=free>=1.25*estimated_peak
    complete(work/"P1_STORAGE_QA.json",dict(status="PASS" if storage_ok else "HOLD_STORAGE",
             free_bytes=free,estimated_added_peak_bytes=estimated_peak,minimum_free_bytes=int(np.ceil(1.25*estimated_peak)),
             pixel_bytes_to_write=pixels_to_write,score_file_bytes=score_file_bytes,output_classes=WIDTH))
    if not storage_ok:
        raise RuntimeError("HOLD_STORAGE")
    # Each archive stays open while its members are extracted in header order.
    from itertools import groupby
    for archive,entries in groupby(extraction,key=lambda task:task[0]):
        with ZipFile(archive) as images:
            for _,info,path in entries:
                temp=path.with_suffix(".partial")
                temp.write_bytes(images.read(info))
                temp.replace(path)
    return train_ids,training,np.r_[dev_ids,bridge_ids],paths


def bridge_features(work,model,all_ids,paths):
    cache=work/"PANOPTIC_BRIDGE_FEATURES.npz"
    bridge=all_ids[5000:]
    if cache.exists():
        with np.load(cache,allow_pickle=False) as file:
            assert np.array_equal(file["ids"],bridge) and file["features"].shape==(len(bridge),1024)
        return
    valid=[]
    seen=[]
    with torch.inference_mode():
        for image,_,ids in loader(bridge,paths,workers=4):
            result=model.encode_global(image.cuda(non_blocking=True))
            assert result.shape==(len(ids),1024) and torch.isfinite(result).all()
            valid.append(result.cpu().numpy())
            seen.extend(ids.tolist())
    assert seen==bridge.tolist()
    temp=cache.with_suffix(".partial")
    with temp.open("wb") as f:
        np.savez(f,ids=bridge,features=np.concatenate(valid))
    temp.replace(cache)


def train_draw(work,model,train_ids,train_pixels,categories,draw,seeds,deadline,text_width,shared_text,state_path):
    with np.load(work/f"PANOPTIC_MASK_{draw}.npz",allow_pickle=False) as data:
        assert data["categories"].tolist()==categories
        row={i:j for j,i in enumerate(data["ids"].tolist())}
        order=[row[i] for i in train_ids.tolist()]
        masks={a:data[a][order] for a in (*ARMS,"LOW")}
    base=int((masks["LOW"]>=0).sum())
    jobs=[]
    for seed in seeds:
        g=torch.Generator().manual_seed(seed)
        value=[torch.empty((16,512)).normal_(0,.02,generator=g) for _ in range(2)]
        for arm in ARMS:
            prompt=copy.deepcopy(model.net.prompt_learner)
            with torch.no_grad():
                for param,init in zip((prompt.ctx_pos,prompt.ctx_neg),value,strict=True):
                    param.copy_(init)
            opt=torch.optim.SGD((prompt.ctx_pos,prompt.ctx_neg),lr=0.,momentum=.9)
            jobs.append((arm,prompt,opt))
    history=[]
    if state_path.exists():
        ck=torch.load(state_path,map_location="cuda",weights_only=True)
        assert ck["draw"]==draw and ck["seeds"]==seeds
        assert ck.get("causal_text_width")==text_width and ck.get("shared_text",False)==shared_text
        history=ck["history"]
        for (_,prompt,opt),(p_state,o_state) in zip(jobs,ck["jobs"],strict=True):
            with torch.no_grad():
                prompt.ctx_pos.copy_(p_state["ctx_pos"])
                prompt.ctx_neg.copy_(p_state["ctx_neg"])
            opt.load_state_dict(o_state)
    ix={i:j for j,i in enumerate(train_ids.tolist())}
    for epoch in range(len(history),50):
        assert time()<deadline-300
        start=perf_counter()
        pixels=loader(train_ids,train_pixels,masks["LOW"],draw,epoch,workers=4)
        seen=[]
        losses=torch.zeros(len(jobs),device="cuda")
        for arm,_,opt in jobs:
            opt.param_groups[0]["lr"]=.002*8976/base*D.factor(epoch)
        for images,_,ids in pixels:
            assert time()<deadline-120
            seen.extend(ids.tolist())
            positions=[ix[i] for i in ids.tolist()]
            with torch.no_grad():
                feat,_=model.net.image_encoder(images.cuda(non_blocking=True))
                feat=normalized(feat,1)
            targets={a:torch.as_tensor(masks[a][positions],device="cuda") for a in ARMS}
            for j,(a,p,o) in enumerate(jobs):
                logits=projected_logits(feat,text_vectors(model.net.text_encoder,p,text_width,shared_text))
                loss=D.loss_fn(logits,targets[a])
                losses[j]+=loss.detach()
                o.zero_grad(set_to_none=True)
                loss.backward()
                o.step()
        del pixels
        assert seen==train_ids[order_for(draw,epoch,len(train_ids))].tolist()
        assert torch.isfinite(losses).all() and all(torch.isfinite(p.ctx_pos).all() and torch.isfinite(p.ctx_neg).all() for _,p,_ in jobs)
        history.append(dict(epoch=epoch+1,steps=126*(epoch+1),loss=losses.tolist(),seconds=perf_counter()-start))
        states=[(dict(ctx_pos=p.ctx_pos.detach(),ctx_neg=p.ctx_neg.detach()),o.state_dict()) for _,p,o in jobs]
        safe_torch(state_path,dict(draw=draw,seeds=seeds,history=history,jobs=states,
                   causal_text_width=text_width,shared_text=shared_text))
        print(draw,epoch+1,history[-1]["seconds"],flush=True)
    return jobs,history


def fit_one(work,model,train_ids,train_pixels,dev_ids,dev_pixels,categories,draw,seeds,deadline,text_width,shared_text):
    result=work/f"PROMPT_DEV_{draw}.npz"
    timing=work/f"PROMPT_DEV_{draw}_TIMING.json"
    if result.exists() and timing.exists():
        with np.load(result,allow_pickle=False) as score:
            assert score["scores"].shape==(12,len(dev_ids),WIDTH) and np.array_equal(score["ids"],dev_ids)
        return json.loads(timing.read_text(encoding="utf-8"))
    state=work/f"PROMPT_DEV_{draw}.pt"
    jobs,history=train_draw(work,model,train_ids,train_pixels,categories,draw,seeds,deadline,text_width,shared_text,state)
    scoring=perf_counter()
    batches=loader(dev_ids,dev_pixels,workers=4)
    score=np.empty((len(jobs),len(dev_ids),WIDTH),np.float32)
    with torch.no_grad():
        contexts=[]
        for _,prompt,_ in jobs:
            contexts.append(text_vectors(model.net.text_encoder,prompt))
        context=torch.cat(contexts)
        seen=[]
        for images,_,ids in batches:
            assert time()<deadline
            j=len(seen)
            seen.extend(ids.tolist())
            feat,_=model.net.image_encoder(images.cuda(non_blocking=True))
            feat=normalized(feat,1)
            raw=20*torch.nn.functional.conv1d(feat,context[:,:,None])
            raw=raw.reshape(len(feat),len(jobs),2,WIDTH,-1)
            wt=raw[:,:,1].softmax(-1)
            logits=5*(raw*wt[:,:,None]).sum(-1)
            if j==0 or len(seen)==len(dev_ids):
                for k,vector in enumerate(contexts):
                    direct=projected_logits(feat,vector)
                    torch.testing.assert_close(logits[:,k],direct,rtol=1e-5,atol=3e-5)
            score[:,j:len(seen)]=logits.softmax(2)[:,:,1].permute(1,0,2).cpu().numpy()
    assert seen==dev_ids.tolist() and np.isfinite(score).all()
    temp=result.with_suffix(".partial")
    with temp.open("wb") as f:
        np.savez(f,ids=dev_ids,categories=categories,scores=score,arms=np.tile(ARMS,4),
                 seeds=np.repeat(np.asarray(seeds,dtype=np.uint64),3))
    temp.replace(result)
    report=dict(draw=draw,endpoints=len(jobs),epochs=len(history),
                visual_and_training_seconds=sum(row["seconds"] for row in history),
                steady_epoch_seconds=median(row["seconds"] for row in history[1:]),
                score_and_seal_seconds=perf_counter()-scoring)
    complete(timing,report)
    return report


if __name__=="__main__":
    parser=ArgumentParser()
    for name in ("work","fixed","a1","bridge"):
        parser.add_argument(name,type=Path)
    parser.add_argument("deadline",type=float)
    parser.add_argument("--gpu-start",type=float)
    parser.add_argument("--budget-hours",type=float,default=16)
    arg=parser.parse_args()
    starting=perf_counter()
    D.setup()
    design=json.loads((arg.work/"PANOPTIC_MASK_SCREEN.json").read_text())
    clas=json.loads((arg.work/"PANOPTIC_CLASS_SPACE_LOCK.json").read_text())["categories"]
    D.CLASSES=[r["query"] for r in clas]
    ids,train,dev,paths=pixel_inputs(arg.work,arg.fixed,arg.a1,arg.bridge,design["dev_seeds"])
    print("FIT",len(ids),"DEV and DEV-BRIDGE",len(dev),"pixel paths bound",flush=True)
    seeds=[int(np.random.SeedSequence([SEED,11,j]).generate_state(1,np.uint64)[0]) for j in (1,2,3,4)]
    model=D.build_dc(fixture_path=arg.work/"PROMPT_SMOKE_INITIAL.pt",init_seed=seeds[0])
    smoke=json.loads((arg.work/"PROMPT_SMOKE_QA.json").read_text())
    for block in model.net.text_encoder.transformer.resblocks:
        block.attn_mask=block.attn_mask.cuda()
    bridge_features(arg.work,model,dev,paths)
    startup_seconds=perf_counter()-starting
    rows=[]
    status="COMPLETE"
    for draw in design["dev_seeds"]:
        rows.append(fit_one(arg.work,model,ids,train,dev,paths,[r["id"] for r in clas],draw,seeds,arg.deadline,
                    smoke.get("causal_text_width"),smoke.get("causal_text_strategy")=="shared"))
        epoch=median(row["steady_epoch_seconds"] for row in rows)
        hold=0 if arg.gpu_start is None else time()-arg.gpu_start
        remainder=len(design["dev_seeds"])-len(rows)
        future=remainder*(50*epoch+median(row["score_and_seal_seconds"] for row in rows))
        tail=1800 if remainder else 0
        projected=(hold+future+tail)/3600/.8
        forecast=dict(completed_masks=len(rows),expected_masks=len(design["dev_seeds"]),
                      gpu_allocation_seconds=hold,remaining_fit_and_score_seconds=future,
                      provisioned_remaining_overhead_seconds=tail,
                      projected_hours_with_recovery=projected,budget_hours=arg.budget_hours,
                      steady_12_endpoint_epoch_seconds=epoch,
                      minimal_12_mask_formal_fit_hours=600*epoch/3600,
                      minimal_joint_development_and_formal_fit_hours=1000*epoch/3600)
        complete(arg.work/"P1_MEASURED_FORECAST.json",forecast)
        print("P1_MEASURED_FORECAST",forecast,flush=True)
        if remainder and arg.gpu_start is not None and projected>arg.budget_hours:
            status="HOLD_BUDGET"
            break
    complete(arg.work/"PROMPT_DEVELOPMENT_QA.json",dict(status=status,development_endpoints=sum(r["endpoints"] for r in rows),
             epochs=50,updates_per_endpoint=6300,development_images=5000,bridge_images=5000,
             preparation_and_bridge_seconds=startup_seconds,runs=rows,confirm_scored=0))
