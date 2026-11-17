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

"""133-output native DualCoOp development throughput, scoring and restart smoke."""
from argparse import ArgumentParser
import copy
import csv
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from zipfile import ZipFile

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
RECOVERY = Path(os.environ.get("FQ_RECOVERY_ROOT", Path(__file__).resolve().parents[2] / "runtime"))
sys.path[:0] = [str(RECOVERY/p) for p in ("src","vendor","vendor/randaugment")]

import dc_model as D
from data import AUG_SEED, Pixels, order_for
import numpy as np
import torch
from torch.nn import functional as F


ARMS = ("DIR1","DIR2","PLC")
WIDTH = 133
SEED = 93120261115


def configure_loader_process():
    if os.name!="nt":
        import resource

        soft,hard=resource.getrlimit(resource.RLIMIT_NOFILE)
        permitted=65536 if hard==resource.RLIM_INFINITY else min(65536,hard)
        if soft<permitted:
            resource.setrlimit(resource.RLIMIT_NOFILE,(permitted,hard))


def loader(ids,paths,visible=None,seed=0,epoch=0,workers=4):
    dataset=Pixels(ids,paths,visible,seed,epoch)
    generator=torch.Generator().manual_seed(int(np.random.SeedSequence([AUG_SEED,seed,epoch,1]).generate_state(1)[0]))
    order=order_for(seed,epoch,len(ids)) if visible is not None else list(range(len(ids)))
    # Each invocation serves one epoch. Joining its workers avoids the pin-memory exit-handler leak.
    return torch.utils.data.DataLoader(dataset,batch_size=32,sampler=order,num_workers=workers,pin_memory=True,
                                      drop_last=False,persistent_workers=False,
                                      prefetch_factor=2 if workers else None,generator=generator)


def complete(path,value):
    tmp = path.with_suffix(".partial")
    tmp.write_text(json.dumps(value,indent=2)+"\n")
    tmp.replace(path)


def normalized(x,dim):
    return x / x.norm(dim=dim,keepdim=True)


def shared_prefix_block(block,prefix,suffix):
    attention=block.attn
    n,s,d=suffix.shape
    p=len(prefix)
    heads=attention.num_heads
    dh=d//heads
    w,b=attention.in_proj_weight,attention.in_proj_bias
    q,k,v=F.linear(block.ln_1(prefix),w,b).reshape(p,3,heads,dh).permute(1,2,0,3)
    sq,sk,sv=F.linear(block.ln_1(suffix),w,b).reshape(n,s,3,heads,dh).permute(2,0,3,1,4)
    shared=F.scaled_dot_product_attention(q,k,v,is_causal=True).transpose(0,1).reshape(p,d)
    key=torch.cat((k.expand(n,-1,-1,-1),sk),dim=2)
    value=torch.cat((v.expand(n,-1,-1,-1),sv),dim=2)
    mask=block.attn_mask[p:p+s,:p+s].to(device=suffix.device,dtype=suffix.dtype).contiguous()
    varied=F.scaled_dot_product_attention(sq,key,value,attn_mask=mask).transpose(1,2).reshape(n,s,d)
    prefix=prefix+attention.out_proj(shared)
    suffix=suffix+attention.out_proj(varied)
    return prefix+block.mlp(block.ln_2(prefix)),suffix+block.mlp(block.ln_2(suffix))


def text_vectors(encoder,prompt,width=None,shared=False):
    text,tokens=prompt()
    if width is None:
        return normalized(encoder(text,tokens),1)
    x=text[:,:width]+encoder.positional_embedding[:width].type(encoder.dtype)
    positions=tokens.argmax(dim=-1)
    rows=torch.arange(len(x),device=x.device)
    if shared:
        counts=(1+prompt.n_ctx_neg,1+prompt.n_ctx_pos)
        results=[]
        for start,count in zip((0,prompt.n_cls),counts,strict=True):
            stop=start+prompt.n_cls
            prefix,suffix=x[start,:count],x[start:stop,count:]
            for block in encoder.transformer.resblocks:
                prefix,suffix=shared_prefix_block(block,prefix,suffix)
            results.append(suffix[rows[:prompt.n_cls],positions[start:stop]-count])
        selected=encoder.ln_final(torch.cat(results)).type(encoder.dtype)
        return normalized(selected @ encoder.text_projection,1)
    # Causal attention makes tokens after EOT irrelevant to the selected output.
    # Slice each block's mask with the sequence and restore the native mask after its forward call.
    x=x.transpose(0,1)
    for block in encoder.transformer.resblocks:
        full=block.attn_mask
        assert full is not None
        block.attn_mask=full[:width,:width].contiguous()
        try:
            x=block(x)
        finally:
            block.attn_mask=full
    x=encoder.ln_final(x.transpose(0,1)).type(encoder.dtype)
    selected=x[rows,positions]
    return normalized(selected @ encoder.text_projection,1)


def projected_logits(image,text):
    classes=len(text)//2
    logits=20*torch.nn.functional.conv1d(image,text[:,:,None])
    weight=logits[:,classes:].softmax(-1)
    return (5*(logits*torch.cat((weight,weight),1)).sum(-1)).reshape(len(image),2,classes)


def text_parity(model,prompt,image,target,strategy):
    encoder=model.net.text_encoder
    with torch.no_grad():
        inputs,tokens=prompt()
        if strategy=="shared":
            for x,k in zip(inputs.chunk(2),(1+prompt.n_ctx_neg,1+prompt.n_ctx_pos),strict=True):
                assert torch.equal(x[:,:k],x[:1,:k].expand_as(x[:,:k]))
    width=int(tokens.argmax(dim=-1).max())+1
    values=[]
    for option in (None,width):
        output=projected_logits(image,text_vectors(encoder,prompt,option,strategy=="shared"))
        loss=D.loss_fn(output,target)
        gradients=torch.autograd.grad(loss,(prompt.ctx_pos,prompt.ctx_neg))
        values.append((output.detach(),loss.detach(),*[g.detach() for g in gradients]))
    comparisons=list(zip(*values,strict=True))
    error=[float((a-b).abs().max()) for a,b in comparisons]
    gradient_relative=[float((a-b).norm()/a.norm().clamp(min=1e-10)) for a,b in comparisons[2:]]
    passing=(torch.allclose(*comparisons[0],atol=3e-5,rtol=1e-5)
             and torch.allclose(*comparisons[1],atol=1e-4,rtol=1e-6)
             and max(gradient_relative)<=5e-5)
    return width if passing else None,dict(strategy=strategy,eot_width=width,accepted=passing,
        logit_error=error[0],loss_error=error[1],gradient_errors=error[2:],gradient_relative_l2=gradient_relative)


def smoke_development_pixels(work,ids):
    paths={}
    scratch=work/"smoke_dev_pixels"
    scratch.mkdir(exist_ok=True)
    roots=[RECOVERY/"datasets/coco"/p for p in ("train2017","val2017")]
    for value in ids.tolist():
        name=f"{value:012d}.jpg"
        path=next((root/name for root in [*roots,scratch] if (root/name).is_file()),None)
        if path is not None:
            paths[str(value)]=str(path)
    for split in ("train2017","val2017"):
        missing=[int(i) for i in ids if str(int(i)) not in paths]
        if not missing:
            break
        with ZipFile(Path(os.environ.get("FQ_COCO2017_ROOT", Path(__file__).resolve().parents[2] / "inputs/coco2017"))/f"{split}.zip") as archive:
            entries=[(i,archive.NameToInfo.get(f"{split}/{i:012d}.jpg")) for i in missing]
            for i,entry in sorted((x for x in entries if x[1] is not None),key=lambda x:x[1].header_offset):
                path=scratch/f"{i:012d}.jpg"
                temp=path.with_suffix(".partial")
                temp.write_bytes(archive.read(entry))
                temp.replace(path)
                paths[str(i)]=str(path)
    assert set(paths)==set(map(str,ids.tolist()))
    return paths


def main(work,fixed,labels,a1):
    configure_loader_process()
    D.setup()
    assert torch.cuda.is_available()
    train = {}
    with fixed.open(encoding="utf-8-sig",newline="") as f:
        for row in csv.DictReader(f):
            if row["role"]!="TRAIN":
                continue
            i = int(row["id"])
            p = RECOVERY/"datasets/coco"/row["existing_subdirectory"]/row["file_name"]
            assert p.is_file()
            train[str(i)]=str(p)
    ids=np.asarray(list(map(int,train)))
    with (np.load(labels,allow_pickle=False) as gold,np.load(a1,allow_pickle=False) as development):
        dev_ids=development["ids"][:128]
        assert len(dev_ids)==128 and len(np.unique(dev_ids))==128 and np.isin(dev_ids,gold["ids"]).all()
    assert len(ids)==4008 and not set(ids)&set(dev_ids)
    dev=smoke_development_pixels(work,dev_ids)
    plan = json.loads((work/"PANOPTIC_MASK_SCREEN.json").read_text())
    classes = json.loads((work/"PANOPTIC_CLASS_SPACE_LOCK.json").read_text())
    D.CLASSES = [row["query"] for row in classes["categories"]]
    draw = plan["dev_seeds"][0]
    with np.load(work/f"PANOPTIC_MASK_{draw}.npz",allow_pickle=False) as archive:
        assert archive["categories"].tolist()==[row["id"] for row in classes["categories"]]
        bind={i:j for j,i in enumerate(archive["ids"].tolist())}
        rows=[bind[i] for i in ids.tolist()]
        masks={a:archive[a][rows] for a in (*ARMS,"LOW")}
    assert all(v.shape==(4008,WIDTH) for v in masks.values())
    seed=int(np.random.SeedSequence([SEED,11,1]).generate_state(1,np.uint64)[0])
    # Bypass the duplicate cold fixture construction for the default shared context.
    generator=torch.Generator().manual_seed(seed)
    fixture={k:torch.empty((16,512)).normal_(0,.02,generator=generator) for k in ("ctx_pos","ctx_neg")}
    torch.save(fixture,work/"PROMPT_SMOKE_INITIAL.pt")
    model=D.build_dc(fixture_path=work/"PROMPT_SMOKE_INITIAL.pt",init_seed=seed)
    prompts=[model.net.prompt_learner,*[copy.deepcopy(model.net.prompt_learner) for _ in ARMS[1:]]]
    optimizer=[torch.optim.SGD((p.ctx_pos,p.ctx_neg),lr=0,momentum=.9) for p in prompts]
    oracle=None
    cropped_width=None
    cropped_strategy=None
    text_qa=[]
    timings=[]
    costs=[]
    index={i:j for j,i in enumerate(ids.tolist())}
    counts=int((masks["LOW"]>=0).sum())
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(2):
        started=perf_counter()
        seen=[]
        events=[]
        train_batches=loader(ids,train,masks["LOW"],draw,epoch,workers=4)
        for optim in optimizer:
            optim.param_groups[0]["lr"] = .002*8976/counts*D.factor(epoch)
        for j,(pixels,_,batch_ids) in enumerate(train_batches):
            pix=pixels.cuda(non_blocking=True)
            positions=[index[i] for i in batch_ids.tolist()]
            seen.extend(batch_ids.tolist())
            begin, visual_done, heads_done = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
            begin.record()
            with torch.no_grad():
                spatial,_=model.net.image_encoder(pix)
                spatial=normalized(spatial,1)
            visual_done.record()
            for i,(arm,prompt,opt) in enumerate(zip(ARMS,prompts,optimizer,strict=True)):
                model.net.prompt_learner=prompt
                target=torch.as_tensor(masks[arm][positions],device="cuda")
                if oracle is None:
                    for strategy in ("shared","short"):
                        try:
                            cropped_width,case=text_parity(model,prompt,spatial,target,strategy)
                        except RuntimeError as exc:
                            cropped_width=None
                            case=dict(strategy=strategy,accepted=False,runtime_error=str(exc))
                            print(case,flush=True)
                        text_qa.append(case)
                        if cropped_width is not None:
                            cropped_strategy=strategy
                            break
                    with torch.no_grad():
                        native=model(pix)
                        projected=projected_logits(spatial,text_vectors(model.net.text_encoder,prompt))
                        torch.testing.assert_close(projected,native,rtol=1e-5,atol=1e-5)
                        oracle=float((projected-native).abs().max())
                elif cropped_width is not None and i==0 and epoch==0 and len(seen)==len(ids):
                    tail_width,case=text_parity(model,prompt,spatial,target,cropped_strategy)
                    text_qa.append(case)
                    if tail_width is None:
                        raise RuntimeError("Causal text optimization failed tail FP32 parity")
                logits=projected_logits(spatial,text_vectors(model.net.text_encoder,prompt,cropped_width,cropped_strategy=="shared"))
                loss=D.loss_fn(logits,target)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                assert torch.isfinite(loss) and torch.isfinite(prompt.ctx_pos.grad).all() and torch.isfinite(prompt.ctx_neg.grad).all()
                opt.step()
            heads_done.record()
            events.append((begin,visual_done,heads_done))
            if j%42==0:
                print(epoch+1,j+1,perf_counter()-started,flush=True)
        del train_batches
        assert seen==ids[order_for(draw,epoch,len(ids))].tolist()
        torch.cuda.synchronize()
        timings.append(perf_counter()-started)
        costs.append(dict(visual_seconds=sum(a.elapsed_time(b) for a,b,_ in events)/1000.,
                          three_head_seconds=sum(b.elapsed_time(c) for _,b,c in events)/1000.))
        checkpoint=work/"PROMPT_SMOKE_CHECKPOINT.pt"
        temp=checkpoint.with_suffix(".partial")
        torch.save(dict(epochs=epoch+1,prompts=[p.state_dict() for p in prompts],
                        optimizer=[o.state_dict() for o in optimizer],width=WIDTH),temp)
        temp.replace(checkpoint)
        previous=[p.ctx_pos.detach().clone() for p in prompts]
        restored=torch.load(checkpoint,map_location="cuda",weights_only=True)
        assert restored["epochs"]==epoch+1 and restored["width"]==WIDTH
        for p,o,pstate,ostate,expected in zip(prompts,optimizer,restored["prompts"],restored["optimizer"],previous,strict=True):
            p.load_state_dict(pstate)
            o.load_state_dict(ostate)
            torch.testing.assert_close(p.ctx_pos,expected,rtol=0,atol=0)
    # Decode a fixed development subset and prove normal model scoring after restore.
    development=loader(dev_ids[:128],dev,workers=4)
    qa=[]
    with torch.no_grad():
        for pixels,_,_ in development:
            model.net.prompt_learner=prompts[0]
            z=model(pixels.cuda()).softmax(1)[:,1]
            assert z.shape[1]==WIDTH and torch.isfinite(z).all()
            qa.append(z.cpu())
    assert torch.cat(qa).shape==(128,WIDTH)
    fit_hours=timings[-1]*96*50/(3*3600)
    formal_floor_hours=timings[-1]*144*50/(3*3600)
    twelve_head_epoch=timings[-1]+3*costs[-1]["three_head_seconds"]
    p1_hours=twelve_head_epoch*8*50/3600
    result=dict(status="PASS_TWO_EPOCH_ENGINEERING",gpu=torch.cuda.get_device_name(),classes=WIDTH,
        epochs=2,steps_per_endpoint=252,endpoints=3,epoch_seconds=timings,parity_max_abs=oracle,
        restore_checked=True,development_pixels=128,peak_gpu_bytes=torch.cuda.max_memory_allocated(),
        causal_text_width=cropped_width,causal_text_strategy=cropped_strategy,causal_text_parity=text_qa,
        confirmation_scores=0,estimated_96_fit_hours_without_extra_sharing=fit_hours,
        image_and_head_timings=costs, twelve_head_epoch_projection_seconds=twelve_head_epoch,
        estimate_with_recovery_p1=p1_hours/.8, estimate_with_recovery_p3_minimum=formal_floor_hours/.8)
    complete(work/"PROMPT_SMOKE_QA.json",result)
    print(result,flush=True)


if __name__=="__main__":
    parser=ArgumentParser()
    for name in ("work","fixed","labels","a1"):
        parser.add_argument(name,type=Path)
    main(**vars(parser.parse_args()))
