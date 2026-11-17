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

"""Train locked DualCoOp endpoints and share one image pass per formal evaluation pool."""
from argparse import ArgumentParser
import csv
import json
from pathlib import Path
import shutil
from time import perf_counter, time

import numpy as np
import torch

from prompt_development import train_draw
from prompt_smoke import ARMS, D, RECOVERY, WIDTH, complete, configure_loader_process, loader, normalized, projected_logits, text_vectors


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def ordered_ids(path):
    with np.load(path,allow_pickle=False) as array:
        ids=array["ids"]
    assert len(np.unique(ids))==len(ids)
    return ids


def training_pixels(manifest):
    with manifest.open(encoding="utf-8-sig",newline="") as stream:
        paths={r["id"]:str(RECOVERY/"datasets/coco"/r["existing_subdirectory"]/r["file_name"])
               for r in csv.DictReader(stream) if r["role"]=="TRAIN"}
    assert len(paths)==4008 and all(Path(path).is_file() for path in paths.values())
    return np.fromiter((int(i) for i in paths),dtype=np.int64),paths


def checkpoints(work,draws):
    return [work/f"M1_DRAW_{draw}.pt" for draw in draws]


def evaluate(work,model,ids,paths,phase,draws,seeds,categories,deadline):
    endpoints=len(ARMS)*len(seeds)
    scoring_receipt=work/f"M1_{phase.upper()}_SCORE_RECEIPT.json"
    if scoring_receipt.exists():
        receipt=load_json(scoring_receipt)
        assert receipt["phase"]==phase and receipt["ids"]==ids.tolist() and receipt["draws"]==draws
        for draw in draws:
            validate_scores(work/f"M1_{phase}_{draw}.npz",ids,categories,seeds)
        return receipt
    shape=(endpoints,len(ids),len(categories))
    bytes_per_file=np.prod(shape,dtype=np.int64).item()*np.dtype(np.float32).itemsize
    pending=[]
    for draw in draws:
        sealed=work/f"M1_{phase}_{draw}.npz"
        if sealed.exists():
            validate_scores(sealed,ids,categories,seeds)
        else:
            pending.append(draw)
    feature=work/"CONF_M_FEATURES.npz"
    need_features=phase=="confirm" and not feature.exists()
    if phase=="confirm" and not need_features:
        np.testing.assert_array_equal(ordered_ids(feature),ids)
    additional=(len(pending)+1)*bytes_per_file+len(ids)*1024*4*need_features+2_000_000
    assert shutil.disk_usage(work).free>=1.25*additional,"HOLD_STORAGE"
    slots=[work/f"M1_{phase}_{draw}.scores_partial.npy" for draw in pending]
    score=[np.lib.format.open_memmap(path,mode="w+",dtype=np.float32,shape=shape) for path in slots]
    contexts=[]
    prompt=model.net.prompt_learner
    start=perf_counter()
    with torch.inference_mode():
        for path in checkpoints(work,pending):
            saved=torch.load(path,map_location="cuda",weights_only=True)
            assert saved["seeds"]==seeds and len(saved["history"])==50
            assert saved["history"][-1]["steps"]==6300 and len(saved["jobs"])==endpoints
            encodings=[]
            for ctx,_ in saved["jobs"]:
                prompt.ctx_pos.copy_(ctx["ctx_pos"])
                prompt.ctx_neg.copy_(ctx["ctx_neg"])
                encodings.append(text_vectors(model.net.text_encoder,prompt))
            contexts.append(torch.stack(encodings))
        position=0 if pending or need_features else len(ids)
        globals_out=np.empty((len(ids),1024),dtype=np.float32) if need_features else None
        batches=loader(ids,paths,workers=6) if pending or need_features else ()
        for pixels,_,batch in batches:
            assert time()<deadline
            stop=position+len(batch)
            assert batch.tolist()==ids[position:stop].tolist()
            spatial,_=model.net.image_encoder(pixels.cuda(non_blocking=True))
            if globals_out is not None:
                leaf=model.net.image_encoder.attnpool.last_global
                assert leaf.shape==(len(batch),1024) and torch.isfinite(leaf).all()
                globals_out[position:stop]=leaf.cpu().numpy()
            spatial=normalized(spatial,1)
            for index,group in enumerate(contexts):
                for first in range(0,endpoints,12):
                    vectors=group[first:first+12]
                    projected=20*torch.nn.functional.conv1d(spatial,vectors.flatten(0,1)[:,:,None])
                    raw=projected.reshape(len(batch),len(vectors),2,WIDTH,-1)
                    attention=raw[:,:,1].softmax(-1)
                    logits=5*(raw*attention[:,:,None]).sum(-1)
                    if position==0 or stop==len(ids):
                        for j,vector in enumerate(vectors):
                            native=projected_logits(spatial,vector)
                            torch.testing.assert_close(logits[:,j],native,rtol=1e-5,atol=3e-5)
                    prob=logits.softmax(2)[:,:,1].permute(1,0,2)
                    assert torch.isfinite(prob).all()
                    score[index][first:first+len(vectors),position:stop]=prob.cpu().numpy()
            position=stop
    assert position==len(ids)
    if globals_out is not None:
        temp=feature.with_suffix(".partial")
        with temp.open("wb") as sink:
            np.savez(sink,ids=ids,features=globals_out)
        temp.replace(feature)
    for i,draw in enumerate(pending):
        path=work/f"M1_{phase}_{draw}.npz"
        temp=path.with_suffix(".partial")
        with temp.open("wb") as sink:
            np.savez(sink,ids=ids,categories=categories,scores=score[i],arms=np.tile(ARMS,len(seeds)),
                     seeds=np.repeat(np.asarray(seeds,dtype=np.uint64),len(ARMS)))
        temp.replace(path)
        score[i].flush()
        score[i]=None
        slots[i].unlink()
    receipt=dict(status="SCORES_SEALED",phase=phase,ids=ids.tolist(),draws=draws,endpoints=len(draws)*endpoints,
                 width=len(categories),wall_seconds=perf_counter()-start,image_encoder_passes_per_image=1,
                 native_first_and_last_batch_checked=True,truth_loaded=False,preexisting_draws=len(draws)-len(pending))
    complete(scoring_receipt,receipt)
    return receipt


def validate_scores(path,ids,categories,seeds):
    with np.load(path,allow_pickle=False) as value:
        assert np.array_equal(value["ids"],ids) and value["categories"].tolist()==categories
        assert value["arms"].tolist()==list(ARMS)*len(seeds)
        assert np.array_equal(value["seeds"],np.repeat(np.asarray(seeds,dtype=np.uint64),len(ARMS)))
        assert value["scores"].shape==(len(seeds)*len(ARMS),len(ids),len(categories))


def run(work,phase,deadline,fit_manifest):
    lock=load_json(work/"PROTOCOL_LOCK.json")
    assert lock["status"]=="READY" and lock["version"]=="v9.3.1-coco133" and len(lock["targets"])>=4
    m1=lock["M1"]
    draws,seeds=m1["draws"],m1["seeds"]
    classes=load_json(work/"PANOPTIC_CLASS_SPACE_LOCK.json")["categories"]
    categories=[item["id"] for item in classes]
    assert categories==lock["categories"] and len(categories)==WIDTH
    smoke=load_json(work.parent/"PROMPT_SMOKE_QA.json")
    assert smoke["status"]=="PASS_TWO_EPOCH_ENGINEERING"
    configure_loader_process()
    D.setup()
    D.CLASSES=[x["query"] for x in classes]
    model=D.build_dc(fixture_path=work.parent/"PROMPT_SMOKE_INITIAL.pt",init_seed=seeds[0])
    for layer in model.net.text_encoder.transformer.resblocks:
        layer.attn_mask=layer.attn_mask.cuda()
    if phase=="fit":
        ids,paths=training_pixels(fit_manifest)
        assert set(ids.tolist())==set(lock["fit_ids"])
        started=perf_counter()
        histories=[]
        for draw,state in zip(draws,checkpoints(work,draws),strict=True):
            _,history=train_draw(work,model,ids,paths,categories,draw,seeds,deadline,
                                 smoke.get("causal_text_width"),smoke.get("causal_text_strategy")=="shared",state)
            histories.append(dict(draw=draw,epochs=len(history),updates=history[-1]["steps"],
                                  fit_seconds=sum(row["seconds"] for row in history)))
        complete(work/"M1_TRAIN_QA.json",dict(status="COMPLETE",endpoints=len(draws)*len(seeds)*len(ARMS),
                 fit_role="FIT",wall_seconds=perf_counter()-started,rows=histories,confirmed_images_scored=0))
        return
    assert load_json(work/"M1_TRAIN_QA.json")["status"]=="COMPLETE"
    if phase=="bridge":
        ids=ordered_ids(work/"PANOPTIC_BRIDGE_LABELS.npz")
        assert ids.tolist()==lock["bridge_ids"]
        pixels=work.parent/"bridge_pixels"
    else:
        sealed=load_json(work/"M3_PREDICTIONS_BEFORE_SCORING.json")
        assert sealed["status"]=="SEALED_SOURCE_PREDICTIONS" and sealed["run_id"]==lock["run_id"]
        ids=ordered_ids(work/"CONF_M_LABELS.npz")
        pixels=work/"confirm_pixels"
        assert ids.tolist()==lock["confirm_ids"]
    assert not set(ids.tolist())&set(lock["fit_ids"])
    paths={str(i):str(pixels/f"{i:012d}.jpg") for i in ids}
    assert all(Path(path).is_file() for path in paths.values())
    evaluate(work,model,ids,paths,phase,draws,seeds,categories,deadline)


if __name__=="__main__":
    parser=ArgumentParser()
    parser.add_argument("work",type=Path)
    parser.add_argument("phase",choices=("fit","bridge","confirm"))
    parser.add_argument("deadline",type=float)
    parser.add_argument("--fit-manifest",type=Path)
    args=parser.parse_args()
    run(args.work,args.phase,args.deadline,args.fit_manifest)
