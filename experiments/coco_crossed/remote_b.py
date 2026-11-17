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

"""Train the 24 N1 endpoints and score the frozen B image pool."""

import csv
import importlib.util
import signal
import sys
import tempfile
import time
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
from torch.nn import functional as F

from remote_v91 import A_PIXELS, ARCHIVE, HERE, S0, read, save

sys.path.insert(0, str(S0))
import pilot as P

spec = importlib.util.spec_from_file_location('v91_training', HERE/'formal_train_shared.py')
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)
D = P.D
N1 = ('N1_TRUE',)


def checkpoint(draw, remote, arms, seeds, epoch):
    file = remote/f'draw_{draw}'/'CHECKPOINT.pt'
    value = torch.load(file, map_location='cpu', weights_only=False)
    assert value['draw']==draw and value['arms']==list(arms) and value['seeds']==seeds
    assert value['steps']==126*value['epoch'] and len(value['history'])==value['epoch']
    assert len(value['heads'])==len(arms)*len(seeds)
    if epoch is not None:
        assert value['epoch']==epoch
    return value


def pixels_for_train():
    train = {}
    with (S0/'PILOT_PIXEL_MANIFEST.csv').open(newline='') as file:
        for row in csv.DictReader(file):
            if row['role']=='TRAIN':
                path=P.RECOVERY/'datasets/coco'/row['existing_subdirectory']/row['file_name']
                assert path.is_file()
                train[row['id']]=str(path)
    assert len(train)==4008
    return np.array(list(map(int,train))),train


def score_b(model, lock, deadline):
    with (HERE/'B_PIXEL_BINDING.csv').open(newline='') as file:
        rows=list(csv.DictReader(file))
    ids=np.array([int(row['id']) for row in rows])
    assert len(ids)==5000 and ids.tolist()==lock['b_ids']
    # These are scratch copies of the already scored A pixels. The official
    # archive, qualification manifest and sealed scores remain available.
    with (HERE/'A_PIXEL_BINDING.csv').open(newline='') as file:
        for row in csv.DictReader(file):
            cache=A_PIXELS/f"{int(row['id']):012d}.jpg"
            cache.unlink(missing_ok=True)
    seed_list=lock['seeds']
    manifest=[]; heads=[]
    for draw in lock['draws']:
        old=checkpoint(draw,S0,P.ARMS,seed_list,50)
        new=checkpoint(draw,HERE,N1,seed_list,50)
        old_ctx=dict(zip(((s,a) for s in seed_list for a in P.ARMS),old['heads'],strict=True))
        new_ctx=dict(zip(seed_list,new['heads'],strict=True))
        for seed in seed_list:
            for arm in ('S0_TRUE','N1_TRUE','S0_PLC'):
                head=P.Head(model)
                src=new_ctx[seed] if arm=='N1_TRUE' else old_ctx[seed,arm]
                T.restore_head(head,src['parameters'])
                heads.append(head)
                manifest.append((draw,seed,arm))
    assert len(heads)==72
    start=time.monotonic()
    with tempfile.TemporaryDirectory(prefix='coco-crossed-') as directory, ZipFile(ARCHIVE) as archive:
        paths={}
        for row in sorted(rows,key=lambda x:archive.getinfo(x['member']).header_offset):
            path=Path(directory)/f"{int(row['id']):012d}.jpg"
            raw=archive.read(row['member'])
            assert len(raw)==int(row['bytes'])
            path.write_bytes(raw)
            paths[row['id']]=str(path)
        out=np.empty((72,5000,20),dtype=np.float32)
        seen=[]
        with torch.no_grad():
            vectors=[]
            for head in heads:
                prompts,tokens=head.learner()
                text=head.encoder(prompts,tokens)
                vectors.append(text/text.norm(dim=-1,keepdim=True))
            for batch,(images,_,ii) in enumerate(T.epoch_loader(ids,paths,workers=3)):
                assert time.time()<deadline and not T.STOP
                offset=len(seen)
                seen.extend(ii.tolist())
                feature,_=model.net.image_encoder(images.cuda(non_blocking=True))
                image=feature/feature.norm(dim=1,keepdim=True)
                for j,(head,text) in enumerate(zip(heads,vectors)):
                    raw=20*F.conv1d(image,text[:,:,None])
                    b,c,_=raw.shape
                    w=raw[:,c//2:].softmax(-1)
                    value=(5*(raw*torch.cat([w,w],1)).sum(-1)).reshape(b,2,c//2).softmax(1)[:,1]
                    if batch in (0,156):
                        assert torch.equal(value,head(feature).softmax(1)[:,1])
                    out[j,offset:len(seen)]=value.cpu().numpy()
                if batch%40==0:
                    print('SCORE_B',len(seen),'/5000',flush=True)
    assert seen==ids.tolist() and np.isfinite(out).all()
    final=HERE/'B_SCORES.npz';tmp=final.with_suffix('.tmp')
    with tmp.open('wb') as file:
        np.savez(file,ids=ids,classes=np.array(D.CLASSES),scores=out,
            draws=[x[0] for x in manifest],seeds=[x[1] for x in manifest],arms=[x[2] for x in manifest])
    tmp.replace(final)
    save('B_SCORE_RECEIPT.json',dict(status='SCORES_SEALED',shape=list(out.shape),endpoints=72,new_endpoints=24,
        truth_loaded=False,training_steps=6300,seconds=time.monotonic()-start,sealed_unix=time.time()))
    print('B_SCORES_SEALED',list(out.shape),flush=True)


def main(mode):
    lock=read(HERE/'B_PROTOCOL_LOCK.json')
    assert lock['status']=='LOCKED' and lock['a_gate']=='PASS_A'
    phase=read(HERE/('B_TRIAL_LEASE.json' if mode=='trial' else 'B_BUDGET_LOCK.json'))
    D.setup()
    P.seed_all()
    signal.signal(signal.SIGTERM,T.request_stop)
    deadline=phase['worker_deadline_unix']
    ids, paths=pixels_for_train()
    assert not set(ids.tolist()) & set(lock['b_ids'])
    model=D.build_dc(init_seed=lock['seeds'][0])
    times=[]
    for draw in lock['draws']:
        if mode=='train':
            checkpoint(draw,HERE,N1,lock['seeds'],None)
        if not T.train_draw(model,ids,paths,HERE/f'MASK_{draw}.npz',draw,lock['seeds'],3,deadline,
                            arms=N1,limit=1 if mode=='trial' else 50):
            raise SystemExit(2)
        cp=checkpoint(draw,HERE,N1,lock['seeds'],1 if mode=='trial' else 50)
        times.append(dict(draw=draw,epoch=cp['epoch'],updates=cp['steps'],
            wall_seconds=sum(x['seconds'] for x in cp['history']),first_epoch_seconds=cp['history'][0]['seconds']))
    if mode=='trial':
        save('B_TRIAL_QA.json',dict(status='PASS',completed_endpoints=24,measured=times,
                                    peak_bytes=torch.cuda.max_memory_allocated()))
        print('B_TRIAL_PASS',times,flush=True)
    else:
        save('B_TRAIN_QA.json',dict(status='COMPLETE',endpoints=24,steps_total=151200,draws=times))
        score_b(model,lock,deadline)


if __name__=='__main__':
    main(sys.argv[1])
