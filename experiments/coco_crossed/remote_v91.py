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

"""Decode the fixed data pools and score the S0 checkpoint endpoints in FP32."""

import csv
import io
import json
import os
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import batched
from pathlib import Path
from zipfile import BadZipFile, ZipFile

HERE = Path(__file__).resolve().parent
ARCHIVE = Path(os.environ.get('FQ_COCO2017_ROOT', Path(__file__).resolve().parents[2] / "inputs/coco2017")) / 'train2017.zip'
S0 = Path(os.environ.get('FQ_COCO_S0_WORK', Path(__file__).resolve().parents[1] / 'coco_fixed_partner'))
A_PIXELS = HERE / "pixel_cache"


def read(path):
    return json.loads(path.read_text())


def save(name, result):
    path = HERE / name
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(result, allow_nan=False))
    temp.replace(path)


def decode():
    from PIL import Image
    start = time.monotonic()
    A_PIXELS.mkdir(exist_ok=True)
    assert shutil.disk_usage(A_PIXELS).free > 1_200_000_000
    pools = {}
    with ZipFile(ARCHIVE) as archive, ThreadPoolExecutor(max_workers=8) as executor:
        for pool in ('A', 'B'):
            with (HERE / f'{pool}_PIXEL_CANDIDATES.csv').open(newline='') as file:
                rows = list(csv.DictReader(file))
            initial, reserve = ([r for r in rows if r['role']==k] for k in ('initial','reserve'))
            assert len(initial)==5000 and len(rows)==len({r['id'] for r in rows})
            verified, failures = {}, {}

            def check(record):
                row, data, error = record
                if error:
                    return row['id'], None, error
                try:
                    with Image.open(io.BytesIO(data)) as image:
                        size = int(row['width']), int(row['height'])
                        assert image.format == 'JPEG' and image.size == size
                        image.load()
                        image.convert('RGB').load()
                    if pool == 'A':
                        file = A_PIXELS / f"{int(row['id']):012d}.jpg"
                        file.write_bytes(data)
                    return row['id'], len(data), None
                except (OSError, ValueError, AssertionError) as exc:
                    return row['id'], None, repr(exc)

            def load(row):
                try:
                    return row, archive.read(row['member']), None
                except (OSError, ValueError, KeyError, BadZipFile) as exc:
                    return row, None, repr(exc)

            def verify_batch(group):
                for i, size, error in executor.map(check, [load(r) for r in group]):
                    if error:
                        failures[i] = error
                    else:
                        verified[i] = size

            ordered = sorted(initial, key=lambda x: archive.getinfo(x['member']).header_offset
                             if x['member'] in archive.NameToInfo else -1)
            for batch_no, group in enumerate(batched(ordered, 32), 1):
                verify_batch(group)
                if batch_no % 31 == 0:
                    print('DECODE', pool, batch_no*32, 'PASS', len(verified), flush=True)
            final = [r for r in initial if r['id'] in verified]
            for row in reserve:
                if len(final) == 5000:
                    break
                verify_batch([row])
                if row['id'] in verified:
                    final.append(row)
            with (HERE / f'{pool}_PIXEL_BINDING.csv').open('w', newline='') as file:
                writer = csv.DictWriter(file, ['id','width','height','member','bytes'])
                writer.writeheader()
                writer.writerows({k:r[k] for k in ('id','width','height','member')} |
                                 {'bytes':verified[r['id']]} for r in final)
            pools[pool] = dict(ids=[int(row['id']) for row in final], full_decode=len(final),
                failures=failures, replacement_ids=[int(r['id']) for r in final if r['role']=='reserve'],
                source=str(ARCHIVE), bytes=sum(verified[r['id']] for r in final))
    save('DECODE_QA.json', dict(status='DECODE_COMPLETE', pools=pools, seconds=time.monotonic()-start))
    print('DECODE_COMPLETE', [p['full_decode'] for p in pools.values()], flush=True)


def scorer(mode):
    sys.path.insert(0, str(S0))
    import pilot as P
    import numpy as np
    import torch
    from torch.nn import functional as F
    D = P.D
    D.setup()
    assert torch.cuda.device_count() == 1
    start = time.monotonic()
    pred = read(HERE/'A_PREDICTIONS.json')
    assert pred['status'] == 'PREDICTIONS_LOCKED' and pred['prediction_direction_pass']
    manifest = [(r['draw'],r['seed'],r['arm']) for r in pred['endpoint_manifest']]
    assert len(manifest) == len(set(manifest)) == 72
    proto = read(S0/'PROTOCOL_LOCK.json')
    model = D.build_dc(init_seed=proto['seeds'][0])
    contexts = {}
    for draw in proto['draws']:
        state = torch.load(S0/f'draw_{draw}'/'CHECKPOINT.pt', map_location='cpu', weights_only=False)
        assert state['draw']==draw and state['steps']==6300 and state['epoch']==50
        assert state['seeds']==proto['seeds'] and state['arms']==proto['arms']
        for (seed, arm), params in zip(((s,a) for s in state['seeds'] for a in state['arms']),
                                      state['heads'],strict=True):
            contexts[draw,seed,arm] = params['parameters']
    heads = []
    for key in manifest:
        head = P.Head(model)
        with torch.no_grad():
            for target, source in zip(P.params(head),contexts[key],strict=True):
                target.copy_(source)
        heads.append(head)

    def text_vectors(head):
        prompts,tokens = head.learner()
        text = head.encoder(prompts,tokens)
        return text / text.norm(dim=-1,keepdim=True)

    def score_fast(feature, text):
        image = feature / feature.norm(dim=1,keepdim=True)
        raw = 20*F.conv1d(image,text[:,:,None])
        b,c,_ = raw.shape
        w = raw[:,c//2:].softmax(-1)
        return (5*(raw*torch.cat([w,w],1)).sum(-1)).reshape(b,2,c//2).softmax(1)[:,1]

    def forward(ids, paths, compare=False, fast=True):
        out = np.empty((len(heads),len(ids),20),dtype=np.float32)
        seen, differences = [], []
        stable = []
        with torch.no_grad():
            vectors = [text_vectors(h) for h in heads] if fast else []
            for batch,(pixels,_,ii) in enumerate(P.loader(ids,paths,None,workers=3)):
                tick=time.monotonic()
                offset=len(seen)
                seen.extend(ii.tolist())
                pixels=pixels.cuda(non_blocking=True)
                feature,_=model.net.image_encoder(pixels)
                for j,h in enumerate(heads):
                    value = score_fast(feature,vectors[j]) if fast else h(feature).softmax(1)[:,1]
                    if compare:
                        reference=h(feature).softmax(1)[:,1]
                        differences.append(float((value-reference).abs().max()))
                    out[j,offset:len(seen)]=value.cpu().numpy()
                stable.append(time.monotonic()-tick)
                if not compare and batch % 40 == 0:
                    print('SCORE_A', len(seen), '/', len(ids), flush=True)
        assert seen==ids.tolist() and np.isfinite(out).all()
        return out, max(differences,default=0.), stable

    if mode=='preflight':
        with (HERE/'OLD_PIXEL_SAMPLE.csv').open(newline='') as file:
            rows=list(csv.DictReader(file))
        sample=np.array([int(row['id']) for row in rows])
        assert len(sample)==104
        with tempfile.TemporaryDirectory(prefix='coco-reference-') as directory, ZipFile(ARCHIVE) as archive:
            paths={}
            for row in sorted(rows,key=lambda x:archive.getinfo(x['member']).header_offset):
                path=Path(directory)/f"{int(row['id']):012d}.jpg"
                raw=archive.read(row['member'])
                assert len(raw)==int(row['bytes'])
                path.write_bytes(raw)
                paths[row['id']]=str(path)
            out,diff,timings=forward(sample,paths,compare=True)
        with np.load(HERE/'OLD_SCORE_SAMPLE.npz') as saved:
            assert np.array_equal(sample,saved['ids'])
            assert manifest==list(zip(saved['draws'].tolist(),saved['seeds'].tolist(),saved['arms'].tolist()))
            native_diff = float(np.max(np.abs(out-saved['scores'])))
        assert diff==native_diff==0.
        save('A_PREFLIGHT.json',dict(status='PASS',device=torch.cuda.get_device_name(),
            saved_scores_exact=True, shared_text_cache_exact=True, cache_max_abs=diff,
            archived_max_abs=native_diff, samples=len(sample), stable_batch_seconds=timings,
            seconds=time.monotonic()-start, peak_bytes=torch.cuda.max_memory_allocated()))
        print('PREFLIGHT_PASS',timings,flush=True)
        return
    budget=read(HERE/'A_BUDGET_LOCK.json')
    assert budget['status']=='LOCKED' and read(HERE/'A_PREFLIGHT.json')['status']=='PASS'
    with (HERE/'A_PIXEL_BINDING.csv').open(newline='') as file:
        rows=list(csv.DictReader(file))
    ids=np.array([int(row['id']) for row in rows])
    assert len(ids)==5000 and ids.tolist()==pred['ids']
    paths={str(i):str(A_PIXELS/f'{i:012d}.jpg') for i in ids}
    assert all(Path(v).is_file() for v in paths.values())
    out,_,timings=forward(ids,paths)
    target=HERE/'A_SCORES.npz';temp=target.with_suffix('.tmp')
    with temp.open('wb') as file:
        np.savez(file,ids=ids,classes=np.array(D.CLASSES),scores=out,
                 draws=[r[0] for r in manifest],seeds=[r[1] for r in manifest],arms=[r[2] for r in manifest])
    temp.replace(target)
    save('A_SCORE_RECEIPT.json',dict(status='SCORES_SEALED',endpoints=len(manifest),images=len(ids),
         shape=list(out.shape),truth_loaded=False,predictions_locked_at=pred['locked_at'],
         elapsed_seconds=time.monotonic()-start, median_batch_seconds=float(np.median(timings)),
         peak_bytes=torch.cuda.max_memory_allocated(),sealed_unix=time.time()))
    print('A_SCORES_SEALED',list(out.shape),flush=True)


if __name__ == '__main__':
    mode=sys.argv[1]
    if mode=='decode':
        decode()
    else:
        scorer(mode)
