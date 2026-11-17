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

"""Run the frozen S0 endpoint matrix with shared native FP32 image features."""

import csv
import json
import os
import signal
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from zipfile import ZipFile

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pilot as P

torch = P.torch
D = P.D
ARMS = P.ARMS
STOP = False


def request_stop(_signal, _frame):
    global STOP
    STOP = True


def atomic_torch(path, value):
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def progress(**fields):
    atomic_json(HERE / 'S0_PROGRESS.json', fields)
    print(json.dumps(fields), flush=True)


def epoch_loader(ids, paths, visible=None, draw=0, epoch=0, workers=6):
    frozen = P.loader(ids, paths, visible, draw, epoch, workers=workers)
    # A fresh loader per epoch does not need persistent workers. Keeping them
    # alive with pinned memory accumulates worker shutdown handles in PyTorch.
    return torch.utils.data.DataLoader(
        frozen.dataset, batch_sampler=frozen.batch_sampler, num_workers=frozen.num_workers,
        collate_fn=frozen.collate_fn, pin_memory=frozen.pin_memory,
        prefetch_factor=frozen.prefetch_factor, generator=frozen.generator,
        persistent_workers=False, timeout=120,
    )


def make_jobs(model, seeds, keys):
    jobs = []
    for seed in seeds:
        rng = torch.Generator().manual_seed(seed)
        initial = [torch.empty((16, 512)).normal_(0, .02, generator=rng) for _ in range(2)]
        for arm in keys:
            head = P.Head(model)
            with torch.no_grad():
                for target, value in zip(P.params(head), initial):
                    target.copy_(value)
            opt = torch.optim.SGD(P.params(head), lr=0., momentum=.9, weight_decay=0.)
            jobs.append((seed, arm, head, opt))
    assert len({p.data_ptr() for _, _, h, _ in jobs for p in P.params(h)}) == 2 * len(jobs)
    return jobs


def load_mask(path, ids, classes):
    with np.load(path) as data:
        assert np.array_equal(data['ids'], ids) and np.array_equal(data['classes'], classes)
        masks = {a: data[a] for a in ARMS}
    base = int((masks['LOW'] != -1).sum())
    for name, matrix in masks.items():
        assert matrix.shape == (4008, 20) and np.isin(matrix, (-1, 0, 1)).all()
        assert int((matrix != -1).sum()) == base + (120 if name.endswith(('TRUE_BG', 'TRUE_PARTNER')) else 0)
    return masks, base


def restore_head(head, parameters):
    with torch.no_grad():
        for current, saved in zip(P.params(head), parameters, strict=True):
            current.copy_(saved)


def train_draw(model, ids, paths, mask_path, draw, seeds, lanes, deadline):
    out = HERE / f'draw_{draw}'
    out.mkdir(exist_ok=True)
    state_path = out / 'CHECKPOINT.pt'
    jobs = make_jobs(model, seeds, ARMS)
    masks, base = load_mask(mask_path, ids, D.CLASSES)
    lr = .002 * 8976 / base
    history = []
    start_epoch = 0
    if state_path.is_file():
        state = torch.load(state_path, map_location='cpu', weights_only=False)
        assert state['draw'] == draw and state['seeds'] == seeds and state['arms'] == list(ARMS)
        history, start_epoch = state['history'], state['epoch']
        assert len(history) == start_epoch and state['steps'] == 126 * start_epoch
        for (_, _, head, opt), saved in zip(jobs, state['heads'], strict=True):
            restore_head(head, saved['parameters'])
            opt.load_state_dict(saved['optimizer'])
        D.rng_restore(state['RNG'])
        assert D.rng_equal(D.rng_snapshot(), state['RNG'])
        progress(stage='RESTORED_CHECKPOINT', draw=draw, epoch=start_epoch,
                 updates=state['steps'], heads=len(jobs))
    if start_epoch == 50:
        return True
    index = {int(i): j for j, i in enumerate(ids)}
    state_thread = threading.local()
    streams = iter(torch.cuda.Stream() for _ in range(lanes))

    def init_stream():
        state_thread.stream = next(streams)

    def step(job, features, targets, known, ready):
        _, arm, head, opt = job
        stream = state_thread.stream
        with torch.cuda.stream(stream):
            stream.wait_event(ready)
            values = P.step(head, opt, features, targets[arm], known[arm])
            stream.synchronize()
            return values.cpu().numpy().tolist()

    with ThreadPoolExecutor(max_workers=lanes, initializer=init_stream) as pool:
        for epoch in range(start_epoch, 50):
            if STOP or time.time() >= deadline:
                return False
            started = time.monotonic()
            for _, _, _, opt in jobs:
                opt.param_groups[0]['lr'] = lr * D.factor(epoch)
            images, batch_counts = [], {arm: [] for arm in ARMS}
            accumulated = np.zeros((len(jobs), 3), dtype=np.float64)
            for batch, (pixels, _, ii) in enumerate(epoch_loader(ids, paths, masks['LOW'], draw, epoch)):
                if STOP or time.time() >= deadline:
                    return False
                images.extend(ii.tolist())
                positions = [index[int(i)] for i in ii]
                with torch.no_grad():
                    features, _ = model.net.image_encoder(pixels.cuda(non_blocking=True))
                labels = {a: mask[positions] for a, mask in masks.items()}
                targets = {a: torch.from_numpy(value).cuda() for a, value in labels.items()}
                for arm, value in labels.items():
                    batch_counts[arm].append([int((value >= 0).sum()), int((value == 0).sum())])
                known = {a: counts[-1][0] > 0 for a, counts in batch_counts.items()}
                ready = torch.cuda.Event()
                ready.record()
                values = np.array(list(pool.map(lambda job: step(job, features, targets, known, ready), jobs)))
                assert values.shape == accumulated.shape and np.isfinite(values).all()
                accumulated[:, 0] += values[:, 0]
                accumulated[:, 1:] = np.maximum(accumulated[:, 1:], values[:, 1:])
                if batch % 42 == 0:
                    progress(stage='TRAIN', draw=draw, epoch=epoch+1, batch=batch+1, heads=len(jobs))
            assert images == ids[P.order_for(draw, epoch, len(ids))].tolist()
            for arm, counts in batch_counts.items():
                assert len(counts) == 126 and sum(x[0] for x in counts) == int((masks[arm] >= 0).sum())
            assert all(torch.isfinite(p).all() for _, _, h, _ in jobs for p in P.params(h))
            history.append(dict(epoch=epoch+1, steps=126*(epoch+1), lr=lr*D.factor(epoch),
                                batches=batch_counts, loss_grad=accumulated.tolist(),
                                seconds=time.monotonic()-started))
            state = dict(draw=draw, seeds=seeds, arms=list(ARMS), epoch=epoch+1, steps=126*(epoch+1),
                         history=history, heads=[dict(parameters=[p.detach().cpu() for p in P.params(h)],
                                                     optimizer=o.state_dict())
                                                  for _, _, h, o in jobs], RNG=D.rng_snapshot())
            atomic_torch(state_path, state)
            progress(stage='CHECKPOINT', draw=draw, epoch=epoch+1, updates=state['steps'], heads=len(jobs),
                     open_fds=len(list(Path('/proc/self/fd').iterdir())))
    return True


def score(model, lock, deadline):
    with (HERE / 'CONFIRM_PIXEL_BINDING.csv').open(newline='') as file:
        pixels = list(csv.DictReader(file))
    ids = np.array([int(row['id']) for row in pixels])
    assert ids.tolist() == lock['confirmation_ids']
    jobs = []
    for draw in lock['draws']:
        run = torch.load(HERE / f'draw_{draw}' / 'CHECKPOINT.pt', map_location='cpu', weights_only=False)
        assert run['epoch'] == 50 and run['steps'] == 6300
        for (seed, arm, head, _), saved in zip(make_jobs(model, run['seeds'], ARMS), run['heads'], strict=True):
            restore_head(head, saved['parameters'])
            jobs.append((draw, seed, arm, head))
    assert len(jobs) == lock['endpoints']
    with tempfile.TemporaryDirectory(prefix='coco-fit-') as directory:
        archive_path = Path(pixels[0]['asset'])
        assert archive_path.is_file() and all(row['asset'] == str(archive_path) for row in pixels)
        paths = {}
        with ZipFile(archive_path) as archive:
            for row in sorted(pixels, key=lambda x: archive.getinfo(x['member']).header_offset):
                path = Path(directory) / f"{int(row['id']):012d}.jpg"
                data = archive.read(row['member'])
                assert len(data) == int(row['bytes'])
                path.write_bytes(data)
                paths[row['id']] = str(path)
        result = np.empty((len(jobs), len(ids), len(D.CLASSES)), np.float32)
        seen = []
        with torch.no_grad():
            for batch, (images, _, ii) in enumerate(epoch_loader(ids, paths, workers=3)):
                if STOP or time.time() >= deadline:
                    return False
                offset = len(seen)
                seen.extend(ii.tolist())
                image = images.cuda(non_blocking=True)
                features, _ = model.net.image_encoder(image)
                for j, (_, _, _, head) in enumerate(jobs):
                    result[j, offset:len(seen)] = head(features).softmax(1)[:, 1].cpu().numpy()
                if batch % 40 == 0:
                    progress(stage='SCORE', rows=len(seen), heads=len(jobs))
    assert seen == ids.tolist() and np.isfinite(result).all()
    destination = HERE / 'S0_SCORES.npz'
    temporary = destination.with_suffix('.tmp')
    with temporary.open('wb') as file:
        np.savez(file, ids=ids, classes=np.array(D.CLASSES), scores=result,
                 draws=[x[0] for x in jobs], seeds=[x[1] for x in jobs], arms=[x[2] for x in jobs])
    temporary.replace(destination)
    atomic_json(HERE / 'S0_SCORE_RECEIPT.json', dict(status='COMPLETE', endpoints=len(jobs), steps=6300,
                scores_shape=list(result.shape), classes=D.CLASSES, truth_loaded=False))
    progress(stage='SCORES_READY', heads=len(jobs), rows=len(ids))
    return True


def main():
    signal.signal(signal.SIGTERM, request_stop)
    D.setup()
    P.seed_all()
    lock = json.loads((HERE / 'PROTOCOL_LOCK.json').read_text())
    deadline = json.loads((HERE / 'RUN_LEASE.json').read_text())['worker_deadline_unix']
    assert lock['status'] == 'LOCKED' and lock['mode'] == 'shared_fp32'
    assert lock['lanes'] in (1, 2, 3) and len(lock['draws']) == 2 and len(lock['seeds']) == lock['n']
    ids, train, old_ids, _, _ = P.inputs()
    assert not set(lock['confirmation_ids']) & set(ids.tolist() + old_ids.tolist())
    model = D.build_dc(init_seed=lock['seeds'][0])
    for draw in lock['draws']:
        if not train_draw(model, ids, train, HERE / f'MASK_{draw}.npz', draw, lock['seeds'],
                          lock['lanes'], deadline):
            progress(stage='CHECKPOINTED_STOP', draw=draw)
            return 2
    return 0 if score(model, lock, deadline) else 2


if __name__ == '__main__':
    raise SystemExit(main())
