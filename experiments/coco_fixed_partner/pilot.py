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

"""FP32 v9 engineering pilot. Reads only the training mask and old development pixels."""

import copy
import csv
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
RECOVERY = Path(os.environ.get('FQ_V9_RECOVERY', os.environ.get('FQ_RECOVERY_ROOT', Path(__file__).resolve().parents[2] / 'runtime')))
sys.path[:0] = [str(RECOVERY / p) for p in ('src', 'vendor', 'vendor/randaugment')]

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import dc_model as D
from data import loader, order_for

WORK = Path(__file__).resolve().parent
CONFIG = json.loads((WORK / 'PILOT_CONFIG.json').read_text())
DRAW, SEED = CONFIG['draw'], CONFIG['seed']
ARMS = ('LOW', 'S0_TRUE', 'S0_PLC', 'S0_TRUE_BG', 'S0_TRUE_PARTNER')
WORKERS = 6


def save_json(name, value):
    path = WORK / name
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def report(stage, **fields):
    result = dict(stage=stage, unix=time.time(), **fields)
    save_json('PROGRESS.json', result)
    print(json.dumps(result), flush=True)


def save_torch(name, value):
    path = WORK / name
    temp = path.with_suffix('.tmp')
    torch.save(value, temp)
    temp.replace(path)


def seed_all():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def inputs():
    train, dev = {}, {}
    with (WORK / 'PILOT_PIXEL_MANIFEST.csv').open(newline='') as file:
        for row in csv.DictReader(file):
            role, i = row['role'], int(row['id'])
            path = RECOVERY / 'datasets/coco' / row['existing_subdirectory'] / row['file_name']
            assert role in ('TRAIN', 'DEVELOPMENT') and path.is_file()
            (train if role == 'TRAIN' else dev)[str(i)] = str(path)
    with np.load(WORK / 'PILOT_MASK.npz') as data:
        ids, classes = data['ids'], data['classes']
        assert ids.tolist() == list(map(int, train)) and classes.tolist() == D.CLASSES
        masks = {arm: data[arm] for arm in ARMS}
    assert len(train) == 4008 and len(dev) == 5000 and not set(train) & set(dev)
    for arm, matrix in masks.items():
        expected = CONFIG['known'] + 120 * arm.endswith(('TRUE_BG', 'TRUE_PARTNER'))
        assert matrix.shape == (4008, 20) and np.isin(matrix, (-1, 0, 1)).all()
        assert int((matrix != -1).sum()) == expected
    assert CONFIG['lr'] == .002 * 8976 / CONFIG['known']
    return ids, train, np.array(list(map(int, dev))), dev, masks


class Head(nn.Module):
    def __init__(self, native):
        super().__init__()
        self.learner = copy.deepcopy(native.net.prompt_learner)
        object.__setattr__(self, 'encoder', native.net.text_encoder)

    def forward(self, feature):
        prompts, tokens = self.learner()
        text = self.encoder(prompts, tokens)
        text = text / text.norm(dim=-1, keepdim=True)
        image = feature / feature.norm(dim=1, keepdim=True)
        raw = 20 * F.conv1d(image, text[:, :, None])
        b, c, _ = raw.shape
        w = raw[:, c // 2:].softmax(-1)
        return (5 * (raw * torch.cat([w, w], 1)).sum(-1)).reshape(b, 2, c // 2)


def params(module):
    return [p for p in module.parameters() if p.requires_grad]


def compare(x, y):
    x, y = x.detach(), y.detach()
    assert x.shape == y.shape
    return dict(exact=bool(torch.equal(x, y)), max_abs=float((x - y).abs().max()))


def tune(opt, epoch):
    for group in opt.param_groups:
        group['lr'] = CONFIG['lr'] * D.factor(epoch)


def step(module, opt, value, target, observe, diagnostics=False):
    opt.zero_grad(set_to_none=True)
    logits = module(value)
    loss = D.loss_fn(logits, target, observe)
    loss.backward()
    opt.step()
    metrics = torch.stack([loss.detach(), *(p.grad.detach().abs().amax() for p in params(module))])
    if diagnostics:
        return metrics, logits.detach()
    return metrics


def parity(native, ids, paths, masks):
    cases = []
    for epoch, arm, tail in ((0, 'LOW', False), (1, 'S0_TRUE_BG', True), (2, 'S0_TRUE', False)):
        reference, shared = None, None
        for row in loader(ids, paths, masks['LOW'], DRAW, epoch, workers=WORKERS):
            reference = row
            if not tail:
                break
        for row in loader(ids, paths, masks[arm], DRAW, epoch, workers=WORKERS):
            shared = row
            if not tail:
                break
        assert torch.equal(reference[0], shared[0]) and torch.equal(reference[2], shared[2])
        pixels, target = shared[0].cuda(), shared[1].cuda()
        before = [x.detach().clone() for x in params(native)]
        head = Head(native)
        with torch.no_grad():
            first, _ = native.net.image_encoder(pixels)
            again, _ = native.net.image_encoder(reference[0].cuda())
            assert torch.equal(first, again)
        opt = D.optimizer_for(native, CONFIG['lr'] * D.factor(epoch))
        head_opt = torch.optim.SGD(params(head), lr=CONFIG['lr'] * D.factor(epoch), momentum=.9)
        observe = bool(target.ne(-1).any())
        native_value, native_logits = step(native, opt, pixels, target, observe, True)
        shared_value, shared_logits = step(head, head_opt, first, target, observe, True)
        grads = [compare(a.grad, b.grad) for a, b in zip(params(native), params(head))]
        updates = [compare(a, b) for a, b in zip(params(native), params(head))]
        row = dict(epoch=epoch, arm=arm, batch=len(target), pixels_exact=True, features_exact=True,
                   logits=compare(native_logits, shared_logits), loss=compare(native_value[:1], shared_value[:1]),
                   gradients=grads, updates=updates)
        cases.append(row)
        with torch.no_grad():
            for parameter, value in zip(params(native), before):
                parameter.copy_(value)
    exact = all(r['logits']['exact'] and r['loss']['exact'] and
                all(x['exact'] for x in r['gradients'] + r['updates']) for r in cases)
    save_json('PARITY_QA.json', dict(exact=exact, cases=cases))
    return exact


def shared_epoch(native, ids, paths, masks, lanes, epoch=0):
    jobs = [(arm, Head(native)) for arm in (ARMS * 3)[:CONFIG['benchmark_heads']]]
    jobs = [(arm, head, torch.optim.SGD(params(head), lr=CONFIG['lr'], momentum=.9)) for arm, head in jobs]
    for _, _, optimizer in jobs:
        tune(optimizer, epoch)
    thread = threading.local()
    index = {int(i): j for j, i in enumerate(ids)}
    streams = [torch.cuda.Stream() for _ in range(lanes)]
    stream_slots = iter(streams)
    losses = [[] for _ in jobs]
    encoder_seconds = 0.
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.monotonic()

    def initialize():
        thread.stream = next(stream_slots)

    with ThreadPoolExecutor(max_workers=lanes, initializer=initialize) as pool:
        for b, (pixels, _, ii) in enumerate(loader(ids, paths, masks['LOW'], DRAW, epoch, workers=WORKERS)):
            t0 = time.monotonic()
            positions = [index[int(i)] for i in ii]
            with torch.no_grad():
                feature, _ = native.net.image_encoder(pixels.cuda(non_blocking=True))
            labels = {arm: masks[arm][positions] for arm in ARMS}
            targets = {arm: torch.from_numpy(arr).cuda() for arm, arr in labels.items()}
            observed = {arm: bool((arr != -1).any()) for arm, arr in labels.items()}
            ready = torch.cuda.Event()
            ready.record()
            encoder_seconds += time.monotonic() - t0

            def run(job):
                arm, head, optimizer = job
                stream = thread.stream
                with torch.cuda.stream(stream):
                    stream.wait_event(ready)
                    value = step(head, optimizer, feature, targets[arm], observed[arm])
                    stream.synchronize()
                    return value.cpu().numpy().tolist()

            for j, value in enumerate(pool.map(run, jobs)):
                losses[j].append(value)
            if b % 42 == 0:
                report('SHARED_BENCHMARK', lanes=lanes, batch=b + 1, heads=len(jobs))
    elapsed = time.monotonic() - started
    assert all(len(trace) == 126 for trace in losses) and np.isfinite(losses).all()
    contexts = [[p.detach().cpu().clone() for p in params(head)] for _, head, _ in jobs]
    return dict(lanes=lanes, epoch=epoch + 1, seconds=elapsed, heads=len(jobs), batches=126,
                encoder_enqueue_seconds=encoder_seconds, peak_bytes=torch.cuda.max_memory_allocated(),
                endpoints_per_gpu_hour=len(jobs) * 3600 / (elapsed * 50)), contexts, np.array(losses)


def native_epoch(model, optimizer, ids, paths, mask, epoch):
    tune(optimizer, epoch)
    started = time.monotonic()
    seen, values, known, batch_neg = [], [], 0, []
    for pixels, target, ii in loader(ids, paths, mask, DRAW, epoch, workers=WORKERS):
        seen.extend(ii.tolist())
        n = int(target.ne(-1).sum())
        batch_neg.append(int(target.eq(0).sum()))
        known += n
        values.append(step(model, optimizer, pixels.cuda(non_blocking=True), target.cuda(non_blocking=True), n > 0))
    result = torch.stack(values).detach().cpu().numpy()
    assert np.isfinite(result).all() and all(torch.isfinite(p).all() for p in params(model))
    assert len(result) == 126 and known == int((mask != -1).sum())
    assert seen == ids[order_for(DRAW, epoch, len(ids))].tolist()
    return dict(epoch=epoch + 1, steps=126 * (epoch + 1), known=known,
                batch_negatives=batch_neg, loss_sum=float(result[:, 0].sum(dtype=np.float64)),
                seconds=time.monotonic() - started), result


def checkpoint(model, opt, history):
    result = D.compact(model, opt)
    result.update(history=history.copy(), seed=SEED, draw=DRAW, steps=len(history) * 126, lr=CONFIG['lr'])
    save_torch('NATIVE_LAST.pt', result)
    save_json('NATIVE_EPOCHS.json', history)


def reload(model, optimizer):
    state = torch.load(WORK / 'NATIVE_LAST.pt', weights_only=False, map_location='cpu')
    assert state['seed'] == SEED and state['draw'] == DRAW and state['steps'] == 126 * len(state['history'])
    D.restore(model, state)
    optimizer.load_state_dict(state['optimizer'])
    D.rng_restore(state['RNG'])
    assert D.rng_equal(state['RNG'], D.rng_snapshot())
    return state['history']


def scoring(model, ids, paths):
    direct, cached, seen = [], [], []
    head = Head(model)
    start = time.monotonic()
    with torch.no_grad():
        for pixels, _, ii in loader(ids, paths, None, workers=WORKERS):
            pixels = pixels.cuda(non_blocking=True)
            direct.append(model(pixels).softmax(1)[:, 1].cpu())
            feature, _ = model.net.image_encoder(pixels)
            cached.append(head(feature).softmax(1)[:, 1].cpu())
            seen.extend(ii.tolist())
    assert seen == ids.tolist()
    a, b = torch.cat(direct), torch.cat(cached)
    assert a.shape == (5000, 20) and a.isfinite().all() and b.isfinite().all()
    return a, b, time.monotonic() - start


def main():
    boot = time.monotonic()
    D.setup()
    torch.set_num_threads(4)
    seed_all()
    assert torch.cuda.device_count() == 1
    ids, train, dev_ids, dev, masks = inputs()
    native = D.build_dc(init_seed=SEED)
    native.eval()
    boot_seconds = time.monotonic() - boot
    initial = [p.detach().cpu().clone() for p in params(native)]
    before = {n: t.detach().cpu().clone() for n, t in native.named_buffers()}
    report('NATIVE_PARITY', gpu=torch.cuda.get_device_name(), boot_seconds=boot_seconds)
    cache_exact = parity(native, ids, train, masks)
    benchmark, shared_reference = [], None
    if cache_exact:
        for lanes in (1, 2, 3):
            try:
                timing, contexts, trace = shared_epoch(native, ids, train, masks, lanes)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                benchmark.append(dict(lanes=lanes, status='OOM'))
                continue
            if shared_reference is None:
                shared_reference = contexts, trace
            ref_contexts, ref_trace = shared_reference
            timing['lane_exact'] = np.array_equal(trace, ref_trace) and all(
                torch.equal(a, b) for pair, ref in zip(contexts, ref_contexts) for a, b in zip(pair, ref))
            timing['status'] = 'MEASURED'
            benchmark.append(timing)
            save_json('BENCHMARK.json', benchmark)
    optimizer = D.optimizer_for(native, CONFIG['lr'])
    assert all(torch.equal(x.cpu(), y) for x, y in zip(params(native), initial))
    history = []
    recovery = None
    for epoch in range(50):
        report('NATIVE_TRAINING', epoch=epoch + 1, completed_steps=126 * len(history))
        row, trace = native_epoch(native, optimizer, ids, train, masks['LOW'], epoch)
        if epoch == 0 and shared_reference is not None:
            ref, ref_trace = shared_reference
            save_json('FULL_EPOCH_PARITY.json', dict(exact=np.array_equal(trace, ref_trace[0]) and
                all(torch.equal(p.cpu(), x) for p, x in zip(params(native), ref[0]))))
        if epoch == 1:
            before_restart = copy.deepcopy(D.compact(native, optimizer))
            restored_history = reload(native, optimizer)
            assert len(restored_history) == epoch
            replay, replay_trace = native_epoch(native, optimizer, ids, train, masks['LOW'], epoch)
            momenta = optimizer.state_dict()['state']
            recovery = dict(steps=252, exact=np.array_equal(trace, replay_trace) and
                all(torch.equal(p.cpu(), t) for p, t in zip(params(native), before_restart['trainable'].values())) and
                all(torch.equal(v['momentum_buffer'], before_restart['optimizer']['state'][i]['momentum_buffer'])
                    for i, v in momenta.items()) and D.rng_equal(D.rng_snapshot(), before_restart['RNG']),
                replay_seconds=replay['seconds'])
            save_json('RECOVERY_QA.json', recovery)
            assert recovery['exact']
        history.append(row)
        checkpoint(native, optimizer, history)
    assert history[-1]['steps'] == 6300
    assert all(torch.equal(before[n], t.cpu()) for n, t in native.named_buffers())
    report('DEVELOPMENT_SCORING', rows=len(dev_ids))
    direct, cached, score_seconds = scoring(native, dev_ids, dev)
    del native, optimizer
    torch.cuda.empty_cache()
    independent = D.build_dc(init_seed=SEED)
    independent_opt = D.optimizer_for(independent, CONFIG['lr'])
    assert len(reload(independent, independent_opt)) == 50
    replay, _, replay_seconds = scoring(independent, dev_ids, dev)
    assert torch.equal(direct, replay)
    np.savez(WORK / 'DEV_SCORES.npz', ids=dev_ids, scores=direct.numpy(), classes=np.array(D.CLASSES))
    scores = compare(direct, cached)
    full_epoch_exact = json.loads((WORK / 'FULL_EPOCH_PARITY.json').read_text())['exact'] if cache_exact else False
    qualified = [x for x in benchmark if x.get('lane_exact')]
    shared_ok = cache_exact and full_epoch_exact and scores['exact'] and bool(qualified)
    selected = max(qualified, key=lambda x: x['endpoints_per_gpu_hour']) if shared_ok else None
    result = dict(status='PILOT_COMPLETE', gpu=torch.cuda.get_device_name(), native_epochs=50, native_updates=6300,
        native_seconds=sum(row['seconds'] for row in history), score_seconds=score_seconds,
        independent_rescore_seconds=replay_seconds, independent_rescore_exact=True,
        score_cache_parity=scores, restart=recovery, shared_exact=shared_ok, selected=selected,
        benchmark=benchmark, boot_seconds=boot_seconds, runtime_seconds=time.monotonic()-boot,
        development_only=True, new_confirmation_scored=False)
    save_json('PILOT_RESULT.json', result)
    report('PILOT_COMPLETE', native_updates=6300, shared_exact=shared_ok, selected=selected)


if __name__ == '__main__':
    main()
