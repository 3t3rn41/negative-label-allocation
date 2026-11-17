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

"""Fit-only qualification, engineering pilot, and sealed formal scoring."""

import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import pilot as P
import formal_train_shared as F

torch = P.torch
HERE = Path(__file__).resolve().parent
RULES = P.RULES
ARMS = P.ARMS


def read(name):
    return json.loads((HERE / name).read_text())


def inputs(role, data):
    ids = np.array(data['role_ids'][role], dtype=np.int64)
    directory = Path(data['source']) / 'JPEGImages'
    return ids, {str(i): str(directory / f'{i:06d}.jpg') for i in ids}


def save_npz(name, **fields):
    path = HERE / name
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as output:
        np.savez_compressed(output, **fields)
    temporary.replace(path)


def difficulty(model, data):
    ids, paths = inputs('fit', data)
    start = time.time()
    seen, scores = [], []
    with torch.no_grad():
        texts = P.clip.tokenize([RULES['difficulty_template'].format(n) for n in P.text_classes()]).cuda()
        reference = torch.nn.functional.normalize(model.clip_model.encode_text(texts))
        for i, (images, _, batch) in enumerate(P.loader(ids, paths)):
            pixels = images.cuda(non_blocking=True)
            features = torch.nn.functional.normalize(model.clip_model.encode_image(pixels))
            scores.append((features @ reference.T).cpu().numpy())
            seen.extend(batch.tolist())
            if i % 42 == 0:
                print('FIT_COSINE', i + 1, len(seen), flush=True)
    assert seen == ids.tolist()
    scores = np.concatenate(scores)
    assert scores.shape == (4008, 20) and np.isfinite(scores).all()
    save_npz('FIT_COSINE.npz', ids=np.array(data['role_ids']['fit']), classes=data['classes'], scores=scores)
    F.atomic_json(HERE/'DIFFICULTY_QA.json', dict(status='PASS', rows=len(ids),
                  seconds=time.time()-start, role='fit', device=torch.cuda.get_device_name()))


def parity(model, data):
    ids, paths = inputs('dev', data)
    batch = next(iter(P.loader(ids[:4], paths, workers=0)))[0].cuda()
    v = model.clip_model.visual
    with torch.no_grad():
        fast, _ = model.net.image_encoder(batch)
        tokens = P.visual_tokens(v, batch)
        actual_cls = v.ln_post(tokens[:, 0]) @ v.proj
        reference_cls = model.clip_model.encode_image(batch)
        captured = []
        hook = v.transformer.register_forward_hook(lambda _, __, output: captured.append(output))
        try:
            model.clip_model.encode_image(batch)
        finally:
            hook.remove()
        last = captured[0][1:].permute(1, 0, 2).contiguous()
        normalized = torch.nn.functional.layer_norm(last, (768,), v.ln_post.weight, v.ln_post.bias,
                                                  v.ln_post.eps)
        reference_patch = torch.einsum('bsc,cd->bds', normalized, v.proj)
    assert fast.shape == (4, 512, 196)
    torch.testing.assert_close(actual_cls, reference_cls, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(fast, reference_patch, rtol=2e-4, atol=2e-5)
    head = F.make_jobs(model, RULES['initializations'][:1], ARMS[:1])[0][2]
    text = head.text()
    primary = P.projected_logits(fast, text)
    image = torch.nn.functional.normalize(reference_patch.transpose(1, 2), dim=-1)
    reference_text = torch.nn.functional.normalize(text, dim=-1)
    similarities = torch.einsum('bsd,kd->bks', image, reference_text) * 20
    attention = similarities[:, 20:].softmax(-1)
    logits = torch.stack([torch.einsum('bcs,bcs->bc', group, attention)
                         for group in similarities.split(20, dim=1)], dim=1) * 5
    torch.testing.assert_close(primary, logits, rtol=3e-5, atol=2e-5)
    y = torch.arange(80, device=batch.device).reshape(4, 20) % 3 - 1
    native_loss = P.D.loss_fn(primary, y)
    pos = logits.softmax(1)[:, 1][y == 1]
    neg = (logits.softmax(1)[:, 0][y == 0] + .05).clamp(max=1)
    other_loss = -(pos.clamp_min(1e-6).log() * (1-pos).detach()).sum() - (
        neg.clamp_min(1e-6).log() * (1-neg).detach().square()).sum()
    torch.testing.assert_close(native_loss, other_loss, rtol=1e-5, atol=2e-5)
    g1 = torch.autograd.grad(native_loss, P.params(head), retain_graph=True)
    g2 = torch.autograd.grad(other_loss, P.params(head))
    for a, b in zip(g1, g2):
        torch.testing.assert_close(a, b, rtol=3e-4, atol=1e-4)
    result = dict(status='PASS', image_count=len(batch), cls_max_abs=float((actual_cls-reference_cls).abs().max()),
                  patch_max_abs=float((fast-reference_patch).abs().max()),
                  logits_max_abs=float((primary-logits).detach().abs().max()),
                  loss_abs=float((native_loss-other_loss).detach().abs()),
                  gradient_max_abs=[float((a-b).abs().max()) for a,b in zip(g1,g2)])
    F.atomic_json(HERE/'PARITY_QA.json', result)
    print('PARITY', result, flush=True)


def checkpoint(draw, epoch):
    state = torch.load(HERE/f'draw_{draw}'/'CHECKPOINT.pt', map_location='cpu', weights_only=False)
    assert state['draw'] == draw and state['seeds'] == RULES['initializations']
    assert state['arms'] == list(ARMS) and state['epoch'] == epoch
    assert state['steps'] == 126 * epoch and len(state['heads']) == 36 and len(state['history']) == epoch
    return state


def resume_replay(model, ids, paths, draw, mask):
    state = checkpoint(draw, 1)
    selected = [0, 1, 2, 33, 34, 35]
    with np.load(mask) as matrices:
        visible = matrices['LOW']
        loader = P.loader(ids, paths, visible, draw, 1, workers=0)
        pixels, _, batch_ids = next(iter(loader))
        index = {v: j for j, v in enumerate(ids.tolist())}
        positions = [index[v] for v in batch_ids.tolist()]
        targets = {a: torch.from_numpy(matrices[a][positions]).cuda() for a in ARMS}
    with torch.no_grad():
        features, _ = model.net.image_encoder(pixels.cuda())
    with np.load(mask) as matrices:
        lr = .002 * 8976 / int((matrices['LOW'] >= 0).sum()) * P.D.factor(1)
    results = []
    for j in selected:
        seed = RULES['initializations'][j // 3]
        arm = ARMS[j % 3]
        outputs = []
        for _ in range(2):
            P.D.rng_restore(state['RNG'])
            _, _, head, opt = F.make_jobs(model, [seed], [arm])[0]
            # A restore may hand a CPU tensor to SGD; clone both momentum
            # buffers to keep the two replays independent.
            import copy
            F.restore_head(head, state['heads'][j]['parameters'])
            opt.load_state_dict(copy.deepcopy(state['heads'][j]['optimizer']))
            opt.param_groups[0]['lr'] = lr
            metrics = P.step(head, opt, features, targets[arm], True)
            outputs.append([metrics.detach(), *[p.detach().clone() for p in P.params(head)]])
        exact = all(torch.equal(a,b) for a,b in zip(*outputs))
        assert exact, (seed, arm)
        results.append(dict(seed=seed, arm=arm, identical=True))
    return results


def evaluate(model, ids, paths, jobs, deadline):
    values = np.empty((len(jobs), len(ids), 20), dtype=np.float32)
    seen = []
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    with torch.no_grad():
        cached = [head.text() for _, _, _, head in jobs]
        for b, (pixels, _, batch_ids) in enumerate(P.loader(ids, paths)):
            if time.time() >= deadline:
                raise TimeoutError('evaluation deadline')
            features, _ = model.net.image_encoder(pixels.cuda(non_blocking=True))
            begin = len(seen)
            seen.extend(batch_ids.tolist())
            for j, (text, (_, _, _, head)) in enumerate(zip(cached, jobs)):
                logits = P.projected_logits(features, text)
                if b == 0 or len(seen) == len(ids):
                    assert torch.equal(logits, head(features))
                values[j, begin:len(seen)] = logits.softmax(1)[:, 1].cpu().numpy()
            if b % 52 == 0:
                print('SCORE', len(seen), '/', len(ids), 'heads', len(jobs), flush=True)
    assert seen == ids.tolist() and np.isfinite(values).all() and ((values>=0)&(values<=1)).all()
    return values, dict(seconds=time.monotonic()-start, rows=len(ids), heads=len(jobs),
                        peak_bytes=torch.cuda.max_memory_allocated(), cached_text_parity='exact_first_and_tail')


def restored_jobs(model, draws, epoch):
    jobs = []
    for draw in draws:
        saved = checkpoint(draw, epoch)['heads']
        for (seed, arm, head, _), item in zip(F.make_jobs(model, RULES['initializations'], ARMS), saved, strict=True):
            F.restore_head(head, item['parameters'])
            jobs.append((draw, seed, arm, head))
    return jobs


def pilot(model, data):
    lock = read('PILOT_LOCK.json')
    assert lock['status'] == 'LOCKED_PILOT' and lock['rules'] == RULES
    deadline = lock['deadline_unix']
    assert time.time() < deadline
    draw = RULES['draws'][0]
    ids, paths = inputs('fit', data)
    mask = HERE / f'MASK_{draw}.npz'
    parity(model, data)
    torch.cuda.reset_peak_memory_stats()
    assert F.train_draw(model, ids, paths, mask, draw, RULES['initializations'], 3,
                        deadline, arms=ARMS, limit=1)
    replay = resume_replay(model, ids, paths, draw, mask)
    assert F.train_draw(model, ids, paths, mask, draw, RULES['initializations'], 3,
                        deadline, arms=ARMS, limit=RULES['pilot_epochs'])
    peak_train = torch.cuda.max_memory_allocated()
    state = checkpoint(draw, RULES['pilot_epochs'])
    dev_ids, dev_paths = inputs('dev', data)
    _, eval_qa = evaluate(model, dev_ids, dev_paths, restored_jobs(model, [draw], state['epoch']), deadline)
    result = dict(status='PASS', draw=draw, heads=36, epochs=state['epoch'], steps_per_head=state['steps'],
                  training_seconds=[r['seconds'] for r in state['history']], peak_training_bytes=peak_train,
                  resume_replay=replay, development_scoring=eval_qa, formal_images_scored=0)
    F.atomic_json(HERE/'PILOT_QA.json', result)
    print('PILOT', result, flush=True)


def formal(model, data):
    lock = read('PROTOCOL_LOCK.json')
    assert lock['status'] == 'LOCKED_FORMAL' and lock['rules'] == RULES
    assert lock['formal_ids'] == data['role_ids']['test']
    assert not (HERE/'TEST_GOLD.npz').exists()
    deadline = lock['worker_deadline_unix']
    train_ids, train_paths = inputs('fit', data)
    for draw in RULES['draws']:
        assert F.train_draw(model, train_ids, train_paths, HERE/f'MASK_{draw}.npz', draw,
                            RULES['initializations'], 3, deadline, arms=ARMS)
    states = [checkpoint(draw, 50) for draw in RULES['draws']]
    qa = dict(status='TRAINING_COMPLETE', endpoints=72, updates=453600,
              draws=[dict(draw=cp['draw'], steps=cp['steps'], history=cp['history']) for cp in states])
    F.atomic_json(HERE/'TRAIN_QA.json', qa)
    jobs = restored_jobs(model, RULES['draws'], 50)
    ids, paths = inputs('test', data)
    scores, perf = evaluate(model, ids, paths, jobs, deadline)
    draws, seeds, arms = zip(*[(draw, seed, arm) for draw,seed,arm,_ in jobs])
    save_npz('SCORES.npz', ids=np.array(data['role_ids']['test']), classes=P.D.CLASSES,
             scores=scores, draws=draws, seeds=seeds, arms=arms)
    F.atomic_json(HERE/'SCORE_RECEIPT.json', dict(status='SCORES_SEALED', shape=list(scores.shape),
                  test_labels_read=False, sealed_unix=time.time(), **perf))


def main():
    mode = sys.argv[1]
    random.seed(90921100)
    np.random.seed(90921100)
    torch.manual_seed(90921100)
    torch.cuda.manual_seed_all(90921100)
    P.D.setup()
    data = read('DATA_ROLE_LOCK.json')
    assert data['status'] == 'PASS_DATA_QUALIFICATION' and data['classes'] == P.D.CLASSES
    model = P.Native().cuda().eval()
    if mode == 'difficulty':
        difficulty(model, data)
        return
    if mode == 'parity':
        parity(model, data)
        return
    if mode == 'pilot':
        pilot(model, data)
        return
    if mode == 'formal':
        formal(model, data)
        return
    raise ValueError(mode)


if __name__ == '__main__':
    main()
