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

"""Minimal adapter around the frozen upstream DualCoOp implementation.

Only the following departures from upstream are made and recorded:
  * 448 input through upstream build_model_conv_proj;
  * frozen image encoder runs under no_grad (mathematically identical for the
    context gradient because the visual branch is frozen);
  * `resize` -> `reshape` (same contiguous tensor);
  * the upstream attention pooling already computes the global x1[0] but does
    not return it.  A subclass stores it for the LIN cache; x2/attn are
    unchanged.
"""
import math
import os
import random
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from suite_common import UPSTREAM, CLASSES, R11

if str(UPSTREAM) not in sys.path:
    sys.path.insert(0, str(UPSTREAM))
from clip.model import build_model_conv_proj, AttentionConv  # noqa: E402
from models.dualcoop import DualCoop, TextEncoder  # noqa: E402

N_CTX = 16
TEXT_WIDTH = 512
IMAGE_SIZE = 448


class GlobalAttentionConv(AttentionConv):
    """Exact upstream AttentionConv plus a captured global x1[0] tensor."""

    def __init__(self, embed_dim, spec_dim, output_dim=None):
        super().__init__(embed_dim, spec_dim, output_dim)
        self.last_global = None

    def forward(self, x):
        _, _, h, w = x.shape
        if h * w + 1 != self.positional_embedding.shape[0]:
            w_spacial = self.positional_embedding[1:].permute(1, 0).reshape(1, -1, self.spec_dim, self.spec_dim)
            w_spacial = F.interpolate(w_spacial, size=(h, w), mode='bicubic')
            w_special = w_spacial.reshape(-1, h * w).permute(1, 0)
            positional_embedding = torch.cat([self.positional_embedding[:1], w_special], dim=0)
        else:
            positional_embedding = self.positional_embedding
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)
        x = x + positional_embedding[:, None, :].to(x.dtype)
        with torch.no_grad():
            x1, attn_weights = F.multi_head_attention_forward(
                query=x, key=x, value=x,
                embed_dim_to_check=x.shape[-1],
                num_heads=32,
                q_proj_weight=self.q_proj.weight,
                k_proj_weight=self.k_proj.weight,
                v_proj_weight=self.v_proj.weight,
                in_proj_weight=None,
                in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
                bias_k=None, bias_v=None, add_zero_attn=False, dropout_p=0,
                out_proj_weight=self.c_proj.weight, out_proj_bias=self.c_proj.bias,
                use_separate_proj_weight=True, training=self.training,
                need_weights=True, average_attn_weights=True)
            self.last_global = x1[0]
        x2 = F.linear(x, self.v_proj.weight, self.v_proj.bias)
        x2 = F.linear(x2, self.c_proj.weight, self.c_proj.bias)
        x2 = x2.permute(1, 2, 0)
        return x2, attn_weights[:, :1]


class NS:
    pass


def cfg_for(image_size=IMAGE_SIZE):
    cfg = NS()
    cfg.INPUT = NS(); cfg.INPUT.SIZE = [image_size, image_size]
    cfg.MODEL = NS(); cfg.MODEL.BACKBONE = NS(); cfg.MODEL.BACKBONE.NAME = 'RN50'
    cfg.TRAINER = NS(); cfg.TRAINER.COOP_MLC = NS()
    cfg.TRAINER.COOP_MLC.N_CTX_POS = N_CTX
    cfg.TRAINER.COOP_MLC.N_CTX_NEG = N_CTX
    cfg.TRAINER.COOP_MLC.CSC = False
    cfg.TRAINER.COOP_MLC.POSITIVE_PROMPT_INIT = ''
    cfg.TRAINER.COOP_MLC.NEGATIVE_PROMPT_INIT = ''
    cfg.TRAINER.COOP_MLC.LS = 1.0
    cfg.TRAINER.FINETUNE_BACKBONE = False
    cfg.TRAINER.FINETUNE_ATTN = False
    cfg.USE_CUDA = True
    return cfg


def build_clip(image_size=IMAGE_SIZE, weights=None):
    weights = Path(weights or (R11 / 'assets/RN50.pt'))
    jit = torch.jit.load(str(weights), map_location='cpu')
    sd = jit.state_dict()
    cfg = cfg_for(image_size)
    clip_model = build_model_conv_proj(sd, cfg)
    clip_model.float().eval()
    return clip_model, cfg


def replace_attnpool(clip_model, image_size=IMAGE_SIZE):
    old = clip_model.visual.attnpool
    new = GlobalAttentionConv(clip_model.visual.embed_dim, image_size // 32, 1024)
    new.load_state_dict(old.state_dict())
    new.float().eval()
    clip_model.visual.attnpool = new
    return new


def make_cold_fixture(init_seed, image_size=IMAGE_SIZE, weights=None):
    clip_model, cfg = build_clip(image_size, weights)
    replace_attnpool(clip_model, image_size)
    model = DualCoop(cfg, CLASSES, clip_model)
    g = torch.Generator().manual_seed(int(init_seed))
    pos = torch.empty(N_CTX, TEXT_WIDTH).normal_(0.0, 0.02, generator=g)
    neg = torch.empty(N_CTX, TEXT_WIDTH).normal_(0.0, 0.02, generator=g)
    model.prompt_learner.ctx_pos = nn.Parameter(pos)
    model.prompt_learner.ctx_neg = nn.Parameter(neg)
    return model, dict(ctx_pos=pos.clone(), ctx_neg=neg.clone(), init_seed=int(init_seed),
                       n_ctx=N_CTX, text_width=TEXT_WIDTH, classes=list(CLASSES))


def load_fixture(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def broadcast_fixture(fix, n_cls=20):
    pos = fix['ctx_pos'].unsqueeze(0).repeat(n_cls, 1, 1).clone()
    neg = fix['ctx_neg'].unsqueeze(0).repeat(n_cls, 1, 1).clone()
    assert pos.shape == (n_cls, N_CTX, TEXT_WIDTH) and neg.shape == (n_cls, N_CTX, TEXT_WIDTH)
    assert pos.data_ptr() != fix['ctx_pos'].data_ptr()
    return pos, neg


class DCModel(nn.Module):
    def __init__(self, csc=False, init_seed=61101, fixture_path=None, image_size=IMAGE_SIZE,
                 weights=None):
        super().__init__()
        clip_model, cfg = build_clip(image_size, weights)
        replace_attnpool(clip_model, image_size)
        self.net = DualCoop(cfg, CLASSES, clip_model)
        self.net.float().eval()
        self.csc = bool(csc)
        self.image_size = image_size
        fix = load_fixture(fixture_path) if fixture_path else None
        if csc:
            if fix is None:
                fix = make_cold_fixture(init_seed, image_size, weights)[1]
            pos, neg = broadcast_fixture(fix)
            self.net.prompt_learner.ctx_pos = nn.Parameter(pos)
            self.net.prompt_learner.ctx_neg = nn.Parameter(neg)
            self.init_source = dict(kind='broadcast_csc_false', init_seed=init_seed,
                                    fixture=str(fixture_path))
        else:
            if fix is None:
                _, fix = make_cold_fixture(init_seed, image_size, weights)
            self.net.prompt_learner.ctx_pos = nn.Parameter(fix['ctx_pos'].clone())
            self.net.prompt_learner.ctx_neg = nn.Parameter(fix['ctx_neg'].clone())
            self.init_source = dict(kind='cold_normal_0_0.02', init_seed=init_seed,
                                    fixture=(str(fixture_path) if fixture_path else None))
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.net.prompt_learner.ctx_pos.requires_grad_(True)
        self.net.prompt_learner.ctx_neg.requires_grad_(True)
        self.net.eval()

    def forward(self, image, return_global=False):
        with torch.no_grad():
            image_features, _ = self.net.image_encoder(image)
            global_feature = self.net.image_encoder.attnpool.last_global
        prompts, tokenized = self.net.prompt_learner()
        text_features = self.net.text_encoder(prompts, tokenized)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        output = 20 * F.conv1d(image_features, text_features[:, :, None])
        b, c, _ = output.shape
        output_half = output[:, c // 2:]
        w_half = F.softmax(output_half, dim=-1)
        w = torch.cat([w_half, w_half], dim=1)
        output = 5 * (output * w).sum(-1)
        logits = output.reshape(b, 2, c // 2)
        if return_global:
            return logits, global_feature
        return logits

    def text_vectors(self):
        prompts, tokenized = self.net.prompt_learner()
        with torch.no_grad():
            tf = self.net.text_encoder(prompts, tokenized)
            tf = tf / tf.norm(dim=-1, keepdim=True)
        return tf

    def encode_global(self, image):
        with torch.no_grad():
            _ = self.net.image_encoder(image)
            return self.net.image_encoder.attnpool.last_global


def build_dc(csc=False, init_seed=61101, fixture_path=None, image_size=448, weights=None, device='cuda'):
    m = DCModel(csc=csc, init_seed=init_seed, fixture_path=fixture_path, image_size=image_size, weights=weights)
    return m.to(device).eval()


def optimizer_for(model, base_lr):
    params = [model.net.prompt_learner.ctx_pos, model.net.prompt_learner.ctx_neg]
    return torch.optim.SGD(params, lr=base_lr, momentum=0.9, weight_decay=0.0,
                           dampening=0.0, nesterov=False)


def loss_fn(logits, target, known_nonzero=True):
    if not known_nonzero:
        return logits.sum() * 0
    known = target.ne(-1)
    prob = logits.softmax(1)
    y = target[known].float()
    pp = prob[:, 1][known]
    pn = (prob[:, 0][known] + 0.05).clamp(max=1)
    base = y * pp.clamp(min=1e-6).log() + (1 - y) * pn.clamp(min=1e-6).log()
    with torch.no_grad():
        focal = (1 - (y * pp + (1 - y) * pn)).pow(y + (1 - y) * 2)
    return -(base * focal).sum()


def factor(e):
    return 0.1 if e == 0 else (1 + math.cos(math.pi * (e - 1) / 49)) / 2


def setup():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def rng_snapshot():
    return dict(Python=random.getstate(), numpy=np.random.get_state(),
                torch_CPU=torch.get_rng_state(),
                torch_CUDA=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def rng_restore(r):
    random.setstate(r['Python'])
    np.random.set_state(r['numpy'])
    torch.set_rng_state(r['torch_CPU'])
    if torch.cuda.is_available() and r.get('torch_CUDA'):
        torch.cuda.set_rng_state_all(r['torch_CUDA'])


def rng_equal(a, b):
    if a['Python'] != b['Python']:
        return False
    if a['numpy'][0] != b['numpy'][0] or not np.array_equal(a['numpy'][1], b['numpy'][1]) or a['numpy'][2:] != b['numpy'][2:]:
        return False
    if not torch.equal(a['torch_CPU'], b['torch_CPU']):
        return False
    return all(torch.equal(x, y) for x, y in zip(a.get('torch_CUDA', []), b.get('torch_CUDA', [])))


def compact(model, opt):
    return dict(
        trainable={n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad},
        buffers={n: b.detach().cpu().clone() for n, b in model.named_buffers()},
        optimizer=opt.state_dict(), RNG=rng_snapshot())


def restore(model, cp):
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    with torch.no_grad():
        for n, v in cp['trainable'].items():
            params[n].copy_(v)
        for n, v in cp['buffers'].items():
            buffers[n].copy_(v)
