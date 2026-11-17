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

"""Official CLIP ViT-B/16 with frozen encoders and the DualCoOp spatial readout."""

import copy
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
RECOVERY = Path(os.environ.get('FQ_RECOVERY_ROOT', Path(__file__).resolve().parents[3] / 'runtime'))
sys.path[:0] = [str(RECOVERY / p) for p in ('src', 'vendor/DualCoOp', 'vendor/randaugment')]

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms

from clip import clip
from clip.model import build_model
from models.dualcoop import MLCPromptLearner, TextEncoder
import dc_model as D
from data import Pixels as V9Pixels, CutoutPIL, MEAN, STD, order_for, AUG_SEED, RandAugment


HERE = Path(__file__).resolve().parent
RULES = json.loads((HERE / 'RULES.json').read_text())
ARMS = tuple(RULES['arms'])


class Pixels(V9Pixels):
    def __init__(self, ids, paths, visible, draw, epoch):
        super().__init__(ids, paths, visible, draw, epoch)
        if visible is None:
            self.ops = clip._transform(224)
        else:
            self.ops = transforms.Compose([transforms.Resize((224, 224)), CutoutPIL(.5), RandAugment(),
                                           transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def loader(ids, paths, visible=None, draw=0, epoch=0, workers=6):
    ds = Pixels(ids, paths, visible, draw, epoch)
    generator = torch.Generator().manual_seed(int(np.random.SeedSequence(
        [AUG_SEED, draw, epoch, 1]).generate_state(1)[0]))
    order = order_for(draw, epoch, len(ids)) if visible is not None else list(range(len(ids)))
    return torch.utils.data.DataLoader(ds, batch_size=32, sampler=order, num_workers=workers,
             pin_memory=True, drop_last=False, persistent_workers=False,
             prefetch_factor=2 if workers else None, generator=generator)


def text_classes():
    return [RULES['text_class_aliases'].get(c, c) for c in D.CLASSES]


def visual_tokens(visual, pixels):
    x = visual.conv1(pixels).flatten(2).transpose(1, 2)
    cls = visual.class_embedding.expand(len(x), 1, -1)
    x = visual.ln_pre(torch.cat((cls, x), dim=1) + visual.positional_embedding)
    return visual.transformer(x.transpose(0, 1)).transpose(0, 1)


class Encoder(nn.Module):
    def __init__(self, visual):
        super().__init__()
        self.visual = visual

    def forward(self, x):
        tokens = visual_tokens(self.visual, x)
        patches = self.visual.ln_post(tokens[:, 1:]) @ self.visual.proj
        return patches.transpose(1, 2).contiguous(), None


class Native(nn.Module):
    def __init__(self):
        super().__init__()
        path = HERE / RULES['weight_name']
        assert path.stat().st_size == RULES['weight_size_bytes']
        archive = torch.jit.load(str(path), map_location='cpu')
        model = build_model(archive.state_dict()).float().eval()
        assert model.visual.input_resolution == 224 and model.visual.conv1.kernel_size == (16, 16)
        assert tuple(model.visual.positional_embedding.shape) == (197, 768)
        model.requires_grad_(False)
        self.clip_model = model
        self.net = nn.Module()
        self.net.image_encoder = Encoder(model.visual)
        self.net.prompt_learner = MLCPromptLearner(D.cfg_for(224), text_classes(), model)
        self.net.text_encoder = TextEncoder(model)


def projected_logits(image, text):
    image = image / image.norm(dim=1, keepdim=True)
    text = text / text.norm(dim=-1, keepdim=True)
    raw = 20 * F.conv1d(image, text[:, :, None])
    weight = raw[:, raw.shape[1] // 2:].softmax(-1)
    return (5 * (raw * torch.cat([weight, weight], 1)).sum(-1)).reshape(len(raw), 2, 20)


class Head(nn.Module):
    def __init__(self, native):
        super().__init__()
        self.learner = copy.deepcopy(native.net.prompt_learner)
        object.__setattr__(self, 'encoder', native.net.text_encoder)

    def text(self):
        return self.encoder(*self.learner())

    def forward(self, feature):
        return projected_logits(feature, self.text())


def params(head):
    return [head.learner.ctx_pos, head.learner.ctx_neg]


def step(head, optimizer, feature, target, known):
    optimizer.zero_grad(set_to_none=True)
    loss = D.loss_fn(head(feature), target, known)
    loss.backward()
    optimizer.step()
    return torch.stack([loss.detach(), *(p.grad.detach().abs().amax() for p in params(head))])
