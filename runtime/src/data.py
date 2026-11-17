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

"""Frozen COCO20 pixel pipeline inherited from R11.

The augmentation seed and scoped RNG are the V3 protocol: seed 91001 keyed by
(draw, epoch, image_id).  No new augmentation policy is introduced.
"""
import random
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from randaugment import RandAugment

MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)
AUG_SEED = 91001


class CutoutPIL:
    def __init__(self, cutout_factor=0.5):
        self.cutout_factor = cutout_factor

    def __call__(self, x):
        img_draw = ImageDraw.Draw(x)
        h, w = x.size[0], x.size[1]
        h_cutout = int(self.cutout_factor * h + 0.5)
        w_cutout = int(self.cutout_factor * w + 0.5)
        y_c = np.random.randint(h)
        x_c = np.random.randint(w)
        y1 = np.clip(y_c - h_cutout // 2, 0, h)
        y2 = np.clip(y_c + h_cutout // 2, 0, h)
        x1 = np.clip(x_c - w_cutout // 2, 0, w)
        x2 = np.clip(x_c + w_cutout // 2, 0, w)
        fill_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        img_draw.rectangle([x1, y1, x2, y2], fill=fill_color)
        return x


class Pixels(Dataset):
    def __init__(self, ids, paths, visible=None, seed=0, epoch=0):
        self.ids = list(ids)
        self.paths = paths
        self.visible = visible
        self.seed = seed
        self.epoch = epoch
        ops = [transforms.Resize((448, 448))]
        if visible is not None:
            ops += [CutoutPIL(.5), RandAugment()]
        ops += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
        self.ops = transforms.Compose(ops)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, j):
        i = self.ids[j]
        with Image.open(self.paths[str(i)]) as im:
            image = im.convert('RGB')
        if self.visible is not None:
            state = (random.getstate(), np.random.get_state(), torch.get_rng_state())
            seed = int(np.random.SeedSequence([AUG_SEED, self.seed, self.epoch, int(i)]).generate_state(1)[0])
            random.seed(seed)
            np.random.seed(seed)
            torch.random.default_generator.manual_seed(seed)
            try:
                image = self.ops(image)
            finally:
                random.setstate(state[0])
                np.random.set_state(state[1])
                torch.set_rng_state(state[2])
        else:
            image = self.ops(image)
        return image, (np.array([]) if self.visible is None else self.visible[j].copy()), int(i)


def order_for(seed, epoch, n):
    return np.random.default_rng(np.random.SeedSequence([AUG_SEED, seed, epoch, 0])).permutation(n).tolist()


def loader(ids, paths, visible=None, seed=0, epoch=0, workers=4):
    ds = Pixels(ids, paths, visible, seed, epoch)
    g = torch.Generator().manual_seed(
        int(np.random.SeedSequence([AUG_SEED, seed, epoch, 1]).generate_state(1)[0]))
    order = order_for(seed, epoch, len(ids)) if visible is not None else list(range(len(ids)))
    return DataLoader(ds, batch_size=32, sampler=order, num_workers=workers, pin_memory=True,
                      drop_last=False, persistent_workers=workers > 0, prefetch_factor=2 if workers > 0 else None,
                      generator=g)


def confirm_loader(ids, paths, workers=4):
    return loader(ids, paths, None, workers=workers)
