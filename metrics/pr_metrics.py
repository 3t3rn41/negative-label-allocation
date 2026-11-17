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

"""Frozen metric arithmetic reused from the audited baseline suite.

This module is a copy-adaptation for the PR sprint.  It does not modify the
parent experiments.  AP_ENV is the audited interpolated-precision AP used by
v11 and the reference baseline.
"""
import numpy as np


class Metric:
    def __init__(self, scores, labels):
        scores = np.asarray(scores)
        labels = np.asarray(labels)
        assert scores.shape == labels.shape
        assert np.isfinite(scores).all() and np.isin(labels, [-1, 0, 1]).all()
        self.valid = np.flatnonzero(labels != -1)
        self.order = self.valid[np.argsort(-scores[self.valid], kind='stable')] if len(self.valid) else np.array([], int)
        if len(self.order):
            ordered = scores[self.order]
            self.starts = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1] if len(ordered) > 1 else np.array([0])
        else:
            self.starts = np.array([], int)
        self.y = labels[self.order].astype(float)
        self.n = len(scores)

    def _prepare(self, weights):
        if weights is None:
            weights = np.ones(self.n)
        w = np.asarray(weights)
        single = w.ndim == 1
        if single:
            w = w[None, :]
        if not len(self.order):
            return w, single, None
        sortedw = w[:, self.order]
        positive = np.add.reduceat(sortedw * self.y, self.starts, axis=1)
        negative = np.add.reduceat(sortedw * (1 - self.y), self.starts, axis=1)
        tp = positive.cumsum(1)
        fp = negative.cumsum(1)
        np_ = positive.sum(1)
        nn = negative.sum(1)
        return w, single, (sortedw, positive, negative, tp, fp, np_, nn)

    @staticmethod
    def _auc(positive, negative, fp, np_, nn):
        with np.errstate(invalid='ignore', divide='ignore'):
            auc = (positive * (nn[:, None] - fp + .5 * negative)).sum(1) / (np_ * nn)
        auc[(np_ == 0) | (nn == 0)] = np.nan
        return auc

    def compute_auc(self, weights=None):
        _, single, pack = self._prepare(weights)
        if pack is None:
            out = np.full(1 if single else len(np.asarray(weights)), np.nan)
            return float(out[0]) if single else out
        _, positive, negative, tp, fp, np_, nn = pack
        auc = self._auc(positive, negative, fp, np_, nn)
        return float(auc[0]) if single else auc

    def compute_ap(self, weights=None, rule='ENV'):
        assert rule in ('ENV', 'NI', 'VOC11')
        w = np.asarray(weights) if weights is not None else np.ones(self.n)
        single = w.ndim == 1
        if single:
            w = w[None, :]
        _, _, pack = self._prepare(w)
        if pack is None:
            out = np.full(len(w), np.nan)
            return float(out[0]) if single else out
        _, positive, negative, tp, fp, np_, nn = pack
        precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
        if rule == 'ENV':
            envelope = np.maximum.accumulate(precision[:, ::-1], axis=1)[:, ::-1]
            with np.errstate(invalid='ignore', divide='ignore'):
                ap = (positive * envelope).sum(1) / np_
        elif rule == 'NI':
            with np.errstate(invalid='ignore', divide='ignore'):
                ap = (positive * precision).sum(1) / np_
        else:
            with np.errstate(invalid='ignore', divide='ignore'):
                recall = tp / np_[:, None]
            vals = []
            for r in np.linspace(0, 1, 11):
                vals.append(np.where(recall >= r - 1e-12, precision, 0.0).max(axis=1))
            ap = np.mean(vals, axis=0)
        ap[np_ == 0] = np.nan
        return float(ap[0]) if single else ap


def conditional_labels(target, partner):
    target = np.asarray(target)
    partner = np.asarray(partner)
    labels = np.full(len(target), -1, np.int8)
    valid = (target != -1) & (partner != -1)
    labels[valid & (target == 1) & (partner == 0)] = 1
    labels[valid & (target == 0) & (partner == 1)] = 0
    return labels


def cell_labels(target, partner, positive_group, negative_group):
    target = np.asarray(target)
    partner = np.asarray(partner)
    labels = np.full(len(target), -1, np.int8)
    pos = (target == int(positive_group[0])) & (partner == int(positive_group[1]))
    neg = (target == int(negative_group[0])) & (partner == int(negative_group[1]))
    labels[pos] = 1
    labels[neg] = 0
    return labels
