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

"""Audit solver output directly from the labels, coordinates, and emitted masks."""

from itertools import combinations
from pathlib import Path
import json

import numpy as np

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent/'voc_design'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    lock = read(PARENT/'DATA_ROLE_LOCK.json')
    parent = read(PARENT/'MASK_QA.json')
    rules = read(HERE/'RULES.json')
    qa = read(HERE/'MASK_QA.json')
    assert lock['status']=='PASS_DATA_QUALIFICATION' and qa['status']=='PASS'
    assert qa['dose']==rules['fixed_dose'] and len(qa['draws'])==2
    assert all(d['offset']==0 for d in qa['draws'])
    with np.load(PARENT/'FIT_GOLD.npz') as g, np.load(PARENT/'FIT_COSINE.npz') as f:
        ids, truth, classes = g['ids'], g['truth'], g['classes'].tolist()
        assert np.array_equal(ids, f['ids'])
        scores = f['scores'].astype(float)
    target_names = [t for t,_ in lock['pairs']['P1']]
    targets = [classes.index(t) for t in target_names]
    partner = [[classes.index(p) for _,p in lock['pairs'][k]] for k in ('P1','P2')]
    position = {v:j for j,v in enumerate(ids.tolist())}
    evidence = []
    for record in qa['draws']:
        draw = record['draw']
        with np.load(HERE/f'MASK_{draw}.npz') as z:
            assert z['ids'].tolist()==[int(i) for i in ids] and z['classes'].tolist()==classes
            base = z['LOW']
            arms = [z[k] for k in rules['arms']]
        known = (np.random.default_rng(draw).uniform(size=truth.shape)<.1)&(truth!=0)
        assert np.array_equal(base, np.where(known,(truth==1).astype(np.int8),-1))
        other_density = np.minimum(np.delete(known,targets,axis=1).sum(axis=1),2)
        rows = []
        for j,t in enumerate(targets):
            dose = qa['dose'][j]
            b1,b2 = partner[0][j],partner[1][j]
            negative = sorted(np.flatnonzero(truth[:,t]==-1),key=lambda i:(scores[i,t],ids[i]))
            bins = np.full(len(ids),-1)
            for k,group in enumerate(np.array_split(negative,10)):
                bins[group]=k
            for layer in record['coordinates'][target_names[j]]:
                samples = [[position[i] for i in v] for v in layer['ids']]
                assert len(set(map(len,samples)))==1
                expected, = [r['quota'] for r in parent['layers'] if
                    (r['draw'],r['dose_index'],r['target'],r['difficulty'],r['other_labels'])==
                    (draw,1,target_names[j],layer['difficulty'],layer['other_labels'])]
                assert len(samples[0])==expected
                for indexes in samples:
                    assert np.all(bins[indexes]==layer['difficulty'])
                    assert np.all(other_density[indexes]==layer['other_labels'])
            for arm in arms:
                assert np.array_equal(arm==1,base==1)
                assert np.array_equal(np.delete(arm,targets,axis=1),np.delete(base,targets,axis=1))
                assert not np.any((arm!=-1)&(truth==0))
                assert np.array_equal(np.where(arm!=-1,truth==1,False),arm==1)
                assert not np.any((arm==0)&(truth!=-1))
            remove = [np.flatnonzero((base[:,t]==0)&(a[:,t]==-1)) for a in arms]
            assert all(np.array_equal(remove[0],r) for r in remove)
            add = [np.flatnonzero((base[:,t]==-1)&(a[:,t]==0)) for a in arms]
            groups = [remove[0],*add]
            for k,g in enumerate(groups):
                declared = sorted(position[i] for r in record['coordinates'][target_names[j]] for i in r['ids'][k])
                assert g.tolist()==declared and len(g)==dose
            assert np.all(truth[groups[0]][:,[b1,b2]]==-1)
            assert np.all(truth[add[0],b1]==1) and np.all(truth[add[1],b2]==1)
            effect = max(abs(scores[a,t].mean()-scores[b,t].mean()) for a,b in combinations(groups,2))
            effect /= np.std(scores[negative,t])
            gap = min(1-float(np.mean(truth[g,p]==1))
                      for p,choices in ((b1,add[1:]),(b2,[add[0],add[2]])) for g in choices)
            assert effect<=rules['max_abs_smd'] and gap>=rules['minimum_directional_gap']-1e-12
            rows.append(dict(target=target_names[j], max_abs_smd=effect, min_partner_gap=gap))
        for a in arms:
            assert np.sum(a>=0)==np.sum(base>=0) and np.sum(a==0)==np.sum(base==0)
            assert ((a==0).any(0)&(a==1).any(0)).all()
        counts = [(a>=0).sum(1) for a in arms]
        dist = [np.bincount(x,minlength=21) for x in counts]
        tv = max(np.sum(abs(x-y))/(2*len(ids)) for x,y in combinations(dist,2))
        active = max(abs(np.count_nonzero(x)-np.count_nonzero(y)) for x,y in combinations(counts,2))
        assert tv<=rules['max_density_tv'] and active<=rules['max_active_difference']
        evidence.append(dict(draw=draw, counts=rows, maximum_tv=float(tv), active_difference=int(active)))
    result = dict(status='PASS', draws=evidence, dose=qa['dose'])
    (HERE/'MASK_INDEPENDENT_QA.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
