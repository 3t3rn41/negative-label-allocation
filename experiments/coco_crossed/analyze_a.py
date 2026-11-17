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

"""Paired two-layer bootstrap for v9.1 stage A, after the score receipt."""

import csv
import itertools
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, t

HERE = Path(__file__).resolve().parent
PHASE0 = HERE.parent / 'coco_design'
sys.path.insert(0, str(PHASE0))
from historical_metrics import Metric, cell_labels
from prepare import PAIR_MAP, read, save

ARMS = ('S0_TRUE', 'S0_PLC', 'LOW')
CELLS = (('10','00'),('10','01'),('11','00'),('11','01'))


def init_worker(truth, weight_file, pstar):
    global TRUTH, WEIGHTS, PSTAR
    TRUTH = truth
    WEIGHTS = np.load(weight_file, mmap_mode='r')
    PSTAR = pstar


def rank_auc(y, s):
    n, m = int(y.sum()), int((1-y).sum())
    return (rankdata(s)[y==1].sum() - n*(n+1)/2)/(n*m)


def endpoint(scores):
    result = np.empty((len(WEIGHTS),3,3), np.float64)
    max_gap = 0.
    cell_points = []
    for j, pair in enumerate(PAIR_MAP['S0']):
        col, part = pair['target'], pair['partner']
        c, q, s = TRUTH[:,col], TRUTH[:,part], scores[:,col]
        models = [Metric(s,cell_labels(c,q,a,b)) for a,b in CELLS]
        fixed_p=PSTAR[j]
        for first in range(0,len(WEIGHTS),16):
            rows=slice(first,min(first+16,len(WEIGHTS)))
            w=WEIGHTS[rows]
            z=np.array([metric.compute_auc(w) for metric in models])
            with np.errstate(invalid='ignore',divide='ignore'):
                p=(w@(c*q))/(w@c)
                r=(w@((1-c)*q))/(w@(1-c))
            natural=(1-p)*((1-r)*z[0]+r*z[1])+p*((1-r)*z[2]+r*z[3])
            bg=(1-fixed_p)*z[0]+fixed_p*z[2]
            pn=(1-fixed_p)*z[1]+fixed_p*z[3]
            result[rows,j]=np.stack((natural,.95*bg+.05*pn,.50*bg+.50*pn),axis=1)
            if first==0:
                alt=np.array([rank_auc(m.y.astype(np.int8),s[m.order]) for m in models])
                max_gap=max(max_gap,float(np.max(np.abs(alt-z[:,0]))),
                            abs(float(natural[0])-rank_auc(c,s)))
                cell_points.append(z[:,0].tolist())
    assert max_gap < 1e-12
    return result, max_gap, cell_points


def bootstrap_summary(x, indices, min_valid):
    point=x[0]
    n=len(point)
    mean=float(point.mean())
    sd=float(point.std(ddof=1))
    t_one=float(t.ppf(.95,n-1))*sd/np.sqrt(n)
    valid_image=x[1:].mean(axis=1)
    joint=x[np.arange(1,len(x))[:,None],indices].mean(axis=1)

    def quantiles(values):
        valid=values[np.isfinite(values)]
        return dict(valid=len(valid),support_pass=len(valid)>=min_valid,
                    q025_05_95_975=np.quantile(valid,[.025,.05,.95,.975]).tolist() if len(valid) else None)
    return dict(mean_pp=mean,by_seed_pp=point.tolist(),t_one_sided=[float(mean-t_one),float(mean+t_one)],
                image=quantiles(valid_image),two_layer=quantiles(joint))


def main():
    pred=read(HERE/'A_PREDICTIONS.json')
    lock=read(HERE/'DATA_ROLE_LOCK.json')
    receipt=read(HERE/'A_SCORE_RECEIPT.json')
    rules=read(HERE/'RULES.json')
    assert pred['status']=='PREDICTIONS_LOCKED' and receipt['status']=='SCORES_SEALED'
    assert receipt['truth_loaded'] is False and receipt['endpoints']==72
    assert receipt['predictions_locked_at']==pred['locked_at']
    assert datetime.fromisoformat(pred['locked_at']).timestamp()<receipt['sealed_unix']
    with np.load(HERE/'A_SCORES.npz') as data:
        ids,classes,scores=data['ids'],data['classes'],data['scores']
        keys=list(zip(data['draws'].tolist(),data['seeds'].tolist(),data['arms'].tolist()))
    expected=[(r['draw'],r['seed'],r['arm']) for r in pred['endpoint_manifest']]
    assert keys==expected and len(set(keys))==72
    assert scores.shape==(72,5000,20) and scores.dtype==np.float32
    assert np.isfinite(scores).all() and ((0<=scores)&(scores<=1)).all()
    assert ids.tolist()==lock['pools']['A']['ids']==pred['ids']
    with np.load(HERE/'A_GOLD.npz') as data:
        assert np.array_equal(ids,data['ids']) and np.array_equal(classes,data['classes'])
        truth=data['truth']
    seeds=sorted({k[1] for k in keys})
    draws=sorted({k[0] for k in keys})
    assert len(seeds)==12 and len(draws)==2
    b=rules['bootstrap_replicates']
    rng=np.random.default_rng(rules['a_bootstrap_seed'])
    out=HERE/'A_analysis'
    out.mkdir(exist_ok=True)
    weight_file=out/'A_SHARED_WEIGHTS.npy'
    weights=np.lib.format.open_memmap(weight_file,mode='w+',dtype=np.uint16,shape=(b+1,len(ids)))
    weights[0]=1
    for j in range(1,b+1):
        weights[j]=np.bincount(rng.integers(len(ids),size=len(ids)),minlength=len(ids))
    weights.flush()
    indices=rng.integers(len(seeds),size=(b,len(seeds)))
    pstar=[r['pstar'] for r in lock['pools']['A']['support']]
    arrays=np.lib.format.open_memmap(out/'A_ENDPOINT_BOOTSTRAP.npy',mode='w+',dtype=np.float64,
                      shape=(b+1,len(seeds),len(draws),len(ARMS),3,3))
    max_error=0.
    cells=[]
    with ProcessPoolExecutor(max_workers=4,initializer=init_worker,initargs=(truth,weight_file,pstar)) as pool:
        for i,(key,(value,gap,cell)) in enumerate(zip(keys,pool.map(endpoint,scores),strict=True),1):
            draw,seed,arm=key
            arrays[:,seeds.index(seed),draws.index(draw),ARMS.index(arm)]=value
            cells.append(dict(draw=draw,seed=seed,arm=arm,cells=cell))
            max_error=max(max_error,gap)
            if i%12==0:
                arrays.flush()
                print('ANALYZED_A',i,'/72',flush=True)
    arrays.flush()
    results={}
    rows=[]
    for k, label in ((1,'TRUE_MINUS_PLC'),(2,'TRUE_MINUS_LOW')):
        diff=100*(arrays[:,:,:,0]-arrays[:,:,:,k]).mean(axis=2)
        level=[]
        for group,x,source,baseline in [
            ('aggregate',diff.mean(axis=2),pred['contrasts'][label]['aggregate']['mean'][:3],
             pred['contrasts'][label]['aggregate']['mean'][3]),
            *((name,diff[:,:,j],pred['contrasts'][label]['target_intervals']['mean'][j][:3],
               pred['contrasts'][label]['target_intervals']['mean'][j][3])
              for j,name in enumerate(pred['target_names']))
        ]:
            observed=[bootstrap_summary(x[:,:,j],indices,rules['minimum_bootstrap_valid']) for j in range(3)]
            errors=[v['mean_pp']-s for v,s in zip(observed,source)]
            level.append(dict(group=group,observed=observed,predicted_pp=source,error_pp=errors,
                 old_natural_baseline_pp=baseline,natural_baseline_error_pp=observed[0]['mean_pp']-baseline))
            for j,mode in enumerate(('natural','r05','r50')):
                q=observed[j]['two_layer']['q025_05_95_975']
                rows.append(dict(contrast=label,group=group,mode=mode,predicted_pp=source[j],
                    observed_pp=observed[j]['mean_pp'],error_pp=errors[j],bootstrap_lower_95=q[1],
                    bootstrap_upper_95=q[2],valid=observed[j]['two_layer']['valid']))
        results[label]=level
    main=results['TRUE_MINUS_PLC'][0]['observed']
    low,high=main[1]['two_layer'],main[2]['two_layer']
    direction=low['support_pass'] and high['support_pass'] and low['q025_05_95_975'][2]<0<high['q025_05_95_975'][1]
    passed=bool(pred['prediction_direction_pass'] and direction)
    with (HERE/'A_RESULTS.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save('A_OBSERVED_CELLS.json',dict(cell_order=['10x00','10x01','11x00','11x01'],rows=cells))
    save('INDEPENDENT_QA.json',dict(status='PASS',aligned_endpoints=72,images=5000,
        independent_rank_auc_max_abs=max_error,prediction_preceded_scores=True))
    save('A_RESULT.json',dict(status='PASS_A' if passed else 'HOLD_B_A_DIRECTION',
        prediction_direction=pred['prediction_direction_pass'],confirmed_direction=bool(direction),
        independent_qa='PASS',bootstrap_replicates=b,pairs=12,contrasts=results,
        main_low_upper_pp=low['q025_05_95_975'][2],main_high_lower_pp=high['q025_05_95_975'][1]))
    print('A_RESULT',passed,'low upper',low['q025_05_95_975'][2],'high lower',high['q025_05_95_975'][1])


if __name__=='__main__':
    main()
