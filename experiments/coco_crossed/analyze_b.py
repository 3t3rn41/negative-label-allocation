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

"""Analyze the sealed N1/S0 training-by-evaluation-partner cross."""

import csv
import json
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

import numpy as np

from analyze_a import CELLS, Metric, bootstrap_summary, cell_labels, rank_auc
from prepare import HERE, PAIR_MAP, read, save

ARMS=('S0_TRUE','N1_TRUE','S0_PLC')
SETTINGS=('S0','N1')
METRICS=('conditional','natural','r05','r50','slope')


def init_worker(truth,weights,pstars):
    global TRUTH, WEIGHTS, PSTARS
    TRUTH=truth
    WEIGHTS=np.load(weights,mmap_mode='r')
    PSTARS=pstars


def endpoint(scores):
    output=np.empty((len(WEIGHTS),2,3,5),np.float64)
    max_error=0.
    for k,setting in enumerate(SETTINGS):
        for j,pair in enumerate(PAIR_MAP[setting]):
            col,part=pair['target'],pair['partner']
            y,q,s=TRUTH[:,col],TRUTH[:,part],scores[:,col]
            metrics=[Metric(s,cell_labels(y,q,a,b)) for a,b in CELLS]
            fixed_p=PSTARS[k][j]
            for first in range(0,len(WEIGHTS),16):
                section=slice(first,min(first+16,len(WEIGHTS)))
                w=WEIGHTS[section]
                z=np.array([metric.compute_auc(w) for metric in metrics])
                with np.errstate(divide='ignore',invalid='ignore'):
                    p=(w@(y*q))/(w@y)
                    r=(w@((1-y)*q))/(w@(1-y))
                nat=(1-p)*((1-r)*z[0]+r*z[1])+p*((1-r)*z[2]+r*z[3])
                bg=(1-fixed_p)*z[0]+fixed_p*z[2]
                part_auc=(1-fixed_p)*z[1]+fixed_p*z[3]
                low,high=.95*bg+.05*part_auc,.50*bg+.50*part_auc
                output[section,k,j]=np.stack((z[1],nat,low,high,(high-low)/.45),axis=1)
                if first==0:
                    independent=[rank_auc(m.y.astype(np.int8),s[m.order]) for m in metrics]
                    max_error=max(max_error,float(np.max(np.abs(z[:,0]-independent))),
                                  abs(float(nat[0])-rank_auc(y,s)))
    assert max_error<1e-12
    return output,max_error


def write_table(name,rows):
    with (HERE/name).open('w',encoding='utf-8-sig',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    a=read(HERE/'A_RESULT.json')
    lock=read(HERE/'B_PROTOCOL_LOCK.json')
    roles=read(HERE/'DATA_ROLE_LOCK.json')
    receipt=read(HERE/'B_SCORE_RECEIPT.json')
    rules=read(HERE/'RULES.json')
    training=read(HERE/'B_TRAIN_QA.json')
    assert a['status']=='PASS_A' and lock['status']=='LOCKED'
    assert receipt['status']=='SCORES_SEALED' and receipt['truth_loaded'] is False
    assert datetime.fromisoformat(lock['locked_at']).timestamp()<receipt['sealed_unix']
    assert training['endpoints']==24 and training['steps_total']==151200 and training['status']=='COMPLETE'
    with np.load(HERE/'B_SCORES.npz') as data:
        ids,classes,scores=data['ids'],data['classes'],data['scores']
        keys=list(zip(data['draws'].tolist(),data['seeds'].tolist(),data['arms'].tolist()))
    seeds,draws=lock['seeds'],lock['draws']
    assert keys==[(d,s,a) for d in draws for s in seeds for a in ARMS]
    assert len(keys)==len(set(keys))==receipt['endpoints']==72
    assert ids.tolist()==lock['b_ids']==roles['pools']['B']['ids']
    assert not set(ids)&set(roles['pools']['A']['ids'])
    assert scores.shape==(72,5000,20) and scores.dtype==np.float32
    assert np.isfinite(scores).all() and ((scores>=0)&(scores<=1)).all()
    with np.load(HERE/'B_GOLD.npz') as data:
        assert np.array_equal(data['ids'],ids) and np.array_equal(data['classes'],classes)
        truth=data['truth']
    assert [p['target'] for p in PAIR_MAP['S0']]==[p['target'] for p in PAIR_MAP['N1']]
    pst=[[row['pstar'] for row in roles['pools']['B']['support'] if row['setting']==s] for s in SETTINGS]
    rng=np.random.default_rng(rules['b_bootstrap_seed'])
    b=rules['bootstrap_replicates']
    out=HERE/'B_analysis'
    out.mkdir(exist_ok=True)
    weights_file=out/'B_SHARED_WEIGHTS.npy'
    w=np.lib.format.open_memmap(weights_file,mode='w+',dtype=np.uint16,shape=(b+1,len(ids)))
    w[0]=1
    for i in range(1,b+1):
        w[i]=np.bincount(rng.integers(len(ids),size=len(ids)),minlength=len(ids))
    w.flush()
    init_indices=rng.integers(len(seeds),size=(b,len(seeds)))
    values=np.lib.format.open_memmap(out/'B_ENDPOINT_BOOTSTRAP.npy',mode='w+',dtype=np.float64,
                      shape=(b+1,len(seeds),len(draws),3,2,3,5))
    worst=0.
    with ProcessPoolExecutor(max_workers=4,initializer=init_worker,initargs=(truth,weights_file,pst)) as pool:
        for i,(key,(result,error)) in enumerate(zip(keys,pool.map(endpoint,scores),strict=True),1):
            draw,seed,arm=key
            values[:,seeds.index(seed),draws.index(draw),ARMS.index(arm)]=result
            worst=max(worst,error)
            if i%12==0:
                values.flush()
                print('ANALYZED_B',i,'/72',flush=True)
    values.flush()
    diff=100*(values[:,:,:,:2]-values[:,:,:,2:3]).mean(axis=2)
    # axes after average: replicate, initialization, training partner,
    # evaluation partner, target, evaluation metric.
    names=['aggregate',*[row['target_name'] for row in PAIR_MAP['S0']]]
    cross=[]
    cross_rows=[]
    min_valid=rules['minimum_bootstrap_valid']
    for train in range(2):
        for evaluation in range(2):
            y=diff[:,:,train,evaluation]
            for group,subset in zip(names,[y.mean(axis=2),*(y[:,:,j] for j in range(3))]):
                for m,metric in enumerate(METRICS):
                    r=bootstrap_summary(subset[:,:,m],init_indices,min_valid)
                    cross.append(dict(train=SETTINGS[train],evaluation=SETTINGS[evaluation],group=group,
                                      metric=metric,**r))
                    q=r['two_layer']['q025_05_95_975']
                    cross_rows.append(dict(train=SETTINGS[train],evaluation=SETTINGS[evaluation],group=group,
                        metric=metric,mean_pp=r['mean_pp'],lower95_one_sided_pp=q[1],upper95_one_sided_pp=q[2],
                        lower95_two_sided_pp=q[0],upper95_two_sided_pp=q[3],valid=r['two_layer']['valid']))
    x=diff[...,0]
    s0=x[:,:,0,0]-x[:,:,1,0]
    n1=x[:,:,1,1]-x[:,:,0,1]
    interaction=.5*(s0+n1)
    direct=.5*100*((values[:,:,:,0,0,: ,0]-values[:,:,:,1,0,:,0])+
                    (values[:,:,:,1,1,:,0]-values[:,:,:,0,1,:,0])).mean(axis=2)
    assert np.allclose(interaction,direct,atol=1e-11,rtol=0,equal_nan=True)
    matched=[]
    for idx,group in enumerate(names):
        selected=(s0.mean(axis=2),n1.mean(axis=2),interaction.mean(axis=2)) if idx==0 else (
                     s0[:,:,idx-1],n1[:,:,idx-1],interaction[:,:,idx-1])
        matched.append(dict(group=group,**{k:bootstrap_summary(v,init_indices,min_valid)
                       for k,v in zip(('s0_column','n1_column','interaction'),selected)}))
    r=matched[0]['interaction']
    lower=r['two_layer']['q025_05_95_975'][1]
    passed=bool(r['two_layer']['support_pass'] and r['mean_pp']>=lock['main_point_threshold_pp']
                and lower>lock['main_bootstrap_lower_threshold_pp'])
    both=bool(matched[0]['s0_column']['mean_pp']>0 and matched[0]['n1_column']['mean_pp']>0)
    write_table('B_CROSS_RESULTS.csv',cross_rows)
    write_table('B_MATCHED_ADVANTAGE.csv',[
        dict(group=entry['group'],component=part,mean_pp=entry[part]['mean_pp'],
             lower95_one_sided_pp=entry[part]['two_layer']['q025_05_95_975'][1],
             valid=entry[part]['two_layer']['valid'])
        for entry in matched for part in ('s0_column','n1_column','interaction')])
    save('B_INDEPENDENT_QA.json',dict(status='PASS',aligned_endpoints=72,new_endpoints=24,images=5000,
        rank_recomputation_max_abs=worst,common_plc_algebra_pass=True,bootstrap_valid=r['two_layer']['valid']))
    save('B_RESULT.json',dict(status='PASS_B' if passed else 'COMPLETE_B_GATE_NOT_MET',
        matched_effect_pp=r['mean_pp'],lower95_one_sided_pp=lower,both_columns_positive=both,
        bootstrap_replicates=b,matched=matched,cross=cross))
    print('B_RESULT',passed,'I',r['mean_pp'],'lower',lower,'both',both,flush=True)


if __name__=='__main__':
    main()
