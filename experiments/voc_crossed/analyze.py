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

"""Analyze the sealed VOC test scores with shared image and initialization draws."""

import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent/'voc_design'
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT/'metrics'))
from pr_metrics import Metric, cell_labels

PAIRS = ('P1','P2')
METRICS = ('conditional','natural','mix_05','mix_50','composition_slope')
CELLS = (('10','00'),('10','01'),('11','00'),('11','01'))


def read(name, folder=HERE):
    return json.loads((folder/name).read_text(encoding='utf-8'))


def save(name, value):
    (HERE/name).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def table(name, rows):
    with (HERE/name).open('w',encoding='utf-8-sig',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sealed_inputs():
    lock, rules, receipt, train = map(read,('PROTOCOL_LOCK.json','RULES.json','SCORE_RECEIPT.json','TRAIN_QA.json'))
    role = read('DATA_ROLE_LOCK.json',PARENT)
    assert lock['status']=='LOCKED_FORMAL' and lock['rules']==rules
    assert receipt['status']=='SCORES_SEALED' and receipt['test_labels_read'] is False
    assert receipt['sealed_unix']>lock['locked_unix']
    assert train['status']=='TRAINING_COMPLETE' and train['endpoints']==72 and train['updates']==453600
    with np.load(HERE/'SCORES.npz') as s, np.load(PARENT/'TEST_GOLD.npz') as g:
        ids, classes, scores = s['ids'], s['classes'].tolist(), s['scores']
        assert np.array_equal(ids,g['ids']) and classes==g['classes'].tolist()==role['classes']
        assert ids.tolist()==lock['formal_ids']==role['role_ids']['test']
        truth = g['truth']
        keys=list(zip(s['draws'].tolist(),s['seeds'].tolist(),s['arms'].tolist()))
    manifest = read('ENDPOINTS.json')['endpoints']
    expected = [(d,i,a) for d in rules['draws'] for i in rules['initializations'] for a in rules['arms']]
    assert keys==expected==[(r['draw'],r['seed'],r['arm']) for r in manifest]
    assert scores.shape==tuple(receipt['shape'])==(72,4952,20)
    assert scores.dtype==np.float32 and np.isfinite(scores).all() and ((scores>=0)&(scores<=1)).all()
    assert truth.shape==(4952,20) and np.isin(truth,(-1,0,1)).all()
    design = []
    for setting in PAIRS:
        row = []
        for t,p in role['pairs'][setting]:
            c = [x for x in role['composition'] if x['key']==f'{setting}:{t}/{p}']
            assert len(c)==2 and all(x['status']=='AVAILABLE' for x in c)
            row.append((classes.index(t),classes.index(p),c[0]['dev_positive_partner_fraction']))
        design.append(row)
    return rules, scores, truth, classes, design


def resampling(rules, size):
    rng=np.random.default_rng(rules['bootstrap_seed'])
    n=rules['bootstrap_replicates']
    weights=np.empty((n+1,size),dtype=np.uint16)
    weights[0]=1
    for i in range(1,n+1):
        weights[i]=np.bincount(rng.integers(size,size=size),minlength=size)
    indices=rng.integers(len(rules['initializations']),size=(n,len(rules['initializations'])))
    return weights,indices


def initialize(truth, design, file):
    global TRUTH, DESIGN, WEIGHTS
    TRUTH=truth
    DESIGN=design
    WEIGHTS=np.load(file,mmap_mode='r')


def voc_ap(weights, scores, labels):
    """Weighted VOC 11-point AP at thresholds with at least one true positive."""
    neg=np.flatnonzero(labels==-1)
    pos=np.flatnonzero(labels==1)
    pos=pos[np.argsort(-scores[pos],kind='stable')]
    neg=neg[np.argsort(-scores[neg],kind='stable')]
    endpoints=np.r_[np.flatnonzero(scores[pos[1:]]!=scores[pos[:-1]]),len(pos)-1]
    neg_right=np.searchsorted(-scores[neg],-scores[pos[endpoints]],side='right')
    tp=np.cumsum(weights[:,pos],axis=1,dtype=np.int32)[:,endpoints]
    negative=np.cumsum(weights[:,neg],axis=1,dtype=np.int32)
    fp=np.where(neg_right,negative[:,np.maximum(0,neg_right-1)],0)
    total=tp[:,-1]
    precision=np.divide(tp,tp+fp,out=np.zeros_like(tp,dtype=float),where=tp+fp>0)
    envelope=np.maximum.accumulate(precision[:,::-1],axis=1)[:,::-1]
    fractions=np.arange(11)
    target=np.maximum(1,(total[:,None]*fractions+9)//10)
    # Positive recall is monotonic. A row-specific threshold index is the
    # count of recall values smaller than its requested threshold.
    position=(tp[:,:,None]<target[:,None,:]).sum(axis=1)
    result=np.take_along_axis(envelope,np.minimum(position,len(endpoints)-1),axis=1).mean(axis=1)
    result[total==0]=np.nan
    return result


def endpoint(scores):
    n=len(WEIGHTS)
    values=np.empty((n,2,3,len(METRICS)))
    all_ap=np.empty((n,20))
    cell_points=np.empty((2,3,4))
    for ci in range(20):
        for start in range(0,n,128):
            index=slice(start,min(start+128,n))
            all_ap[index,ci]=voc_ap(WEIGHTS[index],scores[:,ci],TRUTH[:,ci])
    for e,pairs in enumerate(DESIGN):
        for j,(target,partner,p) in enumerate(pairs):
            a,b=TRUTH[:,target],TRUTH[:,partner]
            y=np.where(a==0,-1,(a==1).astype(np.int8))
            q=np.where(b==0,-1,(b==1).astype(np.int8))
            metric=[Metric(scores[:,target],cell_labels(y,q,i,k)) for i,k in CELLS]
            natural=Metric(scores[:,target],np.where(q==-1,-1,y))
            for start in range(0,n,128):
                ix=slice(start,min(start+128,n))
                w=WEIGHTS[ix]
                z=np.array([m.compute_auc(w) for m in metric])
                bg=(1-p)*z[0]+p*z[2]
                exposed=(1-p)*z[1]+p*z[3]
                values[ix,e,j]=np.stack((z[1],natural.compute_auc(w),
                                         .95*bg+.05*exposed,.5*bg+.5*exposed,exposed-bg),axis=1)
                if start==0:
                    cell_points[e,j]=z[:,0]
    return values,all_ap,cell_points


def summarize(x,indices,minimum):
    point=x[0]
    paired=x[np.arange(1,len(x))[:,None],indices].mean(axis=1)
    good=paired[np.isfinite(paired)]
    assert np.isfinite(point).all()
    q=np.quantile(good,[.025,.05,.95,.975]).tolist() if len(good) else [None]*4
    return dict(mean_pp=float(point.mean()),by_initialization_pp=point.tolist(),valid=len(good),
                sufficient=len(good)>=minimum,lower95_two_sided_pp=q[0],lower95_one_sided_pp=q[1],
                upper95_one_sided_pp=q[2],upper95_two_sided_pp=q[3])


def verdict(main, rules):
    threshold=[0.,0.,rules['interaction_threshold_pp']]
    valid=all(x['sufficient'] for x in main)
    if not valid:
        return 'INCOMPLETE', [False]*3
    passes=[(x['mean_pp']>t if j<2 else x['mean_pp']>=t) and x['lower95_one_sided_pp']>0
            for j,(x,t) in enumerate(zip(main,threshold))]
    if all(passes):
        return 'PASS', passes
    if sum(passes[:2])==1:
        return 'PARTIAL_REPLICATION', passes
    return 'NOT_CONFIRMED',passes


def flatten(rows):
    return [{k:v for k,v in row.items() if k!='by_initialization_pp'} for row in rows]


def main():
    rules,scores,truth,classes,design=sealed_inputs()
    w,indices=resampling(rules,len(truth))
    np.save(HERE/'BOOTSTRAP_IMAGE_WEIGHTS.npy',w)
    np.save(HERE/'BOOTSTRAP_INIT_INDICES.npy',indices)
    n=len(w)
    values=np.empty((n,2,12,3,2,3,len(METRICS)))
    ap=np.empty((n,2,12,3,20))
    cells=np.empty((2,12,3,2,3,4))
    with ProcessPoolExecutor(max_workers=6,initializer=initialize,
                             initargs=(truth,design,HERE/'BOOTSTRAP_IMAGE_WEIGHTS.npy')) as pool:
        for i,(metric,class_ap,cell) in enumerate(pool.map(endpoint,scores)):
            d,i0,a=np.unravel_index(i,(2,12,3))
            values[:,d,i0,a]=metric
            ap[:,d,i0,a]=class_ap
            cells[d,i0,a]=cell
            if (i+1)%12==0:
                print('ANALYZED_VOC',i+1,'/ 72',flush=True)
    minimum=rules['minimum_valid']
    stats=lambda x: summarize(x,indices,minimum)
    effects=100*(values[:,:,:,:2]-values[:,:,:,2:]).mean(axis=1)
    groups=['aggregate',*[classes[t] for t,_,_ in design[0]]]
    rows,match=[],[]
    for e in range(2):
        for t in range(2):
            for j,group in enumerate(groups):
                g=effects[:,:,t,e].mean(axis=2) if j==0 else effects[:,:,t,e,j-1]
                for k,metric in enumerate(METRICS):
                    rows.append(dict(train=PAIRS[t],evaluation=PAIRS[e],group=group,metric=metric,**stats(g[:,:,k])))
    m1=effects[:,:,0,0,:,0]-effects[:,:,1,0,:,0]
    m2=effects[:,:,1,1,:,0]-effects[:,:,0,1,:,0]
    for j,group in enumerate(groups):
        for name,x in zip(('M1','M2','I'),(m1,m2,(m1+m2)/2)):
            y=x.mean(axis=2) if j==0 else x[:,:,j-1]
            match.append(dict(group=group,component=name,**stats(y)))
    status,passes=verdict(match[:3],rules)
    mrows=[]
    class_rows=[]
    for a,arm in enumerate(rules['arms']):
        ap_arm=100*ap[:,:,:,a].mean(axis=1)
        mrows.append(dict(arm=arm,estimand='absolute_mAP',**stats(ap_arm.mean(axis=2))))
        if a<2:
            comparison=ap_arm-100*ap[:,:,:,2].mean(axis=1)
            mrows.append(dict(arm=arm,estimand='mAP_minus_PLC',**stats(comparison.mean(axis=2))))
        for c,name in enumerate(classes):
            class_rows.append(dict(arm=arm,class_name=name,**stats(ap_arm[:,:,c])))
    table('CROSS_RESULTS.csv',flatten(rows))
    table('MATCHED_ADVANTAGE.csv',flatten(match))
    table('MAP_RESULTS.csv',flatten(mrows))
    table('AP_BY_CLASS.csv',flatten(class_rows))
    np.savez_compressed(HERE/'POINTS_BY_ENDPOINT.npz',cell_auc=cells,metrics=values[0],ap=ap[0])
    save('RESULT.json',dict(status=status,primary_passes=passes,matched=match,cross=rows,map=mrows,
                           rows=len(truth),endpoints=len(scores),bootstrap_replicates=len(w)-1))
    print('VOC_RESULT',status,[(r['component'],r['mean_pp'],r['lower95_one_sided_pp']) for r in match[:3]],flush=True)


if __name__=='__main__':
    main()
