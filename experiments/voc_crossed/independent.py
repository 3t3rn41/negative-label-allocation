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

"""Recompute the VOC point estimates and all 5,000 primary paired replicates."""

from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.stats import rankdata

from analyze import HERE, read, save, sealed_inputs


def init(truth, design):
    global TRUTH, DESIGN, W
    TRUTH=truth
    DESIGN=design
    W=np.load(HERE/'BOOTSTRAP_IMAGE_WEIGHTS.npy',mmap_mode='r')


def auc_rank(scores, pos, neg):
    ranks=rankdata(scores[pos|neg],method='average')
    n=int(pos.sum())
    return (ranks[pos[pos|neg]].sum()-n*(n+1)/2)/(n*int(neg.sum()))


def auc_weighted(scores, pos, neg):
    si=np.flatnonzero(pos)
    ni=np.flatnonzero(neg)
    ni=ni[np.argsort(scores[ni],kind='stable')]
    lower=np.searchsorted(scores[ni],scores[si],side='left')
    upper=np.searchsorted(scores[ni],scores[si],side='right')
    result=np.empty(len(W))
    for i in range(0,len(W),128):
        weights=W[i:i+128]
        positives=weights[:,si]
        cumulative=np.c_[np.zeros(len(weights),dtype=np.int32),weights[:,ni].cumsum(1,dtype=np.int32)]
        count=(cumulative[:,lower]+cumulative[:,upper])*0.5
        denominator=cumulative[:,-1].astype(float)*positives.sum(1)
        result[i:i+len(weights)]=np.divide((positives*count).sum(1),denominator,
                      out=np.full(len(weights),np.nan),where=denominator>0)
    return result


def ap_thresholds(scores, y):
    valid=y!=0
    s=scores[valid]
    y=y[valid]==1
    order=np.argsort(s,kind='stable')[::-1]
    s,y=s[order],y[order]
    ends=np.r_[np.flatnonzero(np.diff(s)),len(s)-1]
    tp=np.cumsum(y)[ends]
    recall=tp/tp[-1]
    precision=tp/(ends+1)
    return float(np.mean([precision[recall>=r-1e-12].max(initial=0) for r in np.linspace(0,1,11)]))


def endpoint(matrix):
    cond=np.empty((len(W),2,3))
    all_cells=np.empty((2,3,4))
    point=np.empty((2,3,5))
    for i,setting in enumerate(DESIGN):
        for j,(t,p,ps) in enumerate(setting):
            s=matrix[:,t]
            y,z=TRUTH[:,t],TRUTH[:,p]
            a,b,c,d=(y==1)&(z==-1),(y==1)&(z==1),(y==-1)&(z==-1),(y==-1)&(z==1)
            cond[:,i,j]=auc_weighted(s,a,d)
            cells=[auc_rank(s,v,w) for v,w in ((a,c),(a,d),(b,c),(b,d))]
            all_cells[i,j]=cells
            natural=auc_rank(s,a|b,c|d)
            full=[sum(x*w for x,w in zip(cells,((1-ps)*(1-r),(1-ps)*r,ps*(1-r),ps*r))) for r in (.05,.5)]
            point[i,j]=[cells[1],natural,*full,(full[1]-full[0])/.45]
            assert abs(cond[0,i,j]-cells[1])<1e-13
    ap=[ap_thresholds(matrix[:,c],TRUTH[:,c]) for c in range(20)]
    return cond,all_cells,point,np.array(ap)


def interval(x,indices):
    boot=x[1:][np.arange(len(indices))[:,None],indices].mean(axis=1)
    values=boot[np.isfinite(boot)]
    return x[0],len(values),np.quantile(values,[.025,.05,.95,.975])


def compare(x, record, indices):
    point,n,bounds=interval(x,indices)
    assert n==record['valid']
    ref=[record[k] for k in ('lower95_two_sided_pp','lower95_one_sided_pp','upper95_one_sided_pp','upper95_two_sided_pp')]
    return float(max(abs(point.mean()-record['mean_pp']),
                     np.max(abs(point-np.array(record['by_initialization_pp']))),
                     np.max(abs(bounds-ref))))


def main():
    rules,scores,truth,classes,design=sealed_inputs()
    original=read('RESULT.json')
    w=np.load(HERE/'BOOTSTRAP_IMAGE_WEIGHTS.npy',mmap_mode='r')
    indices=np.load(HERE/'BOOTSTRAP_INIT_INDICES.npy')
    assert w.shape==(rules['bootstrap_replicates']+1,len(truth))
    rng=np.random.Generator(np.random.PCG64(rules['bootstrap_seed']))
    assert (w[0]==1).all()
    for k,counts in enumerate(w[1:]):
        assert np.array_equal(np.bincount(rng.integers(0,len(truth),len(truth)),minlength=len(truth)),counts),k
    assert np.array_equal(rng.integers(0,12,indices.shape),indices)
    conditional=np.empty((len(w),2,12,3,2,3))
    point_gap, ap_gap=0.,0.
    with np.load(HERE/'POINTS_BY_ENDPOINT.npz') as saved:
        cell_reference,metric_reference,ap_reference=(saved[k] for k in ('cell_auc','metrics','ap'))
    with ProcessPoolExecutor(max_workers=6,initializer=init,initargs=(truth,design)) as pool:
        for i,(res,all_cells,point,ap) in enumerate(pool.map(endpoint,scores)):
            d,j,a=np.unravel_index(i,(2,12,3))
            conditional[:,d,j,a]=res
            point_gap=max(point_gap,float(np.max(abs(all_cells-cell_reference[d,j,a]))),
                          float(np.max(abs(point-metric_reference[d,j,a]))))
            ap_gap=max(ap_gap,float(np.max(abs(ap-ap_reference[d,j,a]))))
            if (i+1)%12==0:
                print('INDEPENDENT_VOC',i+1,'/ 72',flush=True)
    diffs=(conditional[:,:,:,:2]-conditional[:,:,:,2:3]).mean(1)*100
    m1=(conditional[:,:,:,0,0]-conditional[:,:,:,1,0]).mean(1)*100
    m2=(conditional[:,:,:,1,1]-conditional[:,:,:,0,1]).mean(1)*100
    errors=[]
    names=['aggregate',*[classes[t] for t,_,_ in design[0]]]
    for j,group in enumerate(names):
        for row in original['cross']:
            if row['group']==group and row['metric']=='conditional':
                t,e=int(row['train'][-1])-1,int(row['evaluation'][-1])-1
                x=diffs[:,:,t,e]
                errors.append(compare(x.mean(2) if j==0 else x[:,:,j-1],row,indices))
        for name,x in (('M1',m1),('M2',m2),('I',(m1+m2)/2)):
            row,=[r for r in original['matched'] if r['group']==group and r['component']==name]
            errors.append(compare(x.mean(2) if j==0 else x[:,:,j-1],row,indices))
    main=[interval(x.mean(2),indices) for x in (m1,m2,(m1+m2)/2)]
    valid=all(n>=rules['minimum_valid'] for _,n,_ in main)
    passes=[bool(n>=rules['minimum_valid'] and q[1]>0 and
                 (point.mean()>0 if j<2 else point.mean()>=rules['interaction_threshold_pp']))
            for j,(point,n,q) in enumerate(main)]
    status=('INCOMPLETE' if not valid else 'PASS' if all(passes) else
            'PARTIAL_REPLICATION' if passes[0]!=passes[1] else 'NOT_CONFIRMED')
    assert status==original['status'] and passes==original['primary_passes']
    assert max(point_gap,ap_gap,max(errors))<1e-10
    save('INDEPENDENT_QA.json',dict(status='PASS',recomputed_status=status,
         verified_endpoints=len(scores),replicates=len(indices),auc_point_max_abs=point_gap,
         ap_point_max_abs=ap_gap,primary_and_conditional_pp_max_abs=max(errors),
         primary_passes=passes,bootstrap_weights_and_initialization_indices_exact=True))
    print('INDEPENDENT_VOC_PASS',status,max(errors),flush=True)


if __name__=='__main__':
    main()
