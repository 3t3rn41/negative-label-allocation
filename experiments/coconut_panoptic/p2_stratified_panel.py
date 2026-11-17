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

"""Calibrate image sampling jointly on a synthetic, fully known ranking population."""
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix


@dataclass
class Population:
    frequencies: list
    operators: list
    gradients: list
    totals: np.ndarray
    truth: np.ndarray
    bins: int
    pairs: int
    donor_match_counts: dict


def signatures(y):
    bits=np.left_shift(np.uint64(1),np.arange(y.shape[1],dtype=np.uint64))
    return (y.astype(np.uint64)*bits).sum(axis=1)


def atom_histogram(development_truth,dev_scores,dev_strata,formal_truth,formal_strata,pairs,bins,rng):
    """Bind synthetic donor-score atoms to label signatures without storing formal image IDs."""
    assert ((development_truth==0)|(development_truth==1)).all()
    assert ((formal_truth==0)|(formal_truth==1)).all()
    dev_code=signatures(development_truth)
    target_columns=sorted({target for target,_ in pairs})
    target_signature=signatures(development_truth[:,target_columns])
    all_codes=signatures(formal_truth)
    rank=np.zeros(dev_scores.shape,dtype=np.int64)
    for j in target_columns:
        reference=np.sort(dev_scores[:,:,j],axis=None)
        rank[:,:,j]=np.minimum(bins-1,(np.searchsorted(reference,dev_scores[:,:,j],side="right")*bins)//len(reference))
    counts=[]
    matrices=[]
    fallback=dict(full_signature=0,target_signature=0,stratum=0)
    dimension=len(pairs)*2*2*bins
    for h in range(2):
        codes,indices,size=np.unique(all_codes[formal_strata==h],return_index=True,return_counts=True)
        representative=formal_truth[formal_strata==h][indices]
        targets=signatures(representative[:,target_columns])
        table=[]
        atom_frequency=[]
        for code,target_code,label,n in zip(codes,targets,representative,size,strict=True):
            donor=np.flatnonzero((dev_code==code)&(dev_strata==h))
            key="full_signature"
            if not len(donor):
                donor=np.flatnonzero((target_signature==target_code)&(dev_strata==h))
                key="target_signature"
            if not len(donor):
                donor=np.flatnonzero(dev_strata==h)
                key="stratum"
            assert len(donor)
            fallback[key]+=int(n)
            occupancy=rng.multinomial(int(n),np.full(len(donor),1/len(donor)))
            used=np.flatnonzero(occupancy)
            table.extend((label,rank[:,donor[k]]) for k in used)
            atom_frequency.extend(occupancy[used].tolist())
        row_index=[]
        column_index=[]
        for i,(label,values) in enumerate(table):
            for j,(t,p) in enumerate(pairs):
                if label[t]==label[p]:
                    continue
                group=0 if label[t]==1 else 1
                for arm_index,arm in enumerate((j%2,1-j%2)):
                    column=((j*2+arm_index)*2+group)*bins+values[arm,t]
                    row_index.append(i)
                    column_index.append(column)
        matrices.append(csr_matrix((np.ones(len(row_index)),(column_index,row_index)),
                                   shape=(dimension,len(table))))
        counts.append(np.asarray(atom_frequency,dtype=np.int64))
    N=np.array([v.sum() for v in counts])
    np.testing.assert_array_equal(N,np.bincount(formal_strata,minlength=2))
    population_weighted=sum((matrix@count).reshape(len(pairs),2,2,bins) for matrix,count in zip(matrices,counts,strict=True))
    truth,gradient=auc_histogram(population_weighted[None],with_gradient=True)
    derivatives=(gradient[0]*np.array([1.,-1.])[None,:,None,None]).reshape(dimension)
    atom_derivatives=[np.asarray(op.T@derivatives)/len(pairs) for op in matrices]
    return Population(counts,matrices,atom_derivatives,N,truth[0],bins,len(pairs),fallback)


def auc_histogram(hist,with_gradient=False):
    pos,neg=hist[...,0,:],hist[...,1,:]
    np_,nn_=pos.sum(-1),neg.sum(-1)
    assert (np_>0).all() and (nn_>0).all()
    left=np.cumsum(neg,axis=-1)-.5*neg
    auc=np.sum(pos*left,axis=-1)/(np_*nn_)
    difference=auc[...,0]-auc[...,1]
    if not with_gradient:
        return difference
    right=np.cumsum(pos[...,::-1],axis=-1)[...,::-1]-.5*pos
    d_pos=(left/nn_[...,None]-auc[...,None])/np_[...,None]
    d_neg=(right/np_[...,None]-auc[...,None])/nn_[...,None]
    return difference,np.stack((d_pos,d_neg),axis=-2)


def combine(model,counts,scales):
    trials=counts[0].shape[0]
    hist=np.zeros((trials,model.pairs,2,2,model.bins))
    for c,scale,op in zip(counts,scales,model.operators,strict=True):
        hist+=(op@(c*scale).T).T.reshape(hist.shape)
    return hist


def sample_atoms(model,quota,reps,rng):
    return [rng.multivariate_hypergeometric(f,int(n),size=reps,method="marginals") if int(n)<int(f.sum())
            else np.broadcast_to(f,(reps,len(f))) for f,n in zip(model.frequencies,quota,strict=True)]


def image_outer(model,quota,reps,rng,save_indices=()):
    errors=[]
    conditional=[]
    saved={}
    keep=set(save_indices)
    for start in range(0,reps,64):
        size=min(64,reps-start)
        counts=sample_atoms(model,quota,size,rng)
        hist=combine(model,counts,model.totals/quota)
        point,gradient=auc_histogram(hist,with_gradient=True)
        errors.append(point-model.truth)
        signed=(gradient*np.array([1.,-1.])[None,None,:,None,None]).reshape(size,-1)/model.pairs
        conditional_var=np.zeros(size)
        for N,n,c,op in zip(model.totals,quota,counts,model.operators,strict=True):
            if n==N:
                continue
            g=(op.T@signed.T).T
            mean=(g*c).sum(1)/n
            s2=(c*(g-mean[:,None])**2).sum(1)/(n-1)
            conditional_var+=N**2*(1-n/N)/n*s2
        conditional.append(conditional_var)
        for i in range(start,start+size):
            if i in keep:
                saved[i]=[c[i-start].copy() for c in counts]
    variance=0.
    for N,n,g,f in zip(model.totals,quota,model.gradients,model.frequencies,strict=True):
        mean=np.dot(g,f)/N
        variance+=N**2*(1-n/N)/n*np.dot(f,(g-mean)**2)/(N-1)
    return np.concatenate(errors)*100,np.concatenate(conditional)*100**2,float(variance*100**2),saved


def image_inner(model,counts,quota,reps,rng):
    """Compare exact and first-order image replicas on identical half samples."""
    base=combine(model,[c[None] for c in counts],model.totals/quota)
    center,gradient=auc_histogram(base,with_gradient=True)
    signed=gradient*np.array([1.,-1.])[None,None,:,None,None]
    chunks=[]
    approx=[]
    for start in range(0,reps,128):
        size=min(128,reps-start)
        weighted=[]
        for N,n,c in zip(model.totals,quota,counts,strict=True):
            if n==N:
                weighted.append(np.broadcast_to(c,(size,len(c))))
                continue
            k=int(n)//2
            fraction=np.sqrt((1-n/N)*k/(n-k))
            half=rng.multivariate_hypergeometric(c,k,size=size,method="marginals")
            weighted.append((1-fraction)*(N/n)*c + fraction*(N/k)*half)
        hist=combine(model,weighted,[1.,1.])
        chunks.append((auc_histogram(hist)-center).mean(1))
        approx.append((signed*(hist-base)).sum((2,3,4)).mean(1))
    return np.concatenate(chunks)*100,np.concatenate(approx)*100
