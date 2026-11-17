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

"""Exact and first-order paired image, mask and initialization replicas."""
import numpy as np


class Ranking:
    def __init__(self,score,positive,negative):
        self.positive=np.flatnonzero(positive)
        self.negative=np.flatnonzero(negative)
        assert len(self.positive) and len(self.negative)
        self.negative=self.negative[np.argsort(score[self.negative],kind="stable")]
        self.left=np.searchsorted(score[self.negative],score[self.positive],side="left")
        self.right=np.searchsorted(score[self.negative],score[self.positive],side="right")

    def calculate(self,weights):
        p,n=weights[:,self.positive],weights[:,self.negative]
        prefix=np.column_stack((np.zeros(len(n)),np.cumsum(n,axis=1)))
        concordant=(prefix[:,self.left]+prefix[:,self.right])*.5
        return (p*concordant).sum(1)/(p.sum(1)*prefix[:,-1])

    def influence(self,weight):
        p,n=weight[self.positive],weight[self.negative]
        prefix=np.r_[0.,np.cumsum(n)]
        pos=(prefix[self.left]+prefix[self.right])*.5/prefix[-1]
        auc=p@pos/p.sum()
        # A negative image is beaten to the right of each positive's right edge
        # when indexed from the high-score end, with half credit on ties.
        terminal=np.zeros(len(n)+1)
        np.add.at(terminal,self.left,.5*p)
        np.add.at(terminal,self.right,.5*p)
        neg=(p.sum()-terminal.cumsum()[:-1])/p.sum()
        g=np.zeros(len(weight))
        g[self.positive]=(pos-auc)/p.sum()
        g[self.negative]=(neg-auc)/n.sum()
        return auc,g


def image_sample_weights(strata,totals,replicas,rng,image_index=None):
    n=np.bincount(strata,minlength=len(totals))
    assert np.array_equal(n>0,totals>0)
    stratum_weight=np.divide(totals,n,out=np.zeros(len(n)),where=n>0)
    base=stratum_weight[strata]
    w=np.broadcast_to(base,(replicas,len(base))).copy()
    for h,(Nh,nh) in enumerate(zip(totals,n,strict=True)):
        if Nh==nh:
            continue
        assert 2<=nh<=Nh
        indexes=np.flatnonzero(strata==h)
        half=nh//2
        rho=np.sqrt((1-nh/Nh)*half/(nh-half))
        w[:,indexes]*=1-rho
        restore=Nh*rho/half
        if image_index is not None:
            frozen=image_index[0][h]
            assert frozen.shape==(replicas,half) and (strata[frozen]==h).all()
        for j,row in enumerate(w):
            positions=rng.choice(indexes,half,replace=False,shuffle=False) if image_index is None else frozen[j]
            row[positions]+=restore
            if j==0 or j==replicas-1:
                assert abs(row[indexes].sum()-Nh)<1e-9
    if image_index is not None:
        rng.bit_generator.state=image_index[1]
    return base,w


def paired_replicas(scores,truth,pairs,strata,N,replicas,rng,return_replicates=False,image_index=None):
    """Return the conditional interaction on a crossed score grid using shared draws."""
    m,r,arms,images,classes=scores.shape
    assert arms in (2,3) and truth.shape==(images,classes) and replicas>=2
    total=np.asarray(N,dtype=np.int64)
    base,weights=image_sample_weights(strata,total,replicas,rng,image_index)
    mm=rng.multinomial(m,np.full(m,1./m),size=replicas)
    rr=rng.multinomial(r,np.full(r,1./r),size=replicas)
    rw=(mm[:,:,None]*rr[:,None,:])/(m*r)
    center=np.zeros((m,r,len(pairs)))
    exact=np.zeros(replicas)
    approx=np.zeros(replicas)
    masks=[((truth[:,t]==1)&(truth[:,p]==0),(truth[:,t]==0)&(truth[:,p]==1)) for t,p in pairs]
    for a in range(m):
        for b in range(r):
            active=np.flatnonzero(rw[:,a,b])
            active_weights=weights[active]
            gradients=np.zeros(images)
            for j,((t,_),(pos,neg)) in enumerate(zip(pairs,masks,strict=True)):
                for arm,sign in ((j%2,1),(1-j%2,-1)):
                    rank=Ranking(scores[a,b,arm,:,t],pos,neg)
                    mu,g=rank.influence(base)
                    center[a,b,j]+=sign*mu
                    gradients+=sign*g
                    for first in range(0,len(active),128):
                        subset=active[first:first+128]
                        exact[subset]+=sign*rw[subset,a,b]*rank.calculate(active_weights[first:first+128])
            for first in range(0,len(active),128):
                subset=active[first:first+128]
                linear=center[a,b].sum()+(active_weights[first:first+128]-base)@gradients
                approx[subset]+=rw[subset,a,b]*linear
    unscaled_center=center.mean()
    endpoints=np.array([.975,.95,.025])
    exact_interval=unscaled_center-np.quantile(exact/len(pairs)-unscaled_center,endpoints)
    approx_interval=unscaled_center-np.quantile(approx/len(pairs)-unscaled_center,endpoints)
    if return_replicates:
        return center,exact_interval,approx_interval,exact/len(pairs)
    return center,exact_interval,approx_interval
