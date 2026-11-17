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

"""Design-weighted ranking primitives and independent algebra checks."""
import json
from pathlib import Path
import numpy as np


def weighted_auc(scores, positive, negative, weights):
    order=np.argsort(scores,kind="stable")
    z=scores[order]
    a,b=(weights[order]*label[order] for label in (positive,negative))
    starts=np.r_[0,np.flatnonzero(z[1:]!=z[:-1])+1]
    ap=np.add.reduceat(a,starts)
    an=np.add.reduceat(b,starts)
    total=ap.sum()*an.sum()
    return float(ap @ (np.cumsum(an)-.5*an)/total) if total else np.nan


def weighted_ap(scores,positive,negative,weights):
    order=np.argsort(-scores,kind="stable")
    z=scores[order]
    a,b=(weights[order]*label[order] for label in (positive,negative))
    starts=np.r_[0,np.flatnonzero(z[1:]!=z[:-1])+1]
    ap=np.add.reduceat(a,starts)
    an=np.add.reduceat(b,starts)
    total=ap.sum()
    precision=np.divide(np.cumsum(ap),np.cumsum(ap+an),out=np.zeros_like(ap),where=np.cumsum(ap+an)>0)
    return float(ap @ np.maximum.accumulate(precision[::-1])[::-1]/total) if total else np.nan


def pairwise_design_auc(scores,positive,negative,stratum,totals):
    """AUC sensitivity using each sampled pair's joint inclusion probability."""
    Nh=np.asarray(totals,dtype=np.int64)
    nh=np.bincount(stratum,minlength=len(Nh))
    assert (nh<=Nh).all() and not np.any(positive&negative)
    mass=0.
    wins=0.
    unit=np.ones(len(scores))
    for i,Ni in enumerate(Nh):
        left=positive&(stratum==i)
        if not left.any():
            continue
        for j,Nj in enumerate(Nh):
            right=negative&(stratum==j)
            pair_count=int(left.sum())*int(right.sum())
            if not pair_count:
                continue
            if i==j:
                inclusion=nh[i]*(nh[i]-1)/(Ni*(Ni-1))
            else:
                inclusion=(nh[i]/Ni)*(nh[j]/Nj)
            pair_mass=pair_count/inclusion
            mass+=pair_mass
            wins+=pair_mass*weighted_auc(scores,left,right,unit)
    return wins/mass if mass else np.nan


def repro_weights(stratum,totals,rng):
    replacement=np.empty(len(stratum),float)
    for h,N in enumerate(totals):
        ids=np.flatnonzero(stratum==h)
        n=len(ids)
        assert 0<=n<=N and (n>=2 or n==N)
        if n==0:
            continue
        w=N/n
        if n==N:
            replacement[ids]=w
            continue
        m=n//2
        a=np.sqrt((1-n/N)*m/(n-m))
        replacement[ids]=w*(1-a)
        replacement[rng.choice(ids,m,replace=False)]+=w*a*n/m
        np.testing.assert_allclose(replacement[ids].sum(),N,atol=1e-10,rtol=1e-12)
    return replacement


def main():
    rng=np.random.default_rng(np.random.SeedSequence([93120261115,30,0]))
    errors=[]
    rank_errors=[]
    pair_design_errors=[]
    reproduction=0.
    for _ in range(200):
        size=int(rng.integers(8,96))
        score=rng.integers(0,9,size)
        y=rng.integers(-1,2,size)
        y[:2]=[1,0]
        w=rng.uniform(.5,10,size)
        sp,sn=score[y==1],score[y==0]
        wp,wn=w[y==1],w[y==0]
        pair=(wp[:,None]*wn[None,:]); obs=(sp[:,None]>sn[None,:])+.5*(sp[:,None]==sn[None,:])
        errors.append(abs(weighted_auc(score,y==1,y==0,w)-(pair*obs).sum()/pair.sum()))
        thresholds=np.unique(score[y==1])[::-1]
        tp=np.array([w[(score>=s)&(y==1)].sum() for s in thresholds]); fp=np.array([w[(score>=s)&(y==0)].sum() for s in thresholds])
        increments=np.r_[tp[0],np.diff(tp)]/tp[-1]
        oracle=sum(v*max((tp/(tp+fp))[j:]) for j,v in enumerate(increments))
        rank_errors.append(abs(oracle-weighted_ap(score,y==1,y==0,w)))
        strata=rng.integers(0,2,size)
        Nh=np.bincount(strata,minlength=2)*3
        nh=Nh//3
        pos=strata[y==1][:,None]
        neg=strata[y==0][None,:]
        p=(nh/np.maximum(Nh,1))
        inverse=1/(p[pos]*p[neg])
        for h in range(len(nh)):
            same=(pos==h)&(neg==h)
            if same.any():
                inverse[same]=Nh[h]*(Nh[h]-1)/(nh[h]*(nh[h]-1))
        oracle_design=(inverse*obs).sum()/inverse.sum()
        pair_design_errors.append(abs(oracle_design-pairwise_design_auc(score,y==1,y==0,strata,Nh)))
        a=np.r_[np.zeros(21,int),np.ones(10,int),np.full(1,2,int)]
        totals=[100,10,1]
        replica=repro_weights(a,totals,rng)
        reproduction=max(reproduction,max(abs(replica[a==j].sum()-n) for j,n in enumerate(totals)))
        assert (replica>0).all() and (replica[a>=1]==1).all()
    assert max(errors+rank_errors+pair_design_errors+[reproduction])<1e-12
    result=dict(status="PASS_ARITHMETIC",trials=200,auc_oracle_error=max(errors),ap_oracle_error=max(rank_errors),
                pairwise_design_auc_oracle_error=max(pair_design_errors),reproduction_mass_error=reproduction,
                coverage_calibration="P2_PENDING",model_score_reads=0)
    (Path(__file__).resolve().parent/"STATISTICS_ARITHMETIC_QA.json").write_text(json.dumps(result,indent=2)+"\n")
    print(result,flush=True)


if __name__=="__main__":
    main()
