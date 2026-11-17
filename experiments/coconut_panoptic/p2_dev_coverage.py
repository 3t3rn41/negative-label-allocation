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

"""Estimate design bias and interval coverage on an actually scored validation population."""
from argparse import ArgumentParser
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.stats import beta, norm

from joint_replica import Ranking
from p2_projection import SEED, data


HERE=Path(__file__).resolve().parent
FRACTIONS=((.20,.45),(.35,.70),(.55,.75),(.75,1.))


def compiled_ranks(heads,truth,pairs):
    ranks=[]
    population=np.ones(len(truth))
    gradient=np.zeros(len(truth))
    interaction=0.
    for j,(target,partner) in enumerate(pairs):
        pos=(truth[:,target]==1)&(truth[:,partner]==0)
        neg=(truth[:,target]==0)&(truth[:,partner]==1)
        for mask in range(heads.shape[0]):
            for init in range(heads.shape[1]):
                for arm,sign in ((j%2,1),(1-j%2,-1)):
                    rank=Ranking(heads[mask,init,arm,:,target],pos,neg)
                    point,g=rank.influence(population)
                    ranks.append((rank,sign))
                    interaction+=sign*point
                    gradient+=sign*g
    divisor=len(pairs)*heads.shape[0]*heads.shape[1]
    return ranks,interaction/divisor,gradient/divisor,divisor


def check_one(ranks,population,g,divisor,stratum,fractions,reps,rng):
    groups=[np.flatnonzero(stratum==h) for h in (0,1)]
    N=np.array([len(h) for h in groups])
    quota=np.rint(N*fractions).astype(int)
    assert (quota>1).all() and (quota<=N).all()
    values=[]
    halfwidth=[]
    for k in range(0,reps,64):
        count=min(reps-k,64)
        sample=[np.array([rng.choice(h,n,replace=False,shuffle=False) for _ in range(count)])
                for h,n in zip(groups,quota,strict=True)]
        weights=np.zeros((count,len(g)))
        rows=np.arange(count)[:,None]
        for inds,weight in zip(sample,N/quota,strict=True):
            weights[rows,inds]=weight
        values.append(np.sum([sign*rank.calculate(weights) for rank,sign in ranks],axis=0)/divisor)
        variance=np.sum([N[h]**2*(1-quota[h]/N[h])*g[positions].var(axis=1,ddof=1)/quota[h]
                         for h,positions in enumerate(sample)],axis=0)
        halfwidth.append(norm.ppf(.975)*np.sqrt(variance))
    estimates=np.concatenate(values)
    widths=np.concatenate(halfwidth)
    available=np.isfinite(estimates)&np.isfinite(widths)
    valid=int(available.sum())
    error=estimates[available]-population
    coverage=int((abs(error)<=widths[available]).sum())
    lo=0. if not coverage else beta.ppf(.025,coverage,valid-coverage+1)
    hi=1. if coverage==valid else beta.ppf(.975,coverage+1,valid-coverage)
    return dict(stratum_fractions=fractions,dev_population=N.tolist(),dev_quotas=quota.tolist(),
                requested_replays=reps,valid_replays=valid,bias_pp=float(error.mean()*100),
                two_sided_coverage=coverage/valid,coverage_mc95=[float(lo),float(hi)],
                mean_halfwidth_pp=float(widths[available].mean()*100))


def run(model,reps):
    started=perf_counter()
    heads,truth,pairs,stratum=data(model)
    ranks,population,g,divisor=compiled_ranks(heads,truth,pairs)
    rng=np.random.default_rng(np.random.SeedSequence([SEED,30,870 if model=="linear" else 871]))
    rows=[]
    with np.errstate(divide="ignore",invalid="ignore"):
        for fractions in FRACTIONS:
            result=check_one(ranks,population,g,divisor,stratum,fractions,reps,rng)
            rows.append(result)
            print(model,result,flush=True)
    data_scope="Exact design-weighted conditional contrasts on A1, fixed development training grid"
    out=dict(status="DEVELOPMENT_IMAGE_COVERAGE_ONLY",model=model,scope=data_scope,
             population_I_pp=population*100,formal_scores_read=0,rows=rows,wall_seconds=perf_counter()-started)
    target=HERE/f"P2_{model.upper()}_DEV_COVERAGE.json"
    temp=target.with_suffix(".partial")
    temp.write_text(json.dumps(out,indent=2)+"\n",encoding="utf-8")
    temp.replace(target)


if __name__=="__main__":
    parser=ArgumentParser()
    parser.add_argument("model",choices=("linear","dualcoop"))
    parser.add_argument("--reps",type=int,default=2500)
    args=parser.parse_args()
    run(args.model,args.reps)
