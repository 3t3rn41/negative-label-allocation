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

"""Stress-test indicative linear power using the observed crossed pilot covariance."""
import json
from pathlib import Path

import numpy as np
from scipy.stats import t, beta

HERE=Path(__file__).resolve().parent


def covariance(values):
    a=np.cov(values,rowvar=False)
    eigen,v=np.linalg.eigh(a)
    return (v*np.maximum(eigen,0))@v.T


if __name__=="__main__":
    p=json.loads((HERE/"LINEAR_PRECISION_50E_6R.json").read_text())
    c=json.loads((HERE/"LINEAR_CROSSED_VARIANCE_QA.json").read_text())
    x=np.array([v["directional_advantage_pp"] for v in p["rows"]]).reshape(8,6,8)
    mt=x.mean(1)
    it=x.mean(0)
    rem=x-mt[:,None,:]-it[None,:,:]+x.mean((0,1))
    cm,ci,cr=map(covariance,(mt,it,rem.reshape(-1,8)))
    rng=np.random.default_rng(np.random.SeedSequence([93120261115,30,400]))
    reps=5000
    rows=[]
    for m in (12,16,24,32,48):
        for r in (4,6,8):
            for n in (10000,16000,30000):
                for stress in (1,2):
                    simulation=rng.multivariate_normal(np.zeros(8),stress*cm,size=(reps,m))
                    simulation+=rng.multivariate_normal(np.zeros(8),cr/r,size=(reps,m))
                    common=rng.multivariate_normal(np.zeros(8),ci/r,size=reps)
                    shared=rng.normal(0,np.sqrt(c["dev_image_variance_pp2"]*5000/n),size=reps)
                    means=simulation.mean(1)+common+shared[:,None]
                    se=np.sqrt(stress*cm.mean()/m+ci.mean()/r+cr.mean()/(m*r)+c["dev_image_variance_pp2"]*5000/n)
                    critical=t.ppf(.95,m-1)
                    empirical_se=simulation.mean(2).std(1,ddof=1)/np.sqrt(m)
                    for effect in (.0,.6):
                        xhat=means.mean(1)+effect
                        this=(xhat>=.30)&(xhat-critical*se>0)&(xhat-critical*empirical_se>0)
                        passed=int(this.sum())
                        rows.append(dict(M=m,R=r,n=n,stress=stress,effect_pp=effect,passed=passed,rate=passed/reps,
                                         upper95=1. if passed==reps else float(beta.ppf(.95,passed+1,reps-passed)),
                                         projected_halfwidth_pp=float(t.ppf(.975,m-1)*se)))
    four=[x for x in rows if x["M"]==12 and x["R"]==4 and x["n"]==10000 and x["stress"]==1]
    result=dict(status="DEVELOPMENT_COVARIANCE_SIMULATION_ONLY",trials_per_case=reps,cases=len(rows),smallest_case=four,
                scope="Gaussian crossed DEV pilot covariance and shared image projection. Formal stratified image-and-training bootstrap power is pending.",
                confirm_m_scores=0,rows=rows)
    (HERE/"LINEAR_GATE_PROJECTION.json").write_text(json.dumps(result,indent=2)+"\n")
    print({k:v for k,v in result.items() if k!="rows"},flush=True)
