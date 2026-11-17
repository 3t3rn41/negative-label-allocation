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

"""Independently verify the 144 linear endpoint scores on the frozen bridge."""
import json
from pathlib import Path

import numpy as np

from prompt_pilot_readout import save_readout, u_auc
from statistics_primitives import weighted_auc


def main():
    here=Path(__file__).resolve().parent
    mask=json.loads((here/"PANOPTIC_MASK_SCREEN.json").read_text(encoding="utf-8"))
    with np.load(here/"PANOPTIC_BRIDGE_LABELS.npz",allow_pickle=False) as data:
        ids=data["ids"]
        classes=data["categories"].tolist()
        truth=data["truth"]
    pairs=[(classes.index(q["target_id"]),classes.index(p)) for q in mask["chosen"] for p in q["partners"]]
    registered=np.asarray([np.random.SeedSequence([93120261115,11,j]).generate_state(1,np.uint64)[0] for j in range(1,7)])
    effects=[]
    max_error=0.
    weight=np.ones(len(ids))
    for draw in mask["dev_seeds"]:
        with np.load(here/f"LINEAR_BRIDGE_{draw}_50E_6R.npz",allow_pickle=False) as score:
            np.testing.assert_array_equal(score["ids"],ids)
            np.testing.assert_array_equal(score["seeds"],np.repeat(registered,3))
            assert score["categories"].tolist()==classes and score["arms"].tolist()==["DIR1","DIR2","PLC"]*6
            case=score["scores"].reshape(6,3,len(ids),len(classes))
        assert np.isfinite(case).all() and ((case>=0)&(case<=1)).all()
        differences=np.empty((6,len(pairs)))
        for j,(target,partner) in enumerate(pairs):
            pos=(truth[:,target]==1)&(truth[:,partner]==0)
            neg=(truth[:,target]==0)&(truth[:,partner]==1)
            assert pos.sum()>=50 and neg.sum()>=50
            for init in range(6):
                auc=[]
                for arm in range(3):
                    values=case[init,arm,:,target]
                    point=u_auc(values,pos,neg)
                    max_error=max(max_error,abs(point-weighted_auc(values,pos,neg,weight)))
                    auc.append(point)
                differences[init,j]=100*(auc[j%2]-auc[1-j%2])
        effects.append(differences)
    array=np.asarray(effects)
    assert len(effects)==8 and max_error<1e-10
    receipt=dict(status="PASS",role="DEV-BRIDGE",images=len(ids),draws=len(effects),endpoints=len(effects)*18,
                 max_abs_auc_error=max_error,mean_partner_interaction_pp=float(array.mean()),
                 target_interactions_pp=array.reshape(8,6,4,2).mean((0,1,3)).tolist(),confirmation_scores_read=0)
    save_readout("LINEAR_BRIDGE_INDEPENDENT_QA.json",receipt)
    print(json.dumps(receipt),flush=True)


if __name__=="__main__":
    main()
