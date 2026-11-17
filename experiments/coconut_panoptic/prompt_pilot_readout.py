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

"""Collect sealed prompt scores and evaluate the two development pools."""
from argparse import ArgumentParser
import json
from pathlib import Path
import sys
from time import monotonic, sleep, time

import numpy as np

from statistics_primitives import weighted_auc


HERE = Path(__file__).resolve().parent
ARMS = ["DIR1", "DIR2", "PLC"]


def get_json(name):
    return json.loads((HERE/name).read_text(encoding="utf-8"))


def save_readout(name,report):
    path=HERE/name
    temp=path.with_suffix(".partial")
    temp.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    temp.replace(path)


def u_auc(score,positive,negative):
    sorted_negatives=np.sort(score[negative])
    a=np.searchsorted(sorted_negatives,score[positive],side="left")
    b=np.searchsorted(sorted_negatives,score[positive],side="right")
    assert len(a) and len(sorted_negatives)
    return (a+b).mean()/(2*len(sorted_negatives))


def curve(scores,target,partner):
    known=(target>=0)&(partner>=0)
    cells=[known & (target==y) & (partner==p) for y,p in ((1,0),(1,1),(0,0),(0,1))]
    counts=[int(c.sum()) for c in cells]
    assert min(counts)>0
    pos=counts[1]/(counts[0]+counts[1])
    neg=counts[3]/(counts[2]+counts[3])
    mixing=np.array([(1-pos)*(1-neg),(1-pos)*neg,pos*(1-neg),pos*neg])
    groups=np.array([u_auc(scores,a,b) for a in cells[:2] for b in cells[2:]])
    pooled=u_auc(scores,cells[0]|cells[1],cells[2]|cells[3])
    assert abs(groups@mixing-pooled)<1e-12
    return groups,pooled


def summarize(draws,dev_gold=None):
    targets=get_json("TARGET_REPAIR_STATUS.json")["selected"]
    columns=[r["id"] for r in get_json("PANOPTIC_CLASS_SPACE_LOCK.json")["categories"]]
    pairs=[(columns.index(r["target_id"]),columns.index(p)) for r in targets for p in r["partners"]]
    exact=np.asarray([np.random.SeedSequence([93120261115,11,i]).generate_state(1,np.uint64)[0] for i in range(1,5)])
    coco=dev_gold or HERE.parent/"coco_reference_labels/DEV80.npz"
    with (np.load(coco,allow_pickle=False) as a,
          np.load(HERE/"PANOPTIC_OFFICIAL_LABELS.npz",allow_pickle=False) as b,
          np.load(HERE/"PANOPTIC_BRIDGE_LABELS.npz",allow_pickle=False) as c):
        np.testing.assert_array_equal(a["ids"],b["dev_ids"])
        assert np.r_[a["category_ids"],b["category_ids"]].tolist()==c["categories"].tolist()==columns
        dev_truth=np.column_stack((a["truth"],b["dev"]))
        dev_lookup={image:i for i,image in enumerate(a["ids"].tolist())}
        bridge_ids=c["ids"]
        bridge={mode:c[mode] for mode in ("truth","original_coco")}
    scores=[]
    max_error=0.
    for draw in draws:
        with np.load(HERE/f"PROMPT_DEV_{draw}.npz",allow_pickle=False) as s:
            assert s["categories"].tolist()==columns
            assert s["arms"].tolist()==ARMS*4
            np.testing.assert_array_equal(s["seeds"],np.repeat(exact,3))
            assert s["seeds"].dtype==np.uint64
            ids=s["ids"]
            data=s["scores"].reshape(4,3,10000,133)
            assert np.isfinite(data).all() and ((data>=0)&(data<=1)).all()
        np.testing.assert_array_equal(ids[5000:],bridge_ids)
        dev=dev_truth[[dev_lookup[i] for i in ids[:5000]]]
        assert len(set(ids.tolist()))==10000
        auc=np.empty((4,3,len(pairs)))
        source={mode:np.empty((4,3,len(pairs),5)) for mode in bridge}
        weight=np.ones(5000)
        for j,(t,p) in enumerate(pairs):
            pos=(dev[:,t]==1)&(dev[:,p]==0)
            neg=(dev[:,t]==0)&(dev[:,p]==1)
            for i in range(4):
                for a in range(3):
                    z=data[i,a,:,t]
                    score=u_auc(z[:5000],pos,neg)
                    max_error=max(max_error,abs(score-weighted_auc(z[:5000],pos,neg,weight)))
                    auc[i,a,j]=score
                    for mode,y in bridge.items():
                        four,natural=curve(z[5000:],y[:,t],y[:,p])
                        source[mode][i,a,j,:4]=four
                        source[mode][i,a,j,4]=natural
        delta=auc[:,:2]-auc[:,2:]
        interaction=np.where(np.arange(len(pairs))%2,delta[:,1]-delta[:,0],delta[:,0]-delta[:,1])
        scores.append(dict(draw=draw,conditional_delta_pp=(delta*100).tolist(),
            direction_pp=(interaction.reshape(4,-1,2)*100).tolist(),
            bridge_auc_pp={mode:(raw*100).tolist() for mode,raw in source.items()}))
    assert max_error<1e-10
    return merge_readouts([dict(rows=scores,independent_auc_max_error=max_error)])


def merge_readouts(reports):
    scores=[row for report in reports for row in report["rows"]]
    assert len({row["draw"] for row in scores})==len(scores)
    error=max(a["independent_auc_max_error"] for a in reports)
    merged=np.array([row["direction_pp"] for row in scores])
    return dict(status="PASS",endpoints=len(scores)*12,development_images=5000,source_bridge_images=5000,
        confirmation_scores_read=0,independent_auc_max_error=error,mean_I_pp=float(merged.mean()),
        by_target_pp=merged.mean(axis=(0,1,3)).tolist(),
        source_cell_order=["10/00","10/01","11/00","11/01","natural"],rows=scores)


def follow_on_host(draws,dev_gold,deadline):
    pending=list(draws)
    verified={}
    final=HERE/"PROMPT_DEVELOPMENT_QA.json"
    while pending:
        done=final.exists()
        for draw in tuple(pending):
            if not (HERE/f"PROMPT_DEV_{draw}_TIMING.json").exists():
                continue
            started=monotonic()
            name=f"PROMPT_DEV_{draw}_INDEPENDENT_QA.json"
            report=summarize([draw],dev_gold)
            report["readout_seconds"]=monotonic()-started
            save_readout(name,report)
            verified[draw]=report
            pending.remove(draw)
            print("Independent on-host development readout",draw,len(verified),"/",len(draws),flush=True)
        if not pending:
            save_readout("PROMPT_PILOT_INDEPENDENT_QA.json",merge_readouts([verified[d] for d in draws]))
            return
        if done and get_json(final.name)["status"]!="COMPLETE":
            print("GPU development ended with a partial matrix",flush=True)
            raise SystemExit(3)
        if time()>=deadline:
            raise TimeoutError("Development score publication did not complete before the registered deadline")
        sleep(5)


if __name__=="__main__":
    parser=ArgumentParser()
    parser.add_argument("--dev-gold",type=Path)
    parser.add_argument("--follow-until",type=float)
    args=parser.parse_args()
    draws=get_json("PANOPTIC_MASK_SCREEN.json")["dev_seeds"]
    if args.follow_until is not None:
        assert args.dev_gold
        follow_on_host(draws,args.dev_gold,args.follow_until)
        sys.exit(0)
    readout=summarize(draws,args.dev_gold)
    save_readout("PROMPT_PILOT_INDEPENDENT_QA.json",readout)
    print({k:v for k,v in readout.items() if k!="rows"},flush=True)
