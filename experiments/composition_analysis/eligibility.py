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

"""Label-only eligibility and local input readiness for later training."""
import csv
from collections import defaultdict

import numpy as np

from replay import AB, G1, HERE, P0, ROOT, V9, Metric, PAIR_MAP, read, save, table


def counts(gold, t, p):
    a, b = gold[:, t], gold[:, p]
    return tuple(int(np.sum((a == y) & (b == q))) for y,q in ((1,0),(0,1),(1,1),(0,0)))


def support(dev, test, test_minima):
    n10, n01, n11, n00 = test
    pstar = dev[2]/(dev[0]+dev[2]) if dev[0]+dev[2] else None
    pos = 1/((1-pstar)**2/n10+pstar**2/n11) if n10 and n11 and pstar is not None else 0.
    neg = [1/((1-r)**2/n00+r**2/n01) if n00 and n01 else 0. for r in (.05,.50)]
    valid = all(n >= 30 for n in dev) and all(n >= m for n,m in zip(test,test_minima,strict=True))
    return dict(**{f"dev_{key}": n for key,n in zip(("10","01","11","00"),dev)},
                **{f"eval_{key}": n for key,n in zip(("10","01","11","00"),test)},
                pstar=pstar, pos_ess=pos, neg_ess_05=neg[0], neg_ess_50=neg[1],
                cell_support=valid, passes=bool(valid and pos>=80 and min(neg)>=250))


def t1():
    with (np.load(V9/"coco_development/DEV_SCORES.npz") as s,
          np.load(G1/"raw/data/COCO_CONFIRM_GOLD.npz") as d,
          np.load(AB/"B_GOLD.npz") as b):
        ids, classes = s["ids"], s["classes"]
        np.testing.assert_array_equal(ids,d["ids"])
        np.testing.assert_array_equal(classes,d["classes"])
        np.testing.assert_array_equal(classes,b["classes"])
        assert not set(ids)&set(b["ids"])
        assert read(V9/"coco_development/PILOT_RESULT.json")["independent_rescore_exact"]
        dev, evaluation, scores = d["truth"], b["truth"], s["scores"]
    historical = {(p["target"],p["partner"]) for pairs in PAIR_MAP.values() for p in pairs}
    rows = []
    for t, target in enumerate(classes):
        for p, partner in enumerate(classes):
            if t==p or (t,p) in historical:
                continue
            n_dev, n_eval = counts(dev,t,p), counts(evaluation,t,p)
            if n_dev[0] and n_dev[1]:
                labels = np.where((dev[:,t]==1)&(dev[:,p]==0),1,
                                  np.where((dev[:,t]==0)&(dev[:,p]==1),0,-1))
                auc = Metric(scores[:,t],labels).compute_auc()
            else:
                auc = None
            rows.append(dict(target=target,partner=partner,dev_conditional_auc=auc,
                             **support(n_dev,n_eval,(50,50,30,200))))
    rows.sort(key=lambda r:(r["dev_conditional_auc"] if r["dev_conditional_auc"] is not None else np.inf,
                            -r["dev_11"],r["target"],r["partner"]))
    eligible = 0
    for k,r in enumerate(rows,1):
        eligible += r["passes"]
        r["rank_development"] = k
        r["rank_eligible"] = eligible if r["passes"] else ""
        r["top_seven"] = bool(r["passes"] and eligible <= 7)
    table("T1_label_screen.csv",rows)
    selected = [r for r in rows if r["top_seven"]]
    save("T1_label_screen.json",dict(eligible=eligible, screened=len(rows), selected=selected,
         role="B reused for labels-only support", status="LABEL_SCREEN_COMPLETE",
         joint_mask_status="PENDING_CROSSED_DESIGN"))
    print("T1 eligible",eligible,"top seven",[(r["target"],r["partner"]) for r in selected],flush=True)


def open_labels(path, ids):
    data = {mid: {} for mid in ids}
    conflicts = {mid: set() for mid in ids}
    reviewed = set()
    with path.open(encoding="utf-8-sig",newline="") as file:
        assert file.readline().strip() == "ImageID,Source,LabelName,Confidence"
        for line in file:
            img,source,mid,confidence = line.rstrip().split(",")
            reviewed.add(img)
            if mid not in data:
                continue
            assert source in ("verification","crowdsource-verification")
            label = float(confidence)
            assert label in (0,1)
            if img in conflicts[mid]:
                continue
            existing = data[mid].get(img)
            if existing is not None and existing != label:
                conflicts[mid].add(img)
                del data[mid][img]
            else:
                data[mid][img] = label
    return data, dict(images=len(reviewed),conflicts={mid:len(x) for mid,x in conflicts.items()})


def t6():
    folder = HERE/"open_images"
    mapping = {"chair":("Chair","/m/01mzpv"),"bottle":("Bottle","/m/04dr76w"),
               "person":("Person","/m/01g317"),"pottedplant":("Houseplant","/m/03fp41"),
               "sofa":("Couch","/m/02crq1"),"tvmonitor":("Television","/m/07c52"),
               "diningtable":("Table","/m/04bcr3")}
    proxies = {"diningtable","pottedplant","tvmonitor"}
    with (folder/"oidv7-class-descriptions.csv").open(encoding="utf-8-sig",newline="") as f:
        names = dict(csv.reader(f))
    for label,mid in mapping.values():
        assert names[mid] == label
    paths = [folder/f"oidv7-{split}-annotations-human-imagelabels.csv" for split in ("val","test")]
    splits, receipts = zip(*(open_labels(path,{mid for _,mid in mapping.values()}) for path in paths),strict=True)
    assert [r["images"] for r in receipts]==[41620,125436]
    all_pairs = {k:[(r["target_name"],r["partner_name"]) for r in PAIR_MAP[k]] for k in ("S0","N1")}
    all_pairs.update(read(V9/"voc_design/DATA_ROLE_LOCK.json")["pairs"])
    rows = []
    for setting,pairs in all_pairs.items():
        for t,p in pairs:
            tm,pm = mapping[t][1],mapping[p][1]
            vectors = []
            for data in splits:
                a,b = data[tm],data[pm]
                common = a.keys() & b.keys()
                counter = defaultdict(int)
                for img in common:
                    counter[a[img],b[img]] += 1
                result = tuple(counter[x] for x in ((1,0),(0,1),(1,1),(0,0)))
                assert sum(result) == len(common)
                assert result == tuple(len({k for k,v in a.items() if v==x} &
                                            {k for k,v in b.items() if v==y})
                                       for x,y in ((1,0),(0,1),(1,1),(0,0)))
                vectors.append(result)
            rows.append(dict(setting=setting,target=t,partner=p,open_images_target=mapping[t][0],target_mid=tm,
                             open_images_partner=mapping[p][0],partner_mid=pm, proxy=bool({t,p}&proxies),
                             **support(*vectors,(30,30,30,30))))
    table("T6_open_images_support.csv",rows)
    save("T6_open_images_support.json",dict(official_source="https://storage.googleapis.com/openimages/web/download_v7.html",
         val=receipts[0],test=receipts[1],passed=sum(r["passes"] for r in rows),
         exact_mapping_passed=sum(r["passes"] and not r["proxy"] for r in rows),
         csv_files=[dict(filename=p.name,bytes=p.stat().st_size) for p in paths]))
    print("T6 support",[(r["setting"],r["target"],r["eval_10"],r["eval_01"],r["passes"]) for r in rows],flush=True)


def t3():
    base = ROOT/"inputs/coco_features"
    archives = [("historical_train",base/"train_ids.npy"),("historical_confirm",base/"confirm_ids.npy")]
    pools = [("fit",G1/"raw/data/COCO_TRAIN_GOLD.npz"),
             ("A1",P0/"A1/CONFIRM_GOLD.npz"),("A",AB/"A_GOLD.npz"),("B",AB/"B_GOLD.npz")]
    rows = []
    for name,path in pools:
        with np.load(path) as d:
            ids = set(d["ids"].tolist())
        for cache,cpath in archives:
            cached = set(np.load(cpath).tolist())
            rows.append(dict(pool=name,cache=cache,pool_count=len(ids),cached_id_count=len(ids&cached),
                             source=cpath.relative_to(ROOT).as_posix()))
    table("T3_feature_coverage.csv",rows)


if __name__ == "__main__":
    t1()
    t3()
    t6()

