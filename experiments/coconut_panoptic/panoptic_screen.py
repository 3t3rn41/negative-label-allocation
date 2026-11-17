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

"""Screen first-use COCO stuff classes while retaining the original image roles."""
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
from time import perf_counter

import ijson
import numpy as np
from PIL import Image
from io import BytesIO
from pyarrow.parquet import ParquetFile


HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "coco_reference_labels"
THRESHOLDS = np.array([50,50,30,200])
# Built-environment/background surfaces with distinct COCO definitions. Historical
# object classes, their synonyms and broad overlapping group classes stay out.
NEW_STUFF = (92,93,95,100,107,109,112,118,119,125,128,130,133,138,141,144,145,148,149,151,
             154,155,156,159,161,166,168,171,175,176,177,178,180,181,184,185,186,187,188,190,191,
             192,193,194,195,197,198,199,200)


def project(path, ids, order):
    positions = {i:j for j,i in enumerate(ids.tolist())}
    result = np.zeros((len(ids),len(order)),dtype=np.int8)
    lookup = {value:i for i,value in enumerate(order)}
    seen = set()
    with path.open("rb") as stream:
        for row in ijson.items(stream,"annotations.item"):
            image = row["image_id"]
            if image not in positions:
                continue
            assert image not in seen
            seen.add(image)
            y = result[positions[image]]
            for segment in row["segments_info"]:
                col = lookup.get(segment["category_id"])
                if col is not None and segment["area"] > 0:
                    y[col] = 1
    return result,np.isin(ids,list(seen))


def coconut(paths, ids, order):
    lookup = {value:i for i,value in enumerate(order)}
    positions = {i:j for j,i in enumerate(ids.tolist())}
    result = np.full((len(ids),len(order)),-1,dtype=np.int8)
    seen = set()
    def decode(row):
        image = row["image_info"]["id"]
        ann = row["segments_info"]
        assert ann["image_id"] == image
        y = result[positions[image]]
        with Image.open(BytesIO(row["mask"]["bytes"])) as im:
            r,g,b = im.convert("RGB").split()
            mask = np.asarray(r)
            if g.getbbox() or b.getbbox():
                mask = mask.astype(np.int32)+256*np.asarray(g,dtype=np.int32)+65536*np.asarray(b,dtype=np.int32)
            present = set(np.flatnonzero(np.bincount(mask.ravel())).tolist()) if mask.max()<65536 else set(np.unique(mask).tolist())
        present.discard(0)
        ann_ids = {s["id"] for s in ann["segments_info"]}
        if present and present <= ann_ids:
            y.fill(0)
        valid, unresolved = set(), set()
        for seg in ann["segments_info"]:
            col = lookup.get(seg["category_id"])
            if col is not None:
                (valid if seg["id"] in present else unresolved).add(col)
        y[list(unresolved)] = -1
        y[list(valid)] = 1
        return image

    with ThreadPoolExecutor(max_workers=6) as workers:
        for p in paths:
            for batch in ParquetFile(p).iter_batches(batch_size=512, columns=["image_info.id","segments_info","mask"]):
                for image in workers.map(decode,(r for r in batch.to_pylist() if r["image_info"]["id"] in positions)):
                    assert image not in seen
                    seen.add(image)
            print("annotation projection",p.name,len(seen),flush=True)
    assert len(seen) == len(ids)
    return result


def counts(data,t,p):
    a,b = data[:,t],data[:,p]
    keep = (a>=0)&(b>=0)
    return np.bincount(2*a[keep]+b[keep],minlength=4)[[2,1,3,0]]


def main(data, parquet):
    start = perf_counter()
    train = data/"panoptic_train2017.json"
    val = data/"panoptic_val2017.json"
    with val.open("rb") as f:
        cats = sorted(ijson.items(f,"categories.item"),key=lambda r:r["id"])
    assert len(cats)==133
    universe = [row["id"] for row in cats if row["isthing"]==0]
    print("stuff categories",[(r["id"],r["name"]) for r in cats if r["isthing"]==0],flush=True)
    selected = sorted(set(NEW_STUFF)&set(universe))
    fit = np.load(PREVIOUS/"FIT80.npz",allow_pickle=False)
    dev = np.load(PREVIOUS/"DEV80.npz",allow_pickle=False)
    bridge = np.load(PREVIOUS/"DEV_BRIDGE80_VALIDATED.npz",allow_pickle=False)
    fi,di,bi = fit["ids"],dev["ids"],bridge["ids"]
    fy,ff = project(train,fi,universe)
    dy,df = project(train,di,universe)
    by,bf = project(val,bi,universe)
    role = [ff.sum().item(),df.sum().item(),bf.sum().item()]
    # The official train panoptic file may omit instance-only COCO images.
    print("official panoptic coverage",role,[len(fi),len(di),len(bi)],flush=True)
    np.savez_compressed(HERE/"PANOPTIC_OFFICIAL_LABELS.npz",fit_ids=fi[ff],fit=fy[ff],dev_ids=di[df],dev=dy[df],
        bridge_ids=bi[bf],bridge=by[bf],category_ids=universe)
    names = {r["id"]:r["name"] for r in cats}
    original = set(fit["category_ids"].tolist())
    targets = [universe.index(c) for c in selected]
    dev_full = np.column_stack((dev["truth"][df],dy[df]))
    fit_full = np.column_stack((fit["truth"][ff],fy[ff]))
    columns = [*fit["category_ids"].tolist(),*universe]
    by_new = coconut([parquet/"relabeled_coco_val.parquet"],bi,universe)
    if int((by_new<0).sum()):
        print("Bridge unknown",int((by_new<0).sum()),flush=True)
    source_full = np.column_stack((bridge["coconut"],by_new))
    np.savez_compressed(HERE/"PANOPTIC_BRIDGE_LABELS.npz",ids=bi,truth=source_full,categories=columns,
                        original_coco=np.column_stack((bridge["coco"],by)))
    with (HERE/"PANOPTIC_BRIDGE_CLASS_QA.csv").open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(["category_id","category","original_positive","coconut_positive","both_known","changed_on_known","positive_jaccard"])
        first=np.column_stack((bridge["coco"],by))
        for j,c in enumerate(columns):
            a,b=first[:,j],source_full[:,j]
            known=b>=0
            overlap=int(((a==1)&(b==1)).sum())
            union=int((known&((a==1)|(b==1))).sum())
            writer.writerow([c,names[c],int((a==1).sum()),int((b==1).sum()),int(known.sum()),int(((a!=b)&known).sum()),overlap/union if union else ""])
    from itertools import combinations
    rows=[]
    c0=len(original)
    for j in targets:
        t=j+c0
        # Thing partners give a stable, auditable ontology and exclude overlaps among stuff labels.
        eligible=[p for p in range(c0) if np.all(counts(dev_full,t,p)>=THRESHOLDS)]
        for a,b in combinations(eligible,2):
            n1=int(((fit_full[:,t]==0)&(fit_full[:,a]==1)&(fit_full[:,b]==0)).sum())
            n2=int(((fit_full[:,t]==0)&(fit_full[:,a]==0)&(fit_full[:,b]==1)).sum())
            if min(n1,n2)<20:
                continue
            source=np.stack([counts(source_full,t,p) for p in (a,b)])
            with np.errstate(divide="ignore",invalid="ignore"):
                n10,n01,n11,n00=source.T
                negative=1/(.25/n00+.25/n01)
            m3=bool(np.all(source>=THRESHOLDS) and np.all(n10+n11>=80) and np.all(negative>=250))
            rows.append(dict(target_id=columns[t],target=names[columns[t]],partner_ids=[columns[a],columns[b]],partners=[names[columns[a]],names[columns[b]]],
                 fit_positive=int(fit_full[:,t].sum()),fit_exclusive=[n1,n2],dev_counts=[counts(dev_full,t,p).tolist() for p in (a,b)],
                 coconut_bridge_counts=source.tolist(),source_support_pass=m3))
    rows.sort(key=lambda r:(r["target_id"],-min(r["fit_exclusive"]),r["partner_ids"]))
    result=dict(status="LABEL_SUPPORT_ONLY", original_roles_covered=role, original_roles=[len(fi),len(di),len(bi)],
        official_stuff_categories=[dict(category_id=c,name=names[c]) for c in universe], reviewed_new_stuff=[dict(category_id=c,name=names[c]) for c in selected],
        targets_passing_fit_dev=sorted({r["target"] for r in rows}), source_supported_targets=sorted({r["target"] for r in rows if r["source_support_pass"]}),
        candidate_triples=len(rows), model_score_reads=0, wall_seconds=perf_counter()-start)
    (HERE/"PANOPTIC_SCREEN.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    with (HERE/"PANOPTIC_CANDIDATES.csv").open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ["target_id","target"])
        writer.writeheader();writer.writerows(rows)
    print({k:v for k,v in result.items() if not isinstance(v,list) or k.endswith("targets")},flush=True)


if __name__=="__main__":
    a=ArgumentParser();a.add_argument("archive",type=Path);a.add_argument("parquet",type=Path);o=a.parse_args()
    main(o.archive,o.parquet)
