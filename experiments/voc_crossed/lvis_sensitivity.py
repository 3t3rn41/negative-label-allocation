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

"""Re-evaluate the 72 sealed LVIS20 scores with explicit LVIS negatives."""

import json
from pathlib import Path
import sys
from zipfile import ZipFile

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'metrics'))
from pr_metrics import Metric,cell_labels


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def strict_labels(ids,classes):
    pos={int(i):r for r,i in enumerate(ids)}
    cls={int(c):j for j,c in enumerate(classes)}
    labels=np.full((len(ids),len(classes)),-1,dtype=np.int8)
    with ZipFile(HERE/'lvis_v1_val.zip') as archive:
        with archive.open('lvis_v1_val.json') as file:
            source=json.load(file)
    selected=set()
    for image in source['images']:
        if image['id'] not in pos:
            continue
        selected.add(image['id'])
        for c in image['neg_category_ids']:
            if c in cls:
                labels[pos[image['id']],cls[c]]=0
    assert selected==set(pos)
    for item in source['annotations']:
        i,c=item['image_id'],item['category_id']
        if i in pos and c in cls:
            row,col=pos[i],cls[c]
            assert labels[row,col]!=0
            labels[row,col]=1
    return labels


def metrics(scores,truth,pairs):
    rows=[]
    for c,q in pairs:
        y,z=truth[:,c],truth[:,q]
        conditional=Metric(scores[:,c],cell_labels(y,z,'10','01')).compute_auc()
        natural=Metric(scores[:,c],np.where(z>=0,y,-1)).compute_auc()
        rows.append((conditional,natural))
    ap=np.array([Metric(scores[:,c],truth[:,c]).compute_ap()
                 if np.any(truth[:,c]==0) and np.any(truth[:,c]==1) else np.nan
                 for c in range(truth.shape[1])])
    return np.array(rows),ap


def common_support(values,names):
    supported=np.isfinite(values).all(axis=(0,1))
    means=values[:,:,supported].mean(axis=(0,2)).tolist() if supported.any() else [None,None]
    per_group=[[float(x) if np.isfinite(x) else None for x in row]
               for row in values.mean(0).T]
    previous=values[:,0]
    old_valid=np.isfinite(previous).all(axis=0)
    return dict(previous_all_supported_pp=float(previous[:,old_valid].mean()) if old_valid.any() else None,
                previous_and_strict_common_pp=means,
                common_support=[name for name,ok in zip(names,supported) if ok],
                missing_under_strict=[name for name,ok in zip(names,supported) if not ok],
                per_group_previous_and_strict_pp=per_group)


def main():
    parent=HISTORY/'data/lvis20'
    info=read(parent/'lvis20_pairs.json')
    pairs=[(r['target'],r['partner']) for r in info['pairs']]
    with np.load(parent/'lvis20_confirm_gold.npz') as data:
        ids,truth,classes=data['ids'],data['truth'],data['classes']
    assert len(ids)==5000 and classes.tolist()==info['classes']
    strict=strict_labels(ids,info['class_ids'])
    assert np.isin(truth,(-1,0,1)).all()
    assert np.all((truth==1)==(strict==1)) and np.all((strict!=0)|(truth==0))
    support=[]
    for c,p in pairs:
        counts=[]
        for labels in (truth,strict):
            counts.append([int(((labels[:,c]==a)&(labels[:,p]==b)).sum())
                           for a,b in ((1,0),(0,1),(1,1),(0,0))])
        support.append(dict(target=str(classes[c]),partner=str(classes[p]),
                            previous_10_01_11_00=counts[0],strict_10_01_11_00=counts[1]))
    ap_support=[dict(class_name=str(cl),positive=int((strict[:,j]==1).sum()),
                     previous_negative=int((truth[:,j]==0).sum()),strict_negative=int((strict[:,j]==0).sum()))
                for j,cl in enumerate(classes)]
    by_method={}
    receipt=read(HISTORY/'run/manifests/LVIS72_SCORES_SEALED.json')
    assert receipt['sealed'] and len(receipt['entries'])==72
    aligned=[]
    for method in ('DC0','LIN'):
        values={}
        for job in (e for e in receipt['entries'] if e['method']==method):
            source=HISTORY/'run/lvis_models'/method/job['job']/Path(job['score_path']).name
            assert source.stat().st_size==job['score_size']
            with np.load(source) as s:
                assert np.array_equal(s['ids'],ids) and np.array_equal(s['classes'],classes)
                score=s['scores']
            assert score.shape==(5000,20) and np.isfinite(score).all()
            values[job['draw'],job['init'],job['stratum'],job['level']]=[
                metrics(score,labels,pairs) for labels in (truth,strict)]
            aligned.append(job['job'])
        comparisons=[]
        for draw in (61001,61002,61003):
            for seed in (61101,61102,61103):
                cell=[]
                for i in range(2):
                    tl,tm,pl,pm=[values[(draw,seed)+key][i] for key in
                              (('TRUE','LOW'),('TRUE','MID'),('PLC','LOW'),('PLC','MID'))]
                    cell.append((100*((tm[0]-tl[0])-(pm[0]-pl[0])),100*(tm[1]-tl[1])))
                comparisons.append(cell)
        b=np.array([[x[0][0],x[1][0]] for x in comparisons])
        maps=np.array([[x[0][1],x[1][1]] for x in comparisons])
        pair_names=[f'{classes[c]}/{classes[p]}' for c,p in pairs]
        by_method[method]=dict(conditional=common_support(b[:,:,:,0],pair_names),
                               full_auc=common_support(b[:,:,:,1],pair_names),
                               true_map_change=common_support(maps,classes.tolist()))
    assert len(set(aligned))==72
    labels_removed=int(((truth==0)&(strict==-1)).sum())
    summary='严格使用 LVIS neg_category_ids，对齐 72 个封存端点。'
    summary+=f'在 5,000 张图像的 20 类标签中，有 {labels_removed:,} 个旧负例坐标转为未决。'
    for method,r in by_method.items():
        conditional=r['conditional']
        old,new=conditional['previous_and_strict_common_pp']
        n=len(conditional['common_support'])
        if n:
            summary+=f'{method} 在两种口径均有支持的 {n} 对中，条件效应由 {old:+.3f} 变为 {new:+.3f} 个百分点。'
        else:
            summary+=f'{method} 的五对条件比较在严格口径下均缺少至少一侧支持。'
    summary+='逐对支持及 AP 结果见回执。 [LVIS 标注协议](https://www.lvisdataset.org/assets/lvis_v0.5.pdf)'
    result=dict(status='COMPLETE',sealed_scores_aligned=len(aligned),label_policy='instance presence positive; neg_category_ids negative; all other cells unknown',
                validation_source='https://dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip',
                known_negatives_reclassified_as_unknown=labels_removed,pair_support=support,ap_support=ap_support,
                methods=by_method,report_text=summary)
    (HERE/'LVIS_QA.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    print(summary,flush=True)


if __name__=='__main__':
    main()
