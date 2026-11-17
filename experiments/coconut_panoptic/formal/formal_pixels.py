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

"""Download and decode the frozen sample from the public COCO image bucket."""
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
import csv
import io
import os
from pathlib import Path
from threading import local
from time import perf_counter, sleep
from zipfile import BadZipFile, ZipFile

from PIL import Image
import requests

from formal_metrics import read, seal


BUCKET="https://s3.amazonaws.com/images.cocodataset.org"
SESSIONS=local()
LEGACY_CACHE=Path(os.environ.get("FQ_COCO_PIXEL_CACHE", Path(__file__).resolve().parents[3] / "inputs/coco_pixels"))


def validate(row,payload):
    with Image.open(io.BytesIO(payload)) as image:
        expected=(int(row["width"]),int(row["height"]))
        if image.format!="JPEG" or image.size!=expected:
            raise ValueError(f"Expected JPEG {expected}, observed {image.format} {image.size}")
        image.convert("RGB").load()


def fetch(row,output):
    identifier=int(row["id"])
    assert row["zip_member"]==f"unlabeled2017/{identifier:012d}.jpg"
    dest=output/f"{identifier:012d}.jpg"
    if dest.exists():
        try:
            payload=dest.read_bytes()
            validate(row,payload)
            return dict(id=identifier,status="PASS",bytes=len(payload),reused=True)
        except (OSError,ValueError):
            pass
    if not hasattr(SESSIONS,"requests"):
        SESSIONS.requests=requests.Session()
    last="not attempted"
    for attempt in range(4):
        try:
            with SESSIONS.requests.get(f"{BUCKET}/{row['zip_member']}",params={"fq_v931":attempt},
                                       headers={"Accept-Encoding":"identity"},timeout=(10,25)) as response:
                if response.status_code!=200:
                    raise OSError(f"HTTP {response.status_code}")
                payload=response.content
            validate(row,payload)
            partial=dest.with_suffix(".partial")
            partial.write_bytes(payload)
            partial.replace(dest)
            return dict(id=identifier,status="PASS",bytes=len(payload),reused=False)
        except (OSError,ValueError,requests.RequestException) as exc:
            last=f"{type(exc).__name__}: {exc}"
            if attempt<3:
                sleep(1.5*attempt)
    return dict(id=identifier,status="FAIL",reason=last)


def stage_cached(rows,output,archives):
    wanted={f"{int(row['id']):012d}.jpg" for row in rows}
    reusable=0
    for archive_path in sorted(archives.glob("coco_*.zip")):
        try:
            with ZipFile(archive_path) as archive:
                for name in wanted.intersection(archive.namelist()):
                    path=output/name
                    if path.exists():
                        continue
                    tmp=path.with_suffix(".partial")
                    tmp.write_bytes(archive.read(name))
                    tmp.replace(path)
                    reusable+=1
        except (OSError,BadZipFile):
            continue
    return reusable


def run(work,request,sampling_lock,workers):
    lock=read(sampling_lock)
    assert lock["confirmation_model_scores_read"]==0
    with request.open(encoding="utf-8-sig",newline="") as source:
        rows=list(csv.DictReader(source))
    ids=[int(row["id"]) for row in rows]
    assert ids==lock["sample_ids"] and len(set(ids))==len(ids)
    assert len({row["flickr_id"] for row in rows})==len(rows) and all(row["flickr_id"] for row in rows)
    output=work/"confirm_pixels"
    output.mkdir(parents=True,exist_ok=True)
    start=perf_counter()
    cached=stage_cached(rows,output,LEGACY_CACHE)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results=[]
        for position,result in enumerate(pool.map(lambda row:fetch(row,output),rows),1):
            results.append(result)
            if position%500==0 or position==len(rows):
                print("FINAL_PIXELS",position,len(rows),round(perf_counter()-start,1),flush=True)
    assert [row["id"] for row in results]==ids
    errors=[row for row in results if row["status"]!="PASS"]
    report=dict(status="PASS" if not errors else "HOLD_PIXEL",source=BUCKET,sample_ids=ids,decoded=len(rows)-len(errors),
                bytes=sum(row.get("bytes",0) for row in results),wall_seconds=perf_counter()-start,
                reused=sum(row.get("reused",False) for row in results),loaded_from_prior_archives=cached,
                failures=errors,score_exposure_count=0)
    seal(work/"CONF_M_PIXEL_QA.json",report)
    if errors:
        raise RuntimeError(f"HOLD_PIXEL on {len(errors)} of {len(rows)} frozen image IDs")
    return report


if __name__=="__main__":
    parser=ArgumentParser()
    for name in ("work","request","sampling_lock"):
        parser.add_argument(name,type=Path)
    parser.add_argument("--workers",type=int,default=24)
    args=parser.parse_args()
    run(args.work,args.request,args.sampling_lock,args.workers)
