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

"""Retrieve the released COCONut-B Parquet shards and project just label metadata."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq
import requests

BASE = "https://huggingface.co/datasets/xdeng77/coconut_b/resolve/main/data"
SIZES = (417218921, 417769003, 400871427, 405293848)
CHUNK = 16 * 1024 * 1024


def fetch_range(url, start, size, total, folder):
    end = start + size - 1
    part = folder / f"{start:012d}.part"
    if part.is_file() and part.stat().st_size == size:
        return part
    for attempt in range(4):
        try:
            endpoint = url if attempt % 2 == 0 else url.replace("huggingface.co", "hf-mirror.com")
            with requests.get(endpoint, params={"download": "true", "offset": start, "retry": attempt},
                              headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
                              stream=True, timeout=(12, 35)) as response:
                if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{total}":
                    raise OSError(f"Unexpected range response: {response.status_code}, {response.headers.get('Content-Range')}")
                with part.open("wb") as stream:
                    for block in response.iter_content(1024 * 1024):
                        stream.write(block)
                if part.stat().st_size != size:
                    raise OSError(f"Truncated range {start}-{end}")
                return part
        except (OSError, requests.RequestException) as exc:
            print("retry", start, attempt, type(exc).__name__, flush=True)
            if attempt == 3:
                raise
            time.sleep(attempt + 1)


def shards(folder, indexes):
    work = []
    for index in indexes:
        filename = f"train-{index:05d}-of-00004.parquet"
        data = folder / filename
        segment_dir = folder / f"parts-{index}"
        segment_dir.mkdir(exist_ok=True)
        if data.is_file() and data.stat().st_size == SIZES[index]:
            continue
        for start in range(0, SIZES[index], CHUNK):
            work.append((f"{BASE}/{filename}", start, min(CHUNK, SIZES[index] - start),
                         SIZES[index], segment_dir))
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, _ in enumerate(pool.map(lambda args: fetch_range(*args), work), 1):
            if i % 10 == 0 or i == len(work):
                print("ranges", i, "/", len(work), flush=True)
    for index in indexes:
        name = folder / f"train-{index:05d}-of-00004.parquet"
        if not name.exists() or name.stat().st_size != SIZES[index]:
            parts = [folder / f"parts-{index}" / f"{start:012d}.part" for start in range(0, SIZES[index], CHUNK)]
            temporary = name.with_suffix(".tmp")
            with temporary.open("wb") as stream:
                for part in parts:
                    stream.write(part.read_bytes())
            assert temporary.stat().st_size == SIZES[index]
            temporary.replace(name)
            for part in parts:
                part.unlink()
        yield name


def project(path, folder):
    out = folder / (path.stem + "_labels.npz")
    source = pq.ParquetFile(path)
    ids, width, height, bits = [], [], [], []
    for batch in source.iter_batches(batch_size=4096,
                                    columns=["image_info.id", "image_info.width", "image_info.height",
                                             "segments_info.image_id", "segments_info.segments_info"]):
        for record in batch.to_pylist():
            img, annotation = record["image_info"], record["segments_info"]
            assert img["id"] == annotation["image_id"]
            category_ids = {s["category_id"] for s in annotation["segments_info"] if s["isthing"] == 1}
            assert all(1 <= category <= 90 for category in category_ids)
            mask = np.zeros(91, dtype=np.uint8)
            mask[list(category_ids)] = 1
            ids.append(img["id"])
            width.append(img["width"])
            height.append(img["height"])
            bits.append(np.packbits(mask))
    assert len(ids) == len(set(ids)) == source.metadata.num_rows
    np.savez_compressed(out, ids=ids, width=width, height=height, categories=bits)
    print(path.name, len(ids), out.stat().st_size, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("shards", type=int, nargs="*", default=list(range(4)))
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    for path in shards(args.directory, args.shards):
        project(path, args.directory)


if __name__ == "__main__":
    main()
