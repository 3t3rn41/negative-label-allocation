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

"""Fully decode each fixed confirmation image from the public COCO assets."""

import csv
import io
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import batched
from pathlib import Path
from urllib.request import urlopen
from zipfile import BadZipFile, ZipFile

from PIL import Image

ROOT = Path(sys.argv[1])
SOURCE = Path(os.environ.get("FQ_COCO2017_ROOT", Path(__file__).resolve().parents[2] / "inputs/coco2017")) / "train2017.zip"
SHARED = Path(os.environ.get("FQ_COCO2017_ROOT", Path(__file__).resolve().parents[2] / "inputs/coco2017")) / "train2017"
OFFICIAL_BUCKETS = ('https://s3.amazonaws.com/images.cocodataset.org/train2017/',
                    'https://images.cocodataset.org/train2017/')


def save_json(path, payload):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(payload), encoding='utf-8')
    tmp.replace(path)


def main():
    start = time.monotonic()
    with (ROOT / 'CONFIRM_PIXELS.csv').open(newline='', encoding='utf-8') as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == len({int(row['id']) for row in rows}) == 5000
    for row in rows:
        assert row['file_name'] == f"{int(row['id']):012d}.jpg"
    archive = ZipFile(SOURCE) if not SHARED.is_dir() else None
    if archive is not None:
        ordered = sorted(rows, key=lambda row: archive.getinfo('train2017/' + row['file_name']).header_offset)
    else:
        ordered = sorted(rows, key=lambda row: row['file_name'])
    payload = dict(source=str(SHARED if archive is None else SOURCE), expected=len(rows),
                   workers=min(8, os.cpu_count() or 1), status='RUNNING')
    save_json(ROOT / 'PIXEL_PROGRESS.json', payload)
    print(json.dumps(payload), flush=True)

    def decode(row, image_bytes, asset, member):
        name = row['file_name']
        with Image.open(io.BytesIO(image_bytes)) as image:
            expected = int(row['width']), int(row['height'])
            assert image.format == 'JPEG' and image.size == expected, (name, image.format, image.size, expected)
            image.load()
            image.convert('RGB').load()
        return dict(id=int(row['id']), width=expected[0], height=expected[1],
                    bytes=len(image_bytes), asset=asset, member=member)

    def read_archive(row):
        name = row['file_name']
        member = name if archive is None else 'train2017/' + name
        try:
            image_bytes = (SHARED / member).read_bytes() if archive is None else archive.read(member)
        except (OSError, BadZipFile) as exc:
            return row, member, None, str(exc)
        return row, member, image_bytes, None

    def verify(record):
        row, member, image_bytes, error = record
        if error:
            return row, None, error
        try:
            return row, decode(row, image_bytes, payload['source'], member), None
        except (OSError, AssertionError, ValueError) as exc:
            return row, None, str(exc)

    def retry(row):
        name = row['file_name']
        failures = []
        override = ROOT / 'official_fallback' / name
        if override.is_file():
            try:
                return row, decode(row, override.read_bytes(), OFFICIAL_BUCKETS[0], name), failures
            except (OSError, AssertionError, ValueError) as exc:
                failures.append(str(exc))
        for url in OFFICIAL_BUCKETS:
            try:
                with urlopen(url + name, timeout=8) as response:
                    image_bytes = response.read()
                return row, decode(row, image_bytes, url, name), failures
            except (OSError, AssertionError, ValueError) as exc:
                failures.append(str(exc))
        return row, None, failures

    checked = {}
    archive_errors = {}
    pending = []
    def write_manifest():
        manifest = ROOT / 'CONFIRM_PIXEL_BINDING.csv'
        if checked:
            with manifest.with_suffix('.tmp').open('w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=list(next(iter(checked.values()))))
                writer.writeheader()
                writer.writerows(checked[int(row['id'])] for row in rows if int(row['id']) in checked)
            manifest.with_suffix('.tmp').replace(manifest)

    try:
        with ThreadPoolExecutor(max_workers=payload['workers']) as workers:
            j = 0
            for batch in batched(ordered, 32):
                for row, item, error in workers.map(verify, [read_archive(row) for row in batch]):
                    j += 1
                    if item is not None:
                        checked[item['id']] = item
                    else:
                        archive_errors[row['id']] = error
                        pending.append(row)
                    if j % 500 == 0:
                        save_json(ROOT / 'PIXEL_PROGRESS.json', dict(payload, checked=j, full_decode_pass=len(checked)))
                        print('DECODE', j, 'PASS', len(checked), flush=True)
    finally:
        if archive is not None:
            archive.close()
    if pending:
        write_manifest()
        save_json(ROOT / 'ARCHIVE_ERRORS.json', archive_errors)
        print('OFFICIAL_FALLBACK', len(pending), flush=True)
    fallback_errors = {}
    with ThreadPoolExecutor(max_workers=8) as workers:
        for row, item, error in workers.map(retry, pending):
            if item is not None:
                checked[item['id']] = item
            else:
                fallback_errors[row['id']] = error
    write_manifest()
    summary = dict(payload, status='HOLD_PIXEL_DECODE' if fallback_errors else 'PASS',
                   full_decode_pass=len(checked), archive_errors=archive_errors,
                   fallback_errors=fallback_errors, fallback_sources=list(OFFICIAL_BUCKETS),
                   total_bytes=sum(row['bytes'] for row in checked.values()),
                   seconds=time.monotonic()-start, materialized_copy=False)
    save_json(ROOT / 'PIXEL_DECODE_QA.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
