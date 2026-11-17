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

"""Extract two public panoptic JSON members without downloading the image archives."""
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import struct
from time import sleep
import zlib

import requests


URL = "https://s3.amazonaws.com/images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip"
TOTAL = 860725834


def segment(start, size):
    end = start+size-1
    for attempt in range(4):
        try:
            with requests.get(URL, params={"offset":start,"attempt":attempt},
                              headers={"Range":f"bytes={start}-{end}","Accept-Encoding":"identity"},
                              timeout=(12,40)) as response:
                assert response.status_code == 206 and response.headers.get("Content-Range") == f"bytes {start}-{end}/{TOTAL}"
                data = response.content
                assert len(data) == size
                return data
        except (AssertionError, requests.RequestException):
            if attempt == 3:
                raise
            sleep(attempt+1)


def members():
    start = TOTAL-65536
    tail = segment(start,65536)
    pointer = tail.rfind(b"PK\x05\x06")
    _,_,_,count,total,directory_size,offset,_ = struct.unpack_from("<4s4H2IH",tail,pointer)
    assert count == total == 11
    metadata = tail[offset-start:offset-start+directory_size]
    pointer = 0
    entries = {}
    while pointer < len(metadata):
        row = struct.unpack_from("<4s6H3I5H2I",metadata,pointer)
        assert row[0] == b"PK\x01\x02"
        name_size, extra, comment = row[10:13]
        name = metadata[pointer+46:pointer+46+name_size].decode()
        entries[name] = (row[4],row[8],row[9],row[-1])
        pointer += 46+name_size+extra+comment
    return entries


def download(directory):
    entries = members()
    for split in ("train", "val"):
        name = f"panoptic_{split}2017.json"
        target = directory / name
        method, compressed, size, begin = entries[f"annotations/{name}"]
        if target.is_file() and target.stat().st_size == size:
            continue
        header = segment(begin,30)
        magic,*_,name_size,extra = struct.unpack("<4s5H3I2H",header)
        assert magic == b"PK\x03\x04" and method == 8
        start = begin+30+name_size+extra
        chunks = [(start+offset,min(4*1024*1024,compressed-offset)) for offset in range(0,compressed,4*1024*1024)]
        prefix = target.with_suffix(".partial")
        decoder = zlib.decompressobj(-15)
        with ThreadPoolExecutor(max_workers=4) as pool, prefix.open("wb") as output:
            for chunk in pool.map(lambda pair: segment(*pair),chunks):
                output.write(decoder.decompress(chunk))
            output.write(decoder.flush())
        assert decoder.eof and prefix.stat().st_size == size
        prefix.replace(target)
        print(name,size,flush=True)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("directory", type=Path)
    folder = parser.parse_args().directory
    folder.mkdir(parents=True,exist_ok=True)
    download(folder)
