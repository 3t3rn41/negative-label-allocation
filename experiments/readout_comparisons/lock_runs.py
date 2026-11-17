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

"""Create local frozen configs for RN50, the ViT replication, and LIN extraction."""
import json
from pathlib import Path
from datetime import datetime,timedelta,timezone
import numpy as np
here=Path(__file__).resolve().parent
v9=here.parent
s0=v9/"coco_fixed_partner"
ab=v9/"coco_crossed"
design=json.loads((here/"COCO_DESIGN.json").read_text(encoding="utf8"))
b=json.loads((ab/"B_PROTOCOL_LOCK.json").read_text(encoding="utf8"))
a1=json.loads((s0/"PROTOCOL_LOCK.json").read_text(encoding="utf8"))
assert design["status"]=="MASKS_QUALIFIED" and design["draws"]==b["draws"] and design["seeds"]==b["seeds"]
assert len(set(b["b_ids"])&set(a1["confirmation_ids"]))==0
base=dict(status="LOCKED",locked_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
          seeds=design["seeds"],draws=design["draws"],epochs=50,steps_per_endpoint=6300,
          lanes=3,b_ids=b["b_ids"],a1_ids=a1["confirmation_ids"])
specs={
 "COCO":dict(backend="RN50",arms=[*design["arms"],"T2_PBCE"],score_original_low=True,worker_time_limit_seconds=90000,
             partial_label_baseline="DualCoOp with archived observed-only BCE on the original LOW mask. Unknown labels excluded. Pair strategy keeps original v9 ASL."),
 "T5":dict(backend="ViT-B/16",arms=["S0_TRUE","N1_TRUE","S0_PLC"],score_original_low=False,worker_time_limit_seconds=36000,
           purpose="Use original COCO B masks, roles, classes and doses with the existing 224px CLIP ViT-B/16 adapter. Compare to COCO B/RN50 and VOC/ViT."),
 "T3":dict(backend="RN50",arms=["LOW","S0_TRUE","S0_PLC"],score_original_low=False,worker_time_limit_seconds=5400,
           feature_protocol="Freeze global CLIP RN50 features, 448px same deterministic evaluation transform for fit and A1. L2-normalize. Paired 12 zero-bias normally initialized linear heads per LOW/TRUE/PLC, seed matched across arms; known-only mean BCE, SGD lr 0.1 cosine to 0, momentum 0.9, 50 epochs, original 32-image shuffled minibatches. No evaluation labels used in fitting.")
}
for key,details in specs.items():
    out=here/key
    out.mkdir(exist_ok=True)
    data={**base,"job":key,**details}
    (out/"RUN_PROTOCOL.json").write_text(json.dumps(data,indent=2),encoding="utf8")
    if key=="T3":
        continue
    for d in base["draws"]:
        path=here/f"COCO_MASK_{d}.npz" if key=="COCO" else v9/"coco_crossed"/f"MASK_{d}.npz"
        with np.load(path) as z:
            arrays={k:z[k] for k in ('ids','classes','LOW',*[a for a in data['arms'] if a!='T2_PBCE'])}
            if key=="COCO":
                arrays["T2_PBCE"]=z["LOW"]
        np.savez_compressed(out/f"MASK_{d}.npz",**arrays)
    if key=="T5":
        rules=json.loads((v9/"voc_crossed/RULES.json").read_text(encoding="utf8"))
        rules.update(draws=base["draws"],initializations=base["seeds"],arms=data["arms"])
        # Class names stay identical to the COCO/RN50 prompts.
        rules["text_class_aliases"]={}
        (out/"RULES.json").write_text(json.dumps(rules,indent=2),encoding="utf8")

