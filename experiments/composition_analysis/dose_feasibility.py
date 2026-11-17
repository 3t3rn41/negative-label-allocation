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

"""Probe T7 mask feasibility with existing v9.2.2 0–1 mask tooling."""

import importlib.util
import sys

import numpy as np

from replay import HERE, P0, VOC, read, save, table


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    config = read(HERE/"T7_FEASIBILITY_PROTOCOL.json")
    # Load COCO source by explicit path. The VOC solver imports its own build_masks.
    coco = load_module("coco_build_masks",P0/"build_masks.py")
    solver = load_module("voc_solve_masks",VOC/"solve_masks.py")
    ids, labels, classes, cosine = coco.load_training()
    truth = (2*labels-1).astype(np.int8)
    pairs = [(a["target"],a["partner"],b["partner"])
             for a,b in zip(coco.PAIR_MAP["S0"],coco.PAIR_MAP["N1"],strict=True)]
    rules = read(VOC/"RULES.json")
    assert config["smd_max"]==rules["max_abs_smd"] and config["minimum_direction_gap"]==rules["minimum_directional_gap"]
    rows, details = [], []
    for draw in coco.RULES["draw_seeds"]:
        with np.load(P0/f"MASK_{draw}.npz") as data:
            np.testing.assert_array_equal(ids,data["ids"])
            np.testing.assert_array_equal(classes,data["classes"])
            low = data["LOW"]
        strata = list(solver.stratify(ids,truth,cosine,low,pairs))
        capacity = [sum(min(map(len,row[2])) for row in strata_j) for strata_j in strata]
        for dose_index,doses in enumerate(config["doses_chair_diningtable_pottedplant"]):
            quotas = [solver.largest_remainder(s,d) for s,d in zip(strata,doses,strict=True)]
            receipt = dict(draw=draw,dose=doses,capacity=capacity,attempts=[])
            status = "CAPACITY_INSUFFICIENT"
            for offset in rules["plc_rank_offsets"] if all(q is not None for q in quotas) else []:
                tie,rng = map(np.random.default_rng,np.random.SeedSequence([draw,922,dose_index]).spawn(2))
                masks = {arm:low.copy() for arm in solver.ARMS}
                groups, statuses = {}, {}
                for (t,a,b),dose,layers,quota in zip(pairs,doses,strata,quotas,strict=True):
                    plc = solver.neutral_placebo(layers,quota,cosine[:,t],ids,tie,offset)
                    selected,message = solver.directed_coordinates(layers,quota,dose,t,a,b,truth,cosine,plc,rng,rules)
                    statuses[str(classes[t])] = message
                    if selected is None:
                        continue
                    groups[t] = selected
                    for _,_,(remove,*add) in selected:
                        for matrix,members in zip(masks.values(),add,strict=True):
                            matrix[remove,t] = -1
                            matrix[members,t] = 0
                if len(groups)==len(pairs):
                    failures,metrics,density = solver.audit(truth,cosine,low,pairs,doses,masks,groups)
                    assert all(x["density_tv"]<=config["density_tv_max"] or failures for x in density)
                else:
                    failures,metrics,density = ["NO_FEASIBLE_COORDINATES"],[],[]
                receipt["attempts"].append(dict(offset=offset,statuses=statuses,failures=failures,metrics=metrics,density=density))
                status = "PASS" if not failures else "NO_PASSING_MASK"
                if status=="PASS":
                    np.savez_compressed(HERE/f"T7_dose{dose_index+1}_mask{draw}.npz",ids=ids,classes=classes,
                                        LOW=low,S0_TRUE=masks["P1_TRUE"],N1_TRUE=masks["P2_TRUE"],PLC=masks["PLC"])
                    break
            details.append(receipt)
            row = dict(draw=draw,dose="/".join(map(str,doses)),capacity="/".join(map(str,capacity)),
                       status=status,attempts=len(receipt["attempts"]))
            rows.append(row)
            print(row,flush=True)
    table("T7_mask_feasibility.csv",rows)
    save("T7_mask_feasibility.json",details)


if __name__=="__main__":
    main()
