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

"""Compute FIT-only CLIP cosine difficulty for the 53 native stuff names."""
from argparse import ArgumentParser
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch


def import_file(name,path):
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(weights, visual, official, screen, output, vendor):
    torch.set_num_threads(1)
    meta = json.loads(screen.read_text(encoding="utf-8"))
    ids = np.load(official,allow_pickle=False)["fit_ids"]
    ordered = meta["official_stuff_categories"]
    names = [x["name"].replace("-stuff","").replace("-merged","").replace("-"," ") for x in ordered]
    prompts = [f"a photo of {n}." for n in names]
    with np.load(visual,allow_pickle=False) as data:
        rows = {int(i): j for j,i in enumerate(data["ids"])}
        assert len(rows) == len(ids) == 4008 and set(rows) == set(ids.tolist())
        features = data["features"][[rows[i] for i in ids]].astype(np.float64)
    assert features.shape == (4008,1024) and np.isfinite(features).all()
    tokenizer = import_file("clip_simple_tokenizer",vendor/"simple_tokenizer.py").SimpleTokenizer()
    transformer_type = import_file("clip_model",vendor/"model.py").Transformer
    checkpoint = torch.jit.load(str(weights),map_location="cpu").eval()
    state = checkpoint.state_dict()
    scale = state["ln_final.weight"].float()
    bias = state["ln_final.bias"].float()
    width = len(scale)
    length = state["positional_embedding"].shape[0]
    attention = torch.full((length,length),float("-inf")).triu_(1)
    layers = len({k.split(".")[2] for k in state if k.startswith("transformer.resblocks.")})
    model = transformer_type(width,layers,width//64,attention).eval()
    model.load_state_dict({k.removeprefix("transformer."):v for k,v in state.items() if k.startswith("transformer.")})
    needed = {k:state[k].float() for k in ("token_embedding.weight","positional_embedding","text_projection")}
    del state
    del checkpoint
    ids_by_prompt = [[tokenizer.encoder["<|startoftext|>"],*tokenizer.encode(s),tokenizer.encoder["<|endoftext|>"]] for s in prompts]
    sequence = torch.zeros((len(prompts),length),dtype=torch.long)
    for i,seq in enumerate(ids_by_prompt):
        sequence[i,:len(seq)] = torch.as_tensor(seq)
    with torch.inference_mode():
        vector = needed["token_embedding.weight"][sequence] + needed["positional_embedding"]
        vector = model(vector.transpose(0,1)).transpose(0,1)
        vector = torch.nn.functional.layer_norm(vector,(width,),scale,bias)
        text = (vector[torch.arange(len(sequence)),sequence.argmax(dim=-1)] @ needed["text_projection"]).numpy().astype(np.float64)
    x = features / np.linalg.norm(features,axis=1,keepdims=True)
    y = text / np.linalg.norm(text,axis=1,keepdims=True)
    score = x @ y.T
    assert score.shape == (len(ids),53) and np.isfinite(score).all() and np.max(np.abs(score)) <= 1
    np.testing.assert_allclose(score[:15],np.einsum("ik,jk->ij",x[:15],y),rtol=1.e-9,atol=1.e-10)
    np.savez_compressed(output/"PANOPTIC_FIT_DIFFICULTY.npz", ids=ids, categories=[x["category_id"] for x in ordered],
                        prompts=prompts, scores=score)
    result = dict(status="PASS", fit_images=len(ids), text_classes=len(names), encoder="CLIP RN50",
                  frozen_t3_visual_cache=str(visual), image_transform="fixed RGB 448 from T3", confirmation_scores_read=0)
    (output/"PANOPTIC_DIFFICULTY_QA.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(result,flush=True)


if __name__ == "__main__":
    parser=ArgumentParser()
    for argument in ("weights","visual","official","screen","output","vendor"):
        parser.add_argument(argument,type=Path)
    args=parser.parse_args()
    main(**vars(args))
