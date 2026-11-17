# Negative Label Allocation

Reproduction code for **Where Negative Labels Matter in Partially Annotated Multi-Label Recognition**.

The experiments hold the annotation budget fixed and vary which verified negative labels are visible during training. They cross training-partner assignments with evaluation-partner assignments to measure how a change in supervision affects target ranking. The analysis also reweights groups of positive and negative examples to study how evaluation composition changes the overall AUC effect.

## Repository layout

| Path | Purpose |
| --- | --- |
| [`experiments/coco_design/`](experiments/coco_design), [`experiments/coco_fixed_partner/`](experiments/coco_fixed_partner), [`experiments/coco_crossed/`](experiments/coco_crossed) | COCO sampling and masks, fixed-partner training, crossed-partner training and evaluation |
| [`experiments/voc_design/`](experiments/voc_design), [`experiments/voc_training/`](experiments/voc_training), [`experiments/voc_crossed/`](experiments/voc_crossed) | VOC2007 partner qualification, mask construction, training and evaluation |
| [`experiments/coconut_inputs/`](experiments/coconut_inputs), [`experiments/coconut_panoptic/`](experiments/coconut_panoptic) | COCONut-B input preparation, panoptic sampling, DualCoOp and linear readouts, and composition prediction |
| [`experiments/composition_analysis/`](experiments/composition_analysis), [`experiments/readout_comparisons/`](experiments/readout_comparisons) | AUC decomposition, sensitivity analyses and readout comparisons |
| [`metrics/`](metrics), [`runtime/`](runtime), [`figures/`](figures) | Shared metrics, model and image adapters, and paper figure generation |

## Environment and inputs

Use Linux and Python 3.12 with a CUDA-enabled installation of PyTorch and torchvision for prompt training. The COCONut-B linear fit and the statistical analyses run on CPU. Install NumPy, SciPy, Pillow, Requests, ftfy, regex, tqdm and RandAugment for the model and data scripts. COCONut-B data preparation uses PyArrow and ijson. The legacy resampling and figure scripts also use Numba and Matplotlib.

Clone the upstream DualCoOp implementation under the runtime directory. Place the CLIP RN50 weights at `$FQ_RECOVERY_ROOT/r11/assets/RN50.pt` and place the ViT-B/16 weights beside the training worker with the filename specified by its `RULES.json`. Use the original sources for the models. [DualCoOp](https://github.com/sunxm2357/DualCoOp) · [CLIP](https://github.com/openai/CLIP)

```bash
export FQ_RECOVERY_ROOT="$PWD/runtime"
mkdir -p "$FQ_RECOVERY_ROOT/vendor" "$FQ_RECOVERY_ROOT/r11/assets"
git clone https://github.com/sunxm2357/DualCoOp.git "$FQ_RECOVERY_ROOT/vendor/DualCoOp"
git -C "$FQ_RECOVERY_ROOT/vendor/DualCoOp" checkout 49e71d75ad6f523aa02a3e763584a8b6b82eaa88
```

Set `FQ_COCO2017_ROOT` to the directory containing the COCO image ZIPs and `FQ_VOC_INPUTS` to the directory containing `VOCdevkit/VOC2007` and the experiment's `VOC_split.json`. Obtain the corresponding images and annotations from the dataset providers. [MS COCO](https://cocodataset.org/#download) · [PASCAL VOC2007](https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2007/) · [COCONut](https://github.com/bytedance/coconut_cvpr2024)

For each experiment, prepare its registered image roles, category order, masks and protocol inputs. Use the input readers in the corresponding scripts for the expected filenames and fields. Exact reruns require the original split and manifest definitions. To recompute metrics from saved predictions, supply the scores, labels, protocol files and any saved resampling indices to the relevant analysis script. Obtain datasets, weights and run artifacts separately.

## Run the analyses

The historical COCO and VOC2007 entry points read from the experiment directories shown above. Populate their inputs before calling these scripts from the repository root.

```bash
python experiments/coco_fixed_partner/analyze_s0.py
python experiments/coco_crossed/analyze_a.py
python experiments/coco_crossed/analyze_b.py
python experiments/voc_crossed/analyze.py
```

For COCONut-B, the formal pipeline takes four arguments in order. Provide the prepared run directory, the image-lineage manifest, the FIT pixel manifest and the FIT feature cache. The run directory must already contain the design lock, masks, and preflight and timing records consumed by the pipeline. The FIT pixel manifest is a CSV with `id`, `role`, `existing_subdirectory` and `file_name` columns. The training loader resolves the image paths under `$FQ_RECOVERY_ROOT/datasets/coco`. The FIT feature cache is an NPZ with `ids` and `features` arrays.

```bash
python experiments/coconut_panoptic/formal_pipeline.py   "$RUN_ROOT" "$LINEAGE_CSV" "$FIT_PIXEL_MANIFEST" "$FIT_FEATURE_CACHE"
```

The pipeline fits the two readouts, scores the bridge pool, saves the composition predictions, then scores and evaluates the confirmation sample. Run the following command to re-evaluate saved formal results after providing their inputs.

```bash
python experiments/coconut_panoptic/formal/formal_metrics.py "$RUN_ROOT/formal"
```

Once the result files exist in the corresponding experiment directories, regenerate the plots with `python figures/make_figures.py`.

## License

The experiment source is licensed under MIT, as stated in the source headers. Upstream code, model weights and datasets retain their respective terms.
