<div align="center">

<h1>GeCo: A Differentiable Geometric Consistency Metric for Video Generation</h1>

<a href="https://geco-geoconsistency.github.io/static/paper.pdf" target="_blank" rel="noopener noreferrer">
  <img src="https://img.shields.io/badge/paper-blue" alt="Paper PDF">
</a>
<a href="https://arxiv.org/abs/2512.22274">
  <img src="https://img.shields.io/badge/arXiv-2512.22274-b31b1b" alt="arXiv">
</a>
<a href="https://geco-geoconsistency.github.io/">
  <img src="https://img.shields.io/badge/Project-Page-green" alt="Project Page">
</a>

<br>

**[Harvard University](https://www.harvard.edu/)** &nbsp;|&nbsp; **[Google DeepMind](https://deepmind.google/)** &nbsp;|&nbsp; **[Massachusetts Institute of Technology](https://www.mit.edu/)**

[Leslie Gu](https://shixuan-gu.me/), [Junhwa Hur](https://hurjunhwa.github.io/), [Charles Herrmann](https://scholar.google.com/citations?user=LQvi5XAAAAAJ), [Fangneng Zhan](https://fnzhan.com/), [Todd Zickler](https://zickler.seas.harvard.edu/pi-bio/), [Deqing Sun](https://deqings.github.io/), [Hanspeter Pfister](https://vcg.seas.harvard.edu/people)

**ECCV 2026**

</div>

GeCo measures the *geometric consistency* of a video by comparing two independent
views of its 3D structure: the **motion** predicted by a dense correspondence model
and the **camera-induced motion** implied by a multi-view geometry estimator (camera
+ depth).  Pixels where the two disagree reveal deformations, "breathing" scenes,
hallucinated occlusions and other geometric artefacts of generated videos.

The metric itself is model-agnostic.  This release ships **pluggable backends**:

| Role | Backend (`--geometry` / `--flow`) | Model | Weights | Notes |
|---|---|---|---|---|
| geometry | `vggt` *(default, used in the paper)* | [VGGT-1B](https://github.com/facebookresearch/vggt) | `facebook/VGGT-1B` (auto-download) | 518 px long side, patch 14 |
| geometry | `vggt_omega` | [VGGT-&Omega;](https://github.com/facebookresearch/vggt-omega) | `facebook/VGGT-Omega` (**gated**, request access) | 512 px "balanced", patch 16; better geometry |
| flow | `ufm` *(default, used in the paper)* | [UFM](https://uniflowmatch.github.io/) | `infinity1096/UFM-Base` (auto-download) | native covisibility head |
| flow | `sea_raft` | [SEA-RAFT](https://github.com/princeton-vl/SEA-RAFT) (Spring-M) | `MemorySlices/Tartan-C-T-TSKH-spring540x960-M` (auto-download) | covisibility from forward/backward consistency |
| flow | `raft` | [RAFT](https://pytorch.org/vision/stable/models/raft.html) (torchvision) | `C_T_SKHT_V2` (auto-download) | zero extra dependencies |
| flow | `ptlflow` *(optional)* | any of the [PTLFlow](https://github.com/hmorimitsu/ptlflow) zoo (DPFlow, MemFlow, ...) | named checkpoints | `pip install 'ptlflow>=0.4'` |

All backends produce the same tensors (see [Backend interface](#backend-interface)),
so `demo_detection.py` and `GeCo-Eval_evaluation.py` work with any combination.

## Installation

**Tested environment:** Linux, Python 3.11, PyTorch 2.8.0 + CUDA 12.8, NVIDIA A100 80 GB
(also runs on H100/H200; any GPU with >= 16 GB works for the demo).

```bash
# 1. Clone (the third-party model code is vendored under external/, no submodules)
git clone https://github.com/ShixuanGu/GeCo.git
cd GeCo

# 2. Environment
conda create -n geco python=3.11 -y
conda activate geco

# 3. PyTorch (pick the CUDA build matching your driver; cu128 is what we tested)
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# 4. UFM (flow backend used in the paper) and its UniCeption dependency
pip install -e external/UFM/UniCeption
pip install -e external/UFM

# 5. GeCo requirements (VGGT, VGGT-Omega and SEA-RAFT are pure-Python and vendored; nothing to build)
pip install -r requirements.txt

# 6. (optional) PTLFlow model zoo as an extra flow backend
pip install -r requirements_optional.txt

# 7. (optional) requirements for the test-time guidance experiment (CogVideoX);
#    the experiment was developed with diffusers 0.35 / transformers 4.56 -- pin them if newer releases break the pipeline
pip install -r requirements_guidance.txt
```

### Model weights

Every backend except `vggt_omega` downloads its weights automatically on first use --
the repositories are public, so **no Hugging Face account, access token or approval is
needed** for `vggt`, `ufm`, `sea_raft`, `raft` and `ptlflow`:

| Backend | How weights are obtained | Account / key |
|---|---|---|
| `vggt` | [`facebook/VGGT-1B`](https://huggingface.co/facebook/VGGT-1B) from the Hugging Face hub (cache: `$HF_HOME`, default `~/.cache/huggingface`) | none |
| `ufm` | [`infinity1096/UFM-Base`](https://huggingface.co/infinity1096/UFM-Base) (or `UFM-Refine`) from the hub; the DINOv2-L encoder comes through `torch.hub` (cache: `$TORCH_HOME`) | none |
| `sea_raft` | [`MemorySlices/Tartan-C-T-TSKH-spring540x960-M`](https://huggingface.co/MemorySlices/Tartan-C-T-TSKH-spring540x960-M) from the hub | none |
| `raft` | torchvision (cache: `$TORCH_HOME`, default `~/.cache/torch`) | none |
| `ptlflow` | PTLFlow's named checkpoints (cache: `$TORCH_HOME`) | none |
| `vggt_omega` | **gated** -- manual approval required, see below | HF account + access request + token |

**`vggt_omega` (approval required).**  The
[`facebook/VGGT-Omega`](https://huggingface.co/facebook/VGGT-Omega) weights are gated with
manual review, so plan for a delay between requesting and receiving access:

1. Sign in to (or create) a Hugging Face account and submit the access request on the
   [model page](https://huggingface.co/facebook/VGGT-Omega); wait for the approval e-mail.
2. Once approved, authenticate the machine that will run GeCo: create a **read** token at
   https://huggingface.co/settings/tokens and run `hf auth login` (or export it as
   `HF_TOKEN`).  `--geometry vggt_omega` then auto-downloads `vggt_omega_1b_512.pt`.
3. Alternatively, download `vggt_omega_1b_512.pt` from the model page in your browser and
   pass it explicitly: `--geometry_ckpt /path/to/vggt_omega_1b_512.pt` (no token needed at
   run time).

Until access is granted, everything else works: simply leave `vggt_omega` out of
`--geometry`.

On clusters whose compute nodes have no internet access, run each backend once
on a login node (or `python scripts/check_backends.py --device cpu --num_frames 2`
is enough to trigger the downloads), then set `HF_HUB_OFFLINE=1` in your jobs.

### Verify the installation

```bash
python scripts/check_backends.py --geometry vggt,vggt_omega --flow ufm,sea_raft,raft --report reports/check.json
```

The checker runs every requested backend on `examples/deform_house`, validates the
output contract (shapes, dtypes, valid masks, camera conventions), cross-feeds the
outputs into the metric and prints timing / peak-memory numbers.  `vggt_omega` is
expected to fail here until your access request for the gated weights is approved
(the checker records the failure and continues; drop it from `--geometry` to verify
the non-gated backends only).  A SLURM template for FASRC-style clusters is in
`scripts/slurm/check_backends_a100.sh`.

## Detecting Deformation on a Single Video
Run GeCo on a directory of frames to generate the per-frame consistency maps
(Motion Map, Structure Map and Fused Map).  Everything for one video is written to
`results/<video name>/` (change the parent with `--outdir`): one consistency-map PNG
per frame plus the rendered video, e.g.

```
results/deform_house/
├── deform_house.mp4      rendered video (or .gif / both, see --save_mode)
├── grid_0000.png         per-frame maps: source | motion | structure | fused
├── grid_0001.png
└── ...
```

```bash
# paper setting (VGGT + UFM) -> results/deform_house/
python demo_detection.py --frame_path examples/deform_house

# better geometry + a different flow model -> results/omega_searaft/deform_house/
python demo_detection.py --frame_path examples/deform_house --outdir results/omega_searaft \
  --geometry vggt_omega --flow sea_raft

# any PTLFlow model, e.g. DPFlow trained on Sintel -> results/dpflow/deform_house/
python demo_detection.py --frame_path examples/deform_house --outdir results/dpflow \
  --flow ptlflow --flow_model dpflow --flow_ckpt sintel
```

Backend options (shared by `demo_detection.py` and `GeCo-Eval_evaluation.py`):

| Flag | Meaning |
|---|---|
| `--geometry {vggt,vggt_omega}` | geometry estimator (default `vggt`) |
| `--geometry_ckpt PATH_OR_HF_ID` | checkpoint override (default: the official weights) |
| `--flow {ufm,sea_raft,raft,ptlflow}` | flow / correspondence estimator (default `ufm`) |
| `--flow_model NAME` | backend-specific variant: UFM `base`/`refine`, SEA-RAFT config (`spring-M`, `sintel-M`, ...), torchvision `large`/`small`, PTLFlow model name |
| `--flow_ckpt PATH_OR_NAME` | checkpoint override (local file, HF id, torchvision weight enum, or PTLFlow checkpoint name such as `sintel`) |
| `--flow_max_side N` | run the flow model on frames downscaled to a long side of `N` px (flow is rescaled back); `0` = no cap; default: UFM = native, RAFT-family = 1024 |
| `--device DEV` | torch device for both backends (default: `cuda` if available, else `cpu`) |

Metric options: `--window_size` / `--window_anchor` (temporal context), `--covis_thresh`
(covisibility threshold, default 0.5), `--conf_percentile` / `--conf_min` (depth-confidence
mask), `--tau_z` (relative depth margin used to detect wrong occlusions).  Each script's
output and windowing flags are listed by `--help`.

## GeCo-Eval Benchmark
To reproduce the benchmark results, ensure the data directory follows the structure below.

```Plaintext
GeCo-Eval
├── Gen_Veo3.1
│   ├── indoor_prompts
│   │   ├── b1_0
│   │   │   ├── frame_000001.png
│   │   │   ├── frame_000002.png
│   │   │   └── ...
│   │   ├── b1_1
│   │   └── ...
│   ├── object_centric_prompts
│   ├── outdoor_prompts
│   └── stress_test_prompts
├── Gen_SORA2
│   └── ...
├── ...
```
### Run Evaluation
This script calculates the aggregate GeCo score for a specific model across all categories.

```bash
python GeCo-Eval_evaluation.py \
  --frames_root path/to/GeCo-Eval/frames \
  --models Gen_Veo3.1 Gen_SORA2            # add e.g. --geometry vggt_omega --flow sea_raft
```

Per-clip scores are written to `<model_key>.csv` (`--csv_dir` to change the location)
and the per-category means are printed as `# SUMMARY,...` lines.  The numbers in the
paper use the default backends (`vggt` + `ufm`).  To benchmark a model that is not in
the built-in FPS table, pass `--default_fps <native fps of your videos>`.

## Test Time Guidance Experiment
Run the following command to generate videos with and without guidance:
```bash
python demo_guidance.py \
  --only both \
  --model-path THUDM/CogVideoX-5b \
  --loss-fn residual_motion \
  --fixed-frames "12,24,36,48" \
  --prompt "A steady 360° orbit around a detailed globe on a stand in a book-lined study."
```
The guidance experiment is kept as in the paper (VGGT + UFM, which are differentiable
w.r.t. the frames); the backend flags above apply to the detection and evaluation scripts.

## Backend interface

Both backend families are small Python classes registered in `geco/geometry/__init__.py`
and `geco/flow/__init__.py`; the metric only touches their outputs.

**Geometry** (`geco.geometry.GeometryBackend.run(frame_paths) -> GeometryOutput`), for
`F` frames on the output grid `(H, W)` (the frames' native resolution by default):

| Field | Shape / dtype | Convention |
|---|---|---|
| `intrinsic` | `(F, 3, 3)` float32 | pinhole `[[fx,0,cx],[0,fy,cy],[0,0,1]]` in pixels of the `(H, W)` grid |
| `extrinsic` | `(F, 3, 4)` float32 | world-to-camera `[R\|t]`, OpenCV axes (x right, y down, z forward) |
| `depth` | `(F, H, W, 1)` float32 | z-depth (> 0 where valid), scale is backend dependent |
| `conf` | `(F, H, W, 1)` float32 | higher = better; VGGT family emits `1 + exp(logit)`, so `--conf_min` is compared to that raw value |
| `valid` | `(F, H, W)` bool | finite depth > 0, finite confidence, inside the region the network saw |

Each backend resizes the frames to its native resolution (VGGT: long side 518, multiple of 14;
VGGT-&Omega;: token-balanced 512, multiple of 16), runs once on the whole window, bilinearly
upsamples depth/confidence to `(H, W)` and rescales the intrinsics per axis
(`geco.geometry.base.map_to_output_grid`).  If a backend has to crop (VGGT-&Omega; crops
aspect ratios outside `[0.5, 2]`), the predictions are pasted back into the full grid and
`valid` is `False` outside the crop.

On landscape frames the default `vggt` backend reproduces the paper pipeline bit-for-bit
(`tests/test_vggt_legacy_equivalence.py`).  Portrait frames are the one deliberate change:
the paper-era loader centre-cropped them to 518&times;518 and stretched the predictions over
the uncropped frame, whereas this release resizes them aspect-preserving, so scores on
portrait clips can differ slightly from (and are more trustworthy than) paper-era runs.

**Flow** (`geco.flow.FlowBackend.predict(src_rgb, tgt_rgb) -> FlowOutput`):

| Field | Shape / dtype | Convention |
|---|---|---|
| `flow` | `(H, W, 2)` float32 | pixels of the source grid, `(dx, dy)` such that `(u, v) -> (u+dx, v+dy)` |
| `covisibility` | `(H, W)` float32 in `[0, 1]` | 1 = visible in the target, 0 = occluded / out of view; thresholded by `--covis_thresh` |
| `valid` | `(H, W)` bool | finite flow and covisibility |

UFM predicts covisibility natively.  For plain flow networks the backend runs the flow in
both directions and converts the forward/backward cycle error into a soft score such that
`covisibility > 0.5` is exactly the classic occlusion test of Sundaram et al.
(`|f + b(x+f)|^2 < 0.01 (|f|^2 + |b|^2) + 0.5`), see `geco/flow/consistency.py`.

**Adding a backend:** subclass `GeometryBackend` (implement `load_images`, `preprocess`,
`infer`) or `FlowBackend` (implement `predict`), call `.validate()` on the output, and
register it with `register_geometry_backend(name, module, cls)` /
`register_flow_backend(...)` (or add it to the `_REGISTRY` dict).  `scripts/check_backends.py`
tells you immediately whether the contract is satisfied.

## Repository layout

```
demo_detection.py          per-video consistency maps (mp4/gif/png)
GeCo-Eval_evaluation.py    GeCo-Eval benchmark driver
demo_guidance.py           test-time guidance of CogVideoX with the GeCo loss (paper setting: VGGT + UFM)
utils.py                   geometry math (ego-flow, depth re-projection), rendering
geco/                      backend registry + metric
  geometry/{base,vggt,vggt_omega}.py
  flow/{base,consistency,ufm,sea_raft,raft_torchvision,ptlflow_backend}.py
  metric.py, cli.py, image_io.py
external/                  vendored third-party code (see each LICENSE / UPSTREAM.txt)
  vggt/  vggt_omega/  UFM/  sea_raft/  guidance_cogvideox/
scripts/check_backends.py  backend contract checker
scripts/slurm/*.sh         SLURM templates used for our A100 tests
tests/                     contract + legacy-equivalence tests (pytest, or run each file directly)
examples/deform_house/     example frames (deforming house)
```

## License
The GeCo code in this repository is released under [CC BY-NC 4.0](LICENSE) for research
purposes.  The vendored third-party components keep their own licenses (see the LICENSE /
UPSTREAM.txt inside each directory): VGGT (CC BY-NC 4.0), VGGT-&Omega;
(FAIR Noncommercial Research License, weights gated on Hugging Face), UFM (BSD-3),
SEA-RAFT (BSD-3); the CogVideoX guidance pipeline is adapted from
[diffusers](https://github.com/huggingface/diffusers) (Apache-2.0) and
[Frame Guidance](https://github.com/agwmon/frame-guidance).
Please check them before any commercial use.

## Citation
If you find this code useful for your research, please cite our paper:

```bibtex
@article{gu2025geco,
  title={GeCo: A Differentiable Geometric Consistency Metric for Video Generation}, 
  author={Gu, Leslie and Hur, Junhwa and Herrmann, Charles and Zhan, Fangneng and Zickler, Todd and Sun, Deqing and Pfister, Hanspeter},
  journal={arXiv preprint arXiv:2512.22274},
  year={2025}
}
```

## Acknowledgements
[Claude](https://claude.com/claude-code) was used for organizing the environment
packages of this release.
