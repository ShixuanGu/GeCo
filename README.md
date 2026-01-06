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

</div>


## Installation
Clone the repository and set up the environment. This implementation relies on [UFM](https://uniflowmatch.github.io/) for motion prediction and [VGGT](https://vgg-t.github.io/) for geometry estimation.

**Tested Environment:** Python 3.11, PyTorch 2.x + CUDA 12.8.

```bash
# Clone repository
git clone https://github.com/ShixuanGu/GeCo.git
cd GeCo

# Create environment
conda create -n geco python=3.11 -y
conda activate geco

# Install UFM dependencies
cd external/UFM/UniCeption
pip install -e .
cd ..
pip install -e .

# Verify UFM installation (Optional)
# This generates `ufm_output.png` which should match `examples/example_ufm_output.png`
python uniflowmatch/models/ufm.py

# Install VGGT & GeCo requirements
cd ../..
pip install -r requirements.txt

# Install requirements for test time guidance experiment
pip install -r requirements_guidance.txt
```

## Detecting Deformation on a Single Video
Run GeCo on a single video to generate consistency maps (i.e., Motion Map, Structure Map, and Fused Map).

```bash
python demo_detection.py \
  --frame_path examples/deform_house \
  --outdir examples/results
```

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

```python
python GeCo-Eval_evaluation.py \
  --frames_root path/to/GeCo-Eval/frames \
  --models Gen_Veo3.1,Gen_SORA2
```

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

## Checklist
- [ ] Organize pair-wise demo code

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