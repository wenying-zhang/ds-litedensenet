# DS-LiteDenseNet

Evaluation code for *DS-LiteDenseNet: a parameter-frugal chest X-ray classifier
evaluated for non-inferiority, cross-hospital transfer, and faithfulness.*

DS-LiteDenseNet is a depthwise-separable DenseNet with 0.95 M parameters, about
one seventh of DenseNet-121. This repository contains the architecture and the
full evaluation protocol used to decide whether a model that small can be
trusted: patient-wise locked splits, five-seed reporting, a pre-specified
non-inferiority test, external validation on two independent hospitals per
pathology, quantified Grad-CAM localisation, and calibration.

Every number in the manuscript is produced by the scripts here. No dataset is
redistributed; each must be obtained from its original distributor under that
distributor's terms, and no trained weights are distributed either, because the
licences of those corpora do not clearly permit redistributing models derived
from them. Section 3 below regenerates the checkpoints from scratch.

**Repository:** <https://github.com/wenying-zhang/ds-litedensenet>
**Archive (v1.0.0):** <https://doi.org/10.5281/zenodo.22479137>
Cite the version DOI of the release you used, not the repository URL, if you
need a reference that is guaranteed to resolve to a fixed state. Zenodo also
mints a concept DOI that always resolves to the newest version; it is shown as
"Cite all versions" on the record page, and is the one to use if you want to
point at the project rather than at a fixed state.

---

## Contents

| Path | Purpose |
|------|---------|
| `config.py` | Paths, hyper-parameters, preprocessing switches |
| `data_manifests.py` | Builds the unified manifests and the patient-wise splits |
| `datasets.py` | Dataset class, image loading, CLAHE and normalisation |
| `models.py` | The four DenseNet variants and the two lightweight baselines |
| `engine.py` | Training loop, inference, metric computation |
| `stats.py` | DeLong, bootstrap CI, paired *t*-test, ECE, non-inferiority |
| `run_ablation.py` | Architecture ablation on a locked test split |
| `run_cross_source.py` | External validation across hospitals |
| `run_gradcam.py` | Quantified Grad-CAM localisation |
| `lung_seg.py` | Lung masks from the ChestX-Det PSPNet |
| `plot_reliability.py` | Reliability diagrams and per-model ECE |
| `export_reliability_curve.py` | Single reliability curve as vector output |
| `diagnose_tb_domain.py` | Input-level vs. pathology-level shift on the TB task |
| `geometric_reference.py` | Area references the Grad-CAM energy shares are read against |
| `gradcam_stats.py` | Significance tests for the localisation measures |
| `bench_inference.py` | Latency and activation cost of a forward pass |
| `eval_external.py` | Scores existing checkpoints on an external cohort |
| `operating_point.py` | Sensitivity-constrained thresholds and prior-shift calibration |
| `paper_figures/` | Scripts that draw the schematic and summary figures |

Detailed per-script documentation, including every argument and output file, is
in [`docs/scripts.md`](docs/scripts.md).

---

## 1. Installation

Python 3.10 or 3.11. A CUDA GPU is needed for the training scripts; the plotting
scripts run on CPU.

```bash
git clone https://github.com/wenying-zhang/ds-litedensenet.git
cd ds-litedensenet
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

Install PyTorch first, choosing the build that matches your CUDA version from
pytorch.org, then the rest:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

The CUDA suffix above is an example; use the one your driver supports. Nothing
here depends on a particular PyTorch version, and `requirements.txt` therefore
gives ranges rather than pins.

`thop` is listed in `requirements.txt` and so installs by default; it is only
used for the FLOP column, which is reported as `None` if the import fails.
`torchxrayvision` is needed only by `run_gradcam.py` and downloads about 260 MB
of segmenter weights on first use.

## 2. Data

Place the corpora under `data/` in the layout below, or edit `RAW_DATA` in
`config.py`.

```
data/
├── rsna/          stage_2_train_images/, stage_2_detailed_class_info.csv,
│                  stage_2_train_labels.csv
├── chexpert/      CheXpert-v1.0-small/{train,valid}.csv,
│                  CheXpert-v1.0-small/train/  (the image tree)
├── vindr/         train/, image_labels_train.csv
├── tb/            ChinaSet_AllFiles/CXR_png/, MontgomerySet/CXR_png/,
│                  MontgomerySet/ManualMask/{leftMask,rightMask}/
└── COVIDGR/       P/, N/
```

| Corpus | Role | Access |
|--------|------|--------|
| RSNA Pneumonia Detection Challenge 2018 | pneumonia, internal | [Kaggle competition data](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data) |
| CheXpert v1.0 small | pneumonia, external | [Stanford AIMI](https://stanfordmlgroup.github.io/competitions/chexpert/), registration and research use agreement; take the downsampled release, not the full-resolution one |
| VinDr-CXR v1.0.0 | external, both tasks | [PhysioNet, credentialed access](https://physionet.org/content/vindr-cxr/1.0.0/) |
| Shenzhen | TB, internal | [US NLM](https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Shenzhen-Hospital-CXR-Set/index.html) |
| Montgomery County | TB, external | [US NLM](https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Montgomery-County-CXR-Set/index.html), includes the manual lung masks |
| COVIDGR-1.0 | COVID, single source | [DaSCI open data](https://dasci.es/opendata/covidgr/), served from [github.com/ari-dasci/OD-covidgr](https://github.com/ari-dasci/OD-covidgr) |

Each corpus must be obtained from the distributor linked above, under that
distributor's own terms. Third-party re-uploads of these datasets circulate on
several platforms; some of them are subsets, some are re-processed, and for the
corpora that require a signed agreement they are not an authorised route.

VinDr-CXR requires credentialed PhysioNet access: CITI "Data or Specimens Only
Research" training and a signed data use agreement. Set
`USE_VINDR_EXTERNAL = False` in `config.py` if it is unavailable; external
validation then runs on CheXpert and Montgomery only.

The adult VinDr-CXR release is required. VinDr-PCXR is a paediatric set and
substituting it confounds a change of hospital with a change of age population.

## 3. Reproducing the manuscript

```bash
python data_manifests.py                      # once; writes manifests/

# Training. Tables 2 and 3 come from the pneumonia ablation, Table 4 from the
# other two tasks, Table 5 from the cross-source runs.
python run_ablation.py --task pneumonia --deterministic
python run_ablation.py --task tb --deterministic
python run_ablation.py --task covid --deterministic
python run_cross_source.py --task pneumonia
python run_cross_source.py --task tb

# Label-efficiency sweep: Table 7 and Figure 8. The 1.0 fraction is the run
# above, which was deterministic; these four were not (see section 4).
# PowerShell: foreach ($f in 0.05,0.1,0.25,0.5) { python run_ablation.py --task pneumonia --label_fraction $f }
for f in 0.05 0.1 0.25 0.5; do
  python run_ablation.py --task pneumonia --label_fraction $f
done

# Analysis and figures. Checkpoints are named by the tag of the run that wrote
# them, so the full-data scratch models are the ones explained here.
python run_gradcam.py --task pneumonia \
    --ckpt checkpoints/DSLiteDenseNet_pneumonia_lf1.0_scratch_seed0.pth
python run_gradcam.py --task tb \
    --ckpt checkpoints/DSLiteDenseNet_tb_lf1.0_scratch_seed0.pth
python run_gradcam.py --task pneumonia --model DenseNet121 \
    --ckpt checkpoints/DenseNet121_pneumonia_lf1.0_scratch_seed0.pth
python plot_reliability.py --task pneumonia   # Figure 6
python plot_reliability.py --task tb          # Supplementary Figure S2
python diagnose_tb_domain.py                  # Figure 7
python paper_figures/label_efficiency.py      # Figure 8
python export_reliability_curve.py            # curve behind the graphical abstract

# Analyses that read what the runs above wrote.
python gradcam_stats.py                       # Section 4.4 significance tests
python geometric_reference.py                 # Table 6 area references
python bench_inference.py                     # Table 2 latency and activation cost
python operating_point.py --probs "results/probs_*.npz" \
    --tag pneumonia_lf1.0_scratch --seeds 0 1 2 3 4    # Section 4.5 operating point

# Section 4.3, the part that separates a change of label definition from a
# change of hospital. Both runs score the same checkpoints against the same
# CheXpert negatives; only the positive column differs. --out_suffix keeps the
# second from overwriting the first.
python eval_external.py --task pneumonia --source chexpert
python eval_external.py --task pneumonia --source chexpert \
    --pos_label "Lung Opacity" --chexpert_csv data/chexpert/CheXpert-v1.0-small/train.csv \
    --out_suffix _opacity
# The leading underscore in _opacity is required. operating_point.py reads the
# suffix only when it begins with a letter after an underscore, which is what
# keeps a stray _seed0_1.npz from an aborted run out of the same mean.

# Schematics, which depend on nothing but matplotlib.
python paper_figures/conv_comparison_standard.py    # Figure 1(a)
python paper_figures/conv_comparison_separable.py   # Figure 1(b)
python paper_figures/dense_block_diagram.py         # Figure 2
python paper_figures/network_architecture.py        # Figure 3
```

Where each manuscript item comes from:

| Item | Produced by |
|------|-------------|
| Table 1 | Dataset description; not generated |
| Table 2 | `run_ablation.py` (parameter and FLOP counts) |
| Table 3 | `run_ablation.py --task pneumonia` |
| Table 4 | `run_ablation.py --task tb` and `--task covid` |
| Table 5 | `run_cross_source.py` |
| Table 6 | `run_gradcam.py` |
| Table 7 | `run_ablation.py --label_fraction ...` |
| Figure 1 | `paper_figures/conv_comparison_{standard,separable}.py`, the two panels combined into one file |
| Figure 2 | `paper_figures/dense_block_diagram.py` |
| Figure 3 | `paper_figures/network_architecture.py` |
| Figures 4, 5 | `run_gradcam.py` |
| Figure 6 | `plot_reliability.py --task pneumonia` |
| Figure 7 | `diagnose_tb_domain.py` |
| Figure 8 | `paper_figures/label_efficiency.py` |
| Graphical abstract | Laid out by hand from generated parts; the calibration curve is traced from `export_reliability_curve.py` and the radiograph panel is taken unaltered from `run_gradcam.py` output |
| Supplementary S1–S4 | `run_cross_source.py` |
| Supplementary S5 | `run_ablation.py --task tb`, `run_ablation.py --task covid` |
| Section 4.4 tests | `gradcam_stats.py` |
| Table 6 geometric references | `geometric_reference.py` |
| Inference cost (latency, activation memory) | `bench_inference.py` |
| Section 4.3 label-definition comparison | `eval_external.py --pos_label "Lung Opacity"`, against the default run |
| Sensitivity-constrained and prior-corrected analysis | `operating_point.py` |
| Supplementary Figs. S1, S2 | `run_gradcam.py`, `plot_reliability.py --task tb` |

The results in the manuscript were produced on a single NVIDIA RTX 5090.

Forward-pass cost is measured, not estimated. Batch 1, 224x224 input, on all 24
threads of a desktop CPU. Each figure is the shortest of **eighteen** repetitions
of `python bench_inference.py`, each repetition itself the median of 50 passes
after 10 warm-up passes:

| Model | Params (M) | FLOPs (G) | CPU (ms) | Activation (MB) | Weights (MB) |
|-------|-----------:|----------:|---------:|----------------:|-------------:|
| DenseNet-121 | 7.22 | 5.79 | 39.4 | 95.0 | 28.9 |
| DenseNet-121-DS | 5.39 | 3.48 | 57.0 | 148.3 | 21.6 |
| LiteDenseNet | 1.70 | 2.31 | 15.8 | 41.4 | 6.8 |
| DS-LiteDenseNet | 0.95 | 1.02 | 22.7 | 72.0 | 3.8 |
| MobileNetV2 | 2.23 | 0.65 | 13.7 | 77.9 | 8.9 |
| ShuffleNetV2 | 1.26 | 0.30 | 14.8 | 21.9 | 5.0 |

Table 2 of the manuscript is the first five columns; the sixth, weight-file
size, is not in it. The FLOPs column is the only one `bench_inference.py` does
not produce: it comes from `models.count_flops_g`, which `run_ablation.py` calls
and writes into its summary CSV.

**One run of this script will not reproduce the CPU column, and is not meant
to.**
Across those eighteen runs on an idle machine, the per-run median for
DenseNet-121 ranged from 39.4 to 56.3 ms, a spread of 37% of its own median;
every model varied by 22–37%. `results/inference_cost.csv` holds whichever run
finished last, so it is a sample, not the table.

The parameter, activation and weight columns are properties of the architecture
and the input size and reproduce exactly on any machine, first time. What
reproduces about the latency is the *ratios*, and those are what the manuscript
argues from. Across all eighteen runs:

- the depthwise-separable variant was slower than its dense counterpart in
  **18/18**, by a per-run median of 44% (DenseNet-121-DS over DenseNet-121) and
  43% (DS-LiteDenseNet over LiteDenseNet);
- DS-LiteDenseNet was faster than DenseNet-121 in **18/18**, by a per-run median
  of 1.8x;
- the model ordering was the one tabulated above in **16/18**, the two exceptions
  swapping only models within 2 ms of each other.

If you want a number to compare against, run the script several times and take
the minimum, not the first result. Pin `--threads` and report it.

Training wall time is not tabulated, because it depends on the GPU, the data
loader and the disk holding the corpora far more than on the architecture. Budget
by running one configuration first: add `--epochs 2 --models DSLiteDenseNet
--seeds 0` to any training script to check the plumbing, then scale by epochs,
models and seeds.

The average precision in the last column of Table 5 is the `auprc` column of
`results/crosssource_<task>_raw.csv`, filtered to `eval_source == "vindr"` and
averaged over seeds; the summary CSV aggregates AUROC only.

Two figures are assembled by hand from generated parts. Figure 1 combines the
two panels above into a single file, and the graphical abstract is laid out in a
vector editor. Its calibration curve is drawn from the coordinates in
`reliability_curve_DSLiteDenseNet_pneumonia.csv` and its radiograph panel is
copied unaltered from a `run_gradcam.py` overlay, so every element plots or shows
measured output even though the layout is manual. No part of it is drawn from
imagination or retouched.

All figure scripts set `pdf.fonttype = 42` so that text is embedded as TrueType
rather than matplotlib's Type 3 default, which publishers do not accept.

## 4. Two properties of the protocol worth stating explicitly

**`run_ablation.py` and `run_cross_source.py` train separate models.** Both
report an internal test AUROC and the two agree to within 0.002 AUROC on
pneumonia, but they are different runs and the values are not identical. Tables 3
and 4 of the manuscript come from the ablation; Table 5 and Supplementary
Tables S1–S4 come from the cross-source pipeline.

**Reproducibility: `--deterministic`.** By default cuDNN picks convolution
algorithms by timing them and several backward kernels accumulate in a
non-deterministic order, so repeating an identical command shifts AUROC by about
as much as the seed-to-seed spread. Passing `--deterministic` to
`run_ablation.py` or `run_cross_source.py` disables autotuning, requests
deterministic algorithms, and seeds the sampler and the DataLoader workers from a
common generator. Training then costs roughly one third to one half again in
time, and the same command on the same machine returns the same numbers.

`CUBLAS_WORKSPACE_CONFIG` must be in the environment before the CUDA context is
created. Both scripts set it themselves when `--deterministic` is present, so
nothing is required of you; to set it explicitly instead:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8       # Linux and macOS
```
```cmd
set CUBLAS_WORKSPACE_CONFIG=:4096:8          REM cmd.exe
```
```powershell
$env:CUBLAS_WORKSPACE_CONFIG=":4096:8"       # PowerShell -- `set` is NOT the same command here
```

Two caveats. `torch.use_deterministic_algorithms(..., warn_only=True)` warns
rather than raises when an operation has no deterministic kernel, so check the
warnings on a first short run. And determinism holds for one machine and one
software stack only: a different GPU, CUDA or PyTorch version will not reproduce
the numbers bit for bit. What makes a result robust to that is reporting a mean
and standard deviation over five seeds, which every table here does.

**Which manuscript tables were produced deterministically.** Tables 3 and 4, the
100 % column of Table 7, and the Grad-CAM and calibration analyses that read
their checkpoints, were run with `--deterministic`. Table 5, Supplementary Tables
S1–S4 and the 5–50 % columns of Table 7 predate that flag and were not. Every
value within a given table comes from a single execution of the corresponding
script rather than a mixture of runs.

**Pooling scripts take an explicit seed list.** `plot_reliability.py` and
`export_reliability_curve.py` read the per-sample probability files by naming the
seeds (`--seeds`, default `config.SEEDS`) rather than globbing `seed*`. A glob
silently absorbs anything left in `results/` — a two-epoch diagnostic run, an
aborted sweep, an earlier seed range — and one model then ends up pooled over
more runs than the others with nothing in the output to show it. Files matching
the pattern but outside `--seeds` are listed and skipped, and
`plot_reliability.py` refuses to write a table in which the models were pooled
over unequal numbers of files.

## 5. Citation

```bibtex
@unpublished{Zhang2026DSLiteDenseNet,
  title  = {{DS-LiteDenseNet}: a parameter-frugal chest {X}-ray classifier
            evaluated for non-inferiority, cross-hospital transfer,
            and faithfulness},
  author = {Zhang, Wenying and Xue, Ling and Li, Jiahang and Wang, Ting},
  year   = {2026},
  note   = {Manuscript submitted for publication}
}
```

This entry will be replaced with the published reference once the paper is
accepted.

Please also cite the corpora and the segmenter under their own terms:
RSNA (Shih et al., 2019), CheXpert (Irvin et al., 2019), VinDr-CXR
(Nguyen et al., 2022), Shenzhen and Montgomery (Jaeger et al., 2014),
COVIDGR-1.0 (Tabik et al., 2020), ChestX-Det PSPNet (Lian et al., 2021)
distributed through TorchXRayVision (Cohen et al., 2022).

## 6. Licence

Code released under the MIT Licence; see `LICENSE`. The licence covers this
repository only. Each dataset remains under the terms of its own distributor,
and none of them is redistributed here.

## 7. Contact

Questions about the code or the protocol: open an issue, or write to the
corresponding authors listed in the paper.
