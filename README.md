# MVGAE-LGNet
A traffic flow forecasting codebase that combines **multi-head variational graph autoencoder (MVGAE)** spatial representation pretraining with a **hybrid spatiotemporal predictor (HybridSTPredictor / LGNet)**.

## Method Overview
Main entry point: `train_hybrid.py` (for full experiments, stage-1 pretraining runs automatically when `z_init.npy` is missing).

## Experimental Environment

Recommended: Linux + NVIDIA GPU (typical setup used in development/experiments; adjust to your machine).

| Item | Suggested version / notes |
|---|---|
| OS | Linux (e.g., Ubuntu) |
| Python | 3.7+ (conda env commonly named `mvgae-lgnet`) |
| PyTorch | GPU build matching your CUDA driver (e.g., CUDA 12.x → `cu124` wheels) |
| torch-geometric | ≥ 2.3 |
| Other dependencies | See `requirements.txt` (`numpy`, `scikit-learn`, `tensorboardX`, etc.) |
| GPU | A single GPU is sufficient; PEMS07 (883 nodes) is memory-heavy—recommend ≥ 24GB; watch for OOM when running multiple jobs in parallel |

## Data Preparation

Place datasets under `data/`, for example:

```text
data/
├── PEMS04/   # PEMS04.npz, distance.csv
├── PEMS07/   # PEMS07.npz, PEMS07.csv
└── PEMS08/   # PEMS08.npz, PEMS08.csv
```

## How to Run

Run all commands from the **project root**:

```bash
cd /path/to/MVGAE-LGNet
conda activate mvgae-lgnet
```

```bash
python train_hybrid.py --config configurations/PEMS04_multi_period.conf
python train_hybrid.py --config configurations/PEMS07_multi_period.conf
python train_hybrid.py --config configurations/PEMS08_multi_period.conf
```

## Outputs

- Experiment directory: `experiments/<dataset>/<model_name>_..._<run_tag>/`  
  Contains weights (e.g., `best.params`), `results_summary.json`, TensorBoard logs, etc.
- Summary table: `experiments/<dataset>/runs_summary.csv`
- `[Training] model_name` distinguishes runs; the default for full experiments is `hybrid_st_period_best`.

## Repository Layout (brief)

```text
MVGAE-LGNet/
├── train_hybrid.py          # Main training entry
├── train_mvgae_pretrain.py  # MVGAE pretraining only
├── prepareData.py           # Data slicing / preparation
├── configurations/          # Experiment configs + per-folder READMEs
├── model/                   # MVGAE, Hybrid, Graph WaveNet
├── lib/                     # Data, losses, pretrain, experiment I/O
├── tools/                   # Utility scripts
├── data/                    # Datasets, z_init, checkpoints
├── experiments/             # Training outputs (created at runtime)
└── requirements.txt
```
