# Fashion-MNIST MLP (Manual Autodiff)

## Environment

- **Python**: 3.9+ recommended (3.8+ should work).
- **Required** (install from repo root):
  ```bash

  python -m pip install -r requirements.txt

  ```
  | Package    | Purpose                                      |
  |------------|----------------------------------------------|
  | `numpy`    | Arrays, training, checkpoints (`.npz`)       |
  | `matplotlib` | Plots (curves, confusion matrix, weights)  |
- **Optional (GPU)**: [CuPy](https://cupy.dev/) matching your CUDA version, e.g.:
  ```bash

  python -m pip install cupy-cuda11x

  ```
  Then pass `--backend gpu` when running `main.py`.
- **Data**: Fashion-MNIST is downloaded automatically on first run into `--data_dir` (default: `data/`).

## How to run

All entry points use `**main.py`** from the repository root.

### Training (single run)

Saves the best-by-validation weights to `artifacts/best_model.npz` and `artifacts/best_model_meta.json`, plus `artifacts/history.json` and `artifacts/training_curves.png`.

```bash

python main.py --mode train --epochs 100 --batch_size 1024 --hidden_dim 384 --activation relu --lr 0.12 --weight_decay 0.0001

```

Adjust `--epochs`, `--batch_size`, `--lr`, etc. as needed. Use `--backend gpu` if CuPy is installed.

### Hyperparameter search (grid search)

Writes each run under `artifacts/search/` (e.g. `run_001_best.npz`, `run_001_best_meta.json`, …) and summary files such as `artifacts/search/best_result.json`.

```bash

python main.py --mode search --epochs 500 --batch_size 1024

```

`--epochs` / `--batch_size` / `--lr_gamma` / `--seed` apply to every grid configuration (learning rates and architecture grid are chosen inside `main.py` from `build_search_space`).

### Checkpoint from Hugging Face Hub (optional)

If you use weights published on the Hub (the same pair as local training: `best_model.npz` and `best_model_meta.json`), download **both** files into **`artifacts/`** at the project root (same folder as `main.py`, i.e. `artifacts/best_model.npz` and `artifacts/best_model_meta.json`, not inside a subfolder).

**Browser:** Open the published model page → [**hhhhhhhhhhhhhhhhhhhhh345/cv**](https://huggingface.co/hhhhhhhhhhhhhhhhhhhhh345/cv) → **Files and versions** → download each file → move or save them into **`artifacts/`** (create the folder if needed).

**CLI** (with `huggingface_hub` installed):

```bash
huggingface-cli download hhhhhhhhhhhhhhhhhhhhh345/cv --local-dir artifacts --include "best_model.npz" --include "best_model_meta.json"
```

Then run `python main.py --mode test` as usual.

### Testing

#### Test the **best single-checkpoint** (default test)

Loads `**artifacts/best_model.npz`** (and `**artifacts/best_model_meta.json**` if present). Prints test loss/accuracy, writes confusion matrix CSV/PNG, first-layer weights figure, and misclassified samples under `artifacts/`.

```bash

python main.py --mode test --batch_size 1024

```

Run `**--mode train**` first so `best_model.npz` exists, or copy a chosen checkpoint from search, for example:

- `artifacts/search/run_016_best.npz` → `artifacts/best_model.npz`
- `artifacts/search/run_016_best_meta.json` → `artifacts/best_model_meta.json`

(`*_meta.json` must sit next to the `.npz` basename so hidden size and activation load correctly.)

After `**--mode full**`, the test phase uses `**artifacts/search/best_result.json**` if it exists (validation winner from search); otherwise it falls back to `best_model.npz`.

#### Test **all** search checkpoints (`--test_all_search`)

Evaluates every `artifacts/search/run_*_best.npz` on the test set, sorts by test accuracy, prints a short top-5 summary, and saves the full ranking to `**artifacts/search/test_ranking.json`**. No confusion matrices or extra plots are produced for this path.

```bash

python main.py --mode test --batch_size 1024 --test_all_search

```

Requires a prior `**--mode search**` (or existing checkpoints under `artifacts/search/`).

### End-to-end (`full`)

Runs **train → search → test** in one command (test step uses search best when `best_result.json` is available):

```bash

python main.py --mode full --epochs 100 --batch_size 1024

```

## Output Files

Under `artifacts/`:

- `best_model.npz`
- `best_model_meta.json` (when training saves meta)
- `history.json`
- `training_curves.png`
- `confusion_matrix.csv`
- `confusion_matrix.png`
- `first_layer_weights.png`
- `misclassified_examples.png`
- `misclassified_all/` (all misclassified images)
- `search/search_results.json`
- `search/best_result.json`
- `search/test_ranking.json` (after `python main.py --mode test --test_all_search`)

