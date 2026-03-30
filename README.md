# Failure-Driven Adaptation (CBR Reuse)

This repository implements a case-based reasoning (CBR) pipeline for structured solution reuse, with experiments on:

- `MultiWOZ 2.2` (task-oriented dialogue)
- `O*NET` (task-to-skill mapping)

At a high level, the pipeline:

1. Builds cases (`problem`, `needs`, `solution`)
2. Retrieves neighbours from a case base
3. Reuses/adapts solutions (`bm`, `gsa`, `nda`, `carm`, `llm_zero`, `llm_fewshot`, `llm_rag`)
4. Applies acceptance/stopping logic (`alpha`, detector-based, or hybrid)
5. Reports Node-F1 / Edge-F1 and related metrics

## Project Entry Points

- `run_multiwoz_experiment.py`: MultiWOZ experiments and sweep runner
- `run_onet_experiment.py`: O*NET experiments and sweep runner

## Setup

Use Python 3.10+ (3.12 works in this repo).

Install common dependencies:

```bash
pip install numpy networkx scikit-learn sentence-transformers
```

Notes:

- `sentence-transformers` is needed for SBERT retrieval; if unavailable, use TF-IDF mode.
- LLM baselines expect a local Ollama server when enabled.

## How To Run Sweeps

Both runners are configured from their `if __name__ == "__main__":` block.
Set `RUN_SWEEP = True`, edit the sweep lists, then run the script.

---

## MultiWOZ Sweeper

File: `run_multiwoz_experiment.py`

### 1) Configure base settings (`cfg = Config(...)`)

Common options to set:

- dataset path: `multiwoz_root`
- case-base sampling: `case_base_size`, `case_base_seed`, `case_base_stratified`
- CBR holdout split: `cbr_mode`, `cbr_holdout_test_size`
- retrieval and detector defaults: `retriever`, `top_k`, `stopping_detector`

### 2) Configure sweep combinations (`run_option_sweep(...)`)

Edit:

- `reuse_methods` (e.g., all 7 methods)
- `affinity_methods` (e.g., `["embedding_cosine", "condprob", "pmi"]`)
- `stopping_modes` (e.g., `["alpha", "detector", "hybrid"]`)
- `stopping_detectors` (e.g., `["NTAD"]` or all detectors)
- `results_path`
- `max_workers`

### 3) Run

```bash
python run_multiwoz_experiment.py
```

Results are written to the CSV set in `results_path`.

---

## O*NET Sweeper

File: `run_onet_experiment.py`

### 1) Configure split (`split_cfg = ONetSplitConfig(...)`)

Set:

- dataset file: `dataset_path`
- split sizes: `train_size`, `test_size`
- seed and stratification: `split_seed`, `stratified`
- optional persisted split IDs: `split_ids_path`

### 2) Configure base run (`cfg = Config(...)`)

Set:

- retrieval: `retriever`, `top_k`
- stopping and detector defaults: `stopping_mode`, `stopping_detector`
- reuse default: `reuse_method`

### 3) Configure sweep (`run_onet_option_sweep(...)`)

Edit:

- `reuse_methods`
- `affinity_methods`
- `stopping_modes`
- `stopping_detectors`
- `results_path`

### 4) Run

```bash
python run_onet_experiment.py
```

Results are written to the CSV set in `results_path`.

## Typical Paper-Style Sweep Example

Use the same setup across methods, then vary only method and selection configuration:

- `reuse_methods = ["bm", "gsa", "nda", "carm", "llm_zero", "llm_fewshot", "llm_rag"]`
- `affinity_methods = ["condprob"]`
- `stopping_modes = ["detector"]`
- `stopping_detectors = ["NTAD"]`

For seed averaging, repeat runs with different sampling/split seeds and average the CSV outputs externally.

## Outputs

Sweep CSV files include per-run:

- `node_f1`
- `edge_f1`
- `avg_neighbours_used`
- `int_node_f1`
- `coverage`
- `status` / `error`

Result files are stored under `results/` for reproducibility.
