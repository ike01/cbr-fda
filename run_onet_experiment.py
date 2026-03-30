import csv
import json
import logging
import os
from dataclasses import dataclass, replace
from itertools import product
from typing import Dict, List, Optional

from data_onet import ONetCaseBuilder, split_cases
from run_multiwoz_experiment import Config, main as run_engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger("run_onet")


@dataclass
class ONetSplitConfig:
    dataset_path: str = "./onet_data/dataset_tasks_to_skills.json"
    train_size: int = 250
    test_size: int = 0
    split_seed: int = 42
    stratified: bool = True
    split_ids_path: Optional[str] = None
    lowercase: bool = True
    dedupe_lists: bool = True


def _load_or_create_split(split_cfg: ONetSplitConfig):
    builder = ONetCaseBuilder(
        lowercase=split_cfg.lowercase,
        dedupe_lists=split_cfg.dedupe_lists,
    )
    all_cases = builder.load_cases(split_cfg.dataset_path, split_name="onet")
    by_id = {c.case_id: c for c in all_cases}

    if split_cfg.split_ids_path and os.path.exists(split_cfg.split_ids_path):
        with open(split_cfg.split_ids_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        tr_ids = [str(x) for x in obj.get("train_case_ids", [])]
        te_ids = [str(x) for x in obj.get("test_case_ids", [])]
        train_cases = [by_id[cid] for cid in tr_ids if cid in by_id][: split_cfg.train_size]
        test_cases = [by_id[cid] for cid in te_ids if cid in by_id][: split_cfg.test_size]
        if len(train_cases) == split_cfg.train_size and len(test_cases) == split_cfg.test_size:
            logger.info(
                "Loaded O*NET split from %s (train=%d test=%d)",
                split_cfg.split_ids_path, len(train_cases), len(test_cases),
            )
            return train_cases, test_cases
        logger.warning("Existing split file is incomplete/mismatched; regenerating split.")

    if split_cfg.test_size == 0:
        train_cases, _ = split_cases(
            all_cases,
            train_size=split_cfg.train_size,
            test_size=1,
            seed=split_cfg.split_seed,
            stratified=split_cfg.stratified,
        )
        test_cases = []
    else:
        train_cases, test_cases = split_cases(
            all_cases,
            train_size=split_cfg.train_size,
            test_size=split_cfg.test_size,
            seed=split_cfg.split_seed,
            stratified=split_cfg.stratified,
        )
    logger.info(
        "Created O*NET split train=%d test=%d from total=%d",
        len(train_cases), len(test_cases), len(all_cases),
    )

    if split_cfg.split_ids_path:
        out_dir = os.path.dirname(split_cfg.split_ids_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        payload = {
            "train_case_ids": [c.case_id for c in train_cases],
            "test_case_ids": [c.case_id for c in test_cases],
            "meta": {
                "dataset_path": split_cfg.dataset_path,
                "train_size": split_cfg.train_size,
                "test_size": split_cfg.test_size,
                "seed": split_cfg.split_seed,
                "stratified": split_cfg.stratified,
            },
        }
        with open(split_cfg.split_ids_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    return train_cases, test_cases


def run_single(engine_cfg: Config, split_cfg: ONetSplitConfig) -> Dict[str, float]:
    train_cases, test_cases = _load_or_create_split(split_cfg)
    # No external val split for this dataset runner.
    return run_engine(engine_cfg, train_cases=train_cases, test_cases=test_cases, val_cases=[])


def run_onet_option_sweep(
    base_cfg: Config,
    split_cfg: ONetSplitConfig,
    affinity_methods: List[str],
    stopping_modes: List[str],
    stopping_detectors: List[str],
    reuse_methods: Optional[List[str]] = None,
    results_path: str = "results/onet_sweep_results.csv",
):
    if reuse_methods is None:
        reuse_methods = [base_cfg.reuse_method]

    train_cases, test_cases = _load_or_create_split(split_cfg)
    out_dir = os.path.dirname(results_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    rows = []
    run_id = 0
    for reuse_method, affinity_method, stopping_mode in product(reuse_methods, affinity_methods, stopping_modes):
        detector_choices = [base_cfg.stopping_detector] if stopping_mode == "alpha" else stopping_detectors
        for stopping_detector in detector_choices:
            run_id += 1
            cfg = replace(
                base_cfg,
                reuse_method=reuse_method,
                affinity_method=affinity_method,
                stopping_mode=stopping_mode,
                stopping_detector=stopping_detector,
                tune_alpha_on_val=(base_cfg.tune_alpha_on_val and stopping_mode in ("alpha", "hybrid")),
            )
            logger.info(
                "ONET SWEEP run=%d reuse=%s affinity=%s stopping=%s detector=%s",
                run_id, cfg.reuse_method, cfg.affinity_method, cfg.stopping_mode, cfg.stopping_detector,
            )
            try:
                m = run_engine(cfg, train_cases=train_cases, test_cases=test_cases, val_cases=[])
                row = {
                    "run": run_id,
                    "reuse": cfg.reuse_method,
                    "affinity": cfg.affinity_method,
                    "stopping": cfg.stopping_mode,
                    "detector": (cfg.stopping_detector if cfg.stopping_mode != "alpha" else "-"),
                    "node_f1": m.get("node_f1", 0.0),
                    "edge_f1": m.get("edge_f1", 0.0),
                    "avg_neighbours_used": m.get("avg_neighbours_used", 0.0),
                    "int_node_f1": m.get("int_node_f1", 0.0),
                    "coverage": m.get("coverage", 0.0),
                    "status": "ok",
                    "error": "",
                }
            except Exception as e:
                row = {
                    "run": run_id,
                    "reuse": cfg.reuse_method,
                    "affinity": cfg.affinity_method,
                    "stopping": cfg.stopping_mode,
                    "detector": (cfg.stopping_detector if cfg.stopping_mode != "alpha" else "-"),
                    "node_f1": 0.0,
                    "edge_f1": 0.0,
                    "avg_neighbours_used": 0.0,
                    "int_node_f1": 0.0,
                    "coverage": 0.0,
                    "status": "error",
                    "error": str(e),
                }
            rows.append(row)

    fields = [
        "run", "reuse", "affinity", "stopping", "detector",
        "node_f1", "edge_f1", "avg_neighbours_used", "int_node_f1", "coverage", "status", "error",
    ]
    with open(results_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    logger.info("ONET sweep results written to %s", results_path)


if __name__ == "__main__":
    onet_split_seed = 46  # fixed seed for reproducible train/test split of O*NET dataset; separate from case base sampling seed
    onet_case_base_seed = onet_split_seed  # fixed seed for reproducible case base sampling from the training portion of O*NET dataset; separate from train/test split seed

    split_cfg = ONetSplitConfig(
        dataset_path="./onet_data/dataset_tasks_to_skills.json",
        train_size=250,
        test_size=0,
        split_seed=onet_split_seed,
        stratified=True,
        split_ids_path=f"results/onet_split_train250_test0_seed{onet_split_seed}.json",
    )

    cfg = Config(
        multiwoz_root="./MultiWOZ_2.2",  # unused when train/test cases are passed explicitly
        retriever="sbert",
        top_k=5,
        alpha=0.45,
        affinity_method="condprob",  # "embedding_cosine" | "condprob" | "pmi"
        stopping_mode="detector",  # "alpha" | "detector" | "hybrid"
        stopping_detector="NTAD",  # "NCD", "WNCD", "END", "NTAD"
        max_pool_actions=80,
        max_pred_actions=20,
        debug_trace=True,
        debug_first_n=3,
        reuse_method="nda",  # "bm", "gsa", "nda", "carm", "llm_zero", "llm_fewshot", "llm_rag"
        lambda_complexity=0.01,
        prompt_style="onet",
        tune_alpha_on_val=True,
        # Keep O*NET runner simple: fixed split from onet dataset, no extra internal split here.
        case_base_size=250,
        case_base_seed=onet_case_base_seed,
        case_base_ids_path=f"results/onet_casebase_ids_n250_seed{onet_case_base_seed}_strat.json",
        cbr_mode=True,
        cbr_holdout_test_size=50,
        cbr_holdout_ids_path=f"results/onet_cbr_holdout_total250_test50_seed{onet_case_base_seed}_strat.json",
        integrate_detector=True,
        threshold_mode="val_tune",   # "fixed" | "percentile" | "val_tune"
        detector_thresholds=None,
    )

    RUN_SWEEP = True
    if RUN_SWEEP:
        run_onet_option_sweep(
            base_cfg=cfg,
            split_cfg=split_cfg,
            affinity_methods=["embedding_cosine"],
            stopping_modes=["alpha"],
            stopping_detectors=["NTAD"],
            reuse_methods=["bm", "gsa", "nda", "carm"],
            results_path=f"results/sweep_results-onet-alpha_embedding_cosine_seed{onet_split_seed}.csv",
        )
    else:
        run_single(cfg, split_cfg)
