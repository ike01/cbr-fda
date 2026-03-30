# run_multiwoz_experiment.py
import logging
from dataclasses import dataclass, replace
from typing import List, Dict, Optional, Tuple
from itertools import product
import csv
import os
import json
import re
from urllib import request as urlrequest
from urllib import error as urlerror
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from data_multiwoz22 import MultiWOZ22Loader, MultiWOZ22CaseBuilder
from graph_structures import Case
from retrieval import TfidfRetriever, SBERTRetriever, Retrieved
from stable_matching import TextSim, stable_match_needs_to_actions
from merge_graph import merge_matched_actions_into_graph, build_action_sequence_graph
from evaluation import eval_solution_graph

# NEW: failure detection + detector evaluation
from failure_detection import (
    NeighbourhoodConsistencyDetector,
    CaseAlignmentIntegrityDetector,
    EmbeddingNoveltyDetector,
    AlignmentStabilityDetector,
    NeighbourhoodTransformationAlignmentDetector,
)
from detector_eval import eval_detector

logging.basicConfig(
    level=logging.INFO,  # change to DEBUG for more detail
    format="%(asctime)s %(name)s %(levelname)s: %(message)s"
)
logger = logging.getLogger("run")


def _canonical_reuse_method(name: str) -> str:
    # Backward-compatible alias for the paper-aligned method name.
    return "carm" if name == "gsa_card" else name


def _canonical_detector_name(name: str) -> str:
    # Backward-compatible aliases for paper-aligned detector labels.
    aliases = {
        "NCFD": "NCD",
        "CAI": "WNCD",
        "EMND": "END",
    }
    return aliases.get(name, name)


@dataclass
class Config:
    multiwoz_root: str
    retriever: str = "sbert"          # "tfidf" or "sbert"
    top_k: int = 5                    # retrieve K neighbour cases
    alpha: float = 0.45               # stable matching acceptance threshold
    affinity_method: str = "embedding_cosine"  # "embedding_cosine" | "condprob" | "pmi"
    condprob_smoothing: float = 1.0
    pmi_smoothing: float = 0.1
    stopping_mode: str = "alpha"      # "alpha" | "detector" | "hybrid"
    stopping_detector: str = "NTAD"   # used when stopping_mode != "alpha"
    # Alpha tuning
    tune_alpha_on_val: bool = True
    alpha_grid: List[float] = None     # if None, use default grid
    alpha_max_cases: int = 2000        # cap val cases for speed
    max_pool_actions: int = 60        # cap candidate pool size
    max_pred_actions: int = 12        # cap predicted actions
    debug_trace: bool = False         # print matching/merge trace for first N examples
    debug_first_n: int = 5            # number of test cases to trace
    reuse_method: str = "gsa"         # "bm", "gsa", "nda", "carm", "llm_zero", "llm_fewshot", "llm_rag"
    lambda_complexity: float = 0.10   # cardinality penalty used by carm

    # Failure detection evaluation
    low_quality_cut: float = 0.2      # "true failure" if node-F1 < this
    detector_thresholds: Dict[str, float] = None
    enable_asfd: bool = False         # ASFD is expensive; enable only when needed

    # Threshold selection
    threshold_mode: str = "val_tune"      # "fixed" | "percentile" | "val_tune"
    flag_rate: float = 0.10            # used by "percentile": flag bottom X%
    tune_grid_size: int = 30           # used by "val_tune": number of candidate thresholds per detector
    max_calib_cases: int = 2000        # cap calibration workload (val_tune/percentile)

    # Integrating failure detection
    integrate_detector: bool = True
    integration_mode: str = "fallback_bm"   # "fallback_bm" or "selective"
    integration_detector: str = "NTAD"      # "NCD", "WNCD", "END", "NTAD" (ASFD if enabled)
    # Reproducible case-base sampling
    case_base_size: Optional[int] = 250    # e.g., 100 or 200; None uses full train set
    case_base_seed: int = 42
    case_base_stratified: bool = True       # stratify by service when sampling
    case_base_ids_path: Optional[str] = None  # load/save sampled case_ids for exact reruns
    persist_case_base_ids: bool = True
    # CBR-style internal tuning (no external val/dev split)
    cbr_mode: bool = False
    cv_folds: int = 5
    cv_max_cases: int = 2000          # cap cases used during CV tuning
    tune_lambda_on_cv: bool = True
    lambda_grid: List[float] = None   # if None, use default grid
    # Optional CBR holdout split from sampled case base (train -> train/test).
    # Example: case_base_size=250, cbr_holdout_test_size=50 => 200/50 split.
    cbr_holdout_test_size: Optional[int] = 50
    cbr_holdout_ids_path: Optional[str] = "results/cbr_holdout_total250_test50_seed42_strat.json" # None
    # Ollama LLM baselines
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"  # "llama3:8b", "llama3.1:8b", "gemma3:12b"
    ollama_timeout_s: int = 120
    llm_temperature: float = 0.0
    llm_max_tokens: int = 512
    llm_seed: int = 42
    llm_fewshot_k: int = 3
    llm_rag_k: int = 3
    prompt_style: str = "multiwoz"   # "multiwoz" | "onet"


# Worker-global datasets for ProcessPoolExecutor childs
_WORKER_SHARED_DATA = {
    "train_cases": None,
    "test_cases": None,
    "val_cases": None,
}


def _load_data_cases(multiwoz_root: str):
    loader = MultiWOZ22Loader(multiwoz_root)
    builder = MultiWOZ22CaseBuilder(include_state=True)

    train_dialogs = loader.load_split("train")
    test_dialogs = loader.load_split("test")

    try:
        try:
            val_dialogs = loader.load_split("val")
        except Exception:
            val_dialogs = loader.load_split("dev")
    except Exception:
        val_dialogs = []

    train_cases = builder.build_cases(train_dialogs, "train")
    test_cases = builder.build_cases(test_dialogs, "test")
    val_cases = builder.build_cases(val_dialogs, "val") if val_dialogs else []

    return train_cases, test_cases, val_cases


def _init_worker_data(multiwoz_root: str):
    global _WORKER_SHARED_DATA
    train_cases, test_cases, val_cases = _load_data_cases(multiwoz_root)
    _WORKER_SHARED_DATA["train_cases"] = train_cases
    _WORKER_SHARED_DATA["test_cases"] = test_cases
    _WORKER_SHARED_DATA["val_cases"] = val_cases


def _default_case_base_ids_path(cfg: Config, n: int) -> str:
    mode = "strat" if cfg.case_base_stratified else "random"
    return os.path.join("results", f"casebase_ids_n{n}_seed{cfg.case_base_seed}_{mode}.json")


def _default_cbr_holdout_ids_path(cfg: Config, n_total: int, n_test: int) -> str:
    mode = "strat" if cfg.case_base_stratified else "random"
    return os.path.join(
        "results",
        f"cbr_holdout_total{n_total}_test{n_test}_seed{cfg.case_base_seed}_{mode}.json",
    )


def _save_case_ids(path: str, case_ids: List[str], cfg: Config, total_available: int, sampled_n: int) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    payload = {
        "case_ids": case_ids,
        "meta": {
            "sampled_n": int(sampled_n),
            "total_available": int(total_available),
            "seed": int(cfg.case_base_seed),
            "stratified": bool(cfg.case_base_stratified),
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _load_case_ids(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict) and "case_ids" in obj:
        return [str(x) for x in obj["case_ids"]]
    if isinstance(obj, list):
        return [str(x) for x in obj]
    raise ValueError(f"Unsupported case id file format: {path}")


def _sample_train_case_base(train_cases: List[Case], cfg: Config) -> List[Case]:
    cases_sorted = sorted(train_cases, key=lambda c: c.case_id)
    total = len(cases_sorted)
    n_req = int(cfg.case_base_size) if cfg.case_base_size is not None else total
    if n_req <= 0 or n_req >= total:
        return cases_sorted

    ids_path = cfg.case_base_ids_path or _default_case_base_ids_path(cfg, n_req)
    if os.path.exists(ids_path):
        loaded_ids = _load_case_ids(ids_path)
        by_id = {c.case_id: c for c in cases_sorted}
        selected = [by_id[cid] for cid in loaded_ids if cid in by_id]
        if len(selected) >= n_req:
            logger.info("Loaded reproducible case-base sample from %s (n=%d)", ids_path, n_req)
            return selected[:n_req]
        logger.warning(
            "Loaded ids from %s but found only %d/%d in current train set; refilling deterministically.",
            ids_path, len(selected), n_req
        )
        selected_ids = set(c.case_id for c in selected)
        remaining = [c for c in cases_sorted if c.case_id not in selected_ids]
        rng = np.random.default_rng(int(cfg.case_base_seed))
        if remaining and len(selected) < n_req:
            extra_idx = rng.choice(len(remaining), size=min(n_req - len(selected), len(remaining)), replace=False)
            for j in np.sort(extra_idx):
                selected.append(remaining[int(j)])
        selected = selected[:n_req]
        if cfg.persist_case_base_ids:
            _save_case_ids(ids_path, [c.case_id for c in selected], cfg, total_available=total, sampled_n=len(selected))
        return selected

    rng = np.random.default_rng(int(cfg.case_base_seed))
    selected: List[Case] = []

    if cfg.case_base_stratified:
        groups: Dict[str, List[Case]] = {}
        for c in cases_sorted:
            groups.setdefault(c.service, []).append(c)

        services = sorted(groups.keys())
        exact = {s: (n_req * len(groups[s]) / float(total)) for s in services}
        quota = {s: min(len(groups[s]), int(np.floor(exact[s]))) for s in services}
        picked = sum(quota.values())
        remaining_budget = n_req - picked

        frac_order = sorted(
            services,
            key=lambda s: (exact[s] - np.floor(exact[s]), len(groups[s])),
            reverse=True
        )
        while remaining_budget > 0:
            progressed = False
            for s in frac_order:
                if quota[s] < len(groups[s]):
                    quota[s] += 1
                    remaining_budget -= 1
                    progressed = True
                    if remaining_budget == 0:
                        break
            if not progressed:
                break

        for s in services:
            g = groups[s]
            q = int(quota[s])
            if q <= 0:
                continue
            idx = rng.choice(len(g), size=q, replace=False)
            for j in np.sort(idx):
                selected.append(g[int(j)])
    else:
        idx = rng.choice(total, size=n_req, replace=False)
        for j in np.sort(idx):
            selected.append(cases_sorted[int(j)])

    selected = sorted(selected, key=lambda c: c.case_id)[:n_req]
    if cfg.persist_case_base_ids:
        _save_case_ids(ids_path, [c.case_id for c in selected], cfg, total_available=total, sampled_n=len(selected))
    logger.info(
        "Sampled reproducible train case-base n=%d/%d seed=%d stratified=%s ids=%s",
        len(selected), total, cfg.case_base_seed, cfg.case_base_stratified, ids_path
    )
    return selected


def _split_cbr_holdout_train_test(pool_cases: List[Case], cfg: Config) -> Tuple[List[Case], List[Case]]:
    """
    Deterministically split sampled pool into CBR train/test holdout.
    Respects `case_base_seed`, optional stratification, and persisted ids.
    """
    n_test = int(cfg.cbr_holdout_test_size or 0)
    if n_test <= 0:
        return pool_cases, []

    pool_cases = sorted(pool_cases, key=lambda c: c.case_id)
    n_total = len(pool_cases)
    if n_test >= n_total:
        raise ValueError(f"cbr_holdout_test_size={n_test} must be < pool size ({n_total}).")

    ids_path = cfg.cbr_holdout_ids_path or _default_cbr_holdout_ids_path(cfg, n_total, n_test)
    by_id = {c.case_id: c for c in pool_cases}

    if os.path.exists(ids_path):
        obj = None
        with open(ids_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        test_ids = []
        train_ids = []
        if isinstance(obj, dict):
            test_ids = [str(x) for x in obj.get("test_case_ids", [])]
            train_ids = [str(x) for x in obj.get("train_case_ids", [])]
        test_cases = [by_id[cid] for cid in test_ids if cid in by_id]
        train_cases = [by_id[cid] for cid in train_ids if cid in by_id]
        if len(test_cases) == n_test and len(test_cases) + len(train_cases) <= n_total:
            used = set(c.case_id for c in test_cases + train_cases)
            if len(train_cases) < (n_total - n_test):
                for c in pool_cases:
                    if c.case_id not in used and len(train_cases) < (n_total - n_test):
                        train_cases.append(c)
            logger.info(
                "Loaded CBR holdout split from %s (train=%d test=%d)",
                ids_path, len(train_cases), len(test_cases)
            )
            return sorted(train_cases, key=lambda c: c.case_id), sorted(test_cases, key=lambda c: c.case_id)

    rng = np.random.default_rng(int(cfg.case_base_seed))
    test_cases: List[Case] = []

    if cfg.case_base_stratified:
        groups: Dict[str, List[Case]] = {}
        for c in pool_cases:
            groups.setdefault(c.service, []).append(c)

        services = sorted(groups.keys())
        exact = {s: (n_test * len(groups[s]) / float(n_total)) for s in services}
        quota = {s: min(len(groups[s]), int(np.floor(exact[s]))) for s in services}
        remaining = n_test - sum(quota.values())
        frac_order = sorted(
            services,
            key=lambda s: (exact[s] - np.floor(exact[s]), len(groups[s])),
            reverse=True,
        )
        while remaining > 0:
            progressed = False
            for s in frac_order:
                if quota[s] < len(groups[s]):
                    quota[s] += 1
                    remaining -= 1
                    progressed = True
                    if remaining == 0:
                        break
            if not progressed:
                break

        for s in services:
            g = groups[s]
            q = int(quota[s])
            if q <= 0:
                continue
            idx = rng.choice(len(g), size=q, replace=False)
            for j in np.sort(idx):
                test_cases.append(g[int(j)])
    else:
        idx = rng.choice(n_total, size=n_test, replace=False)
        for j in np.sort(idx):
            test_cases.append(pool_cases[int(j)])

    test_ids_set = set(c.case_id for c in test_cases)
    train_cases = [c for c in pool_cases if c.case_id not in test_ids_set]
    train_cases = sorted(train_cases, key=lambda c: c.case_id)
    test_cases = sorted(test_cases, key=lambda c: c.case_id)

    if cfg.persist_case_base_ids:
        out_dir = os.path.dirname(ids_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        payload = {
            "train_case_ids": [c.case_id for c in train_cases],
            "test_case_ids": [c.case_id for c in test_cases],
            "meta": {
                "seed": int(cfg.case_base_seed),
                "stratified": bool(cfg.case_base_stratified),
                "total": int(n_total),
                "test_size": int(n_test),
            },
        }
        with open(ids_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    logger.info(
        "Created CBR holdout split train=%d test=%d from pool=%d ids=%s",
        len(train_cases), len(test_cases), n_total, ids_path
    )
    return train_cases, test_cases


def _build_retriever(cfg: Config):
    if cfg.retriever == "sbert":
        try:
            return SBERTRetriever()
        except Exception:
            logger.warning("SBERT retriever unavailable. Falling back to TF-IDF.")
            return TfidfRetriever()
    return TfidfRetriever()


def _make_casebase_cv_folds(cases: List[Case], n_folds: int, seed: int) -> List[List[int]]:
    n_folds = max(2, int(n_folds))
    by_service: Dict[str, List[int]] = {}
    for idx, c in enumerate(cases):
        by_service.setdefault(c.service, []).append(idx)

    rng = np.random.default_rng(int(seed))
    folds = [[] for _ in range(n_folds)]
    for svc in sorted(by_service.keys()):
        idxs = np.array(by_service[svc], dtype=int)
        rng.shuffle(idxs)
        for j, idx in enumerate(idxs.tolist()):
            folds[j % n_folds].append(int(idx))
    for f in folds:
        f.sort()
    return folds


def tune_hparams_on_casebase_cv(cfg: Config, train_cases: List[Case]) -> Tuple[float, float]:
    """
    Tune alpha (+ lambda for carm) with K-fold CV on the case base.
    Returns (best_alpha, best_lambda).
    """
    cases = sorted(train_cases, key=lambda c: c.case_id)
    if cfg.cv_max_cases and len(cases) > int(cfg.cv_max_cases):
        cases = cases[: int(cfg.cv_max_cases)]

    if len(cases) < 2:
        logger.warning("CBR CV tuning skipped: not enough cases (%d).", len(cases))
        return float(cfg.alpha), float(cfg.lambda_complexity)

    alpha_grid = cfg.alpha_grid if cfg.alpha_grid is not None else [round(x, 2) for x in np.arange(0.20, 0.71, 0.05)]
    if _canonical_reuse_method(cfg.reuse_method) == "carm" and cfg.tune_lambda_on_cv:
        lambda_grid = cfg.lambda_grid if cfg.lambda_grid is not None else [0.00, 0.05, 0.10, 0.20, 0.40]
    else:
        lambda_grid = [float(cfg.lambda_complexity)]

    folds = _make_casebase_cv_folds(cases, cfg.cv_folds, cfg.case_base_seed)
    n_folds = len(folds)
    results: List[Tuple[float, float, float]] = []

    # Build CV components once and reuse across all folds/hyperparameter settings.
    cv_retriever = _build_retriever(cfg)
    try:
        cv_sim_backend = TextSim(
            prefer_sbert=True,
            affinity_method=cfg.affinity_method,
            condprob_smoothing=cfg.condprob_smoothing,
            pmi_smoothing=cfg.pmi_smoothing,
        )
    except Exception:
        logger.warning("CBR-CV SBERT init failed once; falling back to non-SBERT affinity backend for CV.")
        cv_sim_backend = TextSim(
            prefer_sbert=False,
            affinity_method=cfg.affinity_method,
            condprob_smoothing=cfg.condprob_smoothing,
            pmi_smoothing=cfg.pmi_smoothing,
        )

    for a in alpha_grid:
        for lam in lambda_grid:
            fold_f1 = []
            for fi in range(n_folds):
                val_idx = set(folds[fi])
                fold_val = [cases[i] for i in sorted(val_idx)]
                fold_train = [cases[i] for i in range(len(cases)) if i not in val_idx]
                if not fold_val or not fold_train:
                    continue

                cv_retriever.fit(fold_train)
                cv_sim_backend.fit(fold_train)

                f1s = []
                for q in fold_val:
                    retrieved = cv_retriever.query(q.problem_text, top_k=cfg.top_k)
                    if cfg.reuse_method == "bm":
                        if not retrieved:
                            pred_graph = build_action_sequence_graph([])
                        else:
                            top_case = fold_train[retrieved[0].idx]
                            pred_graph = build_action_sequence_graph(top_case.solution_actions[:cfg.max_pred_actions])
                    else:
                        pred_graph, _ = stable_matching_reuse_with_graph_merge(
                            query_case=q,
                            retrieved=retrieved,
                            case_base=fold_train,
                            alpha=float(a),
                            sim_backend=cv_sim_backend,
                            max_pool_actions=cfg.max_pool_actions,
                            max_pred_actions=cfg.max_pred_actions,
                            match_method=cfg.reuse_method,
                            debug=False,
                            stopping_mode="alpha",
                            stop_detector=None,
                            stop_threshold=0.0,
                            lambda_complexity=float(lam),
                        )
                    m = eval_solution_graph(pred_graph, q.solution_actions)
                    f1s.append(float(m["node_f1"]))

                if f1s:
                    fold_f1.append(float(np.mean(f1s)))

            avg_f1 = float(np.mean(fold_f1)) if fold_f1 else 0.0
            results.append((float(a), float(lam), avg_f1))
            logger.info(
                "[cbr-cv] alpha=%.2f lambda=%.3f | folds=%d | Avg node-F1=%.4f",
                float(a), float(lam), n_folds, avg_f1
            )

    if not results:
        logger.warning("CBR CV tuning produced no results. Using current alpha/lambda.")
        return float(cfg.alpha), float(cfg.lambda_complexity)

    # Tie-breaker: lower lambda then lower alpha for simpler settings.
    results.sort(key=lambda t: (-t[2], t[1], t[0]))
    best_alpha, best_lambda, best_f1 = results[0]
    logger.info(
        "[cbr-cv] BEST alpha=%.2f lambda=%.3f | Avg node-F1=%.4f",
        best_alpha, best_lambda, best_f1
    )
    return best_alpha, best_lambda


def collect_pool_actions(retrieved: List[Retrieved], case_base: List[Case], upto_i: int, cap: int) -> List[str]:
    """
    Pool deduplicated action labels from the first upto_i neighbours.
    """
    pool = []
    seen = set()
    for r in retrieved[:upto_i]:
        for a in case_base[r.idx].solution_actions:
            if a not in seen:
                seen.add(a)
                pool.append(a)
            if len(pool) >= cap:
                return pool
    return pool


def stable_matching_reuse_with_graph_merge(
    query_case: Case,
    retrieved: List[Retrieved],
    case_base: List[Case],
    alpha: float,
    sim_backend: TextSim,
    max_pool_actions: int,
    max_pred_actions: int,
    match_method: str = "gsa",
    debug: bool = False,
    stopping_mode: str = "alpha",
    stop_detector=None,
    stop_threshold: float = 0.0,
    lambda_complexity: float = 0.10,
):
    """
    Failure-driven reuse loop with configurable stopping policy:
      - alpha: stop when matching score >= alpha
      - detector: stop when detector score >= stop_threshold
      - hybrid: stop when both conditions hold
    Then graph-merge matched actions into a predicted SolutionGraph.
    """
    match_method = _canonical_reuse_method(match_method)

    if not retrieved:
        return build_action_sequence_graph([]), {"final_score": 0.0, "pairs": {}}

    best_score = -1.0
    best_rank_score = -1.0
    best_stop_score = -1.0
    best_pairs = {}
    best_pool = []
    best_i = 1

    def _matched_actions_from_pairs(
        pairs: Dict[int, int],
        pool_actions: List[str],
        sim_mat: np.ndarray,
        retrieved_subset: List[Retrieved],
    ) -> List[str]:
        # Seed core from stable matching using deterministic need order.
        matched_actions = []
        used = set()
        for need_i in sorted(pairs.keys()):
            a = pool_actions[pairs[need_i]]
            if a not in used:
                used.add(a)
                matched_actions.append(a)
        seed_order = {a: i for i, a in enumerate(matched_actions)}

        if match_method != "carm":
            if len(matched_actions) < 1:
                # Fallback if no pair survives matching.
                matched_actions = case_base[retrieved[0].idx].solution_actions[:max_pred_actions]
            return matched_actions[:max_pred_actions]

        # carm: stable matching provides seeds, then post-matching
        # cardinality is selected by a quality-complexity objective.
        n_needs, m_actions = sim_mat.shape
        action_sim = np.zeros(m_actions, dtype=np.float32)
        if n_needs > 0 and m_actions > 0:
            action_sim = np.max(sim_mat, axis=0)

        weights = np.array([max(float(r.score), 0.0) for r in retrieved_subset], dtype=np.float32)
        if len(weights) == 0:
            weights = np.array([1.0], dtype=np.float32)
        if float(np.sum(weights)) <= 1e-12:
            weights = np.ones_like(weights)

        lengths = np.array(
            [len(case_base[r.idx].solution_actions) for r in retrieved_subset],
            dtype=np.float32
        ) if retrieved_subset else np.array([len(case_base[retrieved[0].idx].solution_actions)], dtype=np.float32)
        if len(lengths) != len(weights):
            weights = np.ones_like(lengths, dtype=np.float32)

        mu = float(np.average(lengths, weights=weights)) if len(lengths) > 0 else 1.0
        low = int(np.floor(np.percentile(lengths, 25))) if len(lengths) > 0 else 1
        high = int(np.ceil(np.percentile(lengths, 75))) if len(lengths) > 0 else max_pred_actions
        low = max(1, low)
        high = max(low, high)
        high = min(high, max_pred_actions)

        support_count = {}
        for ridx, r in enumerate(retrieved_subset):
            w = float(weights[ridx]) if ridx < len(weights) else 1.0
            for a in set(case_base[r.idx].solution_actions):
                support_count[a] = support_count.get(a, 0.0) + w
        support_norm = max(support_count.values()) if support_count else 1.0

        action_score = {}
        for j, a in enumerate(pool_actions):
            s_aff = float(action_sim[j]) if j < len(action_sim) else 0.0
            s_sup = float(support_count.get(a, 0.0) / max(support_norm, 1e-9))
            action_score[a] = 0.8 * s_aff + 0.2 * s_sup

        # Rank core by matched-pair affinity (higher first), tie-break by seed order.
        core_affinity = {}
        for need_i, act_j in pairs.items():
            a = pool_actions[act_j]
            s = float(sim_mat[need_i, act_j]) if sim_mat.size else 0.0
            prev = core_affinity.get(a)
            if prev is None or s > prev:
                core_affinity[a] = s

        ranked_core = sorted(
            matched_actions,
            key=lambda a: (-core_affinity.get(a, 0.0), seed_order.get(a, 10**9)),
        )
        rest = [a for a in pool_actions if a not in set(ranked_core)]
        rest.sort(key=lambda a: action_score.get(a, 0.0), reverse=True)
        ordered = ranked_core + rest
        if not ordered:
            ordered = case_base[retrieved[0].idx].solution_actions[:max_pred_actions]

        max_k = min(len(ordered), max_pred_actions, high)
        min_k = min(max(1, low), max_k)
        if min_k > max_k:
            min_k = max_k

        denom = max(float(max_k - min_k), 1.0)
        best_k = min_k
        best_obj = -1e9
        for k in range(min_k, max_k + 1):
            q = float(np.mean([action_score.get(a, 0.0) for a in ordered[:k]]))
            c = float(((k - mu) / denom) ** 2)
            obj = q - float(lambda_complexity) * c
            if obj > best_obj:
                best_obj = obj
                best_k = k

        return ordered[:best_k]

    K = len(retrieved)
    for i in range(1, K + 1):
        pool_actions = collect_pool_actions(retrieved, case_base, upto_i=i, cap=max_pool_actions)

        sm_method = "gsa" if match_method == "carm" else match_method
        pairs, score, sim_mat = stable_match_needs_to_actions(
            query_case.needs,
            pool_actions,
            sim_backend,
            debug=debug,
            method=sm_method
        )

        candidate_actions = _matched_actions_from_pairs(pairs, pool_actions, sim_mat, retrieved[:i])
        candidate_quality = 0.0
        if candidate_actions and len(pool_actions) > 0 and sim_mat.shape[0] > 0:
            idx_map = {a: j for j, a in enumerate(pool_actions)}
            vals = []
            for a in candidate_actions:
                j = idx_map.get(a)
                if j is not None:
                    vals.append(float(np.max(sim_mat[:, j])))
            if vals:
                candidate_quality = float(np.mean(vals))

        detector_score = None
        should_stop_alpha = (candidate_quality >= alpha) if match_method == "carm" else (score >= alpha)
        should_stop_detector = False
        if stopping_mode in ("detector", "hybrid"):
            if stop_detector is None:
                raise RuntimeError(
                    f"stopping_mode='{stopping_mode}' requires stop_detector."
                )
            candidate_graph, _, _ = merge_matched_actions_into_graph(
                matched_actions=candidate_actions,
                retrieved=retrieved[:i],
                case_base=case_base,
                debug=False
            )
            detector_score = float(stop_detector.score(query_case, candidate_graph, retrieved[:i], case_base))
            should_stop_detector = (detector_score >= stop_threshold)

        if debug:
            if detector_score is None:
                logger.info("i=%d pool=%d score=%.4f alpha=%.2f", i, len(pool_actions), score, alpha)
            else:
                logger.info(
                    "i=%d pool=%d score=%.4f alpha=%.2f det=%.4f thr=%.3f mode=%s",
                    i, len(pool_actions), score, alpha, detector_score, stop_threshold, stopping_mode
                )
            pair_list = []
            for need_i, act_j in pairs.items():
                pair_list.append((query_case.needs[need_i], pool_actions[act_j], float(sim_mat[need_i, act_j])))
            pair_list.sort(key=lambda t: -t[2])
            for need, act, s in pair_list[:10]:
                logger.info("  pair need='%s' -> action='%s' affinity=%.3f", need, act, s)

        rank_score = (
            candidate_quality if match_method == "carm" else score
        ) if stopping_mode in ("alpha", "hybrid") else float(detector_score or 0.0)
        if rank_score > best_rank_score:
            best_rank_score = rank_score
            best_score = candidate_quality if match_method == "carm" else score
            best_stop_score = float(detector_score or 0.0)
            best_pairs = pairs
            best_pool = pool_actions
            best_i = i

        if stopping_mode == "alpha":
            should_stop = should_stop_alpha
        elif stopping_mode == "detector":
            should_stop = should_stop_detector
        elif stopping_mode == "hybrid":
            should_stop = should_stop_alpha and should_stop_detector
        else:
            raise ValueError(f"Unknown stopping_mode: {stopping_mode}")

        if should_stop:
            break

    _, _, best_sim = stable_match_needs_to_actions(
        query_case.needs,
        best_pool,
        sim_backend,
        debug=False,
        method=("gsa" if match_method == "carm" else match_method)
    )
    matched_actions = _matched_actions_from_pairs(best_pairs, best_pool, best_sim, retrieved[:best_i])

    pred_graph, _, merged_order = merge_matched_actions_into_graph(
        matched_actions=matched_actions,
        retrieved=retrieved[:best_i],
        case_base=case_base,
        debug=debug
    )

    meta = {
        "final_score": float(best_score),
        "final_stop_score": float(best_stop_score),
        "best_i": best_i,
        "matched_actions": matched_actions,
        "merged_order": merged_order
    }
    return pred_graph, meta

def bm_graph(retrieved: List[Retrieved], train_cases: List[Case], max_pred_actions: int):
    if not retrieved:
        return build_action_sequence_graph([])
    top_case = train_cases[retrieved[0].idx]
    return build_action_sequence_graph(top_case.solution_actions[:max_pred_actions])


def resolve_stopping_detector(cfg: Config, detector_map: Dict[str, object]) -> Optional[object]:
    """
    Resolve detector instance used by stopping policy.
    Returns None when stopping mode is alpha-only.
    """
    if cfg.stopping_mode == "alpha":
        return None
    det_name = _canonical_detector_name(cfg.stopping_detector)
    if det_name not in detector_map:
        raise RuntimeError(
            f"Unknown stopping_detector={cfg.stopping_detector}. Available: {list(detector_map.keys())}"
        )
    return detector_map[det_name]


def tune_alpha_on_validation(cfg: Config, val_cases: List[Case], train_cases: List[Case], retriever, sim_backend: TextSim) -> float:
    """
    Grid-search alpha on validation set to maximize Avg node-F1.
    Returns the best alpha.
    """
    if cfg.alpha_grid is None:
        # Good default grid for MultiWOZ
        grid = [round(x, 2) for x in np.arange(0.20, 0.71, 0.05)]
    else:
        grid = cfg.alpha_grid

    n = min(len(val_cases), cfg.alpha_max_cases)
    if n == 0:
        raise RuntimeError("No validation cases available for alpha tuning.")

    if _is_llm_method(cfg.reuse_method):
        logger.info("Skipping alpha tuning for reuse_method=%s (not alpha-driven).", cfg.reuse_method)
        return float(cfg.alpha)

    results = []
    for a in grid:
        node_f1s = []
        best_is = []

        for i in range(n):
            q = val_cases[i]
            retrieved = retriever.query(q.problem_text, top_k=cfg.top_k)

            # BM ignores alpha; we still compute for completeness
            if cfg.reuse_method == "bm":
                if not retrieved:
                    pred_graph = build_action_sequence_graph([])
                    best_i = 0
                else:
                    top_case = train_cases[retrieved[0].idx]
                    pred_graph = build_action_sequence_graph(top_case.solution_actions[:cfg.max_pred_actions])
                    best_i = 1
            else:
                pred_graph, meta = stable_matching_reuse_with_graph_merge(
                    query_case=q,
                    retrieved=retrieved,
                    case_base=train_cases,
                    alpha=a,
                    sim_backend=sim_backend,
                    max_pool_actions=cfg.max_pool_actions,
                    max_pred_actions=cfg.max_pred_actions,
                    lambda_complexity=cfg.lambda_complexity,
                    match_method=cfg.reuse_method,
                    debug=False
                )
                best_i = int(meta.get("best_i", 1))

            metrics = eval_solution_graph(pred_graph, q.solution_actions)
            node_f1s.append(metrics["node_f1"])
            best_is.append(best_i)

        avg_f1 = float(np.mean(node_f1s)) if node_f1s else 0.0
        avg_best_i = float(np.mean(best_is)) if best_is else 0.0
        results.append((a, avg_f1, avg_best_i))
        logger.info("[alpha-tune] alpha=%.2f | val Avg node-F1=%.4f | avg best_i=%.2f", a, avg_f1, avg_best_i)

    # pick best by Avg node-F1 (tie-break: smaller avg_best_i)
    results.sort(key=lambda t: (-t[1], t[2]))
    best_alpha, best_f1, best_bi = results[0]
    logger.info("[alpha-tune] BEST alpha=%.2f | val Avg node-F1=%.4f | avg best_i=%.2f", best_alpha, best_f1, best_bi)
    return float(best_alpha)


def _is_llm_method(reuse_method: str) -> bool:
    return reuse_method in {"llm_zero", "llm_fewshot", "llm_rag"}


def _extract_actions_from_text(raw: str) -> List[str]:
    raw = (raw or "").strip()
    if not raw:
        return []

    def _from_obj(obj) -> List[str]:
        if isinstance(obj, dict):
            acts = obj.get("actions")
            if isinstance(acts, list):
                return [str(x) for x in acts if isinstance(x, (str, int, float))]
            return []
        if isinstance(obj, list):
            return [str(x) for x in obj if isinstance(x, (str, int, float))]
        return []

    # 1) Direct JSON parse.
    try:
        acts = _from_obj(json.loads(raw))
        if acts:
            return acts
    except Exception:
        pass

    # 2) Try extracting largest JSON object/list from text.
    candidates = re.findall(r"(\{.*\}|\[.*\])", raw, flags=re.DOTALL)
    for c in candidates:
        try:
            acts = _from_obj(json.loads(c))
            if acts:
                return acts
        except Exception:
            continue

    # 3) Fallback: lines with action-like tokens.
    out = []
    for ln in raw.splitlines():
        t = ln.strip().strip("-*0123456789. ").strip()
        if ":" in t:
            out.append(t)
    return out


def _extract_action_ids_from_text(raw: str) -> List[int]:
    raw = (raw or "").strip()
    if not raw:
        return []

    def _ids_from_obj(obj) -> List[int]:
        if isinstance(obj, dict):
            xs = obj.get("action_ids")
            if not isinstance(xs, list):
                return []
            out = []
            for x in xs:
                try:
                    out.append(int(x))
                except Exception:
                    continue
            return out
        return []

    # 1) Direct parse
    try:
        ids = _ids_from_obj(json.loads(raw))
        if ids:
            return ids
    except Exception:
        pass

    # 2) Extract JSON object candidates
    candidates = re.findall(r"(\{.*\})", raw, flags=re.DOTALL)
    for c in candidates:
        try:
            ids = _ids_from_obj(json.loads(c))
            if ids:
                return ids
        except Exception:
            continue

    # 3) Fallback: scan for integers in text
    nums = re.findall(r"-?\d+", raw)
    out = []
    for n in nums:
        try:
            out.append(int(n))
        except Exception:
            continue
    return out


def _ollama_chat(cfg: Config, messages: List[Dict[str, str]]) -> str:
    url = cfg.ollama_host.rstrip("/") + "/api/chat"
    payload = {
        "model": cfg.ollama_model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": float(cfg.llm_temperature),
            "num_predict": int(cfg.llm_max_tokens),
            "seed": int(cfg.llm_seed),
        },
    }
    body = json.dumps(payload).encode("utf-8")
    req = urlrequest.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlrequest.urlopen(req, timeout=int(cfg.ollama_timeout_s)) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urlerror.URLError as e:
        raise RuntimeError(f"Ollama request failed: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Ollama response parse failed: {e}") from e

    msg = data.get("message", {}) if isinstance(data, dict) else {}
    content = msg.get("content", "") if isinstance(msg, dict) else ""
    return str(content or "")


def _select_static_fewshots(train_cases: List[Case], k: int, query_case: Optional[Case] = None) -> List[Case]:
    if k <= 0:
        return []
    cases = sorted(train_cases, key=lambda c: c.case_id)
    out = []
    for c in cases:
        if query_case is not None and c.case_id == query_case.case_id:
            continue
        out.append(c)
        if len(out) >= k:
            break
    return out


def _run_ollama_baseline(
    cfg: Config,
    query_case: Case,
    train_cases: List[Case],
    retrieved: List[Retrieved],
) -> Tuple[object, Dict[str, object]]:
    method = cfg.reuse_method
    fewshots: List[Case] = []
    rag_cases: List[Case] = []

    if method == "llm_fewshot":
        fewshots = _select_static_fewshots(train_cases, cfg.llm_fewshot_k, query_case=query_case)
    elif method == "llm_rag":
        top = retrieved[: max(0, int(cfg.llm_rag_k))]
        rag_cases = [train_cases[r.idx] for r in top if 0 <= int(r.idx) < len(train_cases)]

    # Build constrained candidate action set.
    # - llm_rag: actions from retrieved neighbours
    # - llm_fewshot: actions from few-shot examples
    # - llm_zero: most frequent actions from train case base
    candidate_actions: List[str] = []
    if method == "llm_rag":
        seen = set()
        for c in rag_cases:
            for a in c.solution_actions:
                if a not in seen:
                    seen.add(a)
                    candidate_actions.append(a)
    elif method == "llm_fewshot":
        seen = set()
        for c in fewshots:
            for a in c.solution_actions:
                if a not in seen:
                    seen.add(a)
                    candidate_actions.append(a)
    else:
        counts: Dict[str, int] = {}
        for c in train_cases:
            for a in c.solution_actions:
                counts[a] = counts.get(a, 0) + 1
        top_n = max(50, int(cfg.max_pool_actions))
        candidate_actions = [a for a, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n]]

    if not candidate_actions:
        candidate_actions = list(
            dict.fromkeys([a for c in train_cases[: max(1, min(20, len(train_cases)))] for a in c.solution_actions])
        )

    if str(cfg.prompt_style).lower() == "onet":
        system_msg = (
            "You are an occupational analysis assistant. "
            "Given a job profile and task list, select skills required to complete those tasks. "
            "Choose actions ONLY from the provided candidate list. "
            "Output ONLY valid JSON with key 'action_ids' whose value is an ordered list of integer IDs. "
            "Do not add explanations."
        )
        user_parts = []
        user_parts.append("Task: select required skills for the given job/tasks.")
        user_parts.append(f"Job Profile: {query_case.problem_text}")
        user_parts.append("Tasks (needs): " + json.dumps(query_case.needs))
        user_parts.append(f"Return at most {int(cfg.max_pred_actions)} skills.")
    else:
        system_msg = (
            "You are a dialogue policy action planner. "
            "Choose actions ONLY from the provided candidate list. "
            "Output ONLY valid JSON with key 'action_ids' whose value is an ordered list of integer IDs. "
            "Do not add explanations."
        )
        user_parts = []
        user_parts.append("Task: predict ordered system action labels.")
        user_parts.append(f"Problem: {query_case.problem_text}")
        user_parts.append("Needs: " + json.dumps(query_case.needs))
        user_parts.append(f"Return at most {int(cfg.max_pred_actions)} actions.")
    user_parts.append("Candidate actions (id -> action):")
    for i, a in enumerate(candidate_actions):
        user_parts.append(f"{i}: {a}")

    if fewshots:
        user_parts.append("Few-shot examples:")
        for i, ex in enumerate(fewshots, start=1):
            if str(cfg.prompt_style).lower() == "onet":
                user_parts.append(f"Example {i} Job Profile: {ex.problem_text}")
                user_parts.append(f"Example {i} Tasks: {json.dumps(ex.needs)}")
            else:
                user_parts.append(f"Example {i} Problem: {ex.problem_text}")
                user_parts.append(f"Example {i} Needs: {json.dumps(ex.needs)}")
            ex_ids = []
            idx_map = {a: j for j, a in enumerate(candidate_actions)}
            for a in ex.solution_actions[:cfg.max_pred_actions]:
                if a in idx_map:
                    ex_ids.append(idx_map[a])
            if str(cfg.prompt_style).lower() == "onet":
                user_parts.append(f"Example {i} Skill IDs: {json.dumps(ex_ids)}")
            else:
                user_parts.append(f"Example {i} Action IDs: {json.dumps(ex_ids)}")

    if rag_cases:
        user_parts.append("Retrieved relevant cases:")
        for i, ex in enumerate(rag_cases, start=1):
            if str(cfg.prompt_style).lower() == "onet":
                user_parts.append(f"Neighbour {i} Job Profile: {ex.problem_text}")
            else:
                user_parts.append(f"Neighbour {i} Problem: {ex.problem_text}")
            ex_ids = []
            idx_map = {a: j for j, a in enumerate(candidate_actions)}
            for a in ex.solution_actions[:cfg.max_pred_actions]:
                if a in idx_map:
                    ex_ids.append(idx_map[a])
            if str(cfg.prompt_style).lower() == "onet":
                user_parts.append(f"Neighbour {i} Skill IDs: {json.dumps(ex_ids)}")
            else:
                user_parts.append(f"Neighbour {i} Action IDs: {json.dumps(ex_ids)}")

    user_parts.append('Output JSON only, e.g. {"action_ids":[3,7,2]}')
    prompt = "\n".join(user_parts)

    raw = _ollama_chat(
        cfg,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ],
    )
    ids = _extract_action_ids_from_text(raw)
    actions: List[str] = []
    seen = set()
    for idx in ids:
        if 0 <= int(idx) < len(candidate_actions):
            a = candidate_actions[int(idx)]
            if a not in seen:
                seen.add(a)
                actions.append(a)
        if len(actions) >= int(cfg.max_pred_actions):
            break

    # Backward compatibility: if model ignored IDs and emitted labels, try strict
    # matching against constrained candidates only.
    if not actions:
        raw_labels = _extract_actions_from_text(raw)
        cand_map = {a.lower().strip(): a for a in candidate_actions}
        for x in raw_labels:
            t = str(x).strip().lower()
            if t in cand_map:
                a = cand_map[t]
                if a not in seen:
                    seen.add(a)
                    actions.append(a)
            if len(actions) >= int(cfg.max_pred_actions):
                break

    # Safe fallback when decode fails.
    if not actions:
        if method == "llm_rag" and rag_cases:
            for a in rag_cases[0].solution_actions:
                if a not in seen:
                    seen.add(a)
                    actions.append(a)
                if len(actions) >= int(cfg.max_pred_actions):
                    break
        else:
            for a in candidate_actions:
                if a not in seen:
                    seen.add(a)
                    actions.append(a)
                if len(actions) >= int(cfg.max_pred_actions):
                    break

    pred_graph = build_action_sequence_graph(actions)
    meta = {
        "final_score": None,
        "final_stop_score": None,
        "best_i": 0,
        "matched_actions": actions,
        "merged_order": actions,
    }
    return pred_graph, meta


def predict_graph_for_case(
    cfg: Config,
    q: Case,
    retrieved: List[Retrieved],
    train_cases: List[Case],
    sim_backend: TextSim,
    detector_map: Optional[Dict[str, object]] = None,
):
    """
    Runs the configured reuse method to produce a predicted graph.
    """
    if _is_llm_method(cfg.reuse_method):
        return _run_ollama_baseline(cfg, q, train_cases, retrieved)

    if cfg.reuse_method == "bm":
        if not retrieved:
            g = build_action_sequence_graph([])
            return g, {
                "final_score": None,
                "final_stop_score": None,
                "best_i": 0,
                "matched_actions": g.node_labels(),
                "merged_order": g.node_labels(),
            }
        top_case = train_cases[retrieved[0].idx]
        g = build_action_sequence_graph(top_case.solution_actions[:cfg.max_pred_actions])
        return g, {
            "final_score": None,
            "final_stop_score": None,
            "best_i": 1,
            "matched_actions": g.node_labels(),
            "merged_order": g.node_labels(),
        }

    stop_detector = None
    stop_threshold = 0.0
    if cfg.stopping_mode != "alpha":
        if detector_map is None:
            raise RuntimeError("detector_map is required when stopping_mode != 'alpha'.")
        stop_detector = resolve_stopping_detector(cfg, detector_map)
        stop_threshold = float(cfg.detector_thresholds.get(_canonical_detector_name(cfg.stopping_detector), 0.0))

    return stable_matching_reuse_with_graph_merge(
        query_case=q,
        retrieved=retrieved,
        case_base=train_cases,
        alpha=cfg.alpha,
        sim_backend=sim_backend,
        max_pool_actions=cfg.max_pool_actions,
        max_pred_actions=cfg.max_pred_actions,
        lambda_complexity=cfg.lambda_complexity,
        match_method=cfg.reuse_method,
        debug=False,
        stopping_mode=cfg.stopping_mode,
        stop_detector=stop_detector,
        stop_threshold=stop_threshold,
    )


def compute_detector_scores_on_cases(
    cases: List[Case],
    train_cases: List[Case],
    retriever,
    sim_backend: TextSim,
    detectors,
    cfg: Config,
    detector_map: Optional[Dict[str, object]] = None,
) -> Dict[str, Dict[str, List]]:
    """
    For each query case in `cases`, run reuse once to produce pred_graph,
    then compute detector scores + true_fail based on node_f1 < low_quality_cut.
    Returns:
      {detector_name: {"scores": [...], "true_fail": [...]}}
    """
    out = {d.name: {"scores": [], "true_fail": []} for d in detectors}

    n = min(len(cases), cfg.max_calib_cases)
    for i in range(n):
        q = cases[i]
        retrieved = retriever.query(q.problem_text, top_k=cfg.top_k)

        pred_graph, _ = predict_graph_for_case(cfg, q, retrieved, train_cases, sim_backend, detector_map=detector_map)
        metrics = eval_solution_graph(pred_graph, q.solution_actions)
        true_fail = (metrics["node_f1"] < cfg.low_quality_cut)

        for d in detectors:
            s = d.score(q, pred_graph, retrieved, train_cases)
            out[d.name]["scores"].append(float(s))
            out[d.name]["true_fail"].append(bool(true_fail))

    return out


def thresholds_from_percentile(score_dict: Dict[str, List[float]], flag_rate: float) -> Dict[str, float]:
    """
    For each detector, set threshold at the `flag_rate` percentile from the bottom.
    Example: flag_rate=0.10 => threshold=10th percentile (flag lowest 10% scores).
    """
    thr = {}
    q = 100.0 * float(flag_rate)
    for name, scores in score_dict.items():
        if not scores:
            thr[name] = 0.0
        else:
            thr[name] = float(np.percentile(np.array(scores, dtype=float), q))
    return thr


def thresholds_from_val_tuning(
    det_data: Dict[str, Dict[str, List]],
    cfg: Config,
) -> Dict[str, float]:
    """
    Tune each detector threshold on validation data to maximize failure-F1
    (predicting true_fail = node_f1 < low_quality_cut).

    We do a simple 1D search per detector over `tune_grid_size` thresholds
    sampled from the score quantiles.
    """
    def _prf(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f = (2*p*r/(p+r)) if (p+r) else 0.0
        return p, r, f

    tuned = {}
    for name, rec in det_data.items():
        scores = np.array(rec["scores"], dtype=float)
        true_fail = np.array(rec["true_fail"], dtype=bool)

        if len(scores) == 0:
            tuned[name] = 0.0
            continue

        # candidate thresholds = quantiles over observed scores
        qs = np.linspace(0.0, 1.0, cfg.tune_grid_size)
        cand = np.quantile(scores, qs)

        best_thr = float(cand[0])
        best_f1 = -1.0

        for t in cand:
            pred_fail = (scores < t)
            tp = int(np.sum(pred_fail & true_fail))
            fp = int(np.sum(pred_fail & ~true_fail))
            fn = int(np.sum(~pred_fail & true_fail))
            _, _, f1 = _prf(tp, fp, fn)
            if f1 > best_f1:
                best_f1 = f1
                best_thr = float(t)

        tuned[name] = best_thr

    return tuned


def main(
    cfg: Config,
    train_cases: Optional[List[Case]] = None,
    test_cases: Optional[List[Case]] = None,
    val_cases: Optional[List[Case]] = None,
):
    if train_cases is None or test_cases is None:
        loader = MultiWOZ22Loader(cfg.multiwoz_root)
        builder = MultiWOZ22CaseBuilder(include_state=True)

        logger.info("Loading MultiWOZ splits...")
        train_dialogs = loader.load_split("train")
        test_dialogs = loader.load_split("test")

        logger.info("Building cases...")
        train_cases = builder.build_cases(train_dialogs, "train")
        test_cases = builder.build_cases(test_dialogs, "test")
        logger.info("Train cases=%d | Test cases=%d", len(train_cases), len(test_cases))

        # Optional validation split for threshold tuning (disabled in CBR mode).
        if cfg.cbr_mode:
            val_cases = []
            logger.info("CBR mode enabled: skipping external val/dev split loading.")
        else:
            try:
                try:
                    val_dialogs = loader.load_split("val")
                except Exception:
                    val_dialogs = loader.load_split("dev")
                val_cases = builder.build_cases(val_dialogs, "val")
                logger.info("Val cases=%d", len(val_cases))
            except Exception:
                val_cases = []
                logger.warning("No val/dev split found; val_tune will not work.")
    elif val_cases is None:
        val_cases = []

    if cfg.cbr_mode:
        val_cases = []

    train_cases = _sample_train_case_base(train_cases, cfg)

    if cfg.cbr_mode and cfg.cbr_holdout_test_size is not None:
        train_cases, test_cases = _split_cbr_holdout_train_test(train_cases, cfg)
        logger.info(
            "CBR holdout active: train(casebase)=%d | test=%d",
            len(train_cases), len(test_cases)
        )
    else:
        logger.info("Effective train case-base size=%d", len(train_cases))

    # Retriever
    retriever = _build_retriever(cfg)

    logger.info("Fitting retriever: %s", retriever.__class__.__name__)
    retriever.fit(train_cases)

    # Pluggable affinity backend for stable matching (embedding/condprob/pmi).
    sim_backend = TextSim(
        prefer_sbert=True,
        affinity_method=cfg.affinity_method,
        condprob_smoothing=cfg.condprob_smoothing,
        pmi_smoothing=cfg.pmi_smoothing,
    )
    sim_backend.fit(train_cases)
    logger.info("Affinity backend=%s", cfg.affinity_method)

    # --- Alpha / Lambda tuning ---
    # In CBR mode, tune internally with K-fold CV over case base.
    # Otherwise, preserve existing val/dev tuning behavior.
    if cfg.tune_alpha_on_val and cfg.stopping_mode in ("alpha", "hybrid"):
        if _is_llm_method(cfg.reuse_method):
            logger.info("Skipping alpha/lambda tuning for reuse_method=%s.", cfg.reuse_method)
        elif cfg.cbr_mode:
            cfg.alpha, cfg.lambda_complexity = tune_hparams_on_casebase_cv(cfg, train_cases)
        elif len(val_cases) == 0:
            logger.warning("tune_alpha_on_val=True but no val split available. Using cfg.alpha=%.2f", cfg.alpha)
        else:
            cfg.alpha = tune_alpha_on_validation(cfg, val_cases, train_cases, retriever, sim_backend)
    elif cfg.tune_alpha_on_val and cfg.stopping_mode not in ("alpha", "hybrid"):
        logger.info("Skipping alpha tuning because stopping_mode=%s does not use alpha.", cfg.stopping_mode)

    detectors = [
        NeighbourhoodConsistencyDetector(),
        CaseAlignmentIntegrityDetector(),
        EmbeddingNoveltyDetector(),
        NeighbourhoodTransformationAlignmentDetector(),
    ]
    for d in detectors:
        d.fit(train_cases)
    det_map = {d.name: d for d in detectors}

    # Failure detectors
    # if cfg.detector_thresholds is None:
    #     # simple defaults; tune later via validation or percentiles
    #     cfg.detector_thresholds = {"NCD": 0.12, "WNCD": 0.12, "END": 0.35, "NTAD": 0.60, "ASFD": 0.65}

    # Threshold selection
    if cfg.detector_thresholds is None:
        cfg.detector_thresholds = {"NCD": 0.12, "WNCD": 0.12, "END": 0.35, "NTAD": 0.60, "ASFD": 0.65}
    else:
        cfg.detector_thresholds = {
            _canonical_detector_name(k): float(v) for k, v in cfg.detector_thresholds.items()
        }

    if cfg.threshold_mode == "fixed":
        logger.info("Threshold mode: fixed (user-provided or defaults)")

    elif cfg.threshold_mode == "percentile":
        logger.info("Threshold mode: percentile (flag_rate=%.2f) using validation if available else training-subset", cfg.flag_rate)

        calib_source = val_cases if len(val_cases) > 0 else train_cases
        det_data = compute_detector_scores_on_cases(
            cases=calib_source,
            train_cases=train_cases,
            retriever=retriever,
            sim_backend=sim_backend,
            detectors=detectors,
            cfg=cfg,
            detector_map=det_map,
        )
        score_dict = {name: det_data[name]["scores"] for name in det_data.keys()}
        cfg.detector_thresholds = thresholds_from_percentile(score_dict, cfg.flag_rate)

    elif cfg.threshold_mode == "val_tune":
        if len(val_cases) == 0:
            # In CBR mode, or when val is absent, use train case base for detector threshold tuning.
            calib_source = train_cases
            logger.warning("threshold_mode='val_tune' but no val split available; tuning detector thresholds on train case base.")
        else:
            calib_source = val_cases
            logger.info("Threshold mode: val_tune (grid=%d) on val split", cfg.tune_grid_size)

        det_data = compute_detector_scores_on_cases(
            cases=calib_source,
            train_cases=train_cases,
            retriever=retriever,
            sim_backend=sim_backend,
            detectors=detectors,
            cfg=cfg,
            detector_map=det_map,
        )
        cfg.detector_thresholds = thresholds_from_val_tuning(det_data, cfg)

    else:
        raise ValueError(f"Unknown threshold_mode: {cfg.threshold_mode}")

    logger.info("Detector thresholds: %s", {k: round(v, 4) for k, v in cfg.detector_thresholds.items()})

    det_records = {d.name: {"pred_fail": [], "true_fail": []} for d in detectors}
    if cfg.enable_asfd:
        det_records["ASFD"] = {"pred_fail": [], "true_fail": []}

    # Run eval
    node_f1s = []
    edge_f1s = []
    neighbours_used = []
    node_f1s_int = []
    edge_f1s_int = []
    coverage_int = []        # 1 if not abstained, else 0
    fallback_rate = []       # 1 if fallback used, else 0

    for idx, q in enumerate(test_cases):
        retrieved = retriever.query(q.problem_text, top_k=cfg.top_k)

        debug = cfg.debug_trace and (idx < cfg.debug_first_n)
        if debug:
            logger.info("=== DEBUG CASE %d ===", idx)
            logger.info("Query case_id=%s", q.case_id)
            logger.info("Problem text: %s", q.problem_text)
            logger.info("Needs: %s", q.needs)
            logger.info("Gold actions: %s", q.solution_actions)
            logger.info("Retrieved neighbours (idx,score): %s", [(r.idx, round(r.score, 3)) for r in retrieved])

        # --- Predict ---
        pred_graph, meta = predict_graph_for_case(
            cfg=cfg,
            q=q,
            retrieved=retrieved,
            train_cases=train_cases,
            sim_backend=sim_backend,
            detector_map=det_map,
        )

        if debug:
            if meta.get("final_score") is None:
                logger.info("Final stable score=None best_i=%d", meta.get("best_i", 1))
            else:
                logger.info("Final stable score=%.4f best_i=%d", float(meta["final_score"]), int(meta["best_i"]))
            logger.info("Matched actions: %s", meta["matched_actions"])
            logger.info("Merged order: %s", meta["merged_order"])
            logger.info("Pred graph edges: %s", pred_graph.edges_labeled())

        # --- Integrated failure detection (optional) ---
        integrated_graph = pred_graph
        integrated_used_fallback = False
        integrated_abstained = False

        if cfg.integrate_detector:
            det_name = _canonical_detector_name(cfg.integration_detector)
            thr = float(cfg.detector_thresholds.get(det_name, 0.0))

            # compute chosen detector score
            # (use the same detector instances you already created)
            if cfg.enable_asfd and det_name == "ASFD":
                # ASFD needs callback; keep your existing ASFD code path if you want it here later
                raise RuntimeError("ASFD integration not implemented here (you can add it similarly).")

            if det_name not in det_map:
                raise RuntimeError(f"Unknown integration_detector={det_name}. Available: {list(det_map.keys())}")

            s = det_map[det_name].score(q, pred_graph, retrieved, train_cases)
            is_fail = (s < thr)

            if is_fail:
                if cfg.integration_mode == "fallback_bm":
                    integrated_graph = bm_graph(retrieved, train_cases, cfg.max_pred_actions)
                    integrated_used_fallback = True
                elif cfg.integration_mode == "selective":
                    integrated_abstained = True
                else:
                    raise ValueError(f"Unknown integration_mode: {cfg.integration_mode}")
                
        # --- Evaluate reuse quality ---
        # Base metrics (no integration)
        metrics = eval_solution_graph(pred_graph, q.solution_actions)

        # Integrated metrics (with detector)
        if cfg.integrate_detector and (not integrated_abstained):
            metrics_int = eval_solution_graph(integrated_graph, q.solution_actions)
        else:
            metrics_int = None
        node_f1s.append(metrics["node_f1"])
        edge_f1s.append(metrics["edge_f1"])
        neighbours_used.append(int(meta.get("best_i", 0)))

        true_fail = (metrics["node_f1"] < cfg.low_quality_cut)

        # Store integrated results
        if cfg.integrate_detector:
            if metrics_int is None:
                coverage_int.append(0)
                fallback_rate.append(0)
            else:
                coverage_int.append(1)
                fallback_rate.append(1 if integrated_used_fallback else 0)
                node_f1s_int.append(metrics_int["node_f1"])
                edge_f1s_int.append(metrics_int["edge_f1"])

        # --- Failure detection scores -> fail/pass ---
        for d in detectors:
            s = d.score(q, pred_graph, retrieved, train_cases)
            thr = float(cfg.detector_thresholds.get(d.name, 0.0))
            pred_fail = (s < thr)
            det_records[d.name]["pred_fail"].append(pred_fail)
            det_records[d.name]["true_fail"].append(true_fail)

        # --- Optional ASFD: expensive leave-one-out stability ---
        if cfg.enable_asfd:
            def _reuse_with_subretrieval(subretrieved):
                g, _ = predict_graph_for_case(
                    cfg=cfg,
                    q=q,
                    retrieved=subretrieved,
                    train_cases=train_cases,
                    sim_backend=sim_backend,
                    detector_map=det_map,
                )
                return g

            asfd = AlignmentStabilityDetector(reuse_callback=_reuse_with_subretrieval)
            s = asfd.score(q, pred_graph, retrieved, train_cases)
            thr = float(cfg.detector_thresholds.get("ASFD", 0.0))
            pred_fail = (s < thr)
            det_records["ASFD"]["pred_fail"].append(pred_fail)
            det_records["ASFD"]["true_fail"].append(true_fail)

        if (idx + 1) % 500 == 0:
            logger.info("Processed %d/%d", idx + 1, len(test_cases))

    # --- Overall results ---
    logger.info("=== RESULTS ===")
    logger.info(
        "Reuse=%s Retriever=%s topK=%d affinity=%s stopping=%s alpha=%.2f lambda=%.3f",
        cfg.reuse_method, cfg.retriever, cfg.top_k, cfg.affinity_method, cfg.stopping_mode, cfg.alpha, cfg.lambda_complexity
    )
    logger.info("Avg node-F1 (action set): %.4f", float(np.mean(node_f1s)))
    logger.info("Avg edge-F1 (sequence edges): %.4f", float(np.mean(edge_f1s)))
    logger.info("Avg neighbours used: %.2f", float(np.mean(neighbours_used)) if neighbours_used else 0.0)

    # --- Failure detection comparison ---
    logger.info("=== FAILURE DETECTION (true_fail: node-F1 < %.2f) ===", cfg.low_quality_cut)
    for name, rec in det_records.items():
        stats = eval_detector(rec["pred_fail"], rec["true_fail"], node_f1s)
        thr = float(cfg.detector_thresholds.get(name, 0.0))
        logger.info(
            "Detector=%s thr=%.3f | P=%.3f R=%.3f F1=%.3f | passF1=%.3f failF1=%.3f | flag=%.2f",
            name, thr,
            stats["precision"], stats["recall"], stats["f1"],
            stats["avg_quality_pass"], stats["avg_quality_fail"],
            stats["flag_rate"]
        )

    if cfg.integrate_detector:
        cov = float(np.mean(coverage_int)) if coverage_int else 0.0
        fb = float(np.mean(fallback_rate)) if fallback_rate else 0.0

        logger.info("=== INTEGRATED RESULTS ===")
        logger.info(
            "Integration=%s detector=%s thr=%.3f | coverage=%.2f fallback_rate=%.2f",
            cfg.integration_mode,
            cfg.integration_detector,
            float(cfg.detector_thresholds.get(cfg.integration_detector, 0.0)),
            cov,
            fb
        )

        if cfg.integration_mode == "selective":
            # average over kept cases only
            logger.info("Selective Avg node-F1 (kept only): %.4f", float(np.mean(node_f1s_int)) if node_f1s_int else 0.0)
            logger.info("Selective Avg edge-F1 (kept only): %.4f", float(np.mean(edge_f1s_int)) if edge_f1s_int else 0.0)
        else:
            # fallback_bm keeps coverage ~1.0; still average over all test cases:
            # we stored only non-abstained, so to get "overall", just mean of node_f1s_int should match cov=1.0.
            logger.info("Fallback Avg node-F1: %.4f", float(np.mean(node_f1s_int)) if node_f1s_int else 0.0)
            logger.info("Fallback Avg edge-F1: %.4f", float(np.mean(edge_f1s_int)) if edge_f1s_int else 0.0)

    # Return compact metrics so sweep mode can print a single summary table.
    out = {
        "node_f1": float(np.mean(node_f1s)) if node_f1s else 0.0,
        "edge_f1": float(np.mean(edge_f1s)) if edge_f1s else 0.0,
        "avg_neighbours_used": float(np.mean(neighbours_used)) if neighbours_used else 0.0,
    }
    if cfg.integrate_detector:
        out["int_node_f1"] = float(np.mean(node_f1s_int)) if node_f1s_int else 0.0
        out["int_edge_f1"] = float(np.mean(edge_f1s_int)) if edge_f1s_int else 0.0
        out["coverage"] = float(np.mean(coverage_int)) if coverage_int else 0.0
        out["fallback_rate"] = float(np.mean(fallback_rate)) if fallback_rate else 0.0
    return out


def run_option_sweep(
    base_cfg: Config,
    affinity_methods: List[str],
    stopping_modes: List[str],
    stopping_detectors: List[str],
    reuse_methods: Optional[List[str]] = None,
    results_path: str = "results/sweep_results.csv",
    max_workers: Optional[int] = None,
    skip_first_runs: int = 0,
):
    """
    Run a grid of experiment options from one entrypoint.
    """
    if reuse_methods is None:
        reuse_methods = [base_cfg.reuse_method]

    run_specs = []
    run_cfg_by_id = {}
    run_idx = 0
    for reuse_method, affinity_method, stopping_mode in product(reuse_methods, affinity_methods, stopping_modes):
        # stopping_detector is ignored by alpha-only mode; keep one placeholder pass.
        detector_choices = [base_cfg.stopping_detector] if stopping_mode == "alpha" else stopping_detectors
        for stopping_detector in detector_choices:
            run_idx += 1
            cfg = replace(
                base_cfg,
                reuse_method=reuse_method,
                affinity_method=affinity_method,
                stopping_mode=stopping_mode,
                stopping_detector=stopping_detector,
                # Alpha is used as a stopping criterion in alpha/hybrid modes.
                tune_alpha_on_val=(base_cfg.tune_alpha_on_val and stopping_mode in ("alpha", "hybrid")),
            )
            run_specs.append((run_idx, cfg))
            run_cfg_by_id[run_idx] = cfg

    total_runs = len(run_specs)
    skip_first_runs = max(0, int(skip_first_runs))
    if skip_first_runs > 0:
        logger.info("Skipping first %d/%d runs", skip_first_runs, total_runs)
        run_specs = run_specs[skip_first_runs:]
    if not run_specs:
        logger.warning(
            "No runs left after skipping (skip_first_runs=%d, total_runs=%d).",
            skip_first_runs,
            total_runs,
        )
        return

    if max_workers is None:
        # Conservative default: parallelize, but avoid oversubscription by heavy retrievers.
        cpu_count = os.cpu_count() or 1
        max_workers = max(1, min(4, cpu_count))
    max_workers = max(1, int(max_workers))
    logger.info("Sweep configured for %d runs with max_workers=%d", len(run_specs), max_workers)

    out_dir = os.path.dirname(results_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fieldnames = [
        "run", "reuse", "affinity", "stopping", "detector",
        "node_f1", "edge_f1", "avg_neighbours_used", "int_node_f1", "coverage", "status", "error"
    ]
    rows = []
    with open(results_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        f.flush()
        os.fsync(f.fileno())

        if max_workers == 1:
            for run_idx, cfg in run_specs:
                logger.info(
                    "=== SWEEP RUN %d | reuse=%s affinity=%s stopping=%s detector=%s ===",
                    run_idx, cfg.reuse_method, cfg.affinity_method, cfg.stopping_mode, cfg.stopping_detector
                )
                row = _execute_sweep_run(run_idx, cfg)
                rows.append(row)
                writer.writerow(row)
                f.flush()
                os.fsync(f.fileno())
                logger.info(
                    "Completed run %d/%d (%s)",
                    len(rows), len(run_specs), f"run={row['run']} status={row['status']}"
                )
        else:
            with ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=_init_worker_data,
                initargs=(base_cfg.multiwoz_root,)
            ) as executor:
                future_to_run = {
                    executor.submit(_execute_sweep_run, run_idx, cfg): run_idx
                    for run_idx, cfg in run_specs
                }
                for done_idx, future in enumerate(as_completed(future_to_run), start=1):
                    run_id = future_to_run[future]
                    try:
                        row = future.result()
                    except Exception as e:
                        cfg = run_cfg_by_id[run_id]
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
                    writer.writerow(row)
                    f.flush()
                    os.fsync(f.fileno())
                    logger.info(
                        "Completed run %d/%d (%s)",
                        done_idx, len(run_specs), f"run={row['run']} status={row['status']}"
                    )

    # Final concise summary with one row per run.
    rows.sort(key=lambda r: int(r["run"]))
    logger.info("=== SWEEP SUMMARY (%d runs) ===", len(rows))
    logger.info("run | reuse | affinity | stopping | detector | node_f1 | edge_f1 | avg_neighbours_used | int_node_f1 | coverage | status")
    for r in rows:
        logger.info(
            "%d | %s | %s | %s | %s | %.4f | %.4f | %.2f | %.4f | %.2f | %s",
            r["run"], r["reuse"], r["affinity"], r["stopping"], r["detector"],
            r["node_f1"], r["edge_f1"], r["avg_neighbours_used"], r["int_node_f1"], r["coverage"], r["status"]
        )

    logger.info("Sweep results written incrementally to %s", results_path)


def _execute_sweep_run(run_idx: int, cfg: Config) -> Dict[str, object]:
    logger.info(
        "=== SWEEP RUN %d | reuse=%s affinity=%s stopping=%s detector=%s ===",
        run_idx, cfg.reuse_method, cfg.affinity_method, cfg.stopping_mode, cfg.stopping_detector
    )
    try:
        if _WORKER_SHARED_DATA["train_cases"] is not None:
            metrics = main(
                cfg,
                train_cases=_WORKER_SHARED_DATA["train_cases"],
                test_cases=_WORKER_SHARED_DATA["test_cases"],
                val_cases=_WORKER_SHARED_DATA["val_cases"],
            )
        else:
            metrics = main(cfg)

        return {
            "run": run_idx,
            "reuse": cfg.reuse_method,
            "affinity": cfg.affinity_method,
            "stopping": cfg.stopping_mode,
            "detector": (cfg.stopping_detector if cfg.stopping_mode != "alpha" else "-"),
            "node_f1": metrics.get("node_f1", 0.0),
            "edge_f1": metrics.get("edge_f1", 0.0),
            "avg_neighbours_used": metrics.get("avg_neighbours_used", 0.0),
            "int_node_f1": metrics.get("int_node_f1", 0.0),
            "coverage": metrics.get("coverage", 0.0),
            "status": "ok",
            "error": "",
        }
    except Exception as e:
        logger.exception("Sweep run %d failed", run_idx)
        return {
            "run": run_idx,
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


if __name__ == "__main__":
    multiwoz_case_base_seed = 46  # fixed seed for reproducible case base sampling across runs

    cfg = Config(
        multiwoz_root="./MultiWOZ_2.2",
        retriever="sbert",      # falls back to tfidf if sentence-transformers is unavailable
        top_k=5,
        alpha=0.45,             # initial, will be overwritten if tuning enabled
        affinity_method="embedding_cosine",  # "embedding_cosine" | "condprob" | "pmi"
        stopping_mode="detector",  # "alpha" | "detector" | "hybrid"
        stopping_detector="NTAD",  # "NCD", "WNCD", "END", "NTAD"
        max_pool_actions=60,
        max_pred_actions=10,
        debug_trace=True,
        debug_first_n=3,
        reuse_method="carm",         # "bm", "gsa", "nda", "carm", "llm_zero", "llm_fewshot", "llm_rag"
        lambda_complexity=0.05,
        tune_alpha_on_val=True,
        alpha_max_cases=1500,
        # alpha_grid=[0.25, 0.35, 0.45, 0.55],  # optional custom grid
        low_quality_cut=0.2,
        detector_thresholds=None,  #{"NCD": 0.12, "WNCD": 0.12, "END": 0.35, "NTAD": 0.60, "ASFD": 0.65},
        enable_asfd=False,
        # Reproducible small case-base setup examples:
        case_base_size=550,
        case_base_seed=multiwoz_case_base_seed,
        case_base_stratified=True,
        case_base_ids_path=f"results/multiwoz_casebase_ids_n550_seed{multiwoz_case_base_seed}_strat.json",
        # CBR internal tuning mode (no val/dev usage):
        cbr_mode=True,
        cv_folds=5,
        tune_lambda_on_cv=True,
        cbr_holdout_test_size=50,
        cbr_holdout_ids_path=f"results/multiwoz_cbr_holdout_total550_test50_seed{multiwoz_case_base_seed}_strat.json",
    )

    # Toggle this to run a full options sweep instead of a single config.
    RUN_SWEEP = True
    SWEEP_SKIP_FIRST_RUNS = 0  # set >0 to skip this many runs when restarting
    if RUN_SWEEP:
        run_option_sweep(
            base_cfg=cfg,
            affinity_methods= ["condprob"],  #["embedding_cosine", "condprob", "pmi"],
            stopping_modes= ["alpha"],  #["alpha", "detector", "hybrid"],
            stopping_detectors= ["NTAD"],  #["NCD", "WNCD", "END", "NTAD"],
            reuse_methods= ["bm"],  #["bm", "gsa", "nda", "carm", "llm_zero", "llm_fewshot", "llm_rag"],
            results_path=f"results/sweep_results-multiwoz-all_reuse-condprob-500_cases-seed{multiwoz_case_base_seed}-bm.csv",
            max_workers=2,
            skip_first_runs=SWEEP_SKIP_FIRST_RUNS,
        )
    else:
        main(cfg)
