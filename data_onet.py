import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from graph_structures import Case


class ONetCaseBuilder:
    """
    Build generic CBR cases from O*NET JSON records.

    Expected mapping per record:
      - job_code -> case_id
      - x (or L) with {title, description} -> problem_text (structured)
      - L        -> needs
      - S        -> solution_actions
    """

    def __init__(
        self,
        lowercase: bool = True,
        dedupe_lists: bool = True,
        max_needs: Optional[int] = None,
        max_actions: Optional[int] = None,
    ):
        self.lowercase = bool(lowercase)
        self.dedupe_lists = bool(dedupe_lists)
        self.max_needs = max_needs
        self.max_actions = max_actions

    @staticmethod
    def _as_records(obj: Any) -> List[Dict[str, Any]]:
        if isinstance(obj, list):
            return [r for r in obj if isinstance(r, dict)]
        if isinstance(obj, dict):
            for k in ("records", "data", "items", "cases"):
                v = obj.get(k)
                if isinstance(v, list):
                    return [r for r in v if isinstance(r, dict)]
        return []

    def _norm_text(self, s: Any) -> str:
        t = str(s or "").strip()
        t = re.sub(r"\s+", " ", t)
        return t.lower() if self.lowercase else t

    def _norm_list(self, xs: Any, cap: Optional[int] = None) -> List[str]:
        if not isinstance(xs, list):
            return []
        out: List[str] = []
        seen = set()
        for x in xs:
            t = self._norm_text(x)
            if not t:
                continue
            if self.dedupe_lists:
                if t in seen:
                    continue
                seen.add(t)
            out.append(t)
            if cap is not None and len(out) >= int(cap):
                break
        return out

    @staticmethod
    def _infer_service(job_code: str) -> str:
        # O*NET major group (e.g., "11-1011.00" -> "11")
        if not job_code:
            return "onet"
        return job_code.split("-", 1)[0] if "-" in job_code else "onet"

    def load_records(self, path: str) -> List[Dict[str, Any]]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"O*NET dataset not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        recs = self._as_records(obj)
        if not recs:
            raise ValueError(f"No valid records found in: {path}")
        return recs

    def build_cases(self, records: List[Dict[str, Any]], split_name: str = "onet") -> List[Case]:
        cases: List[Case] = []
        for i, rec in enumerate(records):
            job_code = self._norm_text(rec.get("job_code", ""))
            case_key = job_code if job_code else f"row{i}"

            # Support both:
            # - New schema: x={title,description}, L=[needs]
            # - Variant schema: L={title,description}, x=[needs]
            raw_x = rec.get("x")
            raw_L = rec.get("L")

            title = ""
            description = ""
            needs_src = raw_L

            if isinstance(raw_x, dict) and ("title" in raw_x or "description" in raw_x):
                title = self._norm_text(raw_x.get("title", ""))
                description = self._norm_text(raw_x.get("description", ""))
                needs_src = raw_L
            elif isinstance(raw_L, dict) and ("title" in raw_L or "description" in raw_L):
                title = self._norm_text(raw_L.get("title", ""))
                description = self._norm_text(raw_L.get("description", ""))
                needs_src = raw_x
            else:
                # Backward compatibility (older plain x string)
                title = self._norm_text(raw_x)
                description = ""
                needs_src = raw_L

            # Structured problem text enables weighted field retrieval.
            if title or description:
                problem_text = f"title={title} | description={description}"
            else:
                problem_text = self._norm_text(raw_x)

            L = self._norm_list(needs_src, cap=self.max_needs)
            S = self._norm_list(rec.get("S", []), cap=self.max_actions)
            if not problem_text or not S:
                continue

            cases.append(
                Case(
                    case_id=f"{split_name}:{case_key}",
                    problem_text=problem_text,
                    needs=L,
                    service=self._infer_service(job_code),
                    solution_actions=S,
                )
            )
        return cases

    def load_cases(self, path: str, split_name: str = "onet") -> List[Case]:
        return self.build_cases(self.load_records(path), split_name=split_name)


def split_cases(
    cases: List[Case],
    train_size: int,
    test_size: int,
    seed: int = 42,
    stratified: bool = True,
) -> Tuple[List[Case], List[Case]]:
    """
    Deterministic split into train/test by absolute sizes.
    Returns (train_cases, test_cases).
    """
    n = len(cases)
    if train_size <= 0 or test_size <= 0:
        raise ValueError("train_size and test_size must be > 0")
    if train_size + test_size > n:
        raise ValueError(f"Requested train+test={train_size + test_size} > available={n}")

    cases = sorted(cases, key=lambda c: c.case_id)
    rng = np.random.default_rng(int(seed))

    if not stratified:
        idx = np.arange(n, dtype=int)
        rng.shuffle(idx)
        test_idx = set(int(i) for i in idx[:test_size].tolist())
        train = [cases[i] for i in range(n) if i not in test_idx][:train_size]
        test = [cases[i] for i in sorted(test_idx)]
        return train, test

    groups: Dict[str, List[int]] = {}
    for i, c in enumerate(cases):
        groups.setdefault(c.service, []).append(i)

    # Allocate test quota per service proportionally.
    services = sorted(groups.keys())
    exact = {s: (test_size * len(groups[s]) / float(n)) for s in services}
    quota = {s: min(len(groups[s]), int(np.floor(exact[s]))) for s in services}
    remain = test_size - sum(quota.values())
    order = sorted(services, key=lambda s: (exact[s] - np.floor(exact[s]), len(groups[s])), reverse=True)
    while remain > 0:
        progressed = False
        for s in order:
            if quota[s] < len(groups[s]):
                quota[s] += 1
                remain -= 1
                progressed = True
                if remain == 0:
                    break
        if not progressed:
            break

    test_idx_set = set()
    for s in services:
        idxs = np.array(groups[s], dtype=int)
        q = int(quota[s])
        if q <= 0:
            continue
        rng.shuffle(idxs)
        for j in idxs[:q]:
            test_idx_set.add(int(j))

    # Fill if quota rounding left short.
    if len(test_idx_set) < test_size:
        remaining = [i for i in range(n) if i not in test_idx_set]
        remaining = np.array(remaining, dtype=int)
        rng.shuffle(remaining)
        for j in remaining[: (test_size - len(test_idx_set))]:
            test_idx_set.add(int(j))

    test = [cases[i] for i in sorted(test_idx_set)]
    train = [cases[i] for i in range(n) if i not in test_idx_set][:train_size]
    return train, test
