# detector_eval.py
from typing import Dict, List, Tuple
import numpy as np


def prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = (2*p*r/(p+r)) if (p+r) else 0.0
    return p, r, f


def eval_detector(pred_fail: List[bool], true_fail: List[bool], qualities: List[float]) -> Dict[str, float]:
    pf = np.array(pred_fail, dtype=bool)
    tf = np.array(true_fail, dtype=bool)
    q = np.array(qualities, dtype=float)

    tp = int(np.sum(pf & tf))
    fp = int(np.sum(pf & ~tf))
    fn = int(np.sum(~pf & tf))

    p, r, f = prf(tp, fp, fn)

    pass_mask = ~pf
    avg_pass = float(np.mean(q[pass_mask])) if np.any(pass_mask) else 0.0
    avg_fail = float(np.mean(q[pf])) if np.any(pf) else 0.0

    return {
        "precision": p,
        "recall": r,
        "f1": f,
        "avg_quality_pass": avg_pass,
        "avg_quality_fail": avg_fail,
        "flag_rate": float(np.mean(pf)),
        "tp": tp, "fp": fp, "fn": fn
    }