# failure_detection.py
from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np

from graph_structures import Case, SolutionGraph
from retrieval import Retrieved

from sklearn.ensemble import IsolationForest

try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except Exception:
    _HAS_ST = False


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    uni = len(a | b)
    return inter / uni if uni else 0.0


class FailureDetector:
    """
    Base class: higher score = more confident it's OK (NOT failure).
    """
    name: str = "base"

    def fit(self, case_base: List[Case]) -> None:
        return

    def score(
        self,
        query: Case,
        pred_graph: SolutionGraph,
        retrieved: List[Retrieved],
        case_base: List[Case],
    ) -> float:
        raise NotImplementedError


class NeighbourhoodConsistencyDetector(FailureDetector):
    """
    NCFD: Score = mean Jaccard(node-set(pred), node-set(neighbour)).
    Domain-free and cheap.
    """
    name = "NCFD"

    def score(self, query, pred_graph, retrieved, case_base) -> float:
        pred = set(pred_graph.node_labels())
        if not retrieved:
            return 0.0
        sims = []
        for r in retrieved:
            nb = set(case_base[r.idx].solution_actions)
            sims.append(_jaccard(pred, nb))
        return float(np.mean(sims)) if sims else 0.0


class CaseAlignmentIntegrityDetector(FailureDetector):
    """
    CAI: similarity-weighted overlap with neighbours.
    Uses retrieval scores as weights.
    """
    name = "CAI"

    def score(self, query, pred_graph, retrieved, case_base) -> float:
        pred = set(pred_graph.node_labels())
        if not retrieved:
            return 0.0
        num, den = 0.0, 0.0
        for r in retrieved:
            w = max(float(r.score), 0.0)
            nb = set(case_base[r.idx].solution_actions)
            j = _jaccard(pred, nb)
            num += w * j
            den += w
        return (num / den) if den else 0.0


class AlignmentStabilityDetector(FailureDetector):
    """
    ASFD: Score = 1 - mean distance between pred and leave-one-out predictions.

    IMPORTANT: This calls your reuse function multiple times -> expensive.
    To keep it general, we accept a callback that re-runs reuse with a modified retrieved list.
    """
    name = "ASFD"

    def __init__(self, reuse_callback, max_loo: int = 5):
        """
        reuse_callback(retrieved_subset) -> SolutionGraph
        max_loo caps number of leave-one-out trials to avoid huge slowdown.
        """
        self.reuse_callback = reuse_callback
        self.max_loo = max_loo

    def score(self, query, pred_graph, retrieved, case_base) -> float:
        base_nodes = set(pred_graph.node_labels())
        if len(retrieved) <= 1:
            return 1.0  # trivially stable with <=1 neighbour

        # limit LOO runs for speed
        loo_indices = list(range(min(len(retrieved), self.max_loo)))
        dists = []
        for i in loo_indices:
            sub = retrieved[:i] + retrieved[i+1:]
            g2 = self.reuse_callback(sub)
            nodes2 = set(g2.node_labels())
            # distance = 1 - Jaccard
            dists.append(1.0 - _jaccard(base_nodes, nodes2))

        if not dists:
            return 1.0
        # higher score means more stable
        return float(1.0 - np.mean(dists))


class _EmbeddingBackbone:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if not _HAS_ST:
            raise RuntimeError("sentence-transformers is required for embedding-based detectors.")
        self.model = SentenceTransformer(model_name)
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def _encode_mean(self, texts: List[str]) -> np.ndarray:
        clean = [t for t in texts if t]
        if not clean:
            return np.zeros(self.dim, dtype=np.float32)
        emb = self.model.encode(clean, normalize_embeddings=True)
        v = np.mean(emb, axis=0).astype(np.float32)
        n = float(np.linalg.norm(v))
        if n > 0:
            v = v / n
        return v

    @staticmethod
    def _cos(u: np.ndarray, v: np.ndarray) -> float:
        if u.shape[0] == 0 or v.shape[0] == 0:
            return 0.0
        return float(np.dot(u, v))


class EmbeddingNoveltyDetector(FailureDetector):
    """
    EMND: Fit IsolationForest on language-model embeddings of training solution-actions.
    Score = sigmoid(decision_function) in (0,1): higher = less novel (more normal).
    """
    name = "EMND"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", random_state: int = 7):
        self.backbone = _EmbeddingBackbone(model_name=model_name)
        self.iforest = IsolationForest(n_estimators=200, random_state=random_state)
        self._fitted = False

    def fit(self, case_base: List[Case]) -> None:
        X = np.array([self.backbone._encode_mean(c.solution_actions) for c in case_base], dtype=np.float32)
        self.iforest.fit(X)
        self._fitted = True

    def score(self, query, pred_graph, retrieved, case_base) -> float:
        if not self._fitted:
            return 0.0
        X = np.array([self.backbone._encode_mean(pred_graph.node_labels())], dtype=np.float32)
        s = float(self.iforest.decision_function(X)[0])  # higher = more normal
        # map to (0,1) for easier thresholding
        return float(1.0 / (1.0 + np.exp(-s)))


class NeighbourhoodTransformationAlignmentDetector(FailureDetector):
    """
    NTAD: Learn a local neighbour transformation from problem-space similarity
    to solution-space similarity, then score query residual against that mapping.
    """
    name = "NTAD"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", gamma: float = 3.0, min_pairs: int = 3):
        self.backbone = _EmbeddingBackbone(model_name=model_name)
        self.gamma = float(gamma)
        self.min_pairs = int(min_pairs)
        self._fitted = False
        self._problem_case_emb = None
        self._solution_case_emb = None
        self._score_cache: Dict[Tuple[str, Tuple[str, ...], Tuple[int, ...]], float] = {}

    def fit(self, case_base: List[Case]) -> None:
        self._problem_case_emb = np.array(
            [self.backbone._encode_mean(c.needs) for c in case_base],
            dtype=np.float32
        )
        self._solution_case_emb = np.array(
            [self.backbone._encode_mean(c.solution_actions) for c in case_base],
            dtype=np.float32
        )
        self._fitted = True
        self._score_cache = {}

    def _fit_local_map(self, xs: np.ndarray, ys: np.ndarray) -> Tuple[float, float]:
        if len(xs) == 0:
            return 0.0, 0.0
        vx = float(np.var(xs))
        if vx < 1e-8:
            return 0.0, float(np.mean(ys))
        A = np.vstack([xs, np.ones_like(xs)]).T
        a, b = np.linalg.lstsq(A, ys, rcond=None)[0]
        return float(a), float(b)

    def score(self, query, pred_graph, retrieved, case_base) -> float:
        if not self._fitted or len(retrieved) < 2:
            return 0.0

        pred_actions = pred_graph.node_labels()
        if not pred_actions:
            return 0.0

        idxs = tuple(int(r.idx) for r in retrieved)
        cache_key = (str(query.case_id), tuple(pred_actions), idxs)
        if cache_key in self._score_cache:
            return self._score_cache[cache_key]

        P = self._problem_case_emb[list(idxs)]
        S = self._solution_case_emb[list(idxs)]

        xs, ys = [], []
        n = len(idxs)
        for i in range(n):
            for j in range(i + 1, n):
                xs.append(self.backbone._cos(P[i], P[j]))
                ys.append(self.backbone._cos(S[i], S[j]))

        if len(xs) < self.min_pairs:
            self._score_cache[cache_key] = 0.0
            return 0.0

        xs_arr = np.array(xs, dtype=np.float32)
        ys_arr = np.array(ys, dtype=np.float32)
        a, b = self._fit_local_map(xs_arr, ys_arr)

        q_problem = self.backbone._encode_mean(query.needs)
        q_solution = self.backbone._encode_mean(pred_actions)

        errs = []
        ws = []
        for i, r in enumerate(retrieved):
            x_qi = self.backbone._cos(q_problem, P[i])
            y_qi = self.backbone._cos(q_solution, S[i])
            y_hat = float(np.clip(a * x_qi + b, -1.0, 1.0))
            errs.append(abs(y_qi - y_hat))
            ws.append(max(float(r.score), 0.0))

        errs_arr = np.array(errs, dtype=np.float32)
        ws_arr = np.array(ws, dtype=np.float32)
        if float(np.sum(ws_arr)) > 0:
            mean_err = float(np.sum(ws_arr * errs_arr) / np.sum(ws_arr))
        else:
            mean_err = float(np.mean(errs_arr)) if len(errs_arr) else 1.0

        # Map error to confidence in (0, 1], where lower means likely failure.
        score = float(np.exp(-self.gamma * max(mean_err, 0.0)))
        self._score_cache[cache_key] = score
        return score
