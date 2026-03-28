# stable_matching.py
from typing import List, Dict, Tuple, Optional
import numpy as np
import logging
import math
from collections import defaultdict

try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except Exception:
    _HAS_ST = False

from sklearn.feature_extraction.text import TfidfVectorizer


logger = logging.getLogger("stable_matching")


class TextSim:
    """
    Pluggable affinity backend for stable matching preferences.

    Supported affinity methods:
      - embedding_cosine: SBERT/TF-IDF cosine (previous behaviour)
      - condprob: smoothed P(action|need) from train case co-occurrence
      - pmi: normalized PMI from train case co-occurrence
    """
    def __init__(
        self,
        prefer_sbert: bool = True,
        sbert_model: str = "all-MiniLM-L6-v2",
        affinity_method: str = "embedding_cosine",
        condprob_smoothing: float = 1.0,
        pmi_smoothing: float = 0.1,
    ):
        self.use_sbert = prefer_sbert and _HAS_ST
        self.model = SentenceTransformer(sbert_model) if self.use_sbert else None
        self.tfidf = None
        self.affinity_method = affinity_method
        self.condprob_smoothing = float(condprob_smoothing)
        self.pmi_smoothing = float(pmi_smoothing)

        # Statistical affinity state, populated by fit(case_base)
        self._fitted = False
        self._num_cases = 0
        self._need_df = defaultdict(int)
        self._action_df = defaultdict(int)
        self._pair_df = defaultdict(int)
        self._action_vocab = set()

    def fit(self, case_base) -> None:
        """
        Fit statistical affinity backends on training cases.
        Embedding cosine does not require fit.
        """
        if self.affinity_method == "embedding_cosine":
            self._fitted = True
            return

        self._num_cases = len(case_base)
        self._need_df.clear()
        self._action_df.clear()
        self._pair_df.clear()
        self._action_vocab = set()

        # Count binary co-occurrence within each case.
        for c in case_base:
            needs = list(dict.fromkeys(c.needs))
            actions = list(dict.fromkeys(c.solution_actions))
            for n in needs:
                self._need_df[n] += 1
            for a in actions:
                self._action_df[a] += 1
                self._action_vocab.add(a)
            for n in needs:
                for a in actions:
                    self._pair_df[(n, a)] += 1

        self._fitted = True

    def embed(self, texts: List[str]) -> np.ndarray:
        if self.use_sbert:
            return self.model.encode(texts, normalize_embeddings=True)
        # TF-IDF fallback (fit on the fly for the match set)
        self.tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
        X = self.tfidf.fit_transform(texts).astype(np.float32)
        return X.toarray()

    @staticmethod
    def cosine_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        A2 = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
        B2 = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
        return A2 @ B2.T

    def _ensure_fitted(self):
        if self.affinity_method != "embedding_cosine" and not self._fitted:
            raise RuntimeError(
                f"Affinity method '{self.affinity_method}' requires fit(case_base) before matching."
            )

    def _condprob_matrix(self, needs: List[str], actions: List[str]) -> np.ndarray:
        self._ensure_fitted()
        n, m = len(needs), len(actions)
        mat = np.zeros((n, m), dtype=np.float32)
        V = max(len(self._action_vocab), 1)
        s = max(self.condprob_smoothing, 1e-9)
        for i, need in enumerate(needs):
            c_n = float(self._need_df.get(need, 0))
            denom = c_n + s * V
            for j, action in enumerate(actions):
                c_na = float(self._pair_df.get((need, action), 0))
                mat[i, j] = float((c_na + s) / denom)
        return mat

    def _pmi_matrix(self, needs: List[str], actions: List[str]) -> np.ndarray:
        self._ensure_fitted()
        n, m = len(needs), len(actions)
        mat = np.zeros((n, m), dtype=np.float32)
        N = max(float(self._num_cases), 1.0)
        s = max(self.pmi_smoothing, 1e-9)
        for i, need in enumerate(needs):
            c_n = float(self._need_df.get(need, 0))
            p_n = (c_n + s) / (N + 2.0 * s)
            for j, action in enumerate(actions):
                c_a = float(self._action_df.get(action, 0))
                c_na = float(self._pair_df.get((need, action), 0))
                p_a = (c_a + s) / (N + 2.0 * s)
                p_na = (c_na + s) / (N + 2.0 * s)
                pmi = math.log((p_na + 1e-12) / ((p_n * p_a) + 1e-12))
                # NPMI in [-1,1], then map to [0,1] for threshold compatibility.
                npmi = pmi / (-(math.log(p_na + 1e-12)))
                mat[i, j] = float(0.5 * (npmi + 1.0))
        return np.clip(mat, 0.0, 1.0)

    def affinity_matrix(self, needs: List[str], pool_actions: List[str]) -> np.ndarray:
        if not needs or not pool_actions:
            return np.zeros((len(needs), len(pool_actions)), dtype=np.float32)

        if self.affinity_method == "embedding_cosine":
            texts = needs + pool_actions
            emb = self.embed(texts)
            n = len(needs)
            return self.cosine_matrix(emb[:n], emb[n:])
        if self.affinity_method == "condprob":
            return self._condprob_matrix(needs, pool_actions)
        if self.affinity_method == "pmi":
            return self._pmi_matrix(needs, pool_actions)
        raise ValueError(f"Unknown affinity_method={self.affinity_method}")


def gale_shapley_match(sim: np.ndarray) -> Dict[int, int]:
    """
    Stable matching between:
      left side  (size n): query needs
      right side (size m): candidate actions

    Returns a mapping left_index -> right_index for matched pairs.
    """
    n, m = sim.shape
    left_pref = [list(np.argsort(-sim[i])) for i in range(n)]
    right_pref = [list(np.argsort(-sim[:, j])) for j in range(m)]

    right_rank = [np.empty(n, dtype=int) for _ in range(m)]
    for j in range(m):
        for rank, li in enumerate(right_pref[j]):
            right_rank[j][li] = rank

    next_prop = [0] * n
    free_left = list(range(n))
    right_partner: Dict[int, int] = {}
    left_partner: Dict[int, int] = {}

    while free_left:
        li = free_left.pop()
        if next_prop[li] >= m:
            continue

        rj = left_pref[li][next_prop[li]]
        next_prop[li] += 1

        if rj not in right_partner:
            right_partner[rj] = li
            left_partner[li] = rj
        else:
            cur_li = right_partner[rj]
            if right_rank[rj][li] < right_rank[rj][cur_li]:
                right_partner[rj] = li
                left_partner[li] = rj
                del left_partner[cur_li]
                free_left.append(cur_li)
            else:
                free_left.append(li)

    return left_partner


def greedy_nda_match(sim: np.ndarray) -> Dict[int, int]:
    """
    NDA analogue: greedy matching without unpairing.
    For each need (row), pick the best available action (col).
    Once an action is taken, it cannot be reassigned.
    """
    n, m = sim.shape
    taken = set()
    pairs: Dict[int, int] = {}

    for li in range(n):
        # rank actions by similarity for this need
        prefs = list(np.argsort(-sim[li]))
        chosen = None
        for aj in prefs:
            if aj not in taken:
                chosen = int(aj)
                break
        if chosen is not None:
            pairs[li] = chosen
            taken.add(chosen)

    return pairs


def stable_match_needs_to_actions(
    needs: List[str],
    pool_actions: List[str],
    sim_backend: TextSim,
    debug: bool = False,
    method: str = "gsa"  # "gsa" or "nda"
) -> Tuple[Dict[int, int], float, np.ndarray]:
    """
    Builds affinity matrix needs x pool_actions and runs stable matching.
    Returns:
      pairs: mapping need_idx -> action_idx
      score: sum(affinity of pairs) / |needs|
      sim_mat: affinity matrix
    """
    if not needs or not pool_actions:
        return {}, 0.0, np.zeros((len(needs), len(pool_actions)), dtype=np.float32)

    sim_mat = sim_backend.affinity_matrix(needs, pool_actions)

    # pairs = gale_shapley_match(sim_mat)
    if method == "nda":
        pairs = greedy_nda_match(sim_mat)
    else:
        pairs = gale_shapley_match(sim_mat)

    if pairs:
        sims = [sim_mat[li, aj] for li, aj in pairs.items()]
        score = float(np.sum(sims) / max(len(needs), 1))
    else:
        score = 0.0

    if debug:
        logger.info("Stable match score=%.4f needs=%d pool=%d pairs=%d", score, len(needs), len(pool_actions), len(pairs))

    return pairs, score, sim_mat
