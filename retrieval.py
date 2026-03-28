# retrieval.py
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except Exception:
    _HAS_ST = False

from graph_structures import Case


def _split_problem_fields(problem_text: str) -> Tuple[str, str, bool]:
    """
    Parse structured text in the form:
      title=<...> | description=<...>
    Returns (title, description, has_structured_fields).
    """
    t = str(problem_text or "").strip()
    if not t:
        return "", "", False
    lower = t.lower()
    if not (lower.startswith("title=") and "| description=" in lower):
        return t, "", False
    try:
        left, right = t.split("|", 1)
        title = left.split("=", 1)[1].strip()
        description = right.split("=", 1)[1].strip()
        return title, description, True
    except Exception:
        return t, "", False


@dataclass
class Retrieved:
    idx: int
    score: float


class Retriever:
    def fit(self, cases: List[Case]) -> None:
        raise NotImplementedError

    def query(self, problem_text: str, top_k: int) -> List[Retrieved]:
        raise NotImplementedError


class TfidfRetriever(Retriever):
    def __init__(self, max_features: int = 50000, ngram_range=(1, 2)):
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
        self.case_matrix = None
        self.field_mode = False
        self.vectorizer_title = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
        self.vectorizer_desc = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
        self.case_matrix_title = None
        self.case_matrix_desc = None

    def fit(self, cases: List[Case]) -> None:
        triples = [_split_problem_fields(c.problem_text) for c in cases]
        self.field_mode = any(has_fields for _, _, has_fields in triples)

        texts = [c.problem_text for c in cases]
        self.case_matrix = self.vectorizer.fit_transform(texts)

        if self.field_mode:
            titles = [ttl if ttl else " " for ttl, _, _ in triples]
            descs = [desc if desc else " " for _, desc, _ in triples]
            self.case_matrix_title = self.vectorizer_title.fit_transform(titles)
            self.case_matrix_desc = self.vectorizer_desc.fit_transform(descs)

    def query(self, problem_text: str, top_k: int) -> List[Retrieved]:
        if self.field_mode and self.case_matrix_title is not None and self.case_matrix_desc is not None:
            title, desc, _ = _split_problem_fields(problem_text)
            q_title = self.vectorizer_title.transform([title if title else " "])
            q_desc = self.vectorizer_desc.transform([desc if desc else " "])
            sims_title = cosine_similarity(q_title, self.case_matrix_title).ravel()
            sims_desc = cosine_similarity(q_desc, self.case_matrix_desc).ravel()
            # Equal weights by default.
            sims = 0.5 * sims_title + 0.5 * sims_desc
        else:
            q = self.vectorizer.transform([problem_text])
            sims = cosine_similarity(q, self.case_matrix).ravel()
        k = min(top_k, len(sims))
        if k <= 0:
            return []
        idxs = np.argpartition(-sims, k - 1)[:k]
        idxs = idxs[np.argsort(-sims[idxs])]
        return [Retrieved(int(i), float(sims[i])) for i in idxs]


class SBERTRetriever(Retriever):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if not _HAS_ST:
            raise RuntimeError("sentence-transformers not installed.")
        self.model = SentenceTransformer(model_name)
        self.case_emb = None
        self.field_mode = False
        self.case_emb_title = None
        self.case_emb_desc = None

    def fit(self, cases: List[Case]) -> None:
        texts = [c.problem_text for c in cases]
        self.case_emb = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
        triples = [_split_problem_fields(c.problem_text) for c in cases]
        self.field_mode = any(has_fields for _, _, has_fields in triples)
        if self.field_mode:
            titles = [ttl if ttl else " " for ttl, _, _ in triples]
            descs = [desc if desc else " " for _, desc, _ in triples]
            self.case_emb_title = self.model.encode(titles, normalize_embeddings=True, show_progress_bar=False)
            self.case_emb_desc = self.model.encode(descs, normalize_embeddings=True, show_progress_bar=False)

    def query(self, problem_text: str, top_k: int) -> List[Retrieved]:
        if self.field_mode and self.case_emb_title is not None and self.case_emb_desc is not None:
            title, desc, _ = _split_problem_fields(problem_text)
            q_title = self.model.encode([title if title else " "], normalize_embeddings=True)[0]
            q_desc = self.model.encode([desc if desc else " "], normalize_embeddings=True)[0]
            sims_title = np.dot(self.case_emb_title, q_title)
            sims_desc = np.dot(self.case_emb_desc, q_desc)
            # Equal weights by default.
            sims = 0.5 * sims_title + 0.5 * sims_desc
        else:
            q = self.model.encode([problem_text], normalize_embeddings=True)[0]
            sims = np.dot(self.case_emb, q)
        k = min(top_k, len(sims))
        if k <= 0:
            return []
        idxs = np.argpartition(-sims, k - 1)[:k]
        idxs = idxs[np.argsort(-sims[idxs])]
        return [Retrieved(int(i), float(sims[i])) for i in idxs]
