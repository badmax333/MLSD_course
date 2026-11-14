"""Лексические бейзлайны поиска: BM25 и TF-IDF по символьным n-граммам.

Это не наш продукт (поиск строит банк), а «линейка»: на прокси с разметкой пар
она показывает, как ведут себя R3 и NDCG@3 и какие запросы трудные.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from mlsd_course.eda.text_stats import tokens


def stem_tokens(text: str, n: int = 5) -> list[str]:
    """Токены, обрезанные до n символов: грубая псевдооснова для русской морфологии."""
    return [t[:n] for t in tokens(text)]


class BM25:
    """BM25 Okapi на разреженной матрице (быстро для десятков тысяч документов)."""

    def __init__(self, docs: list[str], k1: float = 1.2, b: float = 0.75, stem: int | None = None):
        analyzer = (lambda t: stem_tokens(t, stem)) if stem else tokens
        self.vec = CountVectorizer(analyzer=analyzer)
        tf = self.vec.fit_transform(docs).tocsc().astype(np.float32)
        n_docs = tf.shape[0]
        dl = np.asarray(tf.sum(axis=1)).ravel()
        df = np.diff(tf.indptr)
        self.idf = np.log1p((n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
        norm = k1 * (1 - b + b * dl / dl.mean())
        tf = tf.tocoo()
        data = tf.data * (k1 + 1) / (tf.data + norm[tf.row])
        self.w = sparse.csr_matrix((data, (tf.row, tf.col)), shape=tf.shape).tocsc()

    def scores(self, queries: list[str]) -> np.ndarray:
        """Матрица скоров: запросы × документы."""
        q = self.vec.transform(queries)
        q.data[:] = 1.0
        return np.asarray((q.multiply(self.idf) @ self.w.T).todense())


class CharTfidf:
    """Косинус TF-IDF по символьным n-граммам 2–5: устойчивее к морфологии и опечаткам."""

    def __init__(self, docs: list[str]):
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)
        self.x = self.vec.fit_transform([d.lower() for d in docs])

    def scores(self, queries: list[str]) -> np.ndarray:
        return np.asarray((self.vec.transform([q.lower() for q in queries]) @ self.x.T).todense())


def top_k_run(scores: np.ndarray, query_ids: list, doc_ids: list, k: int = 10) -> pd.DataFrame:
    """Выдача top-k по матрице скоров: query_id, position (с 1), doc_id, score."""
    doc_ids = np.asarray(doc_ids)
    rows = []
    for i, qid in enumerate(query_ids):
        s = scores[i]
        idx = np.argpartition(-s, min(k, len(s) - 1))[:k]
        idx = idx[np.lexsort((idx, -s[idx]))]  # по убыванию скора, при равенстве — по индексу
        for pos, j in enumerate(idx, start=1):
            rows.append({"query_id": qid, "position": pos, "doc_id": doc_ids[j], "score": s[j]})
    return pd.DataFrame(rows)
