"""Modelos de predição de complexidade (sklearn) + wrapper de inferência sobre texto bruto."""
from __future__ import annotations
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import LEVEL_NAMES
from .features import extract

CANON_PREFIXES = ("canonical_", "noncanon_", "shape_")
LENGTH_ONLY = ["n_words", "n_sents", "words_per_sent", "words_per_sent_std", "words_per_sent_max",
               "chars_per_word", "long_word_ratio", "vlong_word_ratio", "syll_per_word", "poly_word_ratio", "flesch_pt"]


def numeric_cols(df: pd.DataFrame) -> list[str]:
    drop = {"chunk_id", "text", "file", "file_idx", "file_rank", "level", "block", "n_words_meta"}
    return [c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])]


def _num(cols):
    return ColumnTransformer([("num", StandardScaler(), cols)], remainder="drop")


def make_models(cols: list[str], seed: int = 0) -> dict[str, Pipeline]:
    no_canon = [c for c in cols if not c.startswith(CANON_PREFIXES)]
    word_tfidf = lambda: TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=150_000, sublinear_tf=True, lowercase=True)
    char_tfidf = lambda: TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=5, max_features=200_000, sublinear_tf=True)
    lr = lambda C=1.0: LogisticRegression(C=C, max_iter=3000, class_weight="balanced")
    return {
        "baseline (classe majoritária)": Pipeline([("clf", DummyClassifier(strategy="most_frequent"))]),
        "LR · só tamanho/silabas": Pipeline([("f", _num(LENGTH_ONLY)), ("clf", lr())]),
        "LR · features (sem canonicidade)": Pipeline([("f", _num(no_canon)), ("clf", lr())]),
        "LR · features": Pipeline([("f", _num(cols)), ("clf", lr())]),
        "RandomForest · features": Pipeline([("f", _num(cols)), ("clf", RandomForestClassifier(
            n_estimators=400, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=2, random_state=seed))]),
        "HistGB · features (sem canonicidade)": Pipeline([("f", _num(no_canon)), ("clf", HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.06, class_weight="balanced", random_state=seed))]),
        "HistGB · features": Pipeline([("f", _num(cols)), ("clf", HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.06, class_weight="balanced", random_state=seed))]),
        "TF-IDF palavras · LR": Pipeline([("f", ColumnTransformer([("t", word_tfidf(), "text")])), ("clf", lr(5.0))]),
        "TF-IDF chars · LR": Pipeline([("f", ColumnTransformer([("t", char_tfidf(), "text")])), ("clf", lr(5.0))]),
        "Híbrido: features + TF-IDF palavras · LR": Pipeline([
            ("f", ColumnTransformer([("num", StandardScaler(), cols), ("t", word_tfidf(), "text")])), ("clf", lr(1.0))]),
    }


class ComplexityClassifier:
    """Texto bruto -> probabilidades por nível + escore contínuo de complexidade (0 a 3)."""

    def __init__(self, pipeline, cols, nlp=None):
        self.pipeline, self.cols, self._nlp = pipeline, cols, nlp

    @property
    def nlp(self):
        if self._nlp is None:
            import spacy
            self._nlp = spacy.load("pt_core_news_sm", disable=["ner"])
        return self._nlp

    def featurize(self, texts: list[str]) -> pd.DataFrame:
        rows = [extract(t, d) for t, d in zip(texts, self.nlp.pipe(texts, batch_size=32))]
        df = pd.DataFrame(rows)
        df["text"] = texts
        return df[self.cols + ["text"]]

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return self.pipeline.predict_proba(self.featurize(texts))

    def score(self, texts: list[str]) -> np.ndarray:
        """Nível esperado E[nível] em [0, 3] – escala contínua, sensível a mudanças pequenas."""
        return self.predict_proba(texts) @ np.arange(4)

    def analyze(self, text: str) -> dict:
        p = self.predict_proba([text])[0]
        return {"level": int(p.argmax()), "label": LEVEL_NAMES[int(p.argmax())],
                "probs": {LEVEL_NAMES[i]: float(p[i]) for i in range(4)}, "score": float(p @ np.arange(4))}

    def save(self, path):
        joblib.dump({"pipeline": self.pipeline, "cols": self.cols}, path)

    @classmethod
    def load(cls, path):
        d = joblib.load(path)
        return cls(d["pipeline"], d["cols"])
