from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import TransformerMixin
import pandas as pd
import numpy as np


class ArabicTfidfVectorizer(TransformerMixin):
    """
    TF-IDF vectorizer tuned for Arabic text.

    Supports both word-level and character n-gram modes.
    Character n-grams are more robust for Arabic due to
    morphological richness — they capture root patterns
    even without stemming.

    Args:
        mode:       'word' | 'char' | 'both'
        max_features: vocabulary size cap
        ngram_range:  n-gram range tuple
    """

    def __init__(
        self,
        mode: str = "word",
        max_features: int = 10000,
        ngram_range: tuple = (1, 2),
    ):
        if mode not in ("word", "char", "both"):
            raise ValueError("mode must be 'word', 'char', or 'both'")

        self.mode = mode
        self.max_features = max_features
        self.ngram_range = ngram_range
        self._vectorizers = self._build_vectorizers()

    def _build_vectorizers(self) -> dict:
        shared = dict(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            sublinear_tf=True,
        )
        vectorizers = {}
        if self.mode in ("word", "both"):
            vectorizers["word"] = TfidfVectorizer(analyzer="word", **shared)
        if self.mode in ("char", "both"):
            vectorizers["char"] = TfidfVectorizer(analyzer="char_wb", **shared)
        return vectorizers

    def fit(self, texts: pd.Series, y=None):
        for vec in self._vectorizers.values():
            vec.fit(texts)
        return self

    def transform(self, texts: pd.Series) -> np.ndarray:
        from scipy.sparse import hstack
        matrices = [vec.transform(texts) for vec in self._vectorizers.values()]
        if len(matrices) == 1:
            return matrices[0]
        return hstack(matrices)

    def fit_transform(self, texts: pd.Series, y=None) -> np.ndarray:
        return self.fit(texts).transform(texts)

    @property
    def vocabulary_size(self) -> int:
        return sum(
            len(vec.vocabulary_) for vec in self._vectorizers.values()
        )