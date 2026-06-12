from abc import ABC, abstractmethod
import pandas as pd


class PipelineStep(ABC):
    """Base class for all preprocessing steps."""

    @abstractmethod
    def transform(self, text: str) -> str:
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class ArabicTextPipeline:
    """
    A modular pipeline that chains preprocessing steps.

    Usage:
        pipeline = ArabicTextPipeline([
            DiacriticsRemover(),
            Normalizer(),
            Elongation Normalizer(),
        ])
        clean_text = pipeline.transform("مرحبااا")
        clean_df = pipeline.transform_series(df["text"])
    """

    def __init__(self, steps: list[PipelineStep]):
        self.steps = steps

    def transform(self, text: str) -> str:
        for step in self.steps:
            text = step.transform(text)
        return text

    def transform_series(self, series: pd.Series) -> pd.Series:
        return series.apply(self.transform)

    def describe(self) -> list[str]:
        """Return names of active steps."""
        return [repr(step) for step in self.steps]