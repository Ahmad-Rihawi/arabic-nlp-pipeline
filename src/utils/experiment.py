from dataclasses import dataclass, field
from typing import Any
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from src.pipeline.base import ArabicTextPipeline
from src.features.vectorizer import ArabicTfidfVectorizer


@dataclass
class ExperimentResult:
    dataset: str
    model_name: str
    vectorizer_mode: str
    f1_macro: float
    f1_weighted: float
    report: str
    extra: dict = field(default_factory=dict)


MODELS = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "LinearSVC": LinearSVC(max_iter=2000),
    "RandomForest": RandomForestClassifier(n_estimators=100, n_jobs=-1),
}


def run_experiment(
    df: pd.DataFrame,
    pipeline: ArabicTextPipeline,
    dataset_name: str,
    vectorizer_mode: str = "word",
    test_size: float = 0.2,
    random_state: int = 42,
) -> list[ExperimentResult]:
    """
    Run full experiment:
    1. Apply preprocessing pipeline
    2. Vectorize
    3. Train and evaluate 3 classifiers
    """
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name} | Vectorizer: {vectorizer_mode}")
    print(f"{'='*60}")

    # preprocess
    print("Preprocessing...")
    df = df.copy()
    df["clean_text"] = pipeline.transform_series(df["text"])
    df = df[df["clean_text"].str.strip().str.len() > 0]

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"],
        df["label"],
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"],
    )

    # vectorize
    print("Vectorizing...")
    vec = ArabicTfidfVectorizer(mode=vectorizer_mode, max_features=15000)
    X_train_vec = vec.fit_transform(X_train)
    X_test_vec = vec.transform(X_test)
    print(f"Vocabulary size: {vec.vocabulary_size}")

    results = []
    for name, model in MODELS.items():
        print(f"Training {name}...")
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)

        f1_macro = f1_score(y_test, y_pred, average="macro")
        f1_weighted = f1_score(y_test, y_pred, average="weighted")
        report = classification_report(y_test, y_pred)

        print(f"  F1 macro: {f1_macro:.4f} | F1 weighted: {f1_weighted:.4f}")

        results.append(ExperimentResult(
            dataset=dataset_name,
            model_name=name,
            vectorizer_mode=vectorizer_mode,
            f1_macro=f1_macro,
            f1_weighted=f1_weighted,
            report=report,
        ))

    return results


def results_to_dataframe(results: list[ExperimentResult]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "dataset": r.dataset,
            "model": r.model_name,
            "vectorizer": r.vectorizer_mode,
            "f1_macro": round(r.f1_macro, 4),
            "f1_weighted": round(r.f1_weighted, 4),
        }
        for r in results
    ])