import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "raw"


def load_hard(filename: str = "ar_reviews_100k.tsv") -> pd.DataFrame:
    """Load HARD hotel reviews dataset."""
    path = DATA_DIR / filename
    df = pd.read_csv(path, sep="\t")
    df.columns = ["label", "text"]
    df = df.dropna(subset=["text"])
    df["source"] = "hard"
    return df


def load_twitter(
    train_neg: str = "train_Arabic_tweets_negative_20190413.tsv",
    train_pos: str = "train_Arabic_tweets_positive_20190413.tsv",
    test_neg: str = "test_Arabic_tweets_negative_20190413.tsv",
    test_pos: str = "test_Arabic_tweets_positive_20190413.tsv",
) -> pd.DataFrame:
    """Load Arabic Twitter sentiment dataset."""
    dfs = []

    for filename, label, split in [
        (train_neg, "Negative", "train"),
        (train_pos, "Positive", "train"),
        (test_neg, "Negative", "test"),
        (test_pos, "Positive", "test"),
    ]:
        path = DATA_DIR / filename
        df = pd.read_csv(path, sep="\t", header=None, names=["label", "text"])
        df["label"] = label
        df["split"] = split
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df = df.dropna(subset=["text"])
    df["source"] = "twitter"
    return df


def load_dataset(name: str) -> pd.DataFrame:
    """Entry point. name: 'hard' or 'twitter'"""
    if name == "hard":
        return load_hard()
    elif name == "twitter":
        return load_twitter()
    else:
        raise ValueError(f"Unknown dataset: {name}. Choose 'hard' or 'twitter'.")