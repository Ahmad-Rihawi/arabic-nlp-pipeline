import torch
from torch.utils.data import Dataset


class ArabicSentimentDataset(Dataset):
    """
    Wraps raw text + labels into a PyTorch Dataset that yields
    tokenized, tensor-ready examples for a transformer model.
    """

    def __init__(self, texts, labels, tokenizer, max_length: int = 128):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenize, pad/truncate to a fixed length, and get PyTorch tensors back directly
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            # squeeze(0) drops the batch dimension the tokenizer adds by default,
            # since we're encoding one example at a time here; the DataLoader
            # re-adds a batch dimension later when it groups examples together.
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }