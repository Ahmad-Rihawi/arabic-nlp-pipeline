# Arabic NLP Pipeline

A modular, production-style preprocessing pipeline for Arabic text with an end-to-end sentiment classification experiment.

## Project Structure
arabic-nlp-pipeline/

├── src/

│   ├── pipeline/       # Core pipeline: base classes and preprocessing steps

│   ├── features/       # Feature extraction (TF-IDF)

│   └── utils/          # Data loading utilities

├── notebooks/          # EDA, experiments, ablation study

├── tests/              # Unit tests for pipeline steps

├── data/raw/           # Raw datasets (gitignored)

├── results/            # Saved experiment results

└── requirements.txt

## Pipeline Steps

- Normalization (hamza, alef, teh marbuta)
- Diacritics removal
- Elongation normalization (تمديد)
- Punctuation and special character removal
- Stopword removal

## Experiments

| Experiment | Dataset | Model | F1 |
|------------|---------|-------|----|
| TBD | HARD | SVM | - |
| TBD | Arabic Twitter | LR | - |

## Ablation Study

Results showing F1 impact of each preprocessing step — see `notebooks/03_ablation.ipynb`

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Author

Ahmad Rihawi — [LinkedIn](https://linkedin.com/in/ahmad-rihawi-ite)