import re
import pyarabic.araby as araby
from src.pipeline.base import PipelineStep


# ------------------------------------------------------------------ #
#  Step 1 — Diacritics Remover                                        #
# ------------------------------------------------------------------ #
class DiacriticsRemover(PipelineStep):
    """Remove Arabic diacritics (tashkeel)."""

    def transform(self, text: str) -> str:
        return araby.strip_tashkeel(text)


# ------------------------------------------------------------------ #
#  Step 2 — Normalizer                                                #
# ------------------------------------------------------------------ #
class Normalizer(PipelineStep):
    """
    Normalize Arabic characters:
    - Unify alef variants (أ إ آ ا) → ا
    - Unify teh marbuta (ة) → ه
    - Unify yeh variants (ى) → ي
    """

    # alef variants
    ALEF_VARIANTS = re.compile(r"[أإآٱ]")
    # alef maqsura
    ALEF_MAQSURA = re.compile(r"ى")
    # teh marbuta
    TEH_MARBUTA = re.compile(r"ة")

    def transform(self, text: str) -> str:
        text = araby.normalize_alef(text)
        text = self.ALEF_MAQSURA.sub("ي", text)
        text = self.TEH_MARBUTA.sub("ه", text)
        return text


# ------------------------------------------------------------------ #
#  Step 3 — Elongation Normalizer                                     #
# ------------------------------------------------------------------ #
class ElongationNormalizer(PipelineStep):
    """Normalize elongated characters e.g. مرحبااااا → مرحبا"""

    # tatweel character
    TATWEEL = re.compile(r"ـ+")
    # repeated arabic letters (more than 2 in a row)
    REPEATED = re.compile(r"(.)\1{2,}")

    def transform(self, text: str) -> str:
        text = araby.strip_tatweel(text)
        text = self.TATWEEL.sub("", text)
        text = self.REPEATED.sub(r"\1\1", text)
        return text


# ------------------------------------------------------------------ #
#  Step 4 — Punctuation Remover                                       #
# ------------------------------------------------------------------ #
class PunctuationRemover(PipelineStep):
    """Remove punctuation, emojis, URLs, and non-Arabic characters."""

    KEEP_PATTERN = re.compile(r"[^\u0600-\u06FF\s]")
    URL_PATTERN = re.compile(r"http\S+|www\S+")

    def transform(self, text: str) -> str:
        text = self.URL_PATTERN.sub(" ", text)
        text = self.KEEP_PATTERN.sub(" ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text


# ------------------------------------------------------------------ #
#  Step 5 — Stopword Remover                                          #
# ------------------------------------------------------------------ #

# Bundled Arabic stopwords — common function words
ARABIC_STOPWORDS = {
    "من", "إلى", "عن", "على", "في", "هذا", "هذه", "ذلك", "تلك",
    "التي", "الذي", "الذين", "اللواتي", "وفي", "وقد", "وكان",
    "كان", "كانت", "يكون", "تكون", "أن", "إن", "لأن", "لكن",
    "كل", "بعض", "غير", "حتى", "إذا", "ثم", "أو", "أم", "لا",
    "ما", "مع", "هو", "هي", "هم", "هن", "أنا", "نحن", "أنت",
    "أنتم", "أنتن", "وهو", "وهي", "فهو", "فهي", "قد", "لم",
    "لن", "كما", "مما", "وأن", "بأن", "أيضا", "فقط", "جدا",
    "كذلك", "هناك", "بين", "حين", "بعد", "قبل", "خلال", "منذ",
    "عند", "لدى", "حول", "تحت", "فوق", "أمام", "وراء", "يوم",
    "هل", "نعم", "لا", "بل", "أي", "كيف", "متى", "أين", "لماذا",
}


class StopwordRemover(PipelineStep):
    """Remove Arabic stopwords."""

    def __init__(self, stopwords: set[str] | None = None):
        self.stopwords = stopwords if stopwords is not None else ARABIC_STOPWORDS

    def transform(self, text: str) -> str:
        tokens = text.split()
        tokens = [t for t in tokens if t not in self.stopwords]
        return " ".join(tokens)