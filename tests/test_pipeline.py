from src.pipeline.base import ArabicTextPipeline
from src.pipeline.steps import (
    DiacriticsRemover,
    Normalizer,
    ElongationNormalizer,
    PunctuationRemover,
    StopwordRemover,
)


def get_full_pipeline() -> ArabicTextPipeline:
    return ArabicTextPipeline([
        DiacriticsRemover(),
        Normalizer(),
        ElongationNormalizer(),
        PunctuationRemover(),
        StopwordRemover(),
    ])


def test_diacritics_removed():
    step = DiacriticsRemover()
    result = step.transform("مرحبًا بِكُمْ")
    assert "ً" not in result
    assert "ِ" not in result
    assert "ْ" not in result


def test_alef_normalized():
    step = Normalizer()
    assert step.transform("أحمد") == "احمد"
    assert step.transform("إنسان") == "انسان"
    assert step.transform("آمن") == "امن"


def test_elongation_normalized():
    step = ElongationNormalizer()
    assert step.transform("مرحبااااا") == "مرحباا"


def test_url_removed():
    step = PunctuationRemover()
    result = step.transform("زوروا موقعنا https://example.com للمزيد")
    assert "https" not in result
    assert "example" not in result


def test_emoji_removed():
    step = PunctuationRemover()
    result = step.transform("الفندق رائع 😍🔥")
    assert "😍" not in result
    assert "🔥" not in result


def test_stopwords_removed():
    step = StopwordRemover()
    result = step.transform("الفندق كان في غاية الروعة")
    tokens = result.split()
    assert "في" not in tokens


def test_full_pipeline():
    pipeline = get_full_pipeline()
    text = "الفُنْدُقُ كааان ممتاززز جداً وخدمة رائعة 😍 https://hotel.com"
    result = pipeline.transform(text)
    assert "😍" not in result
    assert "https" not in result
    assert isinstance(result, str)
    assert len(result) > 0


def test_pipeline_describe():
    pipeline = get_full_pipeline()
    steps = pipeline.describe()
    assert len(steps) == 5