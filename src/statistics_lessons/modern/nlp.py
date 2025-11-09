"""Modern NLP utilities using Transformers."""

from __future__ import annotations

try:
    from transformers import pipeline
except Exception:  # pragma: no cover - handled in tests
    pipeline = None


def summarize_text(text: str) -> str:
    """Generate a summary for the provided text.

    Args:
        text: Text to summarize.

    Returns:
        Generated summary.

    Raises:
        ImportError: If transformers is not installed.
    """
    if pipeline is None:  # pragma: no cover - dependency missing
        raise ImportError("transformers is required for this function")

    summarizer = pipeline(
        "summarization",
        model="sshleifer/tiny-mbart",
        tokenizer="sshleifer/tiny-mbart",
    )
    result = summarizer(text, max_length=20, min_length=5, do_sample=False)[0]
    return result["summary_text"]


def extract_named_entities(text: str) -> list[tuple[str, str]]:
    """Extract named entities from text.

    Uses a tiny transformer model to identify named entities and return their
    labels.

    Args:
        text: Text to analyze.

    Returns:
        List of ``(entity, label)`` tuples.

    Raises:
        ImportError: If transformers is not installed.
    """

    if pipeline is None:  # pragma: no cover - dependency missing
        raise ImportError("transformers is required for this function")

    ner = pipeline(
        "ner",
        model="hf-internal-testing/tiny-bert-for-token-classification",
        tokenizer="hf-internal-testing/tiny-bert-for-token-classification",
        aggregation_strategy="simple",
    )
    results = ner(text)
    return [(r["word"], r["entity_group"]) for r in results]
