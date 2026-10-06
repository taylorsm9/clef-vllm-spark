"""Client-side constants and helpers. No vLLM imports, so clients can use them cheaply."""

ARCHITECTURE = "ClefFlashForDecision"
TASK = "token_classify"


def schema_layout(encoded) -> list:
    """PoolingParams.extra_kwargs["clef_questions"] for an EncodedRecord from encode_record."""
    return [[q.question_type, list(q.question_span), [list(s) for s in q.option_spans]] for q in encoded.questions]
