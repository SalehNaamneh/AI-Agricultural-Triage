from typing import TypedDict


class AgentState(TypedDict):
    question: str          # original or enriched with image prediction
    crop_id: str
    image_bytes: bytes | None
    prediction: dict | None    # ImageAnalyzer result (class, confidence, …)
    answer: str                # LLM answer, may be modified by output guard
    sources: list[dict]        # retrieval sources with metadata
    warnings: list[str]        # output guardrail warnings
    blocked: bool
    block_reason: str
    block_message: str
