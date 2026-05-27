import sys
import io
from pathlib import Path

_here   = Path(__file__).resolve().parent      # Layer3/agent/
_layer3 = _here.parent                          # Layer3/
sys.path.insert(0, str(_layer3))                          # guardrails package
sys.path.insert(0, str(_layer3 / "RAG"))                  # rag_chain, retriever, ingest
sys.path.insert(0, str(_layer3 / "ImageAnalyzer"))        # predict, model, dataset

from PIL import Image

from guardrails import input_guard, output_guard
from rag_chain import ask as rag_ask
from predict import predict_pil

from state import AgentState


def node_guard_input(state: AgentState) -> dict:
    r = input_guard.check(state["question"])
    if not r.passed:
        return {"blocked": True, "block_reason": r.reason, "block_message": r.message_he}
    return {"blocked": False, "block_reason": "", "block_message": ""}


def node_analyze_image(state: AgentState) -> dict:
    raw = state.get("image_bytes")
    if not raw:
        return {"prediction": None}
    img  = Image.open(io.BytesIO(raw)).convert("RGB")
    pred = predict_pil(img, crop_id=state.get("crop_id", "onion"))
    return {"prediction": pred}


def node_enrich_question(state: AgentState) -> dict:
    pred = state.get("prediction")
    if not pred:
        return {}
    enriched = (
        f"בתמונה זוהתה מחלה: {pred['class_he']} ({pred['class_en']}) "
        f"בגידול {pred['crop_he']} (בטחון {pred['confidence']}%). "
        f"{state['question']}"
    )
    return {"question": enriched}


def node_query_rag(state: AgentState) -> dict:
    resp = rag_ask(state["question"])
    return {"answer": resp["answer"], "sources": resp["sources"]}


def node_guard_output(state: AgentState) -> dict:
    out = output_guard.check(state["answer"], state["sources"])
    return {"answer": out.answer, "warnings": out.warnings}
