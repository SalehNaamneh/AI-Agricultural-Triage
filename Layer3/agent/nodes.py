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


_CONFIDENCE_THRESHOLD = 50.0

def node_enrich_question(state: AgentState) -> dict:
    pred = state.get("prediction")
    if not pred:
        return {}

    question = state["question"]

    # Healthy plant — reframe question toward prevention
    if "healthy" in pred["class_en"].lower():
        default_q = question or "האם יש המלצות מניעה לשמירה על בריאות הצמח?"
        enriched = (
            f"[זיהוי תמונה] הגידול: {pred['crop_he']} ({pred['crop_en']}). "
            f"מצב: **בריא** — ציון בריאות {pred['health_score']}%. "
            f"אין מחלה פעילה. {default_q}"
        )
        return {"question": enriched}

    # Low confidence — warn the RAG chain not to treat as definitive
    conf_tag = (
        "✅ זיהוי בטוח"
        if pred["confidence"] >= _CONFIDENCE_THRESHOLD
        else "⚠️ זיהוי לא בטוח — אמת לפני טיפול"
    )
    default_q = question or f"מה הטיפול המומלץ ב{pred['class_he']} וכיצד למנוע התפשטות?"
    enriched = (
        f"[זיהוי תמונה] הגידול: {pred['crop_he']} ({pred['crop_en']}). "
        f"מחלה שזוהתה: **{pred['class_he']}** ({pred['class_en']}). "
        f"בטחון: {pred['confidence']}% — {conf_tag}. "
        f"ציון בריאות: {pred['health_score']}%. "
        f"שאלת המשתמש: {default_q}"
    )
    return {"question": enriched}


def node_query_rag(state: AgentState) -> dict:
    resp = rag_ask(state["question"])
    return {"answer": resp["answer"], "sources": resp["sources"]}


def node_guard_output(state: AgentState) -> dict:
    out = output_guard.check(state["answer"], state["sources"])
    return {"answer": out.answer, "warnings": out.warnings}
