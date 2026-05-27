"""
Prompt Engineering — Surface 2: Agent Enrich-Question Prompt
Tests how image prediction is merged into the user question.
"""
import sys, json
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")

TEST_CASES = [
    ("TC01", {"class_he":"כתם סגול","class_en":"Purple Blotch","crop_he":"בצל","crop_en":"onion","confidence":94.2,"health_score":5.8}, "מה הטיפול?"),
    ("TC02", {"class_he":"כתם סגול","class_en":"Purple Blotch","crop_he":"בצל","crop_en":"onion","confidence":31.5,"health_score":68.5}, "מה הטיפול?"),
    ("TC03", {"class_he":"בריא","class_en":"Healthy","crop_he":"בצל","crop_en":"onion","confidence":89.0,"health_score":89.0},            "מה הטיפול?"),
    ("TC04", {"class_he":"כתם סגול","class_en":"Purple Blotch","crop_he":"בצל","crop_en":"onion","confidence":94.2,"health_score":5.8}, "What treatment should I apply?"),
    ("TC05", {"class_he":"בוטריטיס","class_en":"Botrytis","crop_he":"בצל","crop_en":"onion","confidence":72.1,"health_score":27.9},         "כמה מינון צריך?"),
    ("TC06", {"class_he":"כתם סגול","class_en":"Purple Blotch","crop_he":"בצל","crop_en":"onion","confidence":55.0,"health_score":45.0}, "האם הצמח יחלים?"),
    ("TC07", {"class_he":"פוזריום","class_en":"Fusarium","crop_he":"בצל","crop_en":"onion","confidence":48.3,"health_score":51.7},           "מה קוד FRAC המתאים?"),
    ("TC08", {"class_he":"בריא","class_en":"Healthy","crop_he":"בצל","crop_en":"onion","confidence":67.0,"health_score":67.0},               "האם הצמח בסדר?"),
    ("TC09", {"class_he":"כתם סגול","class_en":"Purple Blotch","crop_he":"בצל","crop_en":"onion","confidence":94.2,"health_score":5.8}, ""),
    ("TC10", {"class_he":"אלטרנריה","class_en":"Alternaria","crop_he":"בצל","crop_en":"onion","confidence":82.0,"health_score":18.0},        "מתי לרסס?"),
]

CONFIDENCE_THRESHOLD = 50.0

# ── VERSION 1 — Baseline ───────────────────────────────────────────────────────
def v1_enrich(pred, question):
    return (
        f"בתמונה זוהתה מחלה: {pred['class_he']} ({pred['class_en']}) "
        f"בגידול {pred['crop_he']} (בטחון {pred['confidence']}%). "
        f"{question}"
    )

# ── VERSION 2 — Fix: flag low confidence ──────────────────────────────────────
def v2_enrich(pred, question):
    conf_note = (
        f"(בטחון **גבוה** {pred['confidence']}%)"
        if pred["confidence"] >= CONFIDENCE_THRESHOLD
        else f"(בטחון **נמוך** {pred['confidence']}% — ייתכן שהזיהוי שגוי)"
    )
    return (
        f"בתמונה זוהתה: {pred['class_he']} ({pred['class_en']}) "
        f"בגידול {pred['crop_he']} {conf_note}. "
        f"{question}"
    )

# ── VERSION 3 — Fix: handle Healthy class separately ─────────────────────────
def v3_enrich(pred, question):
    if "healthy" in pred["class_en"].lower():
        return (
            f"הצמח נראה **בריא** (ציון בריאות {pred['health_score']}%). "
            f"אין עדות לזיהום פעיל. "
            f"{question if question else 'האם יש דבר שצריך לשים לב אליו לצורך מניעה?'}"
        )
    conf_note = (
        f"בטחון גבוה ({pred['confidence']}%)"
        if pred["confidence"] >= CONFIDENCE_THRESHOLD
        else f"בטחון נמוך ({pred['confidence']}%) — הזיהוי עשוי להיות לא מדויק"
    )
    return (
        f"בתמונה זוהתה: **{pred['class_he']}** ({pred['class_en']}) "
        f"בגידול {pred['crop_he']}. {conf_note}. "
        f"ציון בריאות הצמח: {pred['health_score']}%. "
        f"{question}"
    )

# ── VERSION 4 — Fix: default question when empty ─────────────────────────────
def v4_enrich(pred, question):
    if "healthy" in pred["class_en"].lower():
        default_q = question or "האם יש המלצות מניעה לשמירה על בריאות הצמח?"
        return (
            f"הצמח נראה **בריא** (ציון בריאות {pred['health_score']}%). "
            f"אין עדות לזיהום פעיל. {default_q}"
        )
    default_q = question or f"מה הטיפול המומלץ ב{pred['class_he']} וכיצד למנוע התפשטות?"
    conf_note = (
        f"בטחון גבוה ({pred['confidence']}%)"
        if pred["confidence"] >= CONFIDENCE_THRESHOLD
        else f"בטחון נמוך ({pred['confidence']}%) — שקול אימות ויזואלי לפני טיפול"
    )
    return (
        f"בתמונה זוהתה: **{pred['class_he']}** ({pred['class_en']}) "
        f"בגידול {pred['crop_he']}. {conf_note}. "
        f"ציון בריאות: {pred['health_score']}%. "
        f"{default_q}"
    )

# ── VERSION 5 — Final: add crop context + bilingual ──────────────────────────
def v5_enrich(pred, question):
    if "healthy" in pred["class_en"].lower():
        default_q = question or "האם יש המלצות מניעה לשמירה על בריאות הצמח?"
        return (
            f"[זיהוי תמונה] הגידול: {pred['crop_he']} ({pred['crop_en']}). "
            f"מצב: **בריא** — ציון בריאות {pred['health_score']}%. "
            f"אין מחלה פעילה. {default_q}"
        )
    default_q = question or f"מה הטיפול המומלץ ב{pred['class_he']} וכיצד למנוע התפשטות?"
    conf_tag = "✅ זיהוי בטוח" if pred["confidence"] >= CONFIDENCE_THRESHOLD else "⚠️ זיהוי לא בטוח — אמת לפני טיפול"
    return (
        f"[זיהוי תמונה] הגידול: {pred['crop_he']} ({pred['crop_en']}). "
        f"מחלה שזוהתה: **{pred['class_he']}** ({pred['class_en']}). "
        f"בטחון: {pred['confidence']}% — {conf_tag}. "
        f"ציון בריאות: {pred['health_score']}%. "
        f"שאלת המשתמש: {default_q}"
    )

VERSIONS = [("V1", v1_enrich), ("V2", v2_enrich), ("V3", v3_enrich), ("V4", v4_enrich), ("V5", v5_enrich)]

results = {}
for vname, fn in VERSIONS:
    results[vname] = []
    for tc_id, pred, question in TEST_CASES:
        enriched = fn(pred, question)
        results[vname].append({"id": tc_id, "question": question,
                                "confidence": pred["confidence"],
                                "class_en": pred["class_en"],
                                "enriched": enriched})

out = Path(__file__).parent / "enrich_prompt_results.json"
with open(out, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"Saved to {out}")

# Print comparison of TC02 (low confidence) and TC03 (healthy) across versions
for vname in ["V1","V2","V3","V4","V5"]:
    print(f"\n=== {vname} — TC02 (low conf 31.5%) ===")
    print(results[vname][1]["enriched"])
    print(f"\n=== {vname} — TC03 (healthy 89%) ===")
    print(results[vname][2]["enriched"])
    print(f"\n=== {vname} — TC09 (empty question) ===")
    print(results[vname][8]["enriched"])
