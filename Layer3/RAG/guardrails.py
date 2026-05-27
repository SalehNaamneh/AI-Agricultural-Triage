"""
Guardrails for AgriTriage RAG API.

Input guardrails  — run before the LLM call:
  • Prompt-injection detection (pattern matching)
  • Off-topic detection       (keyword allowlist, no extra LLM call)

Output guardrails — run on the LLM answer before it is returned:
  • Low-confidence disclaimer (avg retrieval score below threshold)
  • Resistance-warning enforcement (if sources have ⚠ warnings, answer must too)
  • Hallucination flag        (FRAC codes / product names in answer not in sources)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class InputResult:
    passed: bool
    reason: str = ""        # machine-readable code for logs
    message_he: str = ""    # Hebrew message shown to the user


@dataclass
class OutputResult:
    answer: str
    warnings: list[str] = field(default_factory=list)


# ── Input guardrail ────────────────────────────────────────────────────────────

class InputGuardrail:
    """Fast, zero-LLM input validator."""

    # ── Prompt-injection patterns ──────────────────────────────────────────────
    _INJECTION_EN = re.compile(
        r"\b("
        r"ignore\s+(previous|all|your|the)\s+(instructions?|prompt|rules?)|"
        r"forget\s+(your|all|previous|the)\s+(instructions?|rules?)|"
        r"(you\s+are\s+now|act\s+as|pretend\s+(you\s+are|to\s+be)|roleplay)|"
        r"disregard\s+(your|all|previous)|"
        r"override\s+(your|the)\s+(system|prompt|instructions?)|"
        r"jailbreak|DAN\b|do\s+anything\s+now|"
        r"new\s+(system\s+)?prompt|"
        r"<\s*system\s*>|<\s*/\s*system\s*>"
        r")\b",
        re.IGNORECASE,
    )
    _INJECTION_HE = re.compile(
        r"(תתעלם\s+מ|שכח\s+את|התחזה\s+ל|אתה\s+עכשיו|עקוף\s+את|הוראות\s+חדשות)",
        re.UNICODE,
    )

    # ── Agricultural keyword allowlist ─────────────────────────────────────────
    # If the query has NO keyword from this list AND is long enough to be a
    # real question, it is considered off-topic.
    _AG_KEYWORDS_EN = {
        "disease", "crop", "plant", "fungus", "fungicide", "bacteria",
        "treatment", "spray", "symptom", "leaf", "blight", "mildew",
        "rust", "rot", "virus", "pest", "insect", "caterpillar",
        "onion", "tomato", "wheat", "potato", "agriculture", "agronomic",
        "field", "dose", "dunam", "frac", "active ingredient", "pathogen",
        "healthy", "infection", "spore", "moisture", "humidity",
        # product types
        "systemic", "contact", "protective", "curative",
    }
    _AG_KEYWORDS_HE = {
        "מחלה", "גידול", "צמח", "עלה", "עלים", "ריסוס", "טיפול",
        "תסמין", "תסמינים", "פטריה", "חיידק", "וירוס", "חרק",
        "זחל", "ריקבון", "חלודה", "כימשון", "כתם", "בוטריטיס",
        "סטמפיליום", "אלטרנריה", "פוזריום", "אפובנטורה",
        "בצל", "עגבנייה", "חיטה", "תפוח", "ירק", "פרי",
        "חקלאי", "שדה", "דונם", "מינון", "תכשיר", "חומר פעיל",
        "קוד", "frac", "מניעה", "אבחנה", "ריקבון", "בריאות",
        "זיהום", "נביטה", "עמידות", "ריסוס", "עונה",
    }
    # Minimum word count before we apply the topic filter
    # (very short queries like just a disease name pass through)
    _MIN_WORDS_FOR_TOPIC_CHECK = 4

    def check(self, text: str) -> InputResult:
        text = text.strip()

        # 1. Prompt injection
        if self._INJECTION_EN.search(text) or self._INJECTION_HE.search(text):
            return InputResult(
                passed=False,
                reason="prompt_injection",
                message_he=(
                    "⚠️ הבקשה שלך נחסמה — היא מכילה ניסיון לעקוף את הגדרות המערכת.\n"
                    "אנא שאל שאלה חקלאית רגילה."
                ),
            )

        # 2. Off-topic (only applied to longer queries)
        words = text.split()
        if len(words) >= self._MIN_WORDS_FOR_TOPIC_CHECK:
            lower = text.lower()
            has_ag = (
                any(kw in lower for kw in self._AG_KEYWORDS_EN)
                or any(kw in text for kw in self._AG_KEYWORDS_HE)
            )
            if not has_ag:
                return InputResult(
                    passed=False,
                    reason="off_topic",
                    message_he=(
                        "🌾 אני מתמחה בזיהוי מחלות גידולים חקלאיים וטיפול בהן.\n"
                        "נראה ששאלתך אינה קשורה לחקלאות — אנא שאל על מחלות, "
                        "תסמינים, טיפולים או מניעה של גידולים."
                    ),
                )

        return InputResult(passed=True)


# ── Output guardrail ───────────────────────────────────────────────────────────

_FRAC_PATTERN = re.compile(r"\bFRAC\s+([A-Z]?\d+)\b", re.IGNORECASE)


class OutputGuardrail:
    """Post-processes the LLM answer before it reaches the user."""

    LOW_CONFIDENCE_THRESHOLD = 0.30   # avg of top-3 source scores

    def check(self, answer: str, sources: list[dict]) -> OutputResult:
        """
        sources: list of dicts with keys: type, disease, score,
                 and optionally frac_code, product_en, resistance_warning.
        Returns the (possibly modified) answer and a list of warning strings.
        """
        warnings: list[str] = []

        # ── 1. Low-confidence disclaimer ───────────────────────────────────────
        scores = [s.get("score", 0.0) for s in sources[:3]]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        if avg_score < self.LOW_CONFIDENCE_THRESHOLD:
            disclaimer = (
                "⚠️ **שים לב:** המידע שנמצא עבור שאלה זו היה חלקי. "
                "ייתכן שהתשובה אינה מלאה — מומלץ להתייעץ עם אגרונום.\n\n"
            )
            answer = disclaimer + answer
            warnings.append(f"low_confidence(avg={avg_score:.2f})")

        # ── 2. Resistance-warning enforcement ─────────────────────────────────
        resistance_texts: list[str] = []
        for s in sources:
            rw = s.get("resistance_warning", "") or ""
            if rw and "⚠" in rw and rw not in resistance_texts:
                resistance_texts.append(rw)

        if resistance_texts:
            warning_keywords_he = ["עמידות", "סובב", "לסובב", "frac", "FRAC", "resistance", "warning"]
            answer_lower = answer.lower()
            answer_has_warning = any(kw.lower() in answer_lower for kw in warning_keywords_he)
            if not answer_has_warning:
                joined = "\n• ".join(resistance_texts)
                answer += (
                    f"\n\n---\n⚠️ **אזהרת עמידות (נמצאה בנתוני המקור):**\n• {joined}"
                )
                warnings.append("resistance_warning_appended")

        # ── 3. FRAC-code hallucination check ──────────────────────────────────
        # Collect FRAC codes known from retrieved sources
        known_fracs: set[str] = set()
        for s in sources:
            fc = s.get("frac_code", "") or ""
            # e.g. "FRAC 3"  → "3"
            m = _FRAC_PATTERN.search(fc)
            if m:
                known_fracs.add(m.group(1).upper())

        if known_fracs:
            answer_fracs = {m.group(1).upper() for m in _FRAC_PATTERN.finditer(answer)}
            hallucinated = answer_fracs - known_fracs
            if hallucinated:
                codes = ", ".join(f"FRAC {c}" for c in sorted(hallucinated))
                answer += (
                    f"\n\n---\n🔍 **הערת מערכת:** קודי FRAC הבאים הוזכרו בתשובה "
                    f"אך אינם נמצאים במקורות שאוחזרו: **{codes}**. "
                    f"אנא אמת מידע זה לפני השימוש."
                )
                warnings.append(f"frac_hallucination:{codes}")

        return OutputResult(answer=answer, warnings=warnings)


# ── Singletons (imported by api.py) ───────────────────────────────────────────
input_guard  = InputGuardrail()
output_guard = OutputGuardrail()
