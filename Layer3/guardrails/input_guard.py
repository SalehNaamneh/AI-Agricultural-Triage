"""
Input guardrails — run before the LLM call (zero latency, no extra model).

Checks:
  1. Prompt-injection detection  (regex, EN + HE)
  2. Off-topic detection         (keyword allowlist, EN + HE)
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class InputResult:
    passed: bool
    reason: str = ""        # machine-readable code for logs
    message_he: str = ""    # Hebrew message shown to the user


class InputGuardrail:

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
    _AG_KEYWORDS_EN = {
        "disease", "crop", "plant", "fungus", "fungicide", "bacteria",
        "treatment", "spray", "symptom", "leaf", "blight", "mildew",
        "rust", "rot", "virus", "pest", "insect", "caterpillar",
        "onion", "tomato", "wheat", "potato", "agriculture", "agronomic",
        "field", "dose", "dunam", "frac", "active ingredient", "pathogen",
        "healthy", "infection", "spore", "moisture", "humidity",
        "systemic", "contact", "protective", "curative",
    }
    _AG_KEYWORDS_HE = {
        "מחלה", "גידול", "צמח", "עלה", "עלים", "ריסוס", "טיפול",
        "תסמין", "תסמינים", "פטריה", "חיידק", "וירוס", "חרק",
        "זחל", "ריקבון", "חלודה", "כימשון", "כתם", "בוטריטיס",
        "סטמפיליום", "אלטרנריה", "פוזריום", "אפובנטורה",
        "בצל", "עגבנייה", "חיטה", "תפוח", "ירק", "פרי",
        "חקלאי", "שדה", "דונם", "מינון", "תכשיר", "חומר פעיל",
        "מניעה", "אבחנה", "בריאות", "זיהום", "עמידות", "עונה",
    }
    # Short queries (e.g. just a disease name) bypass the topic filter
    _MIN_WORDS_FOR_TOPIC_CHECK = 4

    def check(self, text: str) -> InputResult:
        text = text.strip()

        if self._INJECTION_EN.search(text) or self._INJECTION_HE.search(text):
            return InputResult(
                passed=False,
                reason="prompt_injection",
                message_he=(
                    "⚠️ הבקשה שלך נחסמה — היא מכילה ניסיון לעקוף את הגדרות המערכת.\n"
                    "אנא שאל שאלה חקלאית רגילה."
                ),
            )

        if len(text.split()) >= self._MIN_WORDS_FOR_TOPIC_CHECK:
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
