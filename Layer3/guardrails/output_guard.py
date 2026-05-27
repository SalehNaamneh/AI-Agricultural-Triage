"""
Output guardrails — post-process the LLM answer before it reaches the user.

Checks:
  1. Low-confidence disclaimer     (avg retrieval score < threshold)
  2. Resistance-warning enforcement (source warnings must appear in answer)
  3. FRAC-code hallucination flag  (codes in answer not found in sources)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class OutputResult:
    answer: str
    warnings: list[str] = field(default_factory=list)


_FRAC_RE = re.compile(r"\bFRAC\s+([A-Z]?\d+)\b", re.IGNORECASE)


class OutputGuardrail:

    LOW_CONFIDENCE_THRESHOLD = 0.30   # avg of top-3 source scores

    def check(self, answer: str, sources: list[dict]) -> OutputResult:
        """
        sources: list of dicts with keys — type, disease, score,
                 and optionally frac_code, resistance_warning.
        """
        warnings: list[str] = []

        # ── 1. Low-confidence disclaimer ───────────────────────────────────────
        scores = [s.get("score", 0.0) for s in sources[:3]]
        avg = sum(scores) / len(scores) if scores else 0.0
        if avg < self.LOW_CONFIDENCE_THRESHOLD:
            answer = (
                "⚠️ **שים לב:** המידע שנמצא עבור שאלה זו היה חלקי. "
                "ייתכן שהתשובה אינה מלאה — מומלץ להתייעץ עם אגרונום.\n\n"
            ) + answer
            warnings.append(f"low_confidence(avg={avg:.2f})")

        # ── 2. Resistance-warning enforcement ─────────────────────────────────
        resistance_texts = list({
            s["resistance_warning"]
            for s in sources
            if s.get("resistance_warning")
        })
        if resistance_texts:
            # Keywords that indicate resistance was already addressed in the answer.
            # "FRAC" alone is excluded — mentioning a FRAC code is not the same as
            # addressing a resistance warning.
            kw = [
                "עמידות", "סובב", "לסובב", "סיבוב", "החלף", "להחליף",
                "resistance", "rotate", "rotation", "alternating",
                "limit", "הגבל", "פעמים בעונה", "יישומים", "applications per",
                "do not apply more", "אל תחזור",
            ]
            if not any(k.lower() in answer.lower() for k in kw):
                joined = "\n• ".join(resistance_texts)
                answer += f"\n\n---\n⚠️ **אזהרת עמידות (מקורות):**\n• {joined}"
                warnings.append("resistance_warning_appended")

        # ── 3. FRAC-code hallucination check ──────────────────────────────────
        known_fracs = {
            m.group(1).upper()
            for s in sources
            for m in [_FRAC_RE.search(s.get("frac_code", "") or "")]
            if m
        }
        if known_fracs:
            answer_fracs = {m.group(1).upper() for m in _FRAC_RE.finditer(answer)}
            hallucinated = answer_fracs - known_fracs
            if hallucinated:
                codes = ", ".join(f"FRAC {c}" for c in sorted(hallucinated))
                answer += (
                    f"\n\n---\n🔍 **הערת מערכת:** קודי FRAC הבאים הוזכרו בתשובה "
                    f"אך אינם במקורות שאוחזרו: **{codes}**. אנא אמת לפני השימוש."
                )
                warnings.append(f"frac_hallucination:{codes}")

        return OutputResult(answer=answer, warnings=warnings)
