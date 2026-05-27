"""
Guardrails HTTP service — port 8003.

POST /check-input   { "text": "..." }
POST /check-output  { "answer": "...", "sources": [...] }
GET  /health
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the guardrails package importable when running as a standalone service
_layer3 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_layer3))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from guardrails import input_guard, output_guard

app = FastAPI(title="AgriTriage Guardrails", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ──────────────────────────────────────────────────

class InputRequest(BaseModel):
    text: str

class InputResponse(BaseModel):
    passed: bool
    reason: str
    message_he: str

class Source(BaseModel):
    type: str = ""
    disease: str = ""
    score: float = 0.0
    frac_code: str = ""
    resistance_warning: str = ""

class OutputRequest(BaseModel):
    answer: str
    sources: list[Source] = []

class OutputResponse(BaseModel):
    answer: str
    warnings: list[str]


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/check-input", response_model=InputResponse)
def check_input(req: InputRequest):
    result = input_guard.check(req.text)
    return InputResponse(
        passed=result.passed,
        reason=result.reason,
        message_he=result.message_he,
    )


@app.post("/check-output", response_model=OutputResponse)
def check_output(req: OutputRequest):
    sources = [s.model_dump() for s in req.sources]
    result = output_guard.check(req.answer, sources)
    return OutputResponse(answer=result.answer, warnings=result.warnings)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8003, reload=False)
