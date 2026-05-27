import sys
from pathlib import Path

_here   = Path(__file__).resolve().parent      # Layer3/agent/
_layer3 = _here.parent                          # Layer3/
sys.path.insert(0, str(_here))                            # graph, nodes, state
sys.path.insert(0, str(_layer3))                          # guardrails package
sys.path.insert(0, str(_layer3 / "RAG"))                  # rag_chain, retriever, ingest
sys.path.insert(0, str(_layer3 / "ImageAnalyzer"))        # predict, model, dataset

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from graph import get_graph

app = FastAPI(title="AgriTriage LangGraph Agent", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/chat")
async def chat(
    question: str = Form(...),
    crop_id:  str = Form(default="onion"),
    image: UploadFile | None = File(default=None),
):
    image_bytes = await image.read() if image else None

    try:
        result = get_graph().invoke({
            "question":      question,
            "crop_id":       crop_id,
            "image_bytes":   image_bytes,
            "prediction":    None,
            "answer":        "",
            "sources":       [],
            "warnings":      [],
            "blocked":       False,
            "block_reason":  "",
            "block_message": "",
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if result["blocked"]:
        return {
            "question":   question,
            "answer":     result["block_message"],
            "sources":    [],
            "warnings":   [],
            "blocked":    True,
            "reason":     result["block_reason"],
            "prediction": None,
        }

    return {
        "question":   question,
        "answer":     result["answer"],
        "sources":    result["sources"],
        "warnings":   result["warnings"],
        "blocked":    False,
        "prediction": result.get("prediction"),
    }


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8001, reload=False)
