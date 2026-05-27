from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag_chain import ask
from retriever import retrieve

app = FastAPI(title="Agricultural RAG API", version="1.0.0")


class QueryRequest(BaseModel):
    question: str
    n_results: int = 5


class RetrieveRequest(BaseModel):
    query: str
    n_results: int = 5
    doc_type: str | None = None


@app.post("/ask")
def ask_question(req: QueryRequest):
    try:
        return ask(req.question, n_results=req.n_results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retrieve")
def retrieve_docs(req: RetrieveRequest):
    try:
        docs = retrieve(req.query, n_results=req.n_results, doc_type=req.doc_type)
        return {"query": req.query, "results": docs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)