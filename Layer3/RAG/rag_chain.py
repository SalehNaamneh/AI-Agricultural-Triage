import os

from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from retriever import retrieve

OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL",    "llama3.1")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

SYSTEM_PROMPT = """אתה עוזר חקלאי מומחה המתמחה במחלות גידולים חקלאיים וטיפולים בהן.
ענה תמיד בעברית בלבד, גם אם השאלה נשאלת באנגלית.
ענה על שאלת החקלאי על פי ההקשר המסופק בלבד.
אם ההקשר אינו מכיל מידע מספיק, ציין זאת בבירור.
היה מדויק לגבי מינונים, קודי FRAC ואזהרות עמידות — אלו קריטיים לשימוש בטוח.

הקשר:
{context}
"""

_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}"),
])
_parser = StrOutputParser()
_llm: OllamaLLM | None = None


def _get_llm() -> OllamaLLM:
    global _llm
    if _llm is None:
        _llm = OllamaLLM(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
    return _llm


def format_context(docs: list[dict]) -> str:
    return "\n\n---\n\n".join(d["content"] for d in docs)


def ask(question: str, n_results: int = 5) -> dict:
    docs = retrieve(question, n_results=n_results)
    context = format_context(docs)

    chain = _prompt | _get_llm() | _parser
    answer = chain.invoke({"context": context, "question": question})

    return {
        "question": question,
        "answer": answer,
        "sources": [
            {
                "type":               d["metadata"]["type"],
                "disease":            d["metadata"].get("disease_en"),
                "disease_he":         d["metadata"].get("disease_he"),
                "score":              d["score"],
                "frac_code":          d["metadata"].get("frac_code", ""),
                "resistance_warning": d["metadata"].get("resistance_warning", ""),
            }
            for d in docs
        ],
    }


if __name__ == "__main__":
    result = ask("What fungicide should I use for Downy Mildew, and what FRAC code does it need?")
    print("Answer:", result["answer"])
    print("\nSources:")
    for s in result["sources"]:
        print(f"  [{s['type']}] {s['disease']} (score={s['score']:.3f})")
