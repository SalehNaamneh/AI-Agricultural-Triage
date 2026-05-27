import os

from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from retriever import retrieve

# ── LLM provider selection ─────────────────────────────────────────────────────
# LLM_PROVIDER = ollama (default) | openai | gemini
LLM_PROVIDER    = os.getenv("LLM_PROVIDER",    "ollama")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL",    "llama3.1")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OPENAI_MODEL    = os.getenv("OPENAI_MODEL",    "gpt-4o")
GEMINI_MODEL    = os.getenv("GEMINI_MODEL",    "gemini-1.5-pro")

SYSTEM_PROMPT = """אתה עוזר חקלאי מומחה המתמחה במחלות גידולים חקלאיים וטיפולים בהן.
ענה תמיד בעברית בלבד, גם אם השאלה נשאלת באנגלית.
השתמש אך ורק במידע המופיע בהקשר שלהלן — אסור להמציא מידע שאינו בהקשר.
אם ההקשר אינו מכיל תשובה לשאלה, השב: "המידע על כך אינו זמין במסד הנתונים שלי."
אל תמציא קודי FRAC, מינונים, או שמות תכשירים שאינם מופיעים בהקשר.
אם השאלה כללית מדי או מתייחסת למספר מחלות, ענה על המחלה הרלוונטית ביותר בהקשר.
אם השאלה אינה חקלאית, השב: "אני מתמחה במחלות גידולים חקלאיים בלבד."

בנה את תשובתך לפי המבנה הבא (אם רלוונטי):
1. **מחלה/בעיה**: שם ותיאור קצר
2. **תסמינים**: הסימנים שיש לחפש
3. **טיפול מומלץ**: שם תכשיר, מינון, קוד FRAC
4. **אזהרת עמידות**: חובה לציין אם קיימת — זהו מידע קריטי לשימוש בטוח

סיים תמיד בשורה: "💡 המידע מבוסס על מסד הנתונים החקלאי של המערכת."

הקשר:
{context}
"""

_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}"),
])
_parser = StrOutputParser()
_llm = None


def _get_llm():
    global _llm
    if _llm is not None:
        return _llm

    if LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        _llm = ChatOpenAI(
            model=OPENAI_MODEL,
            temperature=0,
            api_key=os.environ["OPENAI_API_KEY"],
        )
    elif LLM_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        _llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            temperature=0,
            google_api_key=os.environ["GOOGLE_API_KEY"],
        )
    else:  # ollama (default)
        # Ollama serves GGUF models using llama.cpp under the hood.
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
