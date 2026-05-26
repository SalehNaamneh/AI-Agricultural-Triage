
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from retriever import retrieve

OLLAMA_MODEL = "llama3.1"

SYSTEM_PROMPT = """אתה עוזר חקלאי מומחה המתמחה במחלות בצל וטיפולים בהן.
ענה תמיד בעברית בלבד, גם אם השאלה נשאלת באנגלית.
ענה על שאלת החקלאי על פי ההקשר המסופק בלבד.
אם ההקשר אינו מכיל מידע מספיק, ציין זאת בבירור.
היה מדויק לגבי מינונים, קודי FRAC ואזהרות עמידות — אלו קריטיים לשימוש בטוח.

הקשר:
{context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}"),
])

llm = OllamaLLM(model=OLLAMA_MODEL)
parser = StrOutputParser()


def format_context(docs: list[dict]) -> str:
    return "\n\n---\n\n".join(d["content"] for d in docs)


def ask(question: str, n_results: int = 5) -> dict:
    docs = retrieve(question, n_results=n_results)
    context = format_context(docs)

    chain = prompt | llm | parser
    answer = chain.invoke({"context": context, "question": question})

    return {
        "question": question,
        "answer": answer,
        "sources": [
            {"type": d["metadata"]["type"], "disease": d["metadata"].get("disease_en"), "score": d["score"]}
            for d in docs
        ],
    }


if __name__ == "__main__":
    result = ask("What fungicide should I use for Downy Mildew, and what FRAC code does it need?")
    print("Answer:", result["answer"])
    print("\nSources:")
    for s in result["sources"]:
        print(f"  [{s['type']}] {s['disease']} (score={s['score']:.3f})")