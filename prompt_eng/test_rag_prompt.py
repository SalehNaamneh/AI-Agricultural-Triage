"""
Prompt Engineering — Surface 1: RAG System Prompt
Runs 10 test cases against each prompt version and saves outputs.
"""
import sys, json, textwrap
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Layer3"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Layer3" / "RAG"))

from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from retriever import retrieve

llm = OllamaLLM(model="llama3.1", base_url="http://localhost:11434")
parser = StrOutputParser()

TEST_CASES = [
    ("TC01", "מה הטיפול בכתם סגול בבצל?"),
    ("TC02", "What is purple blotch and how do I treat it?"),
    ("TC03", "What FRAC code prevents resistance for purple blotch?"),
    ("TC04", "הצמח שלי נראה חולה, מה עושים?"),
    ("TC05", "Is it safe to mix two different fungicides on the same day?"),
    ("TC06", "סיכום של כל המחלות בבצל"),
    ("TC07", "מה המינון המומלץ לריסוס Score?"),
    ("TC08", "What diseases affect onions?"),
    ("TC09", "Tell me about Botrytis in onion"),
    ("TC10", "האם אפשר להשתמש באותו תכשיר שלוש עונות ברצף?"),
]

def run_version(version_name: str, system_prompt: str):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}"),
    ])
    chain = prompt | llm | parser
    results = []
    for tc_id, question in TEST_CASES:
        docs = retrieve(question, n_results=3)
        context = "\n\n---\n\n".join(d["content"] for d in docs)
        answer = chain.invoke({"context": context, "question": question})
        results.append({
            "id": tc_id,
            "question": question,
            "answer": answer.strip(),
            "top_source": docs[0]["metadata"].get("disease_en", "?") if docs else "none",
            "top_score": round(docs[0]["score"], 3) if docs else 0,
        })
        print(f"  {tc_id} done")
    return results


# ── VERSION 1 — Baseline ───────────────────────────────────────────────────────
V1 = """אתה עוזר חקלאי מומחה המתמחה במחלות גידולים חקלאיים וטיפולים בהן.
ענה תמיד בעברית בלבד, גם אם השאלה נשאלת באנגלית.
ענה על שאלת החקלאי על פי ההקשר המסופק בלבד.
אם ההקשר אינו מכיל מידע מספיק, ציין זאת בבירור.
היה מדויק לגבי מינונים, קודי FRAC ואזהרות עמידות — אלו קריטיים לשימוש בטוח.

הקשר:
{context}
"""

# ── VERSION 2 — Fix: enforce Hebrew + no hallucination ────────────────────────
V2 = """אתה עוזר חקלאי מומחה המתמחה במחלות גידולים חקלאיים וטיפולים בהן.
ענה תמיד בעברית בלבד, גם אם השאלה נשאלת באנגלית.
השתמש אך ורק במידע המופיע בהקשר שלהלן — אסור להמציא מידע שאינו בהקשר.
אם ההקשר אינו מכיל תשובה לשאלה, השב: "המידע על כך אינו זמין במסד הנתונים שלי."
אל תמציא קודי FRAC, מינונים, או שמות תכשירים שאינם מופיעים בהקשר.

הקשר:
{context}
"""

# ── VERSION 3 — Fix: structured output format ─────────────────────────────────
V3 = """אתה עוזר חקלאי מומחה המתמחה במחלות גידולים חקלאיים וטיפולים בהן.
ענה תמיד בעברית בלבד, גם אם השאלה נשאלת באנגלית.
השתמש אך ורק במידע המופיע בהקשר שלהלן — אסור להמציא מידע שאינו בהקשר.
אם ההקשר אינו מכיל תשובה לשאלה, השב: "המידע על כך אינו זמין במסד הנתונים שלי."
אל תמציא קודי FRAC, מינונים, או שמות תכשירים שאינם מופיעים בהקשר.

בנה את תשובתך לפי המבנה הבא (אם רלוונטי):
1. **מחלה/בעיה**: שם ותיאור קצר
2. **תסמינים**: הסימנים שיש לחפש
3. **טיפול מומלץ**: שם תכשיר, מינון, קוד FRAC
4. **אזהרת עמידות**: אם קיימת בהקשר

הקשר:
{context}
"""

# ── VERSION 4 — Fix: handle vague + multi-disease context ────────────────────
V4 = """אתה עוזר חקלאי מומחה המתמחה במחלות גידולים חקלאיים וטיפולים בהן.
ענה תמיד בעברית בלבד, גם אם השאלה נשאלת באנגלית.
השתמש אך ורק במידע המופיע בהקשר שלהלן — אסור להמציא מידע שאינו בהקשר.
אם ההקשר אינו מכיל תשובה לשאלה, השב: "המידע על כך אינו זמין במסד הנתונים שלי."
אל תמציא קודי FRAC, מינונים, או שמות תכשירים שאינם מופיעים בהקשר.
אם השאלה כללית מדי או מתייחסת למספר מחלות, ענה על המחלה הרלוונטית ביותר בהקשר.

בנה את תשובתך לפי המבנה הבא (אם רלוונטי):
1. **מחלה/בעיה**: שם ותיאור קצר
2. **תסמינים**: הסימנים שיש לחפש
3. **טיפול מומלץ**: שם תכשיר, מינון, קוד FRAC
4. **אזהרת עמידות**: אם קיימת בהקשר

אם השאלה אינה חקלאית, השב: "אני מתמחה במחלות גידולים חקלאיים בלבד."

הקשר:
{context}
"""

# ── VERSION 5 — Final: add confidence + resistance emphasis ──────────────────
V5 = """אתה עוזר חקלאי מומחה המתמחה במחלות גידולים חקלאיים וטיפולים בהן.
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

VERSIONS = [("V1_baseline", V1), ("V2_no_hallucination", V2),
            ("V3_structured", V3), ("V4_vague_handling", V4), ("V5_final", V5)]

all_results = {}
for name, prompt in VERSIONS:
    print(f"\nRunning {name}...")
    all_results[name] = run_version(name, prompt)

out_path = Path(__file__).parent / "rag_prompt_results.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)
print(f"\nSaved to {out_path}")
