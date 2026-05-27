"""Quick spot-check: V1 vs V5 on the 3 failure cases."""
import sys, json
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Layer3"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Layer3" / "RAG"))

from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from retriever import retrieve

llm    = OllamaLLM(model="llama3.1", base_url="http://localhost:11434")
parser = StrOutputParser()

SPOT = [
    ("TC02_en_input", "What is purple blotch and how do I treat it?"),
    ("TC04_vague",    "הצמח שלי נראה חולה, מה עושים?"),
    ("TC07_dosage",   "מה המינון המומלץ לריסוס Score?"),
]

V1 = """אתה עוזר חקלאי מומחה המתמחה במחלות גידולים חקלאיים וטיפולים בהן.
ענה תמיד בעברית בלבד, גם אם השאלה נשאלת באנגלית.
ענה על שאלת החקלאי על פי ההקשר המסופק בלבד.
אם ההקשר אינו מכיל מידע מספיק, ציין זאת בבירור.
היה מדויק לגבי מינונים, קודי FRAC ואזהרות עמידות — אלו קריטיים לשימוש בטוח.
הקשר: {context}"""

V5 = """אתה עוזר חקלאי מומחה המתמחה במחלות גידולים חקלאיים וטיפולים בהן.
ענה תמיד בעברית בלבד, גם אם השאלה נשאלת באנגלית.
השתמש אך ורק במידע המופיע בהקשר שלהלן — אסור להמציא מידע שאינו בהקשר.
אם ההקשר אינו מכיל תשובה לשאלה, השב: "המידע על כך אינו זמין במסד הנתונים שלי."
אל תמציא קודי FRAC, מינונים, או שמות תכשירים שאינם מופיעים בהקשר.
אם השאלה כללית מדי, ענה על המחלה הרלוונטית ביותר בהקשר.
אם השאלה אינה חקלאית, השב: "אני מתמחה במחלות גידולים חקלאיים בלבד."
בנה את תשובתך לפי המבנה הבא (אם רלוונטי):
1. **מחלה/בעיה**: שם ותיאור קצר
2. **תסמינים**: הסימנים שיש לחפש
3. **טיפול מומלץ**: שם תכשיר, מינון, קוד FRAC
4. **אזהרת עמידות**: חובה לציין אם קיימת
סיים תמיד בשורה: "💡 המידע מבוסס על מסד הנתונים החקלאי של המערכת."
הקשר: {context}"""

results = {"V1": {}, "V5": {}}
for version, sys_p in [("V1", V1), ("V5", V5)]:
    prompt = ChatPromptTemplate.from_messages([("system", sys_p), ("human", "{question}")])
    chain  = prompt | llm | parser
    for tc_id, question in SPOT:
        print(f"  {version} {tc_id}...", flush=True)
        docs    = retrieve(question, n_results=3)
        context = "\n\n---\n\n".join(d["content"] for d in docs)
        answer  = chain.invoke({"context": context, "question": question})
        results[version][tc_id] = answer.strip()
        print(f"    → {answer[:80]}...")

out = Path(__file__).parent / "rag_spot_check_results.json"
out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\nSaved to {out}")
