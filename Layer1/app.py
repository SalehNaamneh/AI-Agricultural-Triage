import csv
import requests
import gradio as gr
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE        = Path(__file__).resolve().parents[1]
DATA_DIR    = BASE / "data" / "data" / "onion"
IMAGES_DIR  = DATA_DIR / "images"
DISEASE_CSV = DATA_DIR / "disease_info" / "onion_diseases.csv"
SPRAY_CSV   = DATA_DIR / "disease_info" / "onion_spray_products.csv"
RAG_URL     = "http://localhost:8000"

# ─── Data ─────────────────────────────────────────────────────────────────────
def _csv(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

DISEASE_INFO = {r["disease_name_en"]: r for r in _csv(DISEASE_CSV)}
TREATMENTS: dict = {}
for _r in _csv(SPRAY_CSV):
    TREATMENTS.setdefault(_r["disease_name_en"], []).append(_r)

DISEASES = [
    {"folder": "Purple blotch",           "name": "כתם סגול",           "csv_key": "Purple Blotch"},
    {"folder": "Downy mildew",            "name": "כימשון הבצל",         "csv_key": "Downy Mildew"},
    {"folder": "stemphylium Leaf Blight", "name": "סטמפיליום",           "csv_key": "Stemphylium Leaf Blight"},
    {"folder": "Healthy leaves",          "name": "בצל בריא",            "csv_key": "Healthy Onion"},
    {"folder": "Bulb Rot",                "name": "ריקבון הבצל",         "csv_key": None},
    {"folder": "Bulb_blight-D",           "name": "כימשון הבצלת",        "csv_key": None},
    {"folder": "Caterpillar-P",           "name": "זחל (מזיק)",          "csv_key": None},
    {"folder": "Fusarium-D",              "name": "פוזריום",              "csv_key": None},
    {"folder": "Rust",                    "name": "חלודה",               "csv_key": None},
    {"folder": "Virosis-D",               "name": "וירוס",               "csv_key": None},
]

def _first_img(folder: str):
    p = IMAGES_DIR / folder
    imgs = sorted(p.glob("*.jpg")) if p.exists() else []
    return str(imgs[0]) if imgs else None

GALLERY, GALLERY_MAP = [], []
for _d in DISEASES:
    _img = _first_img(_d["folder"])
    if _img:
        GALLERY.append((_img, _d["name"]))
        GALLERY_MAP.append(_d)

# ─── HTML generators ──────────────────────────────────────────────────────────
def _treatment_card(t: dict) -> str:
    warn  = "⚠" in t.get("resistance_warning", "")
    color = "#c1121f" if warn else "#40916c"
    dose  = t.get("dose_per_dunam", "")
    dose_html = f'<div style="font-size:12px;color:#495057;margin-bottom:5px">📏 {dose}</div>' if dose else ""
    warn_html = f'<div style="font-size:11px;color:{"#c1121f" if warn else "#6c757d"};font-style:italic;margin-top:4px">{"⚠️ " if warn else ""}{t.get("resistance_warning","")}</div>'
    return f"""
<div style="background:white;border-radius:14px;padding:16px;margin:6px;
     box-shadow:0 3px 12px rgba(0,0,0,0.10);border-left:5px solid {color};
     flex:1 1 190px;min-width:190px;max-width:250px;direction:rtl">
  <div style="font-weight:700;font-size:14px;color:#1b4332;margin-bottom:3px">{t['product_name_he']}</div>
  <div style="font-size:11px;color:#888;margin-bottom:2px">{t['product_name_en']}</div>
  <div style="font-size:12px;color:#6c757d;margin-bottom:9px">{t['active_ingredient_he']}</div>
  <div style="display:flex;gap:5px;flex-wrap:wrap;margin-bottom:9px">
    <span style="background:#1b4332;color:white;padding:2px 10px;border-radius:20px;font-size:11px;font-weight:700">{t['frac_code']}</span>
    <span style="background:#d8f3dc;color:#1b4332;padding:2px 10px;border-radius:20px;font-size:11px">{t['product_type_he']}</span>
  </div>
  {dose_html}
  {warn_html}
</div>"""

def _info_grid(info: dict) -> str:
    def bullet(text): return "• " + "<br>• ".join(text.split(" | ")) if text else "—"
    confusion = (
        f'<div style="background:#fff3cd;border-radius:10px;padding:12px 16px;'
        f'font-size:13px;color:#856404;margin-top:16px;direction:rtl">'
        f'<strong>⚠️ סיכון בלבול:</strong> {info["confusion_risk_he"]}</div>'
    ) if info.get("confusion_risk_he") else ""
    return f"""
<div style="margin-bottom:8px;direction:rtl">
  <h2 style="color:#1b4332;font-size:22px;margin:0 0 3px">{info['disease_name_he']}</h2>
  <div style="color:#52b788;font-style:italic;font-size:13px;margin-bottom:14px">
    {info['scientific_name']} &nbsp;·&nbsp; {info['pathogen_type_he']}
  </div>
  <p style="color:#444;line-height:1.75;margin-bottom:20px">{info['description_he']}</p>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
    <div style="background:#f0faf5;border-radius:12px;padding:14px">
      <div style="font-weight:700;color:#2d6a4f;margin-bottom:8px">👁️ תסמינים חזותיים</div>
      <div style="font-size:13px;color:#444;line-height:1.75">{bullet(info['visual_symptoms_he'])}</div>
    </div>
    <div style="background:#f0faf5;border-radius:12px;padding:14px">
      <div style="font-weight:700;color:#2d6a4f;margin-bottom:8px">🔍 סימן אבחנתי</div>
      <div style="font-size:13px;color:#444;line-height:1.75">{info['diagnostic_sign_he']}</div>
    </div>
    <div style="background:#fff8e1;border-radius:12px;padding:14px">
      <div style="font-weight:700;color:#e65100;margin-bottom:8px">⚡ תנאים מעדיפים</div>
      <div style="font-size:13px;color:#444;line-height:1.75">{bullet(info['favorable_conditions_he'])}</div>
    </div>
    <div style="background:#e8f5e9;border-radius:12px;padding:14px">
      <div style="font-weight:700;color:#1b5e20;margin-bottom:8px">🛡️ מניעה</div>
      <div style="font-size:13px;color:#444;line-height:1.75">{bullet(info['agronomic_prevention_he'])}</div>
    </div>
  </div>
  {confusion}
</div>"""

def disease_detail_html(d: dict) -> str:
    info   = DISEASE_INFO.get(d["csv_key"])  if d["csv_key"] else None
    treats = TREATMENTS.get(d["csv_key"], []) if d["csv_key"] else []

    if not info and not treats:
        return f"""
<div style="text-align:center;padding:48px;color:#aaa;direction:rtl">
  <div style="font-size:52px;margin-bottom:12px">🌿</div>
  <h3 style="color:#2d6a4f">{d['name']}</h3>
  <p>מידע מלא יתווסף בעדכון הקרוב.</p>
</div>"""

    info_block  = _info_grid(info) if info else ""
    treat_block = ""
    if treats:
        cards = "".join(_treatment_card(t) for t in treats)
        treat_block = f"""
<div style="margin-top:24px;direction:rtl">
  <h3 style="color:#1b4332;margin-bottom:14px">
    💊 אפשרויות טיפול
    <span style="background:#d8f3dc;color:#1b4332;font-size:13px;padding:2px 10px;
          border-radius:20px;margin-right:8px;font-weight:600">{len(treats)} תכשירים</span>
  </h3>
  <div style="display:flex;flex-wrap:wrap;gap:8px">{cards}</div>
</div>"""

    return f'<div style="padding:4px 4px">{info_block}{treat_block}</div>'

PLACEHOLDER = """
<div style="text-align:center;padding:56px 20px;color:#c0c8c4;
     border:2px dashed #c8e6c9;border-radius:16px;margin-top:8px;direction:rtl">
  <div style="font-size:56px;margin-bottom:14px">🌾</div>
  <div style="font-size:16px;font-weight:500">לחץ על תמונת מחלה כדי לראות פרטים ואפשרויות טיפול</div>
</div>"""

def on_gallery_select(evt: gr.SelectData) -> str:
    idx = evt.index
    if isinstance(idx, (list, tuple)):
        idx = idx[0]
    if 0 <= idx < len(GALLERY_MAP):
        return disease_detail_html(GALLERY_MAP[idx])
    return PLACEHOLDER

# ─── Chat logic ───────────────────────────────────────────────────────────────
WELCOME = (
    "👋 שלום! אני **AgriTriage AI**, העוזר החקלאי שלך.\n\n"
    "שאל אותי כל שאלה על מחלות בצל, תסמינים, טיפולים ומניעה — "
    "או העלה תמונה של הצמח ותאר מה אתה רואה. אשמח לעזור לזהות את הבעיה ולהמליץ על פתרונות."
)

def respond(message, history: list) -> tuple:
    if isinstance(message, dict):
        text  = (message.get("text") or "").strip()
        files = message.get("files") or []
    else:
        text, files = str(message).strip(), []

    if not text and not files:
        return history, gr.MultimodalTextbox(value=None)

    query = text or "נתח את תמונת הצמח הזו — איזו מחלה או בעיה אתה מזהה?"

    if files:
        user_content = [{"type": "image", "value": files[0]}, {"type": "text", "value": query}]
    else:
        user_content = query
    history.append({"role": "user", "content": user_content})

    try:
        r = requests.post(f"{RAG_URL}/ask", json={"question": query}, timeout=90)
        data    = r.json()
        answer  = data.get("answer", "לא התקבלה תשובה.")
        sources = data.get("sources", [])
        if sources:
            diseases = list({s["disease"] for s in sources if s.get("disease")})
            footer = f"\n\n---\n📚 *מקורות: {', '.join(diseases)}*"
        else:
            footer = ""
        reply = answer + footer
    except Exception as e:
        reply = (
            "⚠️ **לא ניתן להתחבר לשירות ה-RAG.**\n\n"
            f"ודא ש-`Layer3/RAG/api.py` פועל על פורט 8000.\n\n`{e}`"
        )

    history.append({"role": "assistant", "content": reply})
    return history, gr.MultimodalTextbox(value=None)

def clear_chat():
    return [{"role": "assistant", "content": WELCOME}]

# ─── CSS ──────────────────────────────────────────────────────────────────────
CSS = """
* { box-sizing: border-box; }

body, .gradio-container { direction: rtl; }

.app-header {
    background: linear-gradient(135deg, #1b4332 0%, #2d6a4f 55%, #52b788 100%);
    color: white; text-align: center; padding: 30px 24px 26px;
    border-radius: 18px; margin-bottom: 10px;
}
.app-header h1 { margin: 0 0 6px; font-size: 30px; font-weight: 800; letter-spacing: -0.5px; }
.app-header p  { margin: 0; opacity: 0.88; font-size: 15px; }

#chatbot { border-radius: 16px !important; }

.clear-btn {
    border-radius: 10px !important;
    color: #6c757d !important;
    border-color: #dee2e6 !important;
}

#disease-detail {
    border: 1.5px solid #d8f3dc;
    border-radius: 18px;
    padding: 20px;
    background: #fafffc;
    margin-top: 10px;
}

.gallery-label {
    font-weight: 600; color: #2d6a4f; font-size: 14px;
    margin: 12px 0 6px; padding-right: 2px;
}

footer { display: none !important; }
"""

# ─── UI ───────────────────────────────────────────────────────────────────────
with gr.Blocks(title="AgriTriage AI") as demo:

    gr.HTML("""
    <div class="app-header">
      <h1>🌾 AgriTriage AI</h1>
      <p>מערכת אינטליגנטית לזיהוי וניהול מחלות בצל</p>
    </div>""")

    with gr.Tabs():

        # ── Tab 1: AI Assistant ───────────────────────────────────────────────
        with gr.Tab("💬 עוזר חכם"):

            chatbot = gr.Chatbot(
                value=[{"role": "assistant", "content": WELCOME}],
                height=460,
                show_label=False,
                elem_id="chatbot",
            )

            msg_box = gr.MultimodalTextbox(
                placeholder="שאל על מחלות, תסמינים, טיפולים... או העלה תמונה של הצמח 📷",
                show_label=False,
                file_types=["image"],
                lines=1,
                submit_btn=True,
            )

            with gr.Row():
                clear_btn = gr.Button("🗑️  נקה שיחה", elem_classes="clear-btn",
                                      size="sm", variant="secondary")

            msg_box.submit(respond, [msg_box, chatbot], [chatbot, msg_box])
            clear_btn.click(clear_chat, outputs=[chatbot])

        # ── Tab 2: Disease Explorer ───────────────────────────────────────────
        with gr.Tab("🔬 סייר מחלות"):

            with gr.Row():
                gr.Dropdown(
                    choices=["🧅 בצל"], value="🧅 בצל",
                    label="גידול", scale=0, min_width=160,
                )
                gr.HTML(
                    '<div style="display:flex;align-items:center;padding:8px 4px;'
                    'color:#6c757d;font-size:13px;direction:rtl">'
                    'גידולים נוספים בקרוב — עגבנייה, חיטה, תפוח אדמה...</div>'
                )

            gr.HTML('<div class="gallery-label">🖼️ לחץ על מחלה לצפייה בפרטים וטיפולים ↓</div>')

            gallery = gr.Gallery(
                value=GALLERY,
                columns=5,
                rows=2,
                height=340,
                show_label=False,
                object_fit="cover",
                allow_preview=False,
            )

            detail_panel = gr.HTML(value=PLACEHOLDER, elem_id="disease-detail")
            gallery.select(on_gallery_select, outputs=[detail_panel])

if __name__ == "__main__":
    demo.launch(
        server_port=7860,
        inbrowser=True,
        css=CSS,
        theme=gr.themes.Soft(primary_hue="green"),
    )