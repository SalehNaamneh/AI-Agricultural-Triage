import io
import sys
import requests
import gradio as gr
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crop_config import load_all_crops, CropConfig

RAG_URL            = "http://localhost:8000"
IMAGE_ANALYZER_URL = "http://localhost:8002"
N8N_WEBHOOK_URL    = "http://localhost:5678/webhook/agritriage"

ALL_CROPS = load_all_crops()
DEFAULT_CROP = next(iter(ALL_CROPS)) if ALL_CROPS else None
CROP_CHOICES = [(f"{c.icon} {c.name_he}", crop_id) for crop_id, c in ALL_CROPS.items()]


def _first_img(crop: CropConfig, folder: str) -> str | None:
    p = crop.images_dir / folder
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"):
        imgs = sorted(p.glob(pattern))
        if imgs:
            return str(imgs[0])
    return None


def build_gallery(crop_id: str):
    if crop_id not in ALL_CROPS:
        return [], []
    crop = ALL_CROPS[crop_id]
    gallery, gallery_map = [], []
    for cls in crop.classes:
        img = _first_img(crop, cls.folder)
        if img:
            gallery.append((img, cls.name_he))
            gallery_map.append((crop_id, cls))
    return gallery, gallery_map


# ─── HTML generators ──────────────────────────────────────────────────────────
def _treatment_card(t: dict) -> str:
    warn      = "⚠" in t.get("resistance_warning", "")
    color     = "#c1121f" if warn else "#40916c"
    dose      = t.get("dose_per_dunam", "")
    dose_html = (
        f'<div style="font-size:12px;color:#495057;margin-bottom:5px">📏 {dose}</div>'
        if dose else ""
    )
    warn_html = (
        f'<div style="font-size:11px;color:{"#c1121f" if warn else "#6c757d"};'
        f'font-style:italic;margin-top:4px">{"⚠️ " if warn else ""}'
        f'{t.get("resistance_warning","")}</div>'
    )
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


def disease_detail_html(crop_id: str, cls) -> str:
    crop         = ALL_CROPS[crop_id]
    disease_info = crop.load_disease_info()
    treatments   = crop.load_treatments()

    info   = disease_info.get(cls.csv_key)  if cls.csv_key else None
    treats = treatments.get(cls.csv_key, []) if cls.csv_key else []

    if not info and not treats:
        return f"""
<div style="text-align:center;padding:48px;color:#aaa;direction:rtl">
  <div style="font-size:52px;margin-bottom:12px">🌿</div>
  <h3 style="color:#2d6a4f">{cls.name_he}</h3>
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


def on_gallery_select(evt: gr.SelectData, gallery_map: list) -> str:
    idx = evt.index
    if isinstance(idx, (list, tuple)):
        idx = idx[0]
    if gallery_map and 0 <= idx < len(gallery_map):
        crop_id, cls = gallery_map[idx]
        return disease_detail_html(crop_id, cls)
    return PLACEHOLDER


def on_crop_change(crop_id: str):
    gallery, gallery_map = build_gallery(crop_id)
    return gallery, gallery_map, PLACEHOLDER, crop_id


# ─── Chat logic ───────────────────────────────────────────────────────────────
WELCOME = (
    "👋 שלום! אני **AgriTriage AI**, העוזר החקלאי שלך.\n\n"
    "שאל אותי כל שאלה על מחלות גידולים, תסמינים, טיפולים ומניעה — "
    "או העלה תמונה של הצמח ואזהה את הבעיה ואמליץ על פתרונות."
)


def respond(message, history: list, crop_id: str) -> tuple:
    if isinstance(message, dict):
        text  = (message.get("text") or "").strip()
        files = message.get("files") or []
    else:
        text, files = str(message).strip(), []

    if not text and not files:
        return history, gr.MultimodalTextbox(value=None)

    prediction_banner = ""
    rag_prefix        = ""

    if files:
        try:
            with open(files[0], "rb") as img_file:
                resp = requests.post(
                    f"{IMAGE_ANALYZER_URL}/predict",
                    files={"file": ("image.jpg", img_file, "image/jpeg")},
                    params={"crop": crop_id},
                    timeout=30,
                )
            if resp.ok:
                pred       = resp.json()
                class_he   = pred.get("class_he", "")
                class_en   = pred.get("class_en", "")
                confidence = pred.get("confidence", 0)
                health     = pred.get("health_score", 0)
                crop_name  = ALL_CROPS[crop_id].name_he if crop_id in ALL_CROPS else crop_id
                prediction_banner = (
                    f"🔍 **זוהה: {class_he}** ({class_en}) — בטחון {confidence}%"
                    + (f" | בריאות: {health}%" if health else "")
                    + "\n\n"
                )
                rag_prefix = f"בתמונה זוהתה מחלה: {class_he} ({class_en}) בגידול {crop_name}. "
        except Exception:
            pass

    query = text or "ספר לי על המחלה הזו וכיצד לטפל בה."

    if files:
        user_content = [{"type": "image", "value": files[0]}, {"type": "text", "value": query}]
    else:
        user_content = query

    history.append({"role": "user", "content": user_content})

    try:
        rag_question = rag_prefix + query
        r       = requests.post(f"{RAG_URL}/ask", json={"question": rag_question}, timeout=90)
        data    = r.json()
        answer  = data.get("answer", "לא התקבלה תשובה.")
        sources = data.get("sources", [])
        footer  = ""
        if sources:
            diseases = list({s["disease"] for s in sources if s.get("disease")})
            footer = f"\n\n---\n📚 *מקורות: {', '.join(diseases)}*"
        reply = prediction_banner + answer + footer
    except Exception as e:
        reply = (
            prediction_banner
            + "⚠️ **לא ניתן להתחבר לשירות ה-RAG.**\n\n"
            + f"ודא ש-`Layer3/RAG/api.py` פועל על פורט 8000.\n\n`{e}`"
        )

    history.append({"role": "assistant", "content": reply})
    return history, gr.MultimodalTextbox(value=None)


def clear_chat():
    return [{"role": "assistant", "content": WELCOME}]


# ─── Form Submission (n8n webhook) ───────────────────────────────────────────
def submit_form(text: str, image, crop_id: str) -> str:
    if not text.strip() and image is None:
        return '<div dir="rtl" style="color:#c1121f;padding:16px">⚠️ יש להזין תיאור או לצרף תמונה.</div>'

    payload: dict = {"text": text.strip(), "crop_id": crop_id}

    if image is not None:
        buf = io.BytesIO()
        img = Image.fromarray(image) if not isinstance(image, Image.Image) else image
        img.save(buf, format="JPEG")
        import base64
        payload["image_base64"] = base64.b64encode(buf.getvalue()).decode()

    try:
        r = requests.post(N8N_WEBHOOK_URL, json=payload, timeout=120)
        if r.ok:
            data    = r.json()
            report  = data.get("report", "")
            warnings = data.get("warnings", [])
            warn_html = ""
            if warnings:
                items = "".join(f"<li>{w}</li>" for w in warnings)
                warn_html = f'<ul style="color:#856404;margin:8px 0 0;font-size:13px">{items}</ul>'
            return (
                f'<div dir="rtl" style="background:#f0faf5;border-radius:14px;padding:20px;'
                f'border-left:5px solid #2d6a4f">'
                f'<div style="white-space:pre-wrap;line-height:1.8;color:#1b4332">{report}</div>'
                f'{warn_html}</div>'
            )
        else:
            return (
                f'<div dir="rtl" style="color:#c1121f;padding:16px">'
                f'⚠️ שגיאה מהשרת: {r.status_code}<br><code>{r.text[:400]}</code></div>'
            )
    except Exception as exc:
        return (
            f'<div dir="rtl" style="color:#c1121f;padding:16px">'
            f'⚠️ לא ניתן להתחבר לשרת n8n.<br>'
            f'ודא ש-n8n פועל ו-<code>N8N_WEBHOOK_URL</code> נכון.<br>'
            f'<code>{exc}</code></div>'
        )


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

.crop-selector-row {
    background: #f0faf5; border-radius: 14px; padding: 14px 20px;
    margin-bottom: 8px; border: 1px solid #d8f3dc;
    display: flex; align-items: center; gap: 12px;
}

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
_init_gallery, _init_map = build_gallery(DEFAULT_CROP) if DEFAULT_CROP else ([], [])

with gr.Blocks(title="AgriTriage AI") as demo:

    gr.HTML("""
    <div class="app-header">
      <h1>🌾 AgriTriage AI</h1>
      <p>מערכת אינטליגנטית לזיהוי וניהול מחלות גידולים</p>
    </div>""")

    # ── Shared crop selector ──────────────────────────────────────────────────
    with gr.Row(elem_classes="crop-selector-row"):
        crop_dropdown = gr.Dropdown(
            choices=CROP_CHOICES,
            value=DEFAULT_CROP,
            label="גידול נוכחי",
            scale=0,
            min_width=180,
        )
        gr.HTML(
            '<div style="color:#52b788;font-size:13px;padding:4px 0;direction:rtl">'
            "בחר גידול — הצ'אט וסייר המחלות יעודכנו בהתאם"
            "</div>"
        )

    crop_state       = gr.State(value=DEFAULT_CROP)
    gallery_map_state = gr.State(value=_init_map)

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
                placeholder="שאל על מחלות, תסמינים, טיפולים... או העלה תמונה 📷",
                show_label=False,
                file_types=["image"],
                lines=1,
                submit_btn=True,
            )

            with gr.Row():
                clear_btn = gr.Button(
                    "🗑️  נקה שיחה", elem_classes="clear-btn",
                    size="sm", variant="secondary"
                )

            msg_box.submit(respond, [msg_box, chatbot, crop_state], [chatbot, msg_box])
            clear_btn.click(clear_chat, outputs=[chatbot])

        # ── Tab 2: Disease Explorer ───────────────────────────────────────────
        with gr.Tab("🔬 סייר מחלות"):

            gr.HTML('<div class="gallery-label">🖼️ לחץ על מחלה לצפייה בפרטים וטיפולים ↓</div>')

            gallery = gr.Gallery(
                value=_init_gallery,
                columns=5,
                rows=2,
                height=340,
                show_label=False,
                object_fit="cover",
                allow_preview=False,
            )

            detail_panel = gr.HTML(value=PLACEHOLDER, elem_id="disease-detail")
            gallery.select(on_gallery_select, [gallery_map_state], [detail_panel])

        # ── Tab 3: Form Submission ────────────────────────────────────────────
        with gr.Tab("📋 שליחת דיווח"):

            gr.HTML(
                '<div dir="rtl" style="background:#fff8e1;border-radius:12px;padding:12px 16px;'
                'margin-bottom:8px;font-size:13px;color:#e65100;border:1px solid #ffe082">'
                '📡 דיווח זה נשלח לשרת n8n ומעובד על ידי כל שכבות המערכת (Guard → AI Agent → Report → Guard).'
                '</div>'
            )

            with gr.Row():
                with gr.Column(scale=2):
                    form_text = gr.Textbox(
                        label="תיאור הבעיה",
                        placeholder="תאר את התסמינים שנצפו — צבע, צורה, מיקום על הצמח...",
                        lines=4,
                        rtl=True,
                    )
                    form_image = gr.Image(
                        label="תמונת הצמח (אופציונלי)",
                        type="numpy",
                        sources=["upload"],
                    )
                    submit_btn = gr.Button("🚀 שלח לניתוח", variant="primary")

                with gr.Column(scale=3):
                    form_output = gr.HTML(
                        value='<div dir="rtl" style="padding:32px;text-align:center;color:#aaa;'
                              'border:2px dashed #d8f3dc;border-radius:16px">'
                              '<div style="font-size:40px;margin-bottom:8px">🌾</div>'
                              '<div>התוצאות יופיעו כאן לאחר שליחת הדיווח</div></div>',
                        label="תוצאת הניתוח",
                    )

            submit_btn.click(
                submit_form,
                inputs=[form_text, form_image, crop_state],
                outputs=[form_output],
            )

    # ── Crop change wires both tabs ───────────────────────────────────────────
    crop_dropdown.change(
        on_crop_change,
        inputs=[crop_dropdown],
        outputs=[gallery, gallery_map_state, detail_panel, crop_state],
    )

if __name__ == "__main__":
    demo.launch(
        server_port=7860,
        inbrowser=True,
        css=CSS,
        theme=gr.themes.Soft(primary_hue="green"),
    )
