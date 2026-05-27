"""Generates the AgriTriage system architecture diagram."""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(18, 13))
ax.set_xlim(0, 18)
ax.set_ylim(0, 13)
ax.axis("off")
fig.patch.set_facecolor("#f8faf9")

# ── Colour palette ─────────────────────────────────────────────────────────────
C = {
    "layer_bg":  ["#e8f5e9", "#fff8e1", "#e3f2fd", "#fce4ec"],
    "layer_bd":  ["#2d6a4f", "#f9a825", "#1565c0", "#ad1457"],
    "layer_txt": ["#1b4332", "#e65100", "#0d47a1", "#880e4f"],
    "box":       "#ffffff",
    "arrow":     "#546e7a",
    "title_bg":  "#1b4332",
}

LAYERS = [
    (0,  "Layer 1 — User Interface",         "Gradio WebUI  ·  Port 7860",                    0),
    (1,  "Layer 2 — n8n Orchestration",       "Workflow Automation  ·  n8n Cloud / Self-hosted", 1),
    (2,  "Layer 3 — AWS EC2 Microservices",   "Docker Containers  ·  FastAPI",                 2),
    (3,  "Layer 4 — LLM Backends",            "Local Ollama  ·  OpenAI  ·  Google Gemini",     3),
]

layer_h  = 2.6
layer_y  = [9.8, 7.0, 4.2, 1.0]   # top-left y of each layer band
band_pad = 0.15

for i, (_, label, sublabel, ci) in enumerate(LAYERS):
    y = layer_y[i]
    rect = FancyBboxPatch(
        (0.3, y - layer_h + band_pad), 17.4, layer_h - band_pad * 2,
        boxstyle="round,pad=0.1",
        linewidth=2, edgecolor=C["layer_bd"][ci],
        facecolor=C["layer_bg"][ci], zorder=1,
    )
    ax.add_patch(rect)
    ax.text(0.7, y - 0.35, label,   fontsize=12, fontweight="bold",
            color=C["layer_txt"][ci], va="top", zorder=2)
    ax.text(0.7, y - 0.72, sublabel, fontsize=9,  color="#555555",
            va="top", zorder=2)

# ── Helper: service box ────────────────────────────────────────────────────────
def sbox(x, y, w, h, title, subtitle, color="#1b4332", bg="#d8f3dc"):
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.12",
        linewidth=1.5, edgecolor=color, facecolor=bg, zorder=3,
    )
    ax.add_patch(rect)
    ax.text(x + w/2, y + h - 0.22, title,    fontsize=9,  fontweight="bold",
            color=color, ha="center", va="top", zorder=4)
    ax.text(x + w/2, y + 0.22,     subtitle, fontsize=7.5, color="#444444",
            ha="center", va="bottom", zorder=4, style="italic")

# ── Helper: arrow ──────────────────────────────────────────────────────────────
def arrow(x1, y1, x2, y2, label=""):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=C["arrow"],
                                lw=1.5, mutation_scale=14),
                zorder=5)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx+0.08, my, label, fontsize=7, color="#37474f", zorder=6)

# ── LAYER 1: WebUI ─────────────────────────────────────────────────────────────
sbox(1.0,  10.1, 3.8, 1.6, "AI Chat Tab",
     "Multimodal chat\nImage upload  |  Crop selector",
     color="#2d6a4f", bg="#c8e6c9")
sbox(5.2,  10.1, 3.8, 1.6, "Disease Explorer Tab",
     "Gallery of diseases\nClick -> symptoms + treatments",
     color="#2d6a4f", bg="#c8e6c9")
sbox(9.4,  10.1, 3.8, 1.6, "Form Submission",
     "Description text\nImage upload  ->  POST to n8n webhook",
     color="#2d6a4f", bg="#c8e6c9")
sbox(13.6, 10.1, 3.6, 1.6, "Gradio App",
     "Port 7860\nHebrew RTL layout",
     color="#2d6a4f", bg="#c8e6c9")

# ── LAYER 2: n8n ──────────────────────────────────────────────────────────────
n8n_boxes = [
    (0.5,  7.3, "① Webhook", "Trigger"),
    (2.4,  7.3, "② Guard\nInput",  "Pass/Fail"),
    (4.3,  7.3, "③ IF\nBranch",    "Route"),
    (6.2,  7.3, "④ Info\nExtract", "Gemini LM"),
    (8.1,  7.3, "⑤ AI Agent",     "GPT-4o / Gemini"),
    (10.0, 7.3, "⑥ LLM Chain",    "Report synthesis"),
    (11.9, 7.3, "⑦ Guard\nOutput","Validate report"),
    (13.8, 7.3, "⑧ Router",       "Crop type"),
]
for x, y, title, sub in n8n_boxes:
    sbox(x, y, 1.65, 1.45, title, sub, color="#f9a825", bg="#fff9c4")

# arrows between n8n nodes
for i in range(len(n8n_boxes)-1):
    x1 = n8n_boxes[i][0] + 1.65
    x2 = n8n_boxes[i+1][0]
    y_mid = 7.3 + 0.72
    arrow(x1, y_mid, x2, y_mid)

# ── LAYER 3: EC2 services ─────────────────────────────────────────────────────
sbox(0.8,  4.5, 3.5, 1.5, "RAG Service",
     "Port 8000\nChromaDB + LangChain\nOllama inference",
     color="#1565c0", bg="#bbdefb")
sbox(4.7,  4.5, 3.5, 1.5, "LangGraph Agent",
     "Port 8001\n5-node StateGraph\nOrchestrates RAG + CNN",
     color="#1565c0", bg="#bbdefb")
sbox(8.6,  4.5, 3.5, 1.5, "Image Analyzer",
     "Port 8002\nEfficientNet-B0 / ResNet50\nDisease + health score",
     color="#1565c0", bg="#bbdefb")
sbox(12.5, 4.5, 3.5, 1.5, "Guardrails Service",
     "Port 8003\nPrompt injection detection\nOutput safety validation",
     color="#1565c0", bg="#bbdefb")

# ── LAYER 4: LLMs ─────────────────────────────────────────────────────────────
sbox(0.8,  1.2, 3.5, 1.4, "Ollama  (llama3.1)",
     "Local inference\nhttp://localhost:11434\nUsed by RAG + Agent",
     color="#ad1457", bg="#fce4ec")
sbox(4.7,  1.2, 3.5, 1.4, "OpenAI  GPT-4o",
     "Cloud API\nOptional LLM provider\nLLM_PROVIDER=openai",
     color="#ad1457", bg="#fce4ec")
sbox(8.6,  1.2, 3.5, 1.4, "Google  Gemini",
     "Cloud API\nOptional LLM provider\nLLM_PROVIDER=gemini",
     color="#ad1457", bg="#fce4ec")
sbox(12.5, 1.2, 3.5, 1.4, "ChromaDB",
     "Vector database\nBilingual 70-doc index\nPersisted volume",
     color="#ad1457", bg="#fce4ec")

# ── Inter-layer arrows ─────────────────────────────────────────────────────────
# Layer 1 → Layer 2
arrow(10.3, 10.1, 10.3, 8.75, "POST /webhook")
# Layer 2 → Layer 3 (n8n calls EC2)
arrow(5.1, 7.3, 2.55, 6.0, "HTTP /ask")
arrow(9.0, 7.3, 6.45, 6.0, "HTTP /chat")
arrow(9.0, 7.3, 10.35, 6.0, "HTTP /predict")
arrow(2.4, 7.3, 14.25, 6.0, "HTTP /check-input\n/check-output")
# Layer 3 → Layer 4
arrow(2.55, 4.5, 2.55, 2.6, "LLM call")
arrow(6.45, 4.5, 6.45, 2.6, "LLM call")
arrow(2.55, 4.5, 14.25, 2.6, "embed +\nretrieve")

# ── Title ──────────────────────────────────────────────────────────────────────
title_rect = FancyBboxPatch((0.3, 12.2), 17.4, 0.65,
    boxstyle="round,pad=0.1", linewidth=0,
    facecolor=C["title_bg"], zorder=6)
ax.add_patch(title_rect)
ax.text(9.0, 12.52,
        "AgriTriage AI — System Architecture",
        fontsize=15, fontweight="bold", color="white",
        ha="center", va="center", zorder=7)
ax.text(9.0, 12.25,
        "AI-Powered Agricultural Plant Disease Triage System  ·  Saleh Naamneh",
        fontsize=9, color="#a5d6a7", ha="center", va="center", zorder=7)

plt.tight_layout(pad=0)
out = "docs/architecture_diagram.png"
import os; os.makedirs("docs", exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved: {out}")
