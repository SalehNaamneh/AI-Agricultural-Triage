import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))   # Layer3/agent/ for state, nodes

from langgraph.graph import StateGraph, END

from state import AgentState
from nodes import (
    node_guard_input,
    node_analyze_image,
    node_enrich_question,
    node_query_rag,
    node_guard_output,
)


def _route_after_guard(state: AgentState) -> str:
    if state["blocked"]:
        return "blocked"
    if state.get("image_bytes"):
        return "analyze_image"
    return "query_rag"


def build_graph():
    g = StateGraph(AgentState)

    g.add_node("guard_input",     node_guard_input)
    g.add_node("analyze_image",   node_analyze_image)
    g.add_node("enrich_question", node_enrich_question)
    g.add_node("query_rag",       node_query_rag)
    g.add_node("guard_output",    node_guard_output)

    g.set_entry_point("guard_input")

    g.add_conditional_edges("guard_input", _route_after_guard, {
        "blocked":       END,
        "analyze_image": "analyze_image",
        "query_rag":     "query_rag",
    })

    g.add_edge("analyze_image",   "enrich_question")
    g.add_edge("enrich_question", "query_rag")
    g.add_edge("query_rag",       "guard_output")
    g.add_edge("guard_output",    END)

    return g.compile()


_compiled = None


def get_graph():
    global _compiled
    if _compiled is None:
        _compiled = build_graph()
    return _compiled
