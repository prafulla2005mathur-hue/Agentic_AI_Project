"""
graph.py
--------
LangGraph state machine for the AIVOA Complaint Copilot.

    router --chitchat--> chitchat_respond --> END
      |
      +--new_complaint/correction
              |
              v
          extraction
              |
              v
      completeness_checker --missing fields--> responder --> END
              |
              (complete)
              v
          risk_assessment
              |
              v
      capa_recommendation
              |
              v
          responder --> END

State persists per `session_id` via LangGraph's MemorySaver, so a
correction on turn 3 is merged on top of whatever was extracted on
turns 1-2 without the frontend needing to resend full history.
"""
from typing import List, Optional, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from schemas import ComplaintExtraction, RouterOutput, RiskAssessmentOutput, CAPAOutput
from llm_utils import get_fast_llm, get_reasoning_llm

REQUIRED_FIELDS = [
    "customer_name",
    "product_name",
    "batch_number",
    "complaint_category",
    "description",
    "affected_quantity",
]

FIELD_LABELS = {
    "complaint_source": "complaint source",
    "customer_name": "customer name",
    "product_name": "product name",
    "product_strength": "product strength",
    "batch_number": "batch/lot number",
    "affected_quantity": "affected quantity",
    "mfg_date": "manufacturing date",
    "exp_date": "expiry date",
    "complaint_category": "complaint category",
    "description": "a description of the issue",
    "severity": "severity",
    "initial_risk_assessment": "initial risk assessment",
    "next_action": "next action",
    "capa_recommendations": "CAPA recommendations",
}


class GraphState(TypedDict, total=False):
    user_input: str
    input_type: str
    complaint_data: dict
    missing_fields: List[str]
    risk_result: Optional[dict]
    capa_recommendations: List[str]
    updated_fields: List[str]
    reply: str


def empty_complaint_dict() -> dict:
    return {
        "complaint_source": None,
        "customer_name": None,
        "product_name": None,
        "product_strength": None,
        "batch_number": None,
        "affected_quantity": None,
        "mfg_date": None,
        "exp_date": None,
        "complaint_category": None,
        "description": None,
        "severity": None,
        "initial_risk_assessment": None,
        "next_action": None,
        "capa_recommendations": [],
    }


# ---------------------------------------------------------------- nodes --

def router_node(state: GraphState) -> GraphState:
    llm = get_fast_llm().with_structured_output(RouterOutput)
    result: RouterOutput = llm.invoke(
        "Classify this message from a pharma quality-complaint intake chat.\n"
        "- 'new_complaint': describes a new customer complaint (email, extracted "
        "document text, or free narration)\n"
        "- 'correction': fixes/updates a field already captured (e.g. 'actually "
        "the batch is...')\n"
        "- 'chitchat': anything else - a question, greeting, or unrelated remark\n\n"
        f"Message:\n{state['user_input']}"
    )
    return {"input_type": result.input_type}


def extraction_node(state: GraphState) -> GraphState:
    llm = get_fast_llm().with_structured_output(ComplaintExtraction)
    extracted: ComplaintExtraction = llm.invoke(
        "Extract complaint details from the message below into the schema. "
        "Only fill a field if it is explicitly stated or very strongly implied. "
        "Leave anything not mentioned as null. Do not guess or invent values.\n\n"
        f"Message:\n{state['user_input']}"
    )

    current = dict(state.get("complaint_data") or empty_complaint_dict())
    updated_fields = []
    for field, value in extracted.model_dump().items():
        if value not in (None, "", "null"):
            if current.get(field) != value:
                updated_fields.append(field)
            current[field] = value

    return {"complaint_data": current, "updated_fields": updated_fields}


def completeness_checker_node(state: GraphState) -> GraphState:
    data = state["complaint_data"]
    missing = [f for f in REQUIRED_FIELDS if not data.get(f)]
    return {"missing_fields": missing}


def risk_assessment_node(state: GraphState) -> GraphState:
    data = state["complaint_data"]
    llm = get_reasoning_llm().with_structured_output(RiskAssessmentOutput)
    result: RiskAssessmentOutput = llm.invoke(
        "You are a pharmaceutical Quality Assurance risk assessor. Given this "
        "customer complaint, determine severity (Minor/Major/Critical), write a "
        "concise initial risk assessment, and recommend a concrete next action.\n\n"
        f"Product: {data.get('product_name')} {data.get('product_strength') or ''}\n"
        f"Batch: {data.get('batch_number')}\n"
        f"Category: {data.get('complaint_category')}\n"
        f"Affected quantity: {data.get('affected_quantity')}\n"
        f"Description: {data.get('description')}\n\n"
        "Guidance: 'Critical' means potential patient safety, adverse event, or "
        "sterility/identity failure. 'Major' means a defect affecting efficacy, "
        "labeling, or a whole batch, but no immediate safety signal. "
        "'Minor' means a cosmetic or packaging issue with no safety or efficacy impact."
    )

    data = dict(data)
    data["severity"] = result.severity
    data["initial_risk_assessment"] = result.initial_risk_assessment
    data["next_action"] = result.next_action
    updated = list(set(state.get("updated_fields", []) + ["severity", "initial_risk_assessment", "next_action"]))
    return {"complaint_data": data, "risk_result": result.model_dump(), "updated_fields": updated}


def capa_recommendation_node(state: GraphState) -> GraphState:
    data = state["complaint_data"]
    llm = get_reasoning_llm().with_structured_output(CAPAOutput)
    result: CAPAOutput = llm.invoke(
        "Based on this pharma complaint and its risk assessment, propose 3-5 "
        "concrete CAPA (Corrective and Preventive Action) recommendations. Be "
        "specific to the defect type, not generic boilerplate.\n\n"
        f"Category: {data.get('complaint_category')}\n"
        f"Severity: {data.get('severity')}\n"
        f"Description: {data.get('description')}\n"
        f"Risk assessment: {data.get('initial_risk_assessment')}\n"
    )
    data = dict(data)
    data["capa_recommendations"] = result.recommendations
    updated = list(set(state.get("updated_fields", []) + ["capa_recommendations"]))
    return {"complaint_data": data, "capa_recommendations": result.recommendations, "updated_fields": updated}


def responder_node(state: GraphState) -> GraphState:
    data = state["complaint_data"]
    missing = state.get("missing_fields", [])
    input_type = state.get("input_type")

    if missing:
        pretty = ", ".join(FIELD_LABELS.get(f, f) for f in missing)
        lead = "Got the update. " if input_type == "correction" else "Thanks, I've started logging this. "
        reply = f"{lead}I still need {pretty} before I can run a risk assessment. Could you share that?"
        return {"reply": reply}

    changed = state.get("updated_fields", [])
    field_bits = ", ".join(FIELD_LABELS.get(f, f).capitalize() for f in changed if f in FIELD_LABELS)
    lead = "Updated" if input_type == "correction" else "Logged the complaint"
    parts = [f"{lead}{(' - ' + field_bits) if field_bits else ''}."]

    if data.get("severity"):
        parts.append(f"Risk assessment: {data['severity']} - {data.get('initial_risk_assessment')}")
    if data.get("next_action"):
        parts.append(f"Next action: {data['next_action']}")
    if data.get("capa_recommendations"):
        capa_lines = "\n".join(f"  {i + 1}. {c}" for i, c in enumerate(data["capa_recommendations"]))
        parts.append(f"Suggested CAPA:\n{capa_lines}")

    return {"reply": "\n\n".join(parts)}


def chitchat_node(state: GraphState) -> GraphState:
    llm = get_fast_llm()
    result = llm.invoke(
        "You are AIVOA, a helpful pharma QMS complaint-intake copilot. Answer "
        f"briefly and helpfully.\n\nUser: {state['user_input']}"
    )
    return {"reply": result.content}


# ---------------------------------------------------------- conditionals --

def route_after_router(state: GraphState) -> str:
    return "chitchat" if state["input_type"] == "chitchat" else "extraction"


def route_after_completeness(state: GraphState) -> str:
    return "incomplete" if state.get("missing_fields") else "complete"


# --------------------------------------------------------------- build ----

def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("router", router_node)
    graph.add_node("extraction", extraction_node)
    graph.add_node("completeness_checker", completeness_checker_node)
    graph.add_node("risk_assessment", risk_assessment_node)
    graph.add_node("capa_recommendation", capa_recommendation_node)
    graph.add_node("responder", responder_node)
    graph.add_node("chitchat_respond", chitchat_node)

    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router", route_after_router, {"chitchat": "chitchat_respond", "extraction": "extraction"}
    )
    graph.add_edge("chitchat_respond", END)

    graph.add_edge("extraction", "completeness_checker")
    graph.add_conditional_edges(
        "completeness_checker",
        route_after_completeness,
        {"incomplete": "responder", "complete": "risk_assessment"},
    )
    graph.add_edge("risk_assessment", "capa_recommendation")
    graph.add_edge("capa_recommendation", "responder")
    graph.add_edge("responder", END)

    return graph.compile(checkpointer=MemorySaver())


complaint_graph = build_graph()


def run_turn(session_id: str, user_input: str, is_document: bool = False) -> GraphState:
    """Invoke the graph for one chat turn, threaded on session_id so
    corrections merge onto prior state automatically.

    is_document=True just prefixes the text with a short hint so the
    router reliably treats long pasted/extracted document text as a new
    complaint rather than trying to classify it as chitchat.
    """
    config = {"configurable": {"thread_id": session_id}}
    text = (
        f"[This text was extracted from an uploaded PDF/document]\n\n{user_input}"
        if is_document
        else user_input
    )
    result = complaint_graph.invoke({"user_input": text}, config=config)
    return result
