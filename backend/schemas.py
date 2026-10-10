"""
schemas.py
----------
Pydantic models shared by the API layer and the LangGraph nodes.

`ComplaintExtraction` doubles as the structured-output target for the
Extraction node (gemma2-9b-it). Every field is Optional because a
single user message rarely contains the whole complaint, and on a
correction turn ("actually the batch number is...") only one or two
fields will be filled in.
"""
from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class ComplaintExtraction(BaseModel):
    """Fields the Extraction node is allowed to fill in from free text."""
    complaint_source: Optional[str] = Field(
        None, description="Where the complaint came from, e.g. 'Apollo Pharmacy', 'Direct Email', 'Call Center'")
    customer_name: Optional[str] = Field(None, description="Name of the customer or reporting party")
    product_name: Optional[str] = Field(None, description="Brand/product name")
    product_strength: Optional[str] = Field(None, description="Dosage strength, e.g. '500mg'")
    batch_number: Optional[str] = Field(None, description="Batch or lot number")
    affected_quantity: Optional[str] = Field(None, description="Quantity affected, e.g. '48 capsules'")
    mfg_date: Optional[str] = Field(None, description="Manufacturing date")
    exp_date: Optional[str] = Field(None, description="Expiry date")
    complaint_category: Optional[str] = Field(
        None, description="e.g. 'Physical Defect', 'Packaging', 'Adverse Event', 'Efficacy', 'Foreign Particle'")
    description: Optional[str] = Field(None, description="Free-text narrative describing what went wrong")


class RouterOutput(BaseModel):
    input_type: Literal["new_complaint", "correction", "chitchat"] = Field(
        ..., description="Classify the user's message")


class RiskAssessmentOutput(BaseModel):
    severity: Literal["Minor", "Major", "Critical"]
    initial_risk_assessment: str = Field(..., description="2-3 sentence risk rationale")
    next_action: str = Field(..., description="Concrete next step, e.g. 'Route to QA for batch investigation'")


class CAPAOutput(BaseModel):
    recommendations: List[str] = Field(..., description="3-5 concrete CAPA recommendations")


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    reply: str
    complaint_data: dict
    missing_fields: List[str]
    updated_fields: List[str]


class SaveComplaintRequest(BaseModel):
    session_id: str
    complaint_data: dict


class ComplaintOut(BaseModel):
    id: str
    session_id: Optional[str] = None
    complaint_source: Optional[str] = None
    customer_name: Optional[str] = None
    product_name: Optional[str] = None
    product_strength: Optional[str] = None
    batch_number: Optional[str] = None
    affected_quantity: Optional[str] = None
    mfg_date: Optional[str] = None
    exp_date: Optional[str] = None
    complaint_category: Optional[str] = None
    description: Optional[str] = None
    severity: Optional[str] = None
    initial_risk_assessment: Optional[str] = None
    next_action: Optional[str] = None
    capa_recommendations: Optional[str] = None
    status: str

    class Config:
        from_attributes = True
