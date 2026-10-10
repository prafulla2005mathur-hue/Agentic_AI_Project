"""
app.py
------
FastAPI entrypoint for the AIVOA Complaint Copilot backend.

Endpoints:
    POST /api/chat              one turn of the copilot conversation
    POST /api/upload            upload a PDF complaint, extract text, run it through the same graph
    POST /api/complaints/save   persist the current form state to the DB
    GET  /api/complaints        list saved complaints
    GET  /api/complaints/{id}   fetch one saved complaint
    GET  /api/health            liveness check

Run with:
    uvicorn app:app --reload --port 8000
"""
import fitz  # PyMuPDF
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from groq import APIError
from sqlalchemy.orm import Session
import logging

from database import Base, engine, get_db
from models import Complaint
from schemas import ChatRequest, ChatResponse, SaveComplaintRequest, ComplaintOut
from graph import run_turn, empty_complaint_dict

Base.metadata.create_all(bind=engine)

app = FastAPI(title="AIVOA Complaint Copilot", version="1.0.0")
logger = logging.getLogger(__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your frontend origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(APIError)
async def groq_error_handler(_, exc: APIError):
    """Expose a stable API error while retaining the provider detail in logs."""
    logger.exception("Groq API request failed: %s", exc)
    return JSONResponse(
        status_code=502,
        content={"detail": "The AI provider could not process this request. Check the backend logs."},
    )


def extract_pdf_text(file_bytes: bytes) -> str:
    """Extract raw text from a PDF using PyMuPDF. If the PDF is a scanned
    image with no text layer, this comes back empty - hook in an OCR step
    (e.g. pytesseract) here if you need to handle scanned complaints."""
    text_chunks = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page in doc:
            text_chunks.append(page.get_text())
    return "\n".join(text_chunks).strip()


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    result = run_turn(req.session_id, req.message)
    return ChatResponse(
        reply=result.get("reply", ""),
        complaint_data=result.get("complaint_data", empty_complaint_dict()),
        missing_fields=result.get("missing_fields", []),
        updated_fields=result.get("updated_fields", []),
    )


@app.post("/api/upload", response_model=ChatResponse)
async def upload(session_id: str = Form(...), file: UploadFile = File(...)):
    if file.content_type != "application/pdf" and not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF uploads are supported right now.")

    file_bytes = await file.read()
    text = extract_pdf_text(file_bytes)
    if not text:
        raise HTTPException(422, "Couldn't extract any text from this PDF, it may be a scanned image.")

    result = run_turn(session_id, text, is_document=True)
    return ChatResponse(
        reply=result.get("reply", ""),
        complaint_data=result.get("complaint_data", empty_complaint_dict()),
        missing_fields=result.get("missing_fields", []),
        updated_fields=result.get("updated_fields", []),
    )


@app.post("/api/complaints/save", response_model=ComplaintOut)
def save_complaint(req: SaveComplaintRequest, db: Session = Depends(get_db)):
    data = req.complaint_data
    record = Complaint(
        session_id=req.session_id,
        complaint_source=data.get("complaint_source"),
        customer_name=data.get("customer_name"),
        product_name=data.get("product_name"),
        product_strength=data.get("product_strength"),
        batch_number=data.get("batch_number"),
        affected_quantity=data.get("affected_quantity"),
        mfg_date=data.get("mfg_date"),
        exp_date=data.get("exp_date"),
        complaint_category=data.get("complaint_category"),
        description=data.get("description"),
        severity=data.get("severity"),
        initial_risk_assessment=data.get("initial_risk_assessment"),
        next_action=data.get("next_action"),
        capa_recommendations="\n".join(data.get("capa_recommendations") or []),
        status="Submitted",
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


@app.get("/api/complaints", response_model=list[ComplaintOut])
def list_complaints(db: Session = Depends(get_db)):
    return db.query(Complaint).order_by(Complaint.created_at.desc()).all()


@app.get("/api/complaints/{complaint_id}", response_model=ComplaintOut)
def get_complaint(complaint_id: str, db: Session = Depends(get_db)):
    record = db.query(Complaint).filter(Complaint.id == complaint_id).first()
    if not record:
        raise HTTPException(404, "Complaint not found")
    return record
