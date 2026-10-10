"""
models.py
---------
SQLAlchemy ORM model for a persisted customer complaint record.
"""
import uuid
import datetime as dt

from sqlalchemy import Column, String, Text, DateTime

from database import Base


def _default_id():
    return str(uuid.uuid4())


class Complaint(Base):
    __tablename__ = "complaints"

    # Stored as a plain string so this works identically on Postgres,
    # MySQL, and SQLite (MySQL/SQLite have no native UUID type).
    id = Column(String(36), primary_key=True, default=_default_id)

    session_id = Column(String(64), index=True, nullable=True)

    complaint_source = Column(String(120), nullable=True)
    customer_name = Column(String(200), nullable=True)
    product_name = Column(String(200), nullable=True)
    product_strength = Column(String(80), nullable=True)
    batch_number = Column(String(80), nullable=True)
    affected_quantity = Column(String(80), nullable=True)
    mfg_date = Column(String(40), nullable=True)
    exp_date = Column(String(40), nullable=True)
    complaint_category = Column(String(120), nullable=True)
    description = Column(Text, nullable=True)

    severity = Column(String(20), nullable=True)
    initial_risk_assessment = Column(Text, nullable=True)
    next_action = Column(Text, nullable=True)
    capa_recommendations = Column(Text, nullable=True)  # newline-joined list

    status = Column(String(30), default="Draft")

    created_at = Column(DateTime, default=dt.datetime.utcnow)
    updated_at = Column(DateTime, default=dt.datetime.utcnow, onupdate=dt.datetime.utcnow)
