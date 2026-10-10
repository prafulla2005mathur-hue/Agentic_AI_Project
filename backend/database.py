"""
database.py
------------
SQLAlchemy engine/session setup for the AIVOA Complaint Copilot.

Works with PostgreSQL or MySQL - just point DATABASE_URL at whichever
you're running. Examples:

    postgresql+psycopg2://user:password@localhost:5432/aivoa_qms
    mysql+pymysql://user:password@localhost:3306/aivoa_qms

Falls back to a local SQLite file if DATABASE_URL isn't set, purely so
the app boots out of the box during development/demo.
"""
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./aivoa_qms.db")

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
