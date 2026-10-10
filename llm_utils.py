"""
llm_utils.py
------------
Thin wrappers around the two Groq-hosted models used by the graph:

  * openai/gpt-oss-120b -> routing, extraction, risk reasoning, and CAPA

Both are wrapped with `.with_structured_output(...)` at the call site so
nodes get back validated Pydantic objects instead of raw text to parse.
"""
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# These IDs are enabled on the configured Groq account. The previous Gemma
# and Llama IDs were decommissioned, which made every chat request fail.
# The 20B model occasionally emits truncated function-call arguments for
# detailed complaint reports. The graph relies on structured output, so use
# the reliably tool-capable model for every schema-constrained stage.
FAST_MODEL = "openai/gpt-oss-120b"
REASONING_MODEL = "openai/gpt-oss-120b"


def get_fast_llm(temperature: float = 0.0) -> ChatGroq:
    # The account's Groq tier allows at most 1,000 output tokens per request.
    # Routing and field extraction require only a compact structured response.
    return ChatGroq(
        model=FAST_MODEL,
        temperature=temperature,
        max_tokens=512,
        api_key=GROQ_API_KEY,
    )


def get_reasoning_llm(temperature: float = 0.2) -> ChatGroq:
    # Risk and CAPA answers are intentionally concise, so this stays within
    # the same account-level output limit.
    return ChatGroq(
        model=REASONING_MODEL,
        temperature=temperature,
        max_tokens=768,
        api_key=GROQ_API_KEY,
    )
