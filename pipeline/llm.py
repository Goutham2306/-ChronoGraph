"""
Centralized LangChain + Gemini LLM configuration for Day 2.

This is deliberately separate from pipeline/extract.py's Gemini setup
(which uses LlamaIndex's GoogleGenAI wrapper for Day 1 extraction). Day 2
uses LangChain's ChatGoogleGenerativeAI instead, because the Day 2
pipeline (Cypher generation, answer generation) is built on LangChain's
`.with_structured_output()` + LCEL, not LlamaIndex's structured-LLM
interface. Both wrap the same underlying Gemini API and read the same
GEMINI_API_KEY, but are separate client objects — this module does not
touch or import pipeline/extract.py, so Day 1 stays untouched.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

_DEFAULT_MODEL = "gemini-3.1-flash-lite"


def _api_key() -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add your Gemini key to .env "
            "(see .env.example)."
        )
    return api_key


def get_qa_llm(temperature: float = 0.0) -> ChatGoogleGenerativeAI:
    """
    LLM used for Day 2 question-answering stages (Cypher generation,
    answer synthesis). Reads CHRONOGRAPH_QA_MODEL if set, otherwise falls
    back to CHRONOGRAPH_EXTRACTION_MODEL (same var Day 1 uses), otherwise
    the hardcoded default.

    temperature=0.0 by default — Cypher generation and grounded answer
    synthesis both want deterministic, non-creative output.
    """
    model = (
        os.environ.get("CHRONOGRAPH_QA_MODEL")
        or os.environ.get("CHRONOGRAPH_EXTRACTION_MODEL")
        or _DEFAULT_MODEL
    )
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=_api_key(),
        temperature=temperature,
    )
