"""
Minimal FastAPI wrapper around pipeline.rag_pipeline.answer_question.

Run:
    uvicorn api:app --host 0.0.0.0 --port 8000

No frontend, no auth - just the two required endpoints per spec.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from pipeline.rag_pipeline import PipelineResult, answer_question

app = FastAPI(title="ChronoGraph Day 2 API")


class QueryRequest(BaseModel):
    question: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/query", response_model=PipelineResult)
def query(request: QueryRequest) -> PipelineResult:
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")
    try:
        return answer_question(request.question)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc
