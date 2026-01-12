from __future__ import annotations

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class TaskInstance(BaseModel):
    id: str
    task_type: str
    input: str
    reference: Optional[str] = None
    evidence: Optional[list[str]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentResult(BaseModel):
    text: str
    tokens_in: int
    tokens_out: int
    latency_ms: int
    metadata: Dict[str, Any] = Field(default_factory=dict)
