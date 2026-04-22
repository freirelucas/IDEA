"""Schema das respostas do classificador LLM (Pydantic)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


Dimensao = Literal["D1", "D2", "D3", "D4", "D5"]


class ClassificacaoLLM(BaseModel):
    """Resposta JSON estruturada do LLM para uma dimensão de um documento."""

    dimensao: Dimensao
    score: int = Field(ge=1, le=5)
    trechos_evidencia: list[str] = Field(min_length=1, max_length=10)
    justificativa: str = Field(min_length=1)

    @field_validator("trechos_evidencia")
    @classmethod
    def _strip_trechos(cls, v: list[str]) -> list[str]:
        cleaned = [t.strip() for t in v if t and t.strip()]
        if not cleaned:
            raise ValueError("trechos_evidencia não pode ser vazio após strip")
        return cleaned
