"""
Persistência de validações humanas em parquet append-only.

Schema (briefing §5 Fase 4):
    document_id, dimensao, score_humano (1-5), justificativa_humana,
    anotador_id, timestamp.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SCHEMA: list[str] = [
    "document_id",
    "dimensao",
    "score_humano",
    "justificativa_humana",
    "anotador_id",
    "timestamp",
]


def append_validacao(
    path: Path,
    document_id: str,
    dimensao: str,
    score_humano: int,
    justificativa_humana: str,
    anotador_id: str,
) -> None:
    """Acrescenta uma linha em `path`. Cria o arquivo se ainda não existir."""
    if dimensao not in {"D1", "D2", "D3", "D4", "D5"}:
        raise ValueError(f"dimensao inválida: {dimensao}")
    if not 1 <= int(score_humano) <= 5:
        raise ValueError(f"score_humano fora de 1-5: {score_humano}")

    row = {
        "document_id": document_id,
        "dimensao": dimensao,
        "score_humano": int(score_humano),
        "justificativa_humana": justificativa_humana,
        "anotador_id": anotador_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        df = pd.read_parquet(path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row], columns=SCHEMA)
    df.to_parquet(path, index=False)


def carregar_validacoes(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=SCHEMA)
    return pd.read_parquet(path)


def ja_anotado(
    path: Path, document_id: str, dimensao: str, anotador_id: str
) -> bool:
    df = carregar_validacoes(path)
    if df.empty:
        return False
    mask = (
        (df["document_id"] == document_id)
        & (df["dimensao"] == dimensao)
        & (df["anotador_id"] == anotador_id)
    )
    return bool(mask.any())
