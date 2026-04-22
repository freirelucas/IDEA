"""
Gerador de classificações sintéticas para desenvolvimento de análises/viz
antes de rodar LLM real.

Não substitui a Fase 3 — serve apenas como placeholder plausível para
iterar em gráficos e testes de pipeline. Os scores têm estrutura realista:
- leve deriva temporal (drift longitudinal plausível em cada dimensão);
- heterogeneidade por tipo documental (ex.: Books mais descritivos, Reports
  mais normativos);
- correlação moderada entre dimensões de um mesmo documento;
- ruído gaussiano truncado a [1, 5].
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.classificacao.prompts import DIMENSOES

# médias-base por dimensão (ponto médio com viés leve, para parecer real)
BASELINE: dict[str, float] = {"D1": 2.6, "D2": 2.7, "D3": 2.3, "D4": 2.5, "D5": 2.8}

# deriva temporal por década (delta a adicionar à baseline por década desde 1970)
# hipótese do briefing: leve crescimento em direção a design em dimensões D1/D2
TEMPORAL: dict[str, float] = {"D1": 0.15, "D2": 0.10, "D3": 0.05, "D4": 0.08, "D5": 0.12}

# modificador por tipo
TIPO_EFFECT: dict[str, dict[str, float]] = {
    "Working paper": {"D1": 0.0, "D2": 0.1, "D3": 0.2, "D4": 0.1, "D5": 0.2},
    "Report":        {"D1": 0.3, "D2": 0.4, "D3": 0.1, "D4": 0.3, "D5": 0.3},
    "Journal article": {"D1": -0.2, "D2": -0.2, "D3": -0.1, "D4": -0.2, "D5": -0.1},
    "Journal Article": {"D1": -0.2, "D2": -0.2, "D3": -0.1, "D4": -0.2, "D5": -0.1},
    "Book":          {"D1": -0.3, "D2": -0.3, "D3": 0.3, "D4": -0.1, "D5": 0.4},
    "Book part":     {"D1": -0.2, "D2": -0.2, "D3": 0.1, "D4": -0.1, "D5": 0.2},
}


def synth_classificacoes(
    metadados: pd.DataFrame,
    *,
    modelo: str = "synth-1.0",
    prompt_version: str = "synth-1.0",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Gera um DataFrame no schema de `classificacoes.parquet` com scores
    sintéticos plausíveis para todos os docs de `metadados`.

    Espera colunas `id`, `ano`, `tipo`. Docs sem ano recebem década mediana.
    """
    rng = np.random.default_rng(random_state)
    df = metadados[["id", "ano", "tipo"]].copy()
    df["ano"] = pd.to_numeric(df["ano"], errors="coerce")
    med = int(df["ano"].dropna().median()) if df["ano"].notna().any() else 2010
    df["ano"] = df["ano"].fillna(med).astype(int)
    df["decada"] = (df["ano"] // 10) * 10
    df["tipo"] = df["tipo"].fillna("").astype(str)

    # fator temporal por doc: décadas desde 1970 / 5 -> roughly 0..1.2
    t_factor = np.clip((df["decada"].values - 1970) / 50.0, 0, 1.2)

    # efeito por doc: gaussiana latente compartilhada entre as dimensões (correlação)
    latente = rng.normal(0, 0.4, size=len(df))

    rows: list[dict] = []
    ts = datetime.now(timezone.utc).isoformat()
    for dim in DIMENSOES:
        tipo_eff = df["tipo"].map(
            lambda t: TIPO_EFFECT.get(t, {}).get(dim, 0.0)
        ).values
        ruido = rng.normal(0, 0.6, size=len(df))
        vals = (
            BASELINE[dim]
            + TEMPORAL[dim] * t_factor * (np.arange(5).mean() / 5)  # reduzido p/ não saturar
            + TEMPORAL[dim] * t_factor  # principal termo temporal
            + tipo_eff
            + latente * 0.5
            + ruido
        )
        scores = np.clip(np.round(vals), 1, 5).astype(int)
        for i, doc_id in enumerate(df["id"].values):
            rows.append({
                "document_id": doc_id,
                "dimensao": dim,
                "score": int(scores[i]),
                "trechos_evidencia": ["[synth]"],
                "justificativa": f"[sintético] baseline={BASELINE[dim]:.2f} + deriva + tipo_eff.",
                "modelo": modelo,
                "prompt_version": prompt_version,
                "tokens_input": 0,
                "tokens_output": 0,
                "timestamp": ts,
            })
    return pd.DataFrame(rows)
