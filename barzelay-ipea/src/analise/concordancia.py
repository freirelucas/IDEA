"""
Métricas de concordância para Fase 5.

- Cohen's kappa LLM↔humano por dimensão (briefing §5 Fase 5).
- Krippendorff's alpha entre anotadores humanos (briefing §5 Fase 4).

Não usamos sklearn.cohen_kappa diretamente para registrar `n_pares`
— o N útil já filtra docs sem score humano e LLM coincidentes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

DIMENSOES = ("D1", "D2", "D3", "D4", "D5")


@dataclass(frozen=True)
class KappaPorDim:
    dimensao: str
    n_pares: int
    kappa: float


def cohen_kappa_llm_vs_humano(
    classificacoes: pd.DataFrame,
    validacoes: pd.DataFrame,
    weights: str | None = "linear",
) -> list[KappaPorDim]:
    """
    Cohen's kappa entre score LLM e mediana dos scores humanos por
    (document_id, dimensao). `weights="linear"` é apropriado para escalas
    Likert ordinais — penaliza mais discordâncias maiores.

    Quando há múltiplos humanos por (doc, dim), agrega por mediana.
    """
    valid = validacoes.dropna(subset=["score_humano"]).copy()
    valid["score_humano"] = valid["score_humano"].astype(int)
    humano = (
        valid.groupby(["document_id", "dimensao"])["score_humano"]
        .median()
        .round()
        .astype(int)
        .reset_index()
    )

    cls = classificacoes.dropna(subset=["score"]).copy()
    cls["score"] = cls["score"].astype(int)
    # Quando há múltiplas execuções LLM (versões/modelos), usa a mais recente
    cls = cls.sort_values("timestamp").drop_duplicates(
        subset=["document_id", "dimensao"], keep="last"
    )

    merged = cls.merge(humano, on=["document_id", "dimensao"], how="inner")

    out: list[KappaPorDim] = []
    for dim in DIMENSOES:
        sub = merged[merged["dimensao"] == dim]
        if len(sub) < 2:
            out.append(KappaPorDim(dim, len(sub), float("nan")))
            continue
        k = cohen_kappa_score(sub["score_humano"], sub["score"], weights=weights)
        out.append(KappaPorDim(dim, len(sub), float(k)))
    return out


def krippendorff_alpha_inter_annotators(
    validacoes: pd.DataFrame,
) -> dict[str, float]:
    """
    Krippendorff's alpha (interval) entre anotadores humanos por dimensão.

    Implementação local para evitar dep extra. Considera só (doc, dim) com
    ≥2 anotadores; valores ausentes são tratados como np.nan.
    """
    out: dict[str, float] = {}
    valid = validacoes.dropna(subset=["score_humano"]).copy()
    valid["score_humano"] = valid["score_humano"].astype(float)

    for dim in DIMENSOES:
        sub = valid[valid["dimensao"] == dim]
        # matriz: linhas = anotadores, colunas = items (document_id)
        pivot = sub.pivot_table(
            index="anotador_id", columns="document_id", values="score_humano",
            aggfunc="mean",
        )
        if pivot.shape[0] < 2 or pivot.shape[1] < 2:
            out[dim] = float("nan")
            continue
        out[dim] = _krippendorff_alpha_interval(pivot.values)
    return out


def _krippendorff_alpha_interval(matrix: np.ndarray) -> float:
    """
    Krippendorff's alpha para nível de medida intervalar.
    Adapta a fórmula clássica: 1 - Do/De.
    """
    m = np.asarray(matrix, dtype=float)
    obs = []  # pares (i,j) na mesma coluna
    for col in range(m.shape[1]):
        vals = m[:, col]
        vals = vals[~np.isnan(vals)]
        if len(vals) < 2:
            continue
        n = len(vals)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                obs.append((vals[i], vals[j]))
    if not obs:
        return float("nan")
    obs_arr = np.asarray(obs)
    Do = float(np.mean((obs_arr[:, 0] - obs_arr[:, 1]) ** 2))

    all_vals = m[~np.isnan(m)]
    if len(all_vals) < 2:
        return float("nan")
    diffs = all_vals[:, None] - all_vals[None, :]
    De = float(np.mean(diffs ** 2))
    if De == 0:
        return float("nan")
    return 1.0 - Do / De
