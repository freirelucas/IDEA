"""
Calibração isotônica dos scores LLM contra padrão-ouro humano.

Briefing §5 Fase 5 e §7: já decidido que a calibração será isotônica
(monotonicamente crescente, não-paramétrica). Aplica `IsotonicRegression`
do sklearn por dimensão; o output é um score calibrado mantendo a ordem
mas alinhando o nível com a anotação humana.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.isotonic import IsotonicRegression

from src.analise.concordancia import DIMENSOES


@dataclass(frozen=True)
class CalibradorPorDim:
    dimensao: str
    n_pares_treino: int
    modelo: IsotonicRegression


def treinar_calibradores(
    classificacoes: pd.DataFrame,
    validacoes: pd.DataFrame,
    score_min: float = 1.0,
    score_max: float = 5.0,
) -> dict[str, CalibradorPorDim]:
    """
    Treina um IsotonicRegression por dimensão usando os pares (LLM, humano)
    disponíveis. Quando há múltiplos humanos por (doc, dim), usa a mediana.
    """
    valid = validacoes.dropna(subset=["score_humano"]).copy()
    valid["score_humano"] = valid["score_humano"].astype(float)
    humano = (
        valid.groupby(["document_id", "dimensao"])["score_humano"]
        .median()
        .reset_index()
    )

    cls = classificacoes.dropna(subset=["score"]).copy()
    cls["score"] = cls["score"].astype(float)
    cls = cls.sort_values("timestamp").drop_duplicates(
        subset=["document_id", "dimensao"], keep="last"
    )

    merged = cls.merge(humano, on=["document_id", "dimensao"], how="inner")
    out: dict[str, CalibradorPorDim] = {}
    for dim in DIMENSOES:
        sub = merged[merged["dimensao"] == dim]
        if len(sub) < 5:
            # poucos pares — não treinar (devolve identidade)
            continue
        iso = IsotonicRegression(
            y_min=score_min, y_max=score_max, out_of_bounds="clip", increasing=True,
        )
        iso.fit(sub["score"].values, sub["score_humano"].values)
        out[dim] = CalibradorPorDim(dim, n_pares_treino=len(sub), modelo=iso)
    return out


def aplicar_calibradores(
    classificacoes: pd.DataFrame,
    calibradores: dict[str, CalibradorPorDim],
) -> pd.DataFrame:
    """Devolve `classificacoes` com coluna `score_calibrado`."""
    out = classificacoes.copy()
    out["score_calibrado"] = out["score"].astype(float)  # default: identidade
    for dim, cal in calibradores.items():
        mask = out["dimensao"] == dim
        out.loc[mask, "score_calibrado"] = cal.modelo.predict(
            out.loc[mask, "score"].astype(float).values
        )
    return out
