"""
Análise longitudinal de scores Barzelay (Fase 5).

Briefing §5 Fase 5:
- trajetória dos scores médios por dimensão ao longo das décadas;
- corte por tipo documental (TDs vs livros vs notas técnicas);
- corte por autor / área temática (palavras-chave).
"""
from __future__ import annotations

import pandas as pd

from src.analise.concordancia import DIMENSOES


def media_por_decada(
    classificacoes: pd.DataFrame,
    metadados: pd.DataFrame,
    use_calibrado: bool = True,
) -> pd.DataFrame:
    """
    Tabela com média do score (calibrado se disponível) por (década, dimensão).
    Linhas: décadas; colunas: dimensões D1-D5.
    """
    col = "score_calibrado" if use_calibrado and "score_calibrado" in classificacoes.columns else "score"
    df = _join_anos(classificacoes, metadados, score_col=col)
    pivot = df.pivot_table(
        index="decada", columns="dimensao", values=col, aggfunc="mean"
    )
    return pivot.reindex(columns=list(DIMENSOES))


def media_por_decada_e_tipo(
    classificacoes: pd.DataFrame,
    metadados: pd.DataFrame,
    use_calibrado: bool = True,
) -> pd.DataFrame:
    """Long-form: década × tipo × dimensão → média."""
    col = "score_calibrado" if use_calibrado and "score_calibrado" in classificacoes.columns else "score"
    df = _join_anos(classificacoes, metadados, score_col=col, with_tipo=True)
    return (
        df.groupby(["decada", "tipo", "dimensao"], as_index=False)[col]
        .mean()
        .rename(columns={col: "score_medio"})
    )


def top_palavras_por_polo(
    classificacoes: pd.DataFrame,
    metadados: pd.DataFrame,
    dimensao: str,
    top_n: int = 30,
    threshold_design: float = 4.0,
    threshold_descritivo: float = 2.0,
    use_calibrado: bool = True,
) -> dict[str, pd.Series]:
    """
    Para uma dimensão, lista as palavras-chave mais frequentes em docs no
    polo design (score ≥ threshold_design) vs polo descritivo (score ≤
    threshold_descritivo). Útil para inspeção qualitativa do que distingue
    os extremos.
    """
    col = "score_calibrado" if use_calibrado and "score_calibrado" in classificacoes.columns else "score"
    cls = classificacoes[classificacoes["dimensao"] == dimensao]
    cls = cls.merge(
        metadados[["id", "palavras_chave"]],
        left_on="document_id", right_on="id", how="inner",
    )

    def _explode(sub: pd.DataFrame) -> pd.Series:
        words = (
            sub["palavras_chave"].fillna("").str.split(",").explode().str.strip()
        )
        return words[words != ""].value_counts().head(top_n)

    return {
        "design": _explode(cls[cls[col] >= threshold_design]),
        "descritivo": _explode(cls[cls[col] <= threshold_descritivo]),
    }


def _join_anos(
    classificacoes: pd.DataFrame,
    metadados: pd.DataFrame,
    *,
    score_col: str,
    with_tipo: bool = False,
    ano_min: int = 1950,
    ano_max: int = 2030,
) -> pd.DataFrame:
    cols = ["id", "ano"] + (["tipo"] if with_tipo else [])
    meta = metadados[cols].dropna(subset=["ano"]).copy()
    meta["ano"] = meta["ano"].astype(int)
    meta = meta[(meta["ano"] >= ano_min) & (meta["ano"] <= ano_max)]
    meta["decada"] = (meta["ano"] // 10) * 10

    cls = classificacoes.dropna(subset=[score_col]).copy()
    cls[score_col] = cls[score_col].astype(float)
    cls = cls.sort_values("timestamp").drop_duplicates(
        subset=["document_id", "dimensao"], keep="last"
    )

    return cls.merge(meta, left_on="document_id", right_on="id", how="inner")
