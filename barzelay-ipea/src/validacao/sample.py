"""
Amostragem para validação humana (Fase 4).

Briefing §5 Fase 4: ~300 documentos, estratificação por **década**, **tipo
documental** e **score LLM** (oversample de extremos e casos de discordância
entre modelos quando a Fase 3 tiver rodado com modelos múltiplos).
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

DEFAULT_TIPOS_ALVO: tuple[str, ...] = (
    "Working paper",
    "Journal article",
    "Journal Article",
    "Book",
    "Book part",
    "Report",
)


def amostra_validacao(
    classificacoes: pd.DataFrame,
    metadados: pd.DataFrame,
    n_target: int = 300,
    tipos_alvo: Iterable[str] = DEFAULT_TIPOS_ALVO,
    oversample_extremos: float = 0.4,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Sorteia ~`n_target` documentos para validação humana, estratificando por
    (década, tipo) e oversampling de extremos (score 1 ou 5 em pelo menos
    uma dimensão).

    `classificacoes` deve ter colunas `document_id`, `dimensao`, `score`.
    `metadados` deve ter `id`, `ano`, `tipo`.

    Devolve DataFrame com 1 linha por documento amostrado, contendo
    `id`, `ano`, `tipo`, `decada`, `n_extremos` (qtd de scores 1 ou 5
    do LLM no doc).
    """
    rng = np.random.default_rng(random_state)

    # 1. Sumariza por documento: n_extremos = #dimensões com score em {1,5}
    classificacoes = classificacoes.dropna(subset=["score"])
    classificacoes["score"] = classificacoes["score"].astype(int)
    extremos = (
        classificacoes.assign(eh_extremo=classificacoes["score"].isin([1, 5]).astype(int))
        .groupby("document_id")["eh_extremo"]
        .sum()
        .rename("n_extremos")
        .reset_index()
        .rename(columns={"document_id": "id"})
    )

    # 2. Junta com metadados
    base = metadados[["id", "ano", "tipo"]].merge(extremos, on="id", how="inner")
    base = base.dropna(subset=["ano"])
    base["ano"] = base["ano"].astype(int)
    base = base[base["tipo"].isin(tipos_alvo)]
    base["decada"] = (base["ano"] // 10) * 10

    if base.empty:
        return base

    # 3. Quotas por estrato (década × tipo) proporcionais a n_target
    strata = base.groupby(["decada", "tipo"]).size()
    pesos = strata / strata.sum()
    quotas = (pesos * n_target).round().astype(int).clip(lower=1)

    # 4. Sample por estrato. Dentro de cada estrato, oversample_extremos da
    #    cota vai para docs com n_extremos >= 1; o resto, sample uniforme.
    selecionados: list[pd.DataFrame] = []
    for (dec, tipo), n in quotas.items():
        stratum = base[(base["decada"] == dec) & (base["tipo"] == tipo)]
        if stratum.empty or n == 0:
            continue
        n_ext = min(int(n * oversample_extremos), len(stratum[stratum["n_extremos"] >= 1]))
        n_uni = n - n_ext
        ext_pool = stratum[stratum["n_extremos"] >= 1]
        uni_pool = stratum.drop(ext_pool.index, errors="ignore")
        ext_pick = (
            ext_pool.sample(n=n_ext, random_state=int(rng.integers(0, 2**31)))
            if n_ext > 0 and len(ext_pool) >= n_ext
            else ext_pool
        )
        n_uni_real = min(n_uni, len(uni_pool))
        uni_pick = (
            uni_pool.sample(n=n_uni_real, random_state=int(rng.integers(0, 2**31)))
            if n_uni_real > 0
            else uni_pool.iloc[0:0]
        )
        selecionados.append(pd.concat([ext_pick, uni_pick], ignore_index=True))

    if not selecionados:
        return base.iloc[0:0]
    return (
        pd.concat(selecionados, ignore_index=True)
        .drop_duplicates(subset=["id"])
        .reset_index(drop=True)
    )
