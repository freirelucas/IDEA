"""
Amostragem estratificada por década sobre o Parquet normalizado.

Briefing §5, Fase 4 já prevê amostragem estratificada (por década + tipo +
score) para a validação humana. Esta função cobre o caso mais simples: um
sample proporcional por década para rodar o pipeline end-to-end num subset
antes de escalar para o corpus completo.
"""
from __future__ import annotations

import pandas as pd


def stratified_sample_by_decade(
    df: pd.DataFrame,
    frac: float = 0.10,
    ano_min: int = 1950,
    ano_max: int = 2030,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Retorna `frac * N` de cada década, amostrado sem reposição.

    - Documentos com `ano` ausente ou fora de `[ano_min, ano_max]` são
      descartados (evita o outlier literal `ano=88` visto no corpus IPEA).
    - A coluna `decada` é acrescentada ao resultado para permitir auditar a
      estratificação downstream.
    - `random_state` fixo garante reprodutibilidade entre rodadas.
    """
    if "ano" not in df.columns:
        raise ValueError("DataFrame precisa ter coluna 'ano'")
    if not 0 < frac <= 1:
        raise ValueError(f"frac deve estar em (0, 1], recebi {frac}")

    mask = df["ano"].notna() & (df["ano"] >= ano_min) & (df["ano"] <= ano_max)
    valid = df.loc[mask].copy()
    valid["decada"] = (valid["ano"].astype(int) // 10) * 10

    sampled = (
        valid.groupby("decada", group_keys=False)
        .sample(frac=frac, random_state=random_state)
        .reset_index(drop=True)
    )
    return sampled
