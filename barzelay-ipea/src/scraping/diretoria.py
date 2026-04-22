"""
Filtros por diretoria do IPEA sobre o Parquet normalizado em `data/interim/`.

O DSpace marca a diretoria de cada item por dois caminhos, que precisam ser
combinados para recuperar todo o corpus histórico:

1. `relation.isOrgUnitOfPublication` — UUID da OrgUnit. Implementado só em itens
   recentes (tipicamente pós-2020). Link autoritativo.
2. `dc.contributor.other` — campo texto livre com o nome da diretoria. Usado em
   todo o corpus histórico; sujeito a variações de grafia.

UUIDs de OrgUnit conhecidos (descobertos via
`discover/search/objects?configuration=orgunit&query={sigla}`):
"""
from __future__ import annotations

import re

import pandas as pd

ORGUNIT_UUID: dict[str, str] = {
    "DIEST": "38b33462-dc75-43f2-9bda-1f3c54757ae7",
}

# Padrões de texto livre em `dc.contributor.other`. Cobrem:
# - sigla "DIEST" ou "Diest"
# - nome atual "Diretoria de Estudos e Políticas do Estado, das Instituições e
#   da Democracia"
# - nomes anteriores documentados: antes de ~2014 DIEST era "Diretoria de
#   Estudos e Políticas do Estado" (sem o trecho "das Instituições e da
#   Democracia")
CONTRIBUTOR_OTHER_PATTERNS: dict[str, re.Pattern[str]] = {
    "DIEST": re.compile(
        r"(?:\bDIEST\b|Diretoria de Estudos e Políticas do Estado)",
        re.IGNORECASE,
    ),
}


def filter_by_diretoria(
    df: pd.DataFrame, sigla: str = "DIEST"
) -> pd.DataFrame:
    """
    Retorna linhas que pertencem à diretoria indicada, combinando os dois
    sinais disponíveis (UUID da OrgUnit + texto de `contributor_other`).

    Exige que o DataFrame tenha as colunas `orgunit_uuid` e `contributor_other`
    (produzidas pela Fase 1 a partir da versão estendida de `extract_fields`).
    """
    missing = {"orgunit_uuid", "contributor_other"} - set(df.columns)
    if missing:
        raise ValueError(
            f"Colunas ausentes: {missing}. Re-rode `python -m src.scraping` "
            "com a versão estendida de extract_fields/clean_item."
        )

    uuid = ORGUNIT_UUID[sigla]
    pattern = CONTRIBUTOR_OTHER_PATTERNS[sigla]

    by_uuid = df["orgunit_uuid"].fillna("").str.contains(uuid, regex=False)
    by_text = df["contributor_other"].fillna("").str.contains(pattern)
    return df[by_uuid | by_text].copy()
