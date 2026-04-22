"""Testes do filtro por diretoria."""
from __future__ import annotations

import pandas as pd
import pytest

from src.scraping.diretoria import ORGUNIT_UUID, filter_by_diretoria

DIEST_UUID = ORGUNIT_UUID["DIEST"]


def _df(rows: list[dict]) -> pd.DataFrame:
    cols = ["id", "contributor_other", "orgunit_uuid"]
    return pd.DataFrame(rows, columns=cols)


def test_match_por_uuid():
    df = _df(
        [
            {"id": "A", "contributor_other": "Ipea", "orgunit_uuid": DIEST_UUID},
            {"id": "B", "contributor_other": "Ipea", "orgunit_uuid": ""},
        ]
    )
    out = filter_by_diretoria(df, "DIEST")
    assert list(out["id"]) == ["A"]


def test_match_por_texto_contributor_other():
    df = _df(
        [
            {
                "id": "A",
                "contributor_other": "Diretoria de Estudos e Políticas do Estado, das Instituições e da Democracia - DIEST",
                "orgunit_uuid": "",
            },
            {"id": "B", "contributor_other": "Diretoria de Estudos Internacionais - DINTE", "orgunit_uuid": ""},
        ]
    )
    out = filter_by_diretoria(df, "DIEST")
    assert list(out["id"]) == ["A"]


def test_match_por_nome_legado_sem_sigla():
    # Nome anterior a ~2014: sem o trecho "das Instituições e da Democracia"
    df = _df(
        [
            {
                "id": "A",
                "contributor_other": "Instituto de Pesquisa Econômica Aplicada. Diretoria de Estudos e Políticas do Estado",
                "orgunit_uuid": "",
            }
        ]
    )
    out = filter_by_diretoria(df, "DIEST")
    assert list(out["id"]) == ["A"]


def test_case_insensitive():
    df = _df(
        [
            {"id": "A", "contributor_other": "diest", "orgunit_uuid": ""},
            {"id": "B", "contributor_other": "Diest", "orgunit_uuid": ""},
            {"id": "C", "contributor_other": "DIEST/Ipea", "orgunit_uuid": ""},
        ]
    )
    out = filter_by_diretoria(df, "DIEST")
    assert set(out["id"]) == {"A", "B", "C"}


def test_nao_confunde_com_outras_diretorias():
    df = _df(
        [
            {"id": "A", "contributor_other": "Diretoria de Estudos Internacionais - DINTE", "orgunit_uuid": ""},
            {"id": "B", "contributor_other": "Diretoria de Estudos e Políticas Sociais - DISOC", "orgunit_uuid": ""},
            {"id": "C", "contributor_other": "DISET", "orgunit_uuid": ""},
        ]
    )
    out = filter_by_diretoria(df, "DIEST")
    assert out.empty


def test_union_uuid_e_texto_sem_dedup():
    # Um item pode ter ambos sinais; filter_by_diretoria não deve duplicá-lo.
    df = _df(
        [
            {"id": "A", "contributor_other": "DIEST", "orgunit_uuid": DIEST_UUID},
            {"id": "B", "contributor_other": "DINTE", "orgunit_uuid": ""},
        ]
    )
    out = filter_by_diretoria(df, "DIEST")
    assert list(out["id"]) == ["A"]


def test_missing_columns_raise():
    df = pd.DataFrame([{"id": "A"}])
    with pytest.raises(ValueError, match="Colunas ausentes"):
        filter_by_diretoria(df, "DIEST")
