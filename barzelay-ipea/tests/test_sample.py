"""Testes de amostragem estratificada por década."""
from __future__ import annotations

import pandas as pd
import pytest

from src.scraping.sample import stratified_sample_by_decade


def _build_df(counts_por_decada: dict[int, int]) -> pd.DataFrame:
    rows = []
    idx = 0
    for decada, n in counts_por_decada.items():
        for i in range(n):
            rows.append({"id": f"d{decada}_{idx}", "ano": decada + (i % 10)})
            idx += 1
    return pd.DataFrame(rows)


def test_proporcao_por_decada():
    df = _build_df({1990: 100, 2000: 200, 2010: 500})
    out = stratified_sample_by_decade(df, frac=0.10, random_state=42)
    counts = (out["ano"].astype(int) // 10 * 10).value_counts().sort_index()
    assert counts.loc[1990] == 10
    assert counts.loc[2000] == 20
    assert counts.loc[2010] == 50
    assert len(out) == 80


def test_reprodutivel_com_random_state():
    df = _build_df({1990: 100, 2000: 100})
    a = stratified_sample_by_decade(df, frac=0.10, random_state=42)
    b = stratified_sample_by_decade(df, frac=0.10, random_state=42)
    assert list(a["id"]) == list(b["id"])


def test_random_state_diferente_muda_amostra():
    df = _build_df({1990: 100, 2000: 100})
    a = stratified_sample_by_decade(df, frac=0.10, random_state=42)
    b = stratified_sample_by_decade(df, frac=0.10, random_state=99)
    assert set(a["id"]) != set(b["id"])


def test_descarta_ano_invalido_e_ausente():
    df = pd.DataFrame(
        [
            {"id": "a", "ano": 2015},
            {"id": "b", "ano": None},
            {"id": "c", "ano": 88},  # outlier real visto no corpus IPEA
            {"id": "d", "ano": 2100},
            {"id": "e", "ano": 2015},
        ]
    )
    out = stratified_sample_by_decade(df, frac=1.0, random_state=42)
    assert set(out["id"]) == {"a", "e"}


def test_acrescenta_coluna_decada():
    df = _build_df({2000: 50})
    out = stratified_sample_by_decade(df, frac=0.20, random_state=42)
    assert "decada" in out.columns
    assert (out["decada"] == 2000).all()


def test_sem_coluna_ano_levanta():
    with pytest.raises(ValueError, match="coluna 'ano'"):
        stratified_sample_by_decade(pd.DataFrame([{"id": "a"}]), frac=0.1)


def test_frac_invalido_levanta():
    df = _build_df({2000: 10})
    with pytest.raises(ValueError, match="frac"):
        stratified_sample_by_decade(df, frac=0)
    with pytest.raises(ValueError, match="frac"):
        stratified_sample_by_decade(df, frac=1.5)


def test_frac_um_devolve_tudo_valido():
    df = _build_df({2000: 30, 2010: 40})
    out = stratified_sample_by_decade(df, frac=1.0, random_state=42)
    assert len(out) == 70
