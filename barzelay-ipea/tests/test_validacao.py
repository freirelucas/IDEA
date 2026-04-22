"""Testes de persistência e amostragem para validação humana."""
from __future__ import annotations

import pandas as pd
import pytest

from src.validacao.persist import (
    append_validacao,
    carregar_validacoes,
    ja_anotado,
)
from src.validacao.sample import amostra_validacao


# ─── persist ───────────────────────────────────────────────────────────────────

class TestPersist:
    def test_carrega_vazio_se_nao_existir(self, tmp_path):
        df = carregar_validacoes(tmp_path / "naoexiste.parquet")
        assert df.empty
        assert "document_id" in df.columns

    def test_append_cria_arquivo(self, tmp_path):
        out = tmp_path / "v.parquet"
        append_validacao(out, "doc1", "D1", 4, "porque sim", "alice")
        df = pd.read_parquet(out)
        assert len(df) == 1
        assert df.iloc[0]["score_humano"] == 4
        assert df.iloc[0]["anotador_id"] == "alice"

    def test_append_acumula(self, tmp_path):
        out = tmp_path / "v.parquet"
        append_validacao(out, "doc1", "D1", 4, "j1", "alice")
        append_validacao(out, "doc1", "D2", 3, "j2", "alice")
        append_validacao(out, "doc1", "D1", 5, "j3", "bob")
        df = pd.read_parquet(out)
        assert len(df) == 3

    def test_dimensao_invalida(self, tmp_path):
        with pytest.raises(ValueError, match="dimensao"):
            append_validacao(tmp_path / "v.parquet", "d", "D7", 3, "j", "a")

    def test_score_fora_de_range(self, tmp_path):
        with pytest.raises(ValueError, match="score"):
            append_validacao(tmp_path / "v.parquet", "d", "D1", 6, "j", "a")
        with pytest.raises(ValueError, match="score"):
            append_validacao(tmp_path / "v.parquet", "d", "D1", 0, "j", "a")

    def test_ja_anotado(self, tmp_path):
        out = tmp_path / "v.parquet"
        assert ja_anotado(out, "doc1", "D1", "alice") is False
        append_validacao(out, "doc1", "D1", 4, "j", "alice")
        assert ja_anotado(out, "doc1", "D1", "alice") is True
        assert ja_anotado(out, "doc1", "D1", "bob") is False
        assert ja_anotado(out, "doc1", "D2", "alice") is False


# ─── sample ────────────────────────────────────────────────────────────────────

def _build_corpus(n_per_decade: dict[int, int], tipos: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows_meta = []
    rows_cls = []
    idx = 0
    for dec, n in n_per_decade.items():
        for i in range(n):
            doc_id = f"d{dec}_{i}"
            tipo = tipos[i % len(tipos)]
            rows_meta.append({"id": doc_id, "ano": dec + (i % 10), "tipo": tipo})
            for d_idx, dim in enumerate(["D1", "D2", "D3", "D4", "D5"]):
                # alterna scores: alguns extremos, outros no meio
                score = 1 if (i + d_idx) % 7 == 0 else (5 if (i + d_idx) % 11 == 0 else 3)
                rows_cls.append({"document_id": doc_id, "dimensao": dim, "score": score})
            idx += 1
    return pd.DataFrame(rows_meta), pd.DataFrame(rows_cls)


class TestAmostra:
    def test_quantidade_proxima_de_n_target(self):
        meta, cls = _build_corpus(
            {2000: 200, 2010: 400, 2020: 300},
            tipos=["Working paper", "Journal article", "Book"],
        )
        out = amostra_validacao(cls, meta, n_target=100, random_state=42)
        # tolera margem porque round + clip por estrato pode somar mais
        assert 60 <= len(out) <= 140

    def test_estratifica_por_decada_e_tipo(self):
        meta, cls = _build_corpus(
            {2000: 200, 2010: 400}, tipos=["Working paper", "Journal article"]
        )
        out = amostra_validacao(cls, meta, n_target=80, random_state=42)
        # Cada combinação (decada, tipo) deve aparecer pelo menos 1 vez
        combos = set(zip(out["decada"], out["tipo"]))
        assert (2000, "Working paper") in combos
        assert (2000, "Journal article") in combos
        assert (2010, "Working paper") in combos
        assert (2010, "Journal article") in combos

    def test_oversample_extremos(self):
        meta, cls = _build_corpus(
            {2010: 200}, tipos=["Working paper"]
        )
        out = amostra_validacao(
            cls, meta, n_target=50, oversample_extremos=0.6, random_state=42
        )
        # esperamos pelo menos algum n_extremos > 0 na amostra
        assert (out["n_extremos"] >= 1).any()

    def test_filtra_tipos_alvo(self):
        meta, cls = _build_corpus(
            {2010: 100}, tipos=["Outro Tipo Não Listado"]
        )
        out = amostra_validacao(
            cls, meta, n_target=10,
            tipos_alvo=("Working paper",),
            random_state=42,
        )
        assert out.empty

    def test_reprodutivel(self):
        meta, cls = _build_corpus(
            {2000: 100, 2010: 100}, tipos=["Working paper", "Book"]
        )
        a = amostra_validacao(cls, meta, n_target=40, random_state=42)
        b = amostra_validacao(cls, meta, n_target=40, random_state=42)
        assert set(a["id"]) == set(b["id"])
