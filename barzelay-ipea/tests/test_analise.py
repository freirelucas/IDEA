"""Testes de concordância, calibração e longitudinal (Fase 5)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analise.calibracao import aplicar_calibradores, treinar_calibradores
from src.analise.concordancia import (
    cohen_kappa_llm_vs_humano,
    krippendorff_alpha_inter_annotators,
)
from src.analise.longitudinal import (
    media_por_decada,
    media_por_decada_e_tipo,
    top_palavras_por_polo,
)


def _cls_df(rows: list[tuple[str, str, int]]) -> pd.DataFrame:
    """rows: (doc_id, dim, score)"""
    out = pd.DataFrame(rows, columns=["document_id", "dimensao", "score"])
    out["timestamp"] = "2026-01-01T00:00:00Z"
    return out


def _val_df(rows: list[tuple[str, str, int, str]]) -> pd.DataFrame:
    """rows: (doc_id, dim, score, anotador)"""
    return pd.DataFrame(rows, columns=["document_id", "dimensao", "score_humano", "anotador_id"])


# ─── concordância ─────────────────────────────────────────────────────────────

class TestCohenKappa:
    def test_perfeita(self):
        cls = _cls_df([(f"d{i}", "D1", s) for i, s in enumerate([1, 2, 3, 4, 5, 1, 2])])
        val = _val_df([(f"d{i}", "D1", s, "alice") for i, s in enumerate([1, 2, 3, 4, 5, 1, 2])])
        out = cohen_kappa_llm_vs_humano(cls, val)
        d1 = next(k for k in out if k.dimensao == "D1")
        assert d1.kappa > 0.95
        assert d1.n_pares == 7

    def test_aleatoria_proximo_de_zero(self):
        rng = np.random.default_rng(42)
        cls = _cls_df([(f"d{i}", "D1", int(rng.integers(1, 6))) for i in range(60)])
        val = _val_df([(f"d{i}", "D1", int(rng.integers(1, 6)), "a") for i in range(60)])
        out = cohen_kappa_llm_vs_humano(cls, val)
        d1 = next(k for k in out if k.dimensao == "D1")
        assert -0.3 < d1.kappa < 0.3

    def test_dimensao_sem_dados_devolve_nan(self):
        cls = _cls_df([("d1", "D1", 3)])
        val = _val_df([("d1", "D1", 3, "alice")])
        out = cohen_kappa_llm_vs_humano(cls, val)
        d2 = next(k for k in out if k.dimensao == "D2")
        assert d2.n_pares == 0
        assert np.isnan(d2.kappa)

    def test_agrega_humanos_por_mediana(self):
        cls = _cls_df([("d1", "D1", 3), ("d2", "D1", 5), ("d3", "D1", 1), ("d4", "D1", 4)])
        val = _val_df([
            ("d1", "D1", 3, "a"), ("d1", "D1", 3, "b"),
            ("d2", "D1", 5, "a"), ("d2", "D1", 5, "b"),
            ("d3", "D1", 1, "a"), ("d3", "D1", 1, "b"),
            ("d4", "D1", 4, "a"), ("d4", "D1", 4, "b"),
        ])
        out = cohen_kappa_llm_vs_humano(cls, val)
        d1 = next(k for k in out if k.dimensao == "D1")
        assert d1.n_pares == 4
        assert d1.kappa > 0.95


class TestKrippendorff:
    def test_concordancia_perfeita_alpha_um(self):
        val = _val_df([
            ("d1", "D1", 3, "a"), ("d1", "D1", 3, "b"),
            ("d2", "D1", 4, "a"), ("d2", "D1", 4, "b"),
            ("d3", "D1", 1, "a"), ("d3", "D1", 1, "b"),
            ("d4", "D1", 5, "a"), ("d4", "D1", 5, "b"),
        ])
        out = krippendorff_alpha_inter_annotators(val)
        assert out["D1"] > 0.99

    def test_um_anotador_eh_nan(self):
        val = _val_df([
            ("d1", "D1", 3, "a"), ("d2", "D1", 4, "a"), ("d3", "D1", 1, "a"),
        ])
        out = krippendorff_alpha_inter_annotators(val)
        assert np.isnan(out["D1"])


# ─── calibração ───────────────────────────────────────────────────────────────

class TestCalibracao:
    def _exemplo(self, n: int = 30):
        # LLM tende a inflar — humano sistematicamente 1 ponto abaixo.
        cls = _cls_df([(f"d{i}", "D1", (i % 5) + 1) for i in range(n)])
        val = _val_df([(f"d{i}", "D1", max(1, (i % 5)), "a") for i in range(n)])
        return cls, val

    def test_treina_e_aplica(self):
        cls, val = self._exemplo()
        cals = treinar_calibradores(cls, val)
        assert "D1" in cals
        assert cals["D1"].n_pares_treino == 30
        out = aplicar_calibradores(cls, cals)
        assert "score_calibrado" in out.columns
        # calibrado deve estar dentro de [1, 5]
        s = out[out["dimensao"] == "D1"]["score_calibrado"]
        assert s.min() >= 1.0 and s.max() <= 5.0

    def test_pares_insuficientes_pula_treino(self):
        cls = _cls_df([("d1", "D1", 3), ("d2", "D1", 4)])
        val = _val_df([("d1", "D1", 3, "a"), ("d2", "D1", 4, "a")])
        cals = treinar_calibradores(cls, val)
        assert "D1" not in cals  # só 2 pares < 5

    def test_aplicar_sem_calibrador_eh_identidade(self):
        cls = _cls_df([("d1", "D1", 3)])
        out = aplicar_calibradores(cls, {})
        assert out["score_calibrado"].iloc[0] == 3.0


# ─── longitudinal ─────────────────────────────────────────────────────────────

class TestLongitudinal:
    def _setup(self):
        cls = pd.DataFrame([
            {"document_id": "d1", "dimensao": "D1", "score": 4, "score_calibrado": 3.5,
             "timestamp": "2026-01-01T00:00:00Z"},
            {"document_id": "d2", "dimensao": "D1", "score": 2, "score_calibrado": 1.5,
             "timestamp": "2026-01-01T00:00:00Z"},
            {"document_id": "d3", "dimensao": "D2", "score": 3, "score_calibrado": 3.0,
             "timestamp": "2026-01-01T00:00:00Z"},
        ])
        meta = pd.DataFrame([
            {"id": "d1", "ano": 2015, "tipo": "Working paper",
             "palavras_chave": "design, política, capacidade"},
            {"id": "d2", "ano": 2018, "tipo": "Working paper",
             "palavras_chave": "evidências, dados"},
            {"id": "d3", "ano": 2005, "tipo": "Book",
             "palavras_chave": "história"},
        ])
        return cls, meta

    def test_media_por_decada(self):
        cls, meta = self._setup()
        out = media_por_decada(cls, meta)
        assert out.loc[2010, "D1"] == pytest.approx((3.5 + 1.5) / 2)
        assert out.loc[2000, "D2"] == pytest.approx(3.0)

    def test_media_por_decada_e_tipo(self):
        cls, meta = self._setup()
        out = media_por_decada_e_tipo(cls, meta)
        assert {"decada", "tipo", "dimensao", "score_medio"} <= set(out.columns)
        wp_2010 = out[(out["decada"] == 2010) & (out["tipo"] == "Working paper") & (out["dimensao"] == "D1")]
        assert wp_2010["score_medio"].iloc[0] == pytest.approx((3.5 + 1.5) / 2)

    def test_top_palavras_por_polo(self):
        cls, meta = self._setup()
        out = top_palavras_por_polo(
            cls, meta, dimensao="D1",
            threshold_design=3.0, threshold_descritivo=2.0,
            top_n=10,
        )
        assert "design" in out["design"].index
        assert "evidências" in out["descritivo"].index
