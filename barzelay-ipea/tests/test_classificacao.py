"""Testes do classificador (mocks do LLM)."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.classificacao.classifier import (
    CallStats,
    _parse_json,
    _split_system_user,
    classificar,
)
from src.classificacao.prompts import DIMENSOES, load_prompt
from src.classificacao.schema import ClassificacaoLLM
from src.classificacao.truncate import truncate_to_chars, truncate_to_tokens


# ─── schema ────────────────────────────────────────────────────────────────────

class TestSchema:
    def test_resposta_valida(self):
        c = ClassificacaoLLM(
            dimensao="D1",
            score=4,
            trechos_evidencia=["recomenda-se", "deveria"],
            justificativa="Forte uso de modais deônticos.",
        )
        assert c.score == 4

    def test_score_fora_de_range(self):
        with pytest.raises(ValidationError):
            ClassificacaoLLM(dimensao="D1", score=6, trechos_evidencia=["a"], justificativa="x")
        with pytest.raises(ValidationError):
            ClassificacaoLLM(dimensao="D1", score=0, trechos_evidencia=["a"], justificativa="x")

    def test_dimensao_invalida(self):
        with pytest.raises(ValidationError):
            ClassificacaoLLM(dimensao="D6", score=3, trechos_evidencia=["a"], justificativa="x")

    def test_trechos_strip(self):
        c = ClassificacaoLLM(
            dimensao="D2",
            score=3,
            trechos_evidencia=["  ok  ", "", "outro"],
            justificativa="..",
        )
        assert c.trechos_evidencia == ["ok", "outro"]


# ─── truncate ──────────────────────────────────────────────────────────────────

class TestTruncate:
    def test_truncate_chars_curto(self):
        out, was = truncate_to_chars("abc", 10)
        assert out == "abc" and was is False

    def test_truncate_chars_longo(self):
        out, was = truncate_to_chars("abcdefghij", 5)
        assert out == "abcde" and was is True

    def test_truncate_tokens(self):
        out, was = truncate_to_tokens("x" * 100, max_tokens=10, chars_per_token=4)
        assert len(out) == 40 and was is True


# ─── prompts ───────────────────────────────────────────────────────────────────

class TestPrompts:
    @pytest.mark.parametrize("dim", DIMENSOES)
    def test_carrega_todas(self, dim):
        tpl = load_prompt(dim)
        assert tpl.dimensao == dim
        assert tpl.versao  # versão extraída
        assert "{texto}" in tpl.template

    def test_render_substitui_placeholders(self):
        tpl = load_prompt("D1")
        rendered = tpl.render(
            titulo="T", autores="A", ano=2020, tipo="Working paper",
            texto="conteúdo", n_tokens=100,
        )
        assert "T" in rendered
        assert "A" in rendered
        assert "2020" in rendered
        assert "conteúdo" in rendered
        assert "{texto}" not in rendered  # nada de placeholder remanescente

    def test_dimensao_invalida(self):
        with pytest.raises(ValueError):
            load_prompt("D6")


# ─── parsers ───────────────────────────────────────────────────────────────────

class TestParsers:
    def test_json_puro(self):
        text = '{"dimensao":"D1","score":3,"trechos_evidencia":["a"],"justificativa":"j"}'
        assert _parse_json(text)["score"] == 3

    def test_json_com_fence(self):
        text = '```json\n{"dimensao":"D2","score":4,"trechos_evidencia":["a"],"justificativa":"j"}\n```'
        assert _parse_json(text)["dimensao"] == "D2"

    def test_json_dentro_de_texto(self):
        text = 'Aqui está:\n{"dimensao":"D3","score":2,"trechos_evidencia":["a"],"justificativa":"j"}\nFim.'
        assert _parse_json(text)["dimensao"] == "D3"

    def test_split_system_user(self):
        rendered = load_prompt("D1").render(
            titulo="t", autores="a", ano=2020, tipo="x", texto="y", n_tokens=100,
        )
        sys_, user = _split_system_user(rendered)
        assert "Você é um analista" in sys_
        assert "## Documento" in user
        assert "y" in user


# ─── classificar (com mock) ────────────────────────────────────────────────────

class TestClassificar:
    def test_chamada_com_mock(self):
        captured = {}

        def fake_caller(system, user, max_tokens, temp):
            captured["system"] = system
            captured["user"] = user
            return (
                '{"dimensao":"D1","score":4,"trechos_evidencia":["recomenda-se"],"justificativa":"forte modal"}',
                CallStats(tokens_input=12345, tokens_output=42),
            )

        res = classificar(
            doc_id="abc",
            titulo="Política de saneamento",
            autores="Silva, J.",
            ano=2020,
            tipo="Working paper",
            texto="Recomenda-se que o ministério... " * 100,
            dimensao="D1",
            caller=fake_caller,
        )
        assert res.classificacao.score == 4
        assert res.stats.tokens_input == 12345
        assert "Você é um analista" in captured["system"]
        assert "Política de saneamento" in captured["user"]

    def test_dimensao_divergente_eh_erro(self):
        def fake_caller(system, user, max_tokens, temp):
            return (
                '{"dimensao":"D2","score":3,"trechos_evidencia":["x"],"justificativa":"y"}',
                CallStats(),
            )
        with pytest.raises(ValueError, match="dimens"):
            classificar(
                doc_id="abc", titulo="t", autores="a", ano=2020, tipo="x",
                texto="...", dimensao="D1", caller=fake_caller,
            )

    def test_json_invalido_eh_erro(self):
        def fake_caller(system, user, max_tokens, temp):
            return ("não é json", CallStats())
        with pytest.raises(Exception):
            classificar(
                doc_id="abc", titulo="t", autores="a", ano=2020, tipo="x",
                texto="...", dimensao="D1", caller=fake_caller,
            )
