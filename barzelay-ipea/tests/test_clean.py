"""Testes unitários de normalização de metadados DSpace."""
from __future__ import annotations

from src.scraping.clean import (
    _join_field,
    _normalize_string,
    _parse_last_modified,
    _parse_year,
    clean_item,
)


class TestNormalizeString:
    def test_none_returns_none(self):
        assert _normalize_string(None) is None

    def test_collapses_whitespace(self):
        assert _normalize_string("  foo   bar\n\tbaz  ") == "foo bar baz"

    def test_empty_becomes_none(self):
        assert _normalize_string("   \n\t ") is None

    def test_list_joined_with_semicolons(self):
        assert _normalize_string(["a", "b", "c"]) == "a; b; c"


class TestJoinField:
    def test_none(self):
        assert _join_field(None) is None

    def test_scalar(self):
        assert _join_field("foo") == "foo"

    def test_list(self):
        assert _join_field(["a", "b"], sep=", ") == "a, b"

    def test_nested_list_flattened(self):
        assert _join_field([["a", "b"], "c"], sep="|") == "a|b|c"


class TestParseYear:
    def test_none(self):
        assert _parse_year(None) is None

    def test_bare_year(self):
        assert _parse_year("1998") == 1998

    def test_iso_date(self):
        assert _parse_year("2010-05-15") == 2010

    def test_embedded_year(self):
        assert _parse_year("janeiro de 2023") == 2023

    def test_no_year(self):
        assert _parse_year("sem ano") is None


class TestParseLastModified:
    def test_none(self):
        assert _parse_last_modified(None) == (None, None)

    def test_empty(self):
        assert _parse_last_modified("") == (None, None)

    def test_naive_datetime_gets_utc(self):
        iso, ts = _parse_last_modified("2024-01-15T10:30:00")
        assert iso is not None and iso.endswith("Z")
        assert isinstance(ts, int)

    def test_aware_datetime_preserved(self):
        iso, ts = _parse_last_modified("2024-01-15T10:30:00+00:00")
        assert iso == "2024-01-15T10:30:00Z"
        assert isinstance(ts, int)

    def test_malformed_returns_none(self):
        assert _parse_last_modified("not a date") == (None, None)


class TestCleanItem:
    def test_full_item(self):
        raw = {
            "id": "abc-123",
            "handle": "https://repositorio.ipea.gov.br/handle/11058/1234",
            "titulo": "  Um estudo   sobre gestão  pública  ",
            "autores": ["Silva, J.", "Souza, M."],
            "ano": "2015-03-01",
            "resumo": ["Parágrafo 1.", "Parágrafo 2."],
            "palavras_chave": ["gestão pública", "avaliação", "Estado"],
            "tipo": ["Texto para Discussão"],
            "last_modified": "2024-01-15T10:30:00",
        }
        out = clean_item(raw)
        assert out["id"] == "abc-123"
        assert out["handle"].startswith("https://")
        assert out["titulo"] == "Um estudo sobre gestão pública"
        assert out["autores"] == "Silva, J.; Souza, M."
        assert out["ano"] == 2015
        assert "Parágrafo 1." in out["resumo"]
        assert "Parágrafo 2." in out["resumo"]
        assert out["palavras_chave"] == "gestão pública, avaliação, Estado"
        assert out["tipo"] == "Texto para Discussão"
        assert out["last_modified"].endswith("Z")
        assert isinstance(out["last_modified_ts"], int)

    def test_missing_fields_default_to_empty_or_none(self):
        out = clean_item({})
        assert out["id"] is None
        assert out["handle"] is None
        assert out["titulo"] == ""
        assert out["resumo"] == ""
        assert out["autores"] == ""
        assert out["palavras_chave"] == ""
        assert out["tipo"] == ""
        assert out["ano"] is None
        assert out["last_modified"] is None
        assert out["last_modified_ts"] is None

    def test_palavras_chave_aceita_chave_alternativa(self):
        out = clean_item({"palavras-chave": ["a", "b"]})
        assert out["palavras_chave"] == "a, b"

    def test_ano_invalido_devolve_none(self):
        out = clean_item({"ano": "sem data"})
        assert out["ano"] is None

    def test_scalar_strings_tambem_aceitas_em_listas(self):
        out = clean_item({"autores": "Único Autor", "tipo": "Livro"})
        assert out["autores"] == "Único Autor"
        assert out["tipo"] == "Livro"

    def test_contributor_other_e_orgunit_uuid(self):
        raw = {
            "contributor_other": [
                "Diretoria de Estudos e Políticas do Estado, das Instituições e da Democracia - DIEST",
                "Instituto de Pesquisa Econômica Aplicada",
            ],
            "orgunit_uuid": ["38b33462-dc75-43f2-9bda-1f3c54757ae7"],
        }
        out = clean_item(raw)
        assert "DIEST" in out["contributor_other"]
        assert " | " in out["contributor_other"]
        assert out["orgunit_uuid"] == "38b33462-dc75-43f2-9bda-1f3c54757ae7"

    def test_contributor_other_e_orgunit_uuid_vazios(self):
        out = clean_item({})
        assert out["contributor_other"] == ""
        assert out["orgunit_uuid"] == ""
