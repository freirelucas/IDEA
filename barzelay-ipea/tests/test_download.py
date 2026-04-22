"""Testes do downloader (mocks de rede)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.extracao.download import _extrair_link_download, baixar_pdf


HTML_OK = """
<html><body>
  <a href="/handle/11058/123/abc">Página</a>
  <a href="/server/api/core/bitstreams/abc-uuid/content/download">Baixar PDF</a>
</body></html>
"""
HTML_SEM_LINK = "<html><body><a href='/foo'>x</a></body></html>"


def test_extrair_link_download():
    url = _extrair_link_download(HTML_OK)
    assert url is not None
    assert url.endswith("/server/api/core/bitstreams/abc-uuid/content/download")
    assert url.startswith("https://repositorio.ipea.gov.br")


def test_extrair_link_download_ausente():
    assert _extrair_link_download(HTML_SEM_LINK) is None


def _mock_session(page_html: str, pdf_bytes: bytes):
    session = MagicMock()
    page_resp = MagicMock(text=page_html)
    page_resp.raise_for_status = MagicMock()
    pdf_resp = MagicMock(content=pdf_bytes)
    pdf_resp.raise_for_status = MagicMock()
    session.get.side_effect = [page_resp, pdf_resp]
    return session


def test_baixar_pdf_grava_no_cache(tmp_path):
    pdf_bytes = b"%PDF-1.4 fake content"
    session = _mock_session(HTML_OK, pdf_bytes)
    with patch("src.extracao.download.create_retry_session", return_value=session):
        result = baixar_pdf("https://x/handle/1", cache_dir=tmp_path)
    assert result is not None
    assert result.pdf_path.exists()
    assert result.pdf_path.read_bytes() == pdf_bytes
    assert result.cache_hit is False


def test_baixar_pdf_cache_hit(tmp_path):
    pdf_bytes = b"%PDF-1.4 fake content"
    import hashlib
    sha = hashlib.sha256(pdf_bytes).hexdigest()
    cache_path = tmp_path / f"{sha}.pdf"
    cache_path.write_bytes(pdf_bytes)
    session = _mock_session(HTML_OK, pdf_bytes)
    with patch("src.extracao.download.create_retry_session", return_value=session):
        result = baixar_pdf("https://x/handle/1", cache_dir=tmp_path)
    assert result is not None
    assert result.cache_hit is True
    assert result.pdf_path == cache_path


def test_baixar_pdf_sem_link(tmp_path):
    session = MagicMock()
    page_resp = MagicMock(text=HTML_SEM_LINK)
    page_resp.raise_for_status = MagicMock()
    session.get.return_value = page_resp
    with patch("src.extracao.download.create_retry_session", return_value=session):
        result = baixar_pdf("https://x/handle/1", cache_dir=tmp_path)
    assert result is None


def test_baixar_pdf_conteudo_nao_pdf(tmp_path):
    session = _mock_session(HTML_OK, b"<html>not a pdf</html>")
    with patch("src.extracao.download.create_retry_session", return_value=session):
        result = baixar_pdf("https://x/handle/1", cache_dir=tmp_path)
    assert result is None
