"""
Download de PDFs do repositório IPEA via DSpace.

Portado de `ipeadata-lab/IpeaPub: ingestao/utils/clean_itens.py:baixar_pdf_real`.
A URL direta do PDF não está exposta no handle DSpace — é preciso GET na
página do documento, parsear o HTML e localizar o `<a href>` cujo `href`
contenha `bitstreams` e `download`.

Diferenças em relação ao IpeaPub original:
- `CACHE_DIR` é parâmetro, não global.
- Sem `print` — usa logging.
- Devolve `DownloadResult` tipado em vez de tupla solta.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from src.scraping.clean import create_retry_session

log = logging.getLogger(__name__)

REPO_BASE = "https://repositorio.ipea.gov.br"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


@dataclass(frozen=True)
class DownloadResult:
    pdf_path: Path
    download_url: str
    sha256: str
    cache_hit: bool


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def baixar_pdf(
    link_pagina: str,
    cache_dir: Path,
    timeout: int = 60,
) -> DownloadResult | None:
    """
    Baixa o PDF de um item DSpace, salvando em `cache_dir/{sha256}.pdf`.

    Retorna `None` se o link de download não for encontrado ou se o conteúdo
    não for um PDF válido.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    session = create_retry_session(total_retries=3, backoff_factor=2.0)

    log.info("Acessando página do documento: %s", link_pagina)
    try:
        resp = session.get(link_pagina, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
    except Exception as e:
        log.error("Falha ao acessar %s: %s", link_pagina, e)
        return None

    download_url = _extrair_link_download(resp.text)
    if not download_url:
        log.warning("Sem link de download em %s", link_pagina)
        return None

    log.info("Baixando PDF: %s", download_url)
    try:
        r = session.get(download_url, headers=HEADERS, timeout=timeout, allow_redirects=True)
        r.raise_for_status()
    except Exception as e:
        log.error("Falha no download de %s: %s", download_url, e)
        return None

    if not r.content.startswith(b"%PDF"):
        log.warning("Conteúdo não é PDF (primeiros bytes: %r)", r.content[:8])
        return None

    sha = _sha256(r.content)
    cache_path = cache_dir / f"{sha}.pdf"

    if cache_path.exists():
        log.info("Cache hit: %s", cache_path)
        return DownloadResult(
            pdf_path=cache_path, download_url=download_url, sha256=sha, cache_hit=True
        )

    cache_path.write_bytes(r.content)
    log.info("PDF salvo: %s (%d bytes)", cache_path, len(r.content))
    return DownloadResult(
        pdf_path=cache_path, download_url=download_url, sha256=sha, cache_hit=False
    )


def _extrair_link_download(html: str) -> str | None:
    """Acha o primeiro `<a href>` com `bitstreams` e `download` no href."""
    soup = BeautifulSoup(html, "html.parser")
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "bitstreams" in href and "download" in href:
            return urljoin(REPO_BASE, href)
    return None
