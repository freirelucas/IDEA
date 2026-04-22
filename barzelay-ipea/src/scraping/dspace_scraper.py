"""
Scraper paginado da API DSpace do repositório institucional do IPEA.

Endpoint:
    https://repositorio.ipea.gov.br/server/api/discover/browses/dateissued/items

Portado de `ipeadata-lab/IpeaPub: ingestao/scraper.py` (Rafael Dias Ghiorzi),
com duas adaptações para o projeto Barzelay:

1. Loop paginado completo via `iterate_pages` — o scraper original processa
   uma página por chamada e delega a iteração ao caller (briefing §5, Fase 1).
2. Remoção do acoplamento com `MetadataDB`/SQLite. Fase 1 escreve só em
   Parquet; controle de ingestão via SQLite pode ser reintroduzido na Fase 2.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Iterator

from src.scraping.clean import create_retry_session

log = logging.getLogger(__name__)

BASE_URL = (
    "https://repositorio.ipea.gov.br/server/api/discover/browses/"
    "dateissued/items?sort=dateissued,DESC"
)


class DSpaceScraper:
    """Itera pela API DSpace do IPEA e extrai campos Dublin Core."""

    def __init__(
        self,
        base_url: str = BASE_URL,
        request_timeout: int = 120,
        sleep_between_pages: float = 0.5,
    ) -> None:
        self.base_url = base_url
        self.request_timeout = request_timeout
        self.sleep_between_pages = sleep_between_pages
        self.session = create_retry_session(total_retries=3, backoff_factor=2.0)

    def fetch_page(self, page_number: int) -> list[dict[str, Any]]:
        """Devolve os itens brutos de uma página da API DSpace."""
        sep = "&" if "?" in self.base_url else "?"
        url = f"{self.base_url}{sep}page={page_number}"
        r = self.session.get(url, timeout=self.request_timeout)
        r.raise_for_status()
        data = r.json()
        return data.get("_embedded", {}).get("items", [])

    @staticmethod
    def extract_fields(item: dict[str, Any]) -> dict[str, Any]:
        """
        Extrai campos Dublin Core de um item DSpace (sem normalizar).

        Campos com cardinalidade >1 em Dublin Core (autores, resumo, palavras,
        tipo) são sempre retornados como lista, mesmo com 0 ou 1 elemento,
        para manter o schema estável ao serializar em Parquet.
        """
        metadata = item.get("metadata", {})

        def get_scalar(field: str) -> str | None:
            arr = metadata.get(field, [])
            if not arr:
                return None
            return arr[0].get("value")

        def get_list(field: str) -> list[str]:
            arr = metadata.get(field, [])
            return [v.get("value") for v in arr if v.get("value") is not None]

        titulo = get_scalar("dc.title.scholar") or item.get("name") or ""

        return {
            "id": item.get("id"),
            "handle": get_scalar("dc.identifier.uri"),
            "titulo": titulo,
            "autores": get_list("dc.contributor.author"),
            "ano": get_scalar("dc.date.issued"),
            "resumo": get_list("dc.description.abstract"),
            "palavras_chave": get_list("dc.subject.keyword"),
            "tipo": get_list("dc.type"),
            "last_modified": item.get("lastModified"),
        }

    def iterate_pages(
        self,
        start: int = 0,
        limit: int | None = None,
    ) -> Iterator[tuple[int, list[dict[str, Any]]]]:
        """
        Gera (page_number, items) até a página retornar lista vazia ou
        `limit` páginas serem processadas.
        """
        page = start
        processed = 0
        while True:
            if limit is not None and processed >= limit:
                log.info("Limite de %d páginas atingido.", limit)
                return
            try:
                items = self.fetch_page(page)
            except Exception as e:
                log.error("Falha ao buscar página %d: %s", page, e)
                return
            if not items:
                log.info("Página %d vazia — fim da paginação.", page)
                return
            yield page, items
            processed += 1
            page += 1
            if self.sleep_between_pages > 0:
                time.sleep(self.sleep_between_pages)

    def run(
        self,
        start: int = 0,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Raspa páginas e devolve lista de dicts com campos extraídos (brutos)."""
        collected: list[dict[str, Any]] = []
        for page, items in self.iterate_pages(start=start, limit=limit):
            log.info("Página %d: %d itens.", page, len(items))
            for raw in items:
                try:
                    collected.append(self.extract_fields(raw))
                except Exception as e:
                    log.warning(
                        "Falha ao extrair item %s na página %d: %s",
                        raw.get("id", "?"),
                        page,
                        e,
                    )
        log.info("Total coletado: %d itens.", len(collected))
        return collected
