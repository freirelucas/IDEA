"""
Normalização de metadados da API DSpace do IPEA.

Portado de `ipeadata-lab/IpeaPub: ingestao/utils/clean_itens.py` (Rafael Dias
Ghiorzi). Conserva apenas a parte de normalização de metadados + sessão HTTP
com retry. Deixa `baixar_pdf_real` de fora (Fase 2).
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Iterable

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def create_retry_session(
    total_retries: int = 2,
    backoff_factor: float = 1.0,
) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=total_retries,
        read=total_retries,
        connect=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def _normalize_string(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        value = "; ".join(str(v) for v in value)
    value = str(value)
    value = re.sub(r"\s+", " ", value).strip()
    return value or None


def _join_field(value: Any, sep: str = "; ") -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        flat: list[str] = []
        for v in value:
            if isinstance(v, list):
                flat.extend(str(x) for x in v)
            else:
                flat.append(str(v))
        return _normalize_string(sep.join(flat))
    return _normalize_string(value)


def _parse_year(ano: Any) -> int | None:
    if ano is None:
        return None
    match = re.search(r"(\d{4})", str(ano))
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _parse_last_modified(iso_str: str | None) -> tuple[str | None, int | None]:
    if not iso_str:
        return None, None
    try:
        dt = datetime.fromisoformat(iso_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        dt_utc = dt.astimezone(timezone.utc)
        return dt_utc.isoformat().replace("+00:00", "Z"), int(dt_utc.timestamp())
    except Exception:
        return None, None


def clean_item(raw: dict[str, Any]) -> dict[str, Any]:
    """Normaliza um item bruto extraído da API DSpace em schema estável."""
    cleaned: dict[str, Any] = {}

    cleaned["id"] = _normalize_string(raw.get("id"))
    cleaned["handle"] = _normalize_string(raw.get("handle"))

    cleaned["titulo"] = _normalize_string(raw.get("titulo")) or ""
    cleaned["resumo"] = _join_field(raw.get("resumo"), sep="\n\n") or ""

    cleaned["autores"] = _join_field(raw.get("autores"), sep="; ") or ""

    palavras = raw.get("palavras_chave") or raw.get("palavras-chave")
    cleaned["palavras_chave"] = _join_field(palavras, sep=", ") or ""

    cleaned["tipo"] = _join_field(raw.get("tipo"), sep="; ") or ""

    cleaned["contributor_other"] = (
        _join_field(raw.get("contributor_other"), sep=" | ") or ""
    )
    cleaned["orgunit_uuid"] = _join_field(raw.get("orgunit_uuid"), sep=",") or ""

    cleaned["ano"] = _parse_year(raw.get("ano"))

    iso_utc, epoch = _parse_last_modified(_normalize_string(raw.get("last_modified")))
    cleaned["last_modified"] = iso_utc
    cleaned["last_modified_ts"] = epoch

    return cleaned


def clean_items(raws: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [clean_item(r) for r in raws]
