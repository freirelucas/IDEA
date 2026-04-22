"""
Fase 2 CLI: download PDF + Docling sobre um Parquet de metadados.

Exemplos:
    # smoke (3 docs)
    python -m src.extracao --input data/interim/metadados_amostra10.parquet --limit 3

    # produção (todos os docs do parquet)
    python -m src.extracao --input data/interim/metadados_amostra10.parquet
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.extracao.download import baixar_pdf
from src.extracao.pdf import ler_pdf_com_docling


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="Parquet de metadados (Fase 1).")
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data/interim/textos.parquet"),
        help="Parquet de saída (texto extraído).",
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("cache/pdfs"),
        help="Diretório de cache de PDFs (sha256.pdf).",
    )
    p.add_argument("--limit", type=int, default=None, help="Limita N documentos (smoke).")
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Pula docs cujo id já existe no parquet de saída.",
    )
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )
    log = logging.getLogger("barzelay.extracao")

    df = pd.read_parquet(args.input)
    log.info("Metadados: %d docs em %s", len(df), args.input)

    if args.skip_existing and args.output.exists():
        existing = set(pd.read_parquet(args.output)["id"].tolist())
        df = df[~df["id"].isin(existing)]
        log.info("Após skip-existing: %d docs novos", len(df))

    if args.limit:
        df = df.head(args.limit)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for _, meta in tqdm(df.iterrows(), total=len(df), desc="extraindo"):
        doc_id = meta["id"]
        handle = meta.get("handle")
        if not handle:
            rows.append(_row_error(doc_id, handle, "sem_handle"))
            continue

        try:
            dl = baixar_pdf(handle, cache_dir=args.cache_dir)
        except Exception as e:
            log.exception("Erro download %s", doc_id)
            rows.append(_row_error(doc_id, handle, f"download_err:{e}"))
            continue
        if dl is None:
            rows.append(_row_error(doc_id, handle, "sem_pdf"))
            continue

        try:
            extract = ler_pdf_com_docling(dl.pdf_path)
        except Exception as e:
            log.exception("Erro Docling %s", doc_id)
            rows.append(_row_error(doc_id, handle, f"docling_err:{e}", sha256=dl.sha256))
            continue

        rows.append({
            "id": doc_id,
            "handle": handle,
            "download_url": dl.download_url,
            "sha256": dl.sha256,
            "n_pages": extract.n_pages,
            "full_text": extract.full_text,
            "tables_md": extract.tables_md,
            "error": extract.error,
            "extracted_at": datetime.now(timezone.utc).isoformat(),
        })

        # flush incremental a cada 25 docs (proteção contra crash em runs longos)
        if len(rows) % 25 == 0:
            _flush(rows, args.output)

    _flush(rows, args.output)
    log.info("Concluído: %d linhas em %s", len(rows), args.output)
    return 0


def _row_error(doc_id: str, handle: str | None, err: str, sha256: str = "") -> dict:
    return {
        "id": doc_id,
        "handle": handle,
        "download_url": "",
        "sha256": sha256,
        "n_pages": 0,
        "full_text": "",
        "tables_md": [],
        "error": err,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
    }


def _flush(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    df_out = pd.DataFrame(rows)
    if path.exists():
        existing = pd.read_parquet(path)
        df_out = pd.concat([existing, df_out], ignore_index=True).drop_duplicates(
            subset=["id"], keep="last"
        )
    df_out.to_parquet(path, index=False)


if __name__ == "__main__":
    raise SystemExit(main())
