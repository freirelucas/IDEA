"""
CLI Fase 1: raspa a API DSpace do IPEA e escreve metadados em Parquet.

Exemplos:
    python -m src.scraping --limit 2
    python -m src.scraping --start-page 0 --raw-dir data/raw --interim-path data/interim/metadados.parquet
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.scraping.clean import clean_item
from src.scraping.dspace_scraper import DSpaceScraper


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--start-page", type=int, default=0, help="Página inicial (0-indexed)."
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Número máximo de páginas (default: raspa até fim).",
    )
    p.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Diretório para metadados brutos.",
    )
    p.add_argument(
        "--interim-path",
        type=Path,
        default=Path("data/interim/metadados.parquet"),
        help="Caminho do Parquet normalizado.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )
    log = logging.getLogger("barzelay.scraping")

    args.raw_dir.mkdir(parents=True, exist_ok=True)
    args.interim_path.parent.mkdir(parents=True, exist_ok=True)

    scraper = DSpaceScraper()
    raw_items = scraper.run(start=args.start_page, limit=args.limit)

    if not raw_items:
        log.warning("Nenhum item coletado — saindo sem escrever.")
        return 1

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    raw_path = args.raw_dir / f"metadados_raw_{ts}.parquet"
    df_raw = pd.DataFrame(raw_items)
    df_raw.to_parquet(raw_path, index=False)
    log.info("Raw escrito em %s (%d linhas).", raw_path, len(df_raw))

    cleaned = [clean_item(r) for r in raw_items]
    df_clean = pd.DataFrame(cleaned)
    df_clean.to_parquet(args.interim_path, index=False)
    log.info(
        "Normalizado escrito em %s (%d linhas).", args.interim_path, len(df_clean)
    )

    with_handle = df_clean["handle"].notna().sum()
    anos = df_clean["ano"].dropna()
    tipos = df_clean["tipo"].value_counts().head(10)
    log.info(
        "Cobertura: %d/%d com handle; ano %s..%s; top tipos: %s",
        int(with_handle),
        len(df_clean),
        int(anos.min()) if not anos.empty else "-",
        int(anos.max()) if not anos.empty else "-",
        dict(tipos),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
