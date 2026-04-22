"""
Fase 3 CLI: classifica documentos em D1–D5 via LLM.

Exemplos:
    # smoke (1 doc, todas as dimensões)
    python -m src.classificacao --input data/interim/textos.parquet --limit 1

    # produção
    python -m src.classificacao --input data/interim/textos.parquet \\
        --output data/processed/classificacoes.parquet \\
        --model claude-sonnet-4-6
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.classificacao.classifier import classificar, make_anthropic_caller
from src.classificacao.prompts import DIMENSOES, load_prompt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="textos.parquet (Fase 2)")
    p.add_argument(
        "--metadados",
        type=Path,
        default=Path("data/interim/metadados.parquet"),
        help="metadados.parquet (Fase 1) para join de titulo/autores/ano/tipo.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/classificacoes.parquet"),
        help="parquet de classificações (append-only).",
    )
    p.add_argument(
        "--model",
        default=os.getenv("MODEL_CLASSIFIER", "claude-sonnet-4-6"),
        help="Modelo Anthropic. Default: env MODEL_CLASSIFIER ou claude-sonnet-4-6.",
    )
    p.add_argument(
        "--dimensoes",
        nargs="+",
        default=list(DIMENSOES),
        choices=list(DIMENSOES),
        help="Quais dimensões rodar (default: todas).",
    )
    p.add_argument("--limit", type=int, default=None, help="Limita N documentos (smoke).")
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Pula (doc, dimensão, prompt_version) já presentes no output.",
    )
    p.add_argument("--sleep", type=float, default=0.5, help="Sleep entre chamadas LLM.")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )
    log = logging.getLogger("barzelay.classificacao")

    if not os.getenv("ANTHROPIC_API_KEY"):
        log.error("ANTHROPIC_API_KEY ausente. Defina antes de rodar.")
        return 2

    textos = pd.read_parquet(args.input)
    textos = textos[(textos["error"].isna()) | (textos["error"] == "")]
    log.info("Textos válidos: %d", len(textos))
    metadados = pd.read_parquet(args.metadados)[["id", "titulo", "autores", "ano", "tipo"]]
    df = textos.merge(metadados, on="id", how="left")

    if args.limit:
        df = df.head(args.limit)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    existing = _load_existing(args.output)
    templates = {d: load_prompt(d) for d in args.dimensoes}
    caller = make_anthropic_caller(args.model)

    rows: list[dict] = []
    pbar = tqdm(total=len(df) * len(args.dimensoes), desc="classificando")
    for _, row in df.iterrows():
        doc_id = row["id"]
        for dim in args.dimensoes:
            tpl = templates[dim]
            key = (doc_id, dim, tpl.versao)
            if args.skip_existing and key in existing:
                pbar.update(1)
                continue
            try:
                res = classificar(
                    doc_id=doc_id,
                    titulo=row.get("titulo") or "",
                    autores=row.get("autores") or "",
                    ano=row.get("ano"),
                    tipo=row.get("tipo") or "",
                    texto=row["full_text"],
                    dimensao=dim,
                    caller=caller,
                    template=tpl,
                )
                rows.append({
                    "document_id": doc_id,
                    "dimensao": dim,
                    "score": res.classificacao.score,
                    "trechos_evidencia": res.classificacao.trechos_evidencia,
                    "justificativa": res.classificacao.justificativa,
                    "modelo": args.model,
                    "prompt_version": tpl.versao,
                    "tokens_input": res.stats.tokens_input,
                    "tokens_output": res.stats.tokens_output,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
            except Exception as e:
                log.exception("Falha em %s/%s: %s", doc_id, dim, e)
                rows.append({
                    "document_id": doc_id,
                    "dimensao": dim,
                    "score": None,
                    "trechos_evidencia": [],
                    "justificativa": f"ERRO: {e}",
                    "modelo": args.model,
                    "prompt_version": tpl.versao,
                    "tokens_input": 0,
                    "tokens_output": 0,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
            pbar.update(1)
            time.sleep(args.sleep)

            # flush a cada 50 linhas (~10 docs)
            if len(rows) % 50 == 0:
                _flush(rows, args.output)
    pbar.close()
    _flush(rows, args.output)
    log.info("Concluído: %d classificações em %s", len(rows), args.output)
    return 0


def _load_existing(path: Path) -> set[tuple[str, str, str]]:
    if not path.exists():
        return set()
    df = pd.read_parquet(path)
    return set(zip(df["document_id"], df["dimensao"], df["prompt_version"]))


def _flush(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    if path.exists():
        df = pd.concat([pd.read_parquet(path), df], ignore_index=True)
    df.to_parquet(path, index=False)


if __name__ == "__main__":
    raise SystemExit(main())
