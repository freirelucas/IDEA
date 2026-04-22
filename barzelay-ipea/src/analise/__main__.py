"""
Fase 5 CLI: relatório consolidado.

Pipeline:
1. Concordância LLM↔humano (Cohen's kappa por dim) — só roda se houver
   validacoes_humanas.parquet.
2. Krippendorff's alpha entre anotadores — só com ≥2 anotadores.
3. Treina calibradores isotônicos (≥5 pares por dim) e aplica.
4. Análise longitudinal: média por década, década × tipo, top palavras
   por polo.
5. Saídas em data/processed/analise/.

Exemplo:
    python -m src.analise --classificacoes data/processed/classificacoes.parquet \\
        --validacoes data/processed/validacoes_humanas.parquet \\
        --metadados data/interim/metadados.parquet
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--classificacoes", type=Path, required=True)
    p.add_argument("--metadados", type=Path, required=True)
    p.add_argument("--validacoes", type=Path, default=None,
                   help="Opcional: validacoes_humanas.parquet (Fase 4).")
    p.add_argument("--out-dir", type=Path, default=Path("data/processed/analise"))
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s [%(levelname)s] %(message)s",
                        stream=sys.stderr)
    log = logging.getLogger("barzelay.analise")

    cls = pd.read_parquet(args.classificacoes)
    meta = pd.read_parquet(args.metadados)
    log.info("Classificações: %d linhas", len(cls))
    log.info("Metadados:      %d docs", len(meta))

    args.out_dir.mkdir(parents=True, exist_ok=True)

    sumario = {"n_classificacoes": int(len(cls)), "n_documentos_meta": int(len(meta))}

    if args.validacoes and args.validacoes.exists():
        valid = pd.read_parquet(args.validacoes)
        log.info("Validações humanas: %d linhas", len(valid))

        kappas = cohen_kappa_llm_vs_humano(cls, valid)
        sumario["cohen_kappa_llm_vs_humano"] = [k.__dict__ for k in kappas]
        log.info("Kappa LLM↔humano: %s", {k.dimensao: round(k.kappa, 3) for k in kappas if k.n_pares > 1})

        alphas = krippendorff_alpha_inter_annotators(valid)
        sumario["krippendorff_alpha_inter_humanos"] = alphas
        log.info("Krippendorff α: %s", {k: round(v, 3) if v == v else "NaN" for k, v in alphas.items()})

        cals = treinar_calibradores(cls, valid)
        sumario["calibradores_treinados"] = {d: c.n_pares_treino for d, c in cals.items()}
        cls = aplicar_calibradores(cls, cals)
        log.info("Calibradores treinados: %s", sumario["calibradores_treinados"])
    else:
        log.info("Sem validações humanas — pulando concordância e calibração.")
        cls["score_calibrado"] = cls["score"].astype(float)

    cls.to_parquet(args.out_dir / "classificacoes_calibradas.parquet", index=False)

    # Longitudinal
    md = media_por_decada(cls, meta)
    md.to_csv(args.out_dir / "media_por_decada.csv")
    sumario["media_por_decada_shape"] = list(md.shape)
    log.info("Médias por década:\n%s", md.round(3).to_string())

    long_dt = media_por_decada_e_tipo(cls, meta)
    long_dt.to_csv(args.out_dir / "media_por_decada_e_tipo.csv", index=False)

    palavras_por_dim: dict[str, dict[str, list[tuple[str, int]]]] = {}
    for dim in ("D1", "D2", "D3", "D4", "D5"):
        pp = top_palavras_por_polo(cls, meta, dimensao=dim)
        palavras_por_dim[dim] = {
            "design": list(pp["design"].items())[:20],
            "descritivo": list(pp["descritivo"].items())[:20],
        }
    (args.out_dir / "top_palavras_por_polo.json").write_text(
        json.dumps(palavras_por_dim, ensure_ascii=False, indent=2, default=str)
    )

    (args.out_dir / "sumario.json").write_text(
        json.dumps(sumario, ensure_ascii=False, indent=2, default=str)
    )
    log.info("Saídas em %s", args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
