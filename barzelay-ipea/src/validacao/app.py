"""
Streamlit de validação humana — Fase 4.

Cada tela mostra um documento e os 5 scores LLM com trechos de evidência;
anotador atribui score humano 1-5 + justificativa para cada dimensão.
Cada anotador trabalha com seu `anotador_id` (slug em sidebar) — `ja_anotado`
suprime documentos já validados.

Briefing §5 Fase 4:
- Cada doc anotado por ≥2 pesquisadores; terceiro em divergência >1.
- Anotador cego (não vê score LLM antes de atribuir o seu).
  → Implementado via toggle `mostrar_llm` em sidebar (default: oculto).

Como rodar:
    uv run streamlit run src/validacao/app.py -- \\
        --classificacoes data/processed/classificacoes.parquet \\
        --textos data/interim/textos.parquet \\
        --metadados data/interim/metadados.parquet \\
        --amostra data/processed/amostra_validacao.parquet \\
        --output data/processed/validacoes_humanas.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from src.validacao.persist import append_validacao, carregar_validacoes, ja_anotado

DIMENSOES = ("D1", "D2", "D3", "D4", "D5")
NOMES = {
    "D1": "D1 — Orientação temporal",
    "D2": "D2 — Posição epistemológica",
    "D3": "D3 — Relação teoria-prática",
    "D4": "D4 — Linguagem de agência",
    "D5": "D5 — Foco analítico",
}


def parse_streamlit_args() -> argparse.Namespace:
    """Lê argumentos passados após `--` na CLI do Streamlit."""
    p = argparse.ArgumentParser()
    p.add_argument("--classificacoes", type=Path, required=True)
    p.add_argument("--textos", type=Path, required=True)
    p.add_argument("--metadados", type=Path, required=True)
    p.add_argument("--amostra", type=Path, required=True)
    p.add_argument(
        "--output", type=Path, default=Path("data/processed/validacoes_humanas.parquet")
    )
    argv = sys.argv[1:]
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    return p.parse_args(argv)


@st.cache_data
def load_data(
    cls_path: Path, txt_path: Path, meta_path: Path, amostra_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cls = pd.read_parquet(cls_path)
    txt = pd.read_parquet(txt_path)
    meta = pd.read_parquet(meta_path)
    amostra = pd.read_parquet(amostra_path)
    return cls, txt, meta, amostra


def render() -> None:
    args = parse_streamlit_args()
    st.set_page_config(page_title="Validação Barzelay-IPEA", layout="wide")
    st.title("Validação humana — Barzelay-IPEA")

    cls, txt, meta, amostra = load_data(
        args.classificacoes, args.textos, args.metadados, args.amostra
    )

    with st.sidebar:
        anotador = st.text_input("Anotador (seu slug)", value="").strip()
        mostrar_llm = st.checkbox(
            "Mostrar scores LLM antes de anotar (NÃO recomendado — viés)",
            value=False,
        )
        st.caption("Anotação cega é o default (briefing §8.4).")

    if not anotador:
        st.warning("Informe um anotador na barra lateral para começar.")
        return

    docs_amostra = amostra["id"].tolist()
    val = carregar_validacoes(args.output)
    pendentes = []
    for doc_id in docs_amostra:
        for dim in DIMENSOES:
            if not ja_anotado(args.output, doc_id, dim, anotador):
                pendentes.append((doc_id, dim))

    st.metric("Validações pendentes", f"{len(pendentes)} / {len(docs_amostra) * 5}")

    if not pendentes:
        st.success("Nada pendente para você.")
        return

    doc_id, dim = pendentes[0]
    meta_row = meta[meta["id"] == doc_id].iloc[0]
    txt_row = txt[txt["id"] == doc_id]
    cls_row = cls[(cls["document_id"] == doc_id) & (cls["dimensao"] == dim)]

    st.subheader(meta_row["titulo"])
    st.caption(
        f"{meta_row.get('autores', '(sem autores)')} · {meta_row.get('ano', '?')} · {meta_row.get('tipo', '?')}"
    )

    with st.expander("Resumo"):
        st.write(meta_row.get("resumo", "(sem resumo)"))

    if not txt_row.empty:
        with st.expander(f"Texto extraído ({txt_row.iloc[0]['n_pages']} páginas)"):
            st.text(txt_row.iloc[0]["full_text"][:8000])

    st.markdown(f"### Avaliando: **{NOMES[dim]}**")

    if mostrar_llm and not cls_row.empty:
        with st.warning("Scores LLM (mostrados a pedido):"):
            row = cls_row.iloc[0]
            st.write(f"Score LLM: **{row['score']}**")
            st.write(f"Justificativa LLM: {row['justificativa']}")
            st.write("Trechos:", row["trechos_evidencia"])
    elif not cls_row.empty:
        st.info("Trechos identificados pelo LLM (sem mostrar score):")
        for t in cls_row.iloc[0]["trechos_evidencia"]:
            st.write(f"- {t}")

    score = st.slider(f"Seu score para {dim}", 1, 5, 3)
    justificativa = st.text_area(
        "Justificativa (2-3 frases, com referência a trechos)",
        height=150,
    )

    if st.button("Salvar e próximo"):
        if not justificativa.strip():
            st.error("Justificativa obrigatória.")
        else:
            append_validacao(
                args.output,
                document_id=doc_id,
                dimensao=dim,
                score_humano=score,
                justificativa_humana=justificativa.strip(),
                anotador_id=anotador,
            )
            st.cache_data.clear()
            st.rerun()


if __name__ == "__main__":
    render()
