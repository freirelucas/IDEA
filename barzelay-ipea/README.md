# barzelay-ipea

Classificação sistemática do repositório institucional do IPEA
(https://repositorio.ipea.gov.br) quanto a cinco dimensões de *idealização*
derivadas de Michael Barzelay, via LLM com validação humana.

Briefing completo: [`../PROJETO_BARZELAY_IPEA.md`](../PROJETO_BARZELAY_IPEA.md).

## Status

- **Fase 0 (Setup):** ✅ estrutura de pastas, dependências, config.
- **Fase 1 (Scraping):** ✅ scraper paginado do DSpace + normalização +
  filtro por diretoria + amostra estratificada + notebooks de cobertura.
- **Fase 2 (Extração):** ✅ download via `baixar_pdf` (cache SHA256) +
  Docling (`do_ocr=True, do_table_structure=True`); CLI com flush
  incremental e skip-existing.
- **Fase 3 (Classificação):** ✅ prompts D1–D5 versionados + classificador
  LLM Anthropic (LLMCaller injetável p/ testes), schema Pydantic, parser
  tolerante a fenced JSON, append-only.
- **Fase 4 (Validação humana):** ✅ Streamlit com anotação cega por padrão
  + amostragem estratificada (década × tipo × oversample extremos).
- **Fase 5 (Análise):** ✅ Cohen's kappa + Krippendorff's α + IsotonicRegression
  por dimensão + médias por década/tipo + top palavras por polo.

**93/93 testes passando.** Smoke real das Fases 2–4 (Docling pesado, custo
LLM, anotação humana) fica para execução do usuário; Fase 5 já foi rodada
end-to-end com dados sintéticos.

## Uso

```bash
# 1. Instalar (Python 3.12)
uv sync --extra dev

# 2. Rodar testes
uv run pytest

# 3. Smoke test do scraper (2 páginas ≈ 40 documentos)
uv run python -m src.scraping --limit 2

# 4. Rodar o scraper completo (~6 min, ~900 páginas, ~18k documentos)
uv run python -m src.scraping

# 5. Notebook de cobertura do corpus completo
uv run jupyter lab notebooks/01_cobertura.ipynb

# 6. Notebook de cobertura do subset DIEST (grava metadados_diest.parquet)
uv run jupyter lab notebooks/02_cobertura_diest.ipynb

# 7. Amostra 10% estratificada por década (~1.758 docs)
uv run jupyter lab notebooks/03_amostra_10pct.ipynb

# 8. Fase 2 — download + Docling sobre a amostra (smoke 3 docs)
uv run python -m src.extracao --input data/interim/metadados_amostra10.parquet --limit 3

# 9. Fase 3 — classificação LLM (precisa ANTHROPIC_API_KEY)
export ANTHROPIC_API_KEY=...
uv run python -m src.classificacao --input data/interim/textos.parquet --limit 3

# 10. Fase 4 — Streamlit de validação humana
uv run streamlit run src/validacao/app.py -- \
    --classificacoes data/processed/classificacoes.parquet \
    --textos data/interim/textos.parquet \
    --metadados data/interim/metadados.parquet \
    --amostra data/processed/amostra_validacao.parquet

# 11. Fase 5 — análise consolidada
uv run python -m src.analise \
    --classificacoes data/processed/classificacoes.parquet \
    --metadados data/interim/metadados.parquet \
    --validacoes data/processed/validacoes_humanas.parquet

# 12. Gerar site estático em docs/ para GitHub Pages
uv run python -m src.analise.build_site
```

## GitHub Pages

A pasta `docs/` contém o site estático (HTML + PNGs). Para publicar:

1. **Settings → Pages** no repositório GitHub.
2. **Source:** `Deploy from a branch`.
3. **Branch:** `main` · **Folder:** `/docs`.
4. Commit em `main` da pasta `docs/` publica automaticamente.

O site exibe KPIs do corpus, cobertura da extração, gráficos de classificação
(reais ou sintéticos), médias por década e instruções para regerar com
dados reais. Regenerar: `uv run python -m src.analise.build_site`.

Saídas:
- `data/raw/metadados_raw_{timestamp}.parquet` — campos brutos da API DSpace.
- `data/interim/metadados.parquet` — campos normalizados (`clean_item`).
- `data/interim/metadados_diest.parquet` — subset DIEST (após rodar o notebook 02).

## Filtro por diretoria

A API DSpace marca a diretoria em dois caminhos, ambos expostos no endpoint
`browses/dateissued/items`:

- `dc.contributor.other` → coluna `contributor_other` (texto livre; usado em
  todo o histórico).
- `relation.isOrgUnitOfPublication` → coluna `orgunit_uuid` (linkagem
  relacional; apenas em itens recentes, tipicamente pós-2020).

`src.scraping.filter_by_diretoria(df, "DIEST")` combina os dois sinais.
UUIDs conhecidos em `src.scraping.diretoria.ORGUNIT_UUID`.

## Layout

```
src/
├── scraping/      Fase 1 — DSpace API + normalização
├── extracao/      Fase 2 — PDF + Docling
├── classificacao/ Fase 3 — prompts + LLM
├── validacao/     Fase 4 — anotação humana
└── analise/       Fase 5 — calibração + longitudinal
data/
├── raw/           saídas não processadas
├── interim/       metadados limpos, textos extraídos
└── processed/     classificações, validações
cache/pdfs/        PDFs baixados (sha256)
prompts/           templates versionados por dimensão
notebooks/         análises exploratórias
tests/             pytest
```

## Origem das funções portadas

O scraper e a normalização são ports limpos de `ipeadata-lab/IpeaPub`
(autor: Rafael Dias Ghiorzi), descartando acoplamentos que não se aplicam
a este projeto (Qdrant, SQLite de ingestão, chunker semântico órfão). Ver
`PROJETO_BARZELAY_IPEA.md` §3 e §11 para o detalhamento.
