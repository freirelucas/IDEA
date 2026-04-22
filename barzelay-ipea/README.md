# barzelay-ipea

Classificação sistemática do repositório institucional do IPEA
(https://repositorio.ipea.gov.br) quanto a cinco dimensões de *idealização*
derivadas de Michael Barzelay, via LLM com validação humana.

Briefing completo: [`../PROJETO_BARZELAY_IPEA.md`](../PROJETO_BARZELAY_IPEA.md).

## Status

- **Fase 0 (Setup):** ✅ estrutura de pastas, dependências, config.
- **Fase 1 (Scraping):** ✅ scraper paginado do DSpace + normalização +
  persistência em Parquet + notebook de cobertura.
- **Fase 2 (Extração):** pendente — download de PDFs + Docling.
- **Fase 3 (Classificação):** pendente — prompts D1–D5 + LLM.
- **Fase 4 (Validação humana):** pendente — Streamlit de anotação.
- **Fase 5 (Análise):** pendente — calibração isotônica + longitudinal.

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
```

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
