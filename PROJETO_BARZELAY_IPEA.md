# Projeto Barzelay-IPEA: Classificação sistemática do repositório do IPEA quanto a marcadores de idealização

**Autor:** Lucas Freire Silva (DIEST/COGIT/IPEA)
**Data deste briefing:** abril/2026
**Status:** Briefing de contexto para implementação via Claude Code.

---

## 1. Objetivo

Classificar sistematicamente todo o repositório institucional do IPEA (https://repositorio.ipea.gov.br) quanto a cinco dimensões operacionalizadas a partir do conceito de *idealização* de Michael Barzelay, usando LLM com validação humana. A pergunta de pesquisa: **o conhecimento produzido historicamente pelo IPEA exibe marcadores de orientação para design (idealização forte) ou predominantemente marcadores descritivos (idealização fraca)?**

A hipótese de trabalho é a de uma falha generalizada na linguagem de gestores, auditores e pesquisadores em gestão pública — linguagem descritiva, backward-looking, sem integração teoria-prática e sem identidade disciplinar coerente. O repositório do IPEA é o corpus para testar essa hipótese no estado brasileiro.

**Não-objetivo:** não é para construir um RAG conversacional. Não é para recomendar leitura. É para gerar um dataset estruturado `(document_id, dimensão, score, justificativa)` que permita análise bibliométrica longitudinal e validação estatística.

---

## 2. Framework teórico (resumo operacional)

Barzelay (em *Public Management as a Design-Oriented Professional Discipline*, 2019, e textos correlatos) distingue disciplinas profissionais bem-idealizadas de disciplinas mal-idealizadas. Uma disciplina bem-idealizada integra conhecimento de pesquisa com conhecimento profissional, orienta-se para design de intervenções, articula teoria e prática, e possui identidade coerente. Gestão pública, segundo Barzelay, sofre de idealização inadequada.

**Cinco dimensões operacionais** (derivadas da leitura de ago/2025 de Barzelay, com atenção à terminologia correta — ele usa *idealization*, não *ideality*):

| Dimensão | Polo "design" | Polo "descritivo" |
|---|---|---|
| **D1. Orientação temporal** | Forward-looking: "deveria", "se implementado", "proposta" | Backward-looking: "foi observado", "houve", "resultou" |
| **D2. Posição epistemológica** | Normativa/prescritiva: recomenda cursos de ação | Descritiva/explicativa: caracteriza fenômenos |
| **D3. Relação teoria-prática** | Integrada: teoria informa desenho de intervenção concreta | Segregada: teoria como referência genérica, prática como ilustração |
| **D4. Linguagem de agência** | Atribui agência a atores institucionais identificados | Passiva/estrutural: agência difusa ou ausente |
| **D5. Foco analítico** | Sintético: articula múltiplas dimensões em design coerente | Fragmentado: analisa variáveis isoladas |

Cada dimensão será pontuada em escala Likert 1–5 pelo LLM, com justificativa textual por documento.

---

## 3. Ativo pré-existente: IpeaPub (ipeadata-lab)

Repositório: https://github.com/ipeadata-lab/IpeaPub
Autor: Rafael Dias Ghiorzi
Status: pré-alpha, desenvolvimento ativo.

### 3.1. O que o IpeaPub oferece

- Scraper da API DSpace do repositório IPEA (endpoint `/server/api/discover/browses/dateissued/items`), com extração de metadados Dublin Core.
- Crawler de PDF que resolve o padrão do DSpace (a URL direta do PDF não está no handle — é preciso raspar a página do documento e localizar o `<a href>` com `bitstreams` e `download`).
- Pipeline Docling com OCR (`easyocr`, pt+en), fragmentação de PDFs longos em blocos de 5 páginas para evitar estouro de memória, e extração de estrutura de tabelas.
- Controle operacional via SQLite com status de ingestão (`pendente`, `em processamento`, `sem_pdf`, `processado`, `erro`).

### 3.2. O que o IpeaPub **NÃO** oferece (apesar do que o README sugere)

- **Uma coleção completa indexada.** O `create_ingestion.py` termina com um filtro ad-hoc (`AUTOR = "Danilo"`, `interesse = "inteligência"`) e chama um método `db_metadata.buscar_interesse()` que **não existe** no `banco_metadados.py`. O script versionado não roda. A cobertura real da coleção Qdrant é incerta.
- **Chunker semântico integrado.** O README descreve um `SemanticChunker` com HDBSCAN como principal. No código, `create_ingestion.py` usa `HybridChunker` do `docling_core`. O `SemanticChunker` está em `ingestao/utils/semantic_chunker.py` como código órfão. E se for reutilizado, removê-lo do `.lower()` que destrói capitalização.
- **Prefixos do E5-large.** O modelo `intfloat/multilingual-e5-large` **exige** `"passage: "` em textos indexados e `"query: "` em buscas. Nem `create_ingestion.py` nem `services/embeddings.py` aplicam. Degradação silenciosa de qualidade.
- **`pyproject.toml` sincronizado.** O arquivo lista ChromaDB, LangChain, Streamlit, PyPDF2 — dependências do `src_legado`. A versão atual usa Qdrant + FastEmbed + Docling + FastAPI. `poetry install` não traz o stack real.
- **Proteção do `create_collection.py`.** Destroi a coleção sempre que executa, sem guarda.

### 3.3. Conclusão

Não é infraestrutura pronta para consumir — é um conjunto de utilitários úteis e uma pipeline parcialmente implementada. Para este projeto, aproveitamos o **scraper** e o **crawler de PDF** como bibliotecas, e **dispensamos a indexação vetorial**. Classificação sistemática não precisa de RAG.

---

## 4. Arquitetura proposta (sem Qdrant)

```
[API DSpace IPEA]
      │
      ▼
[Scraper paginado] ──► metadados.parquet (id, título, autores, ano, tipo, palavras-chave, handle)
      │
      ▼
[Crawler PDF] ──► cache/pdfs/{sha256}.pdf
      │
      ▼
[Docling OCR+estrutura] ──► textos_completos.parquet (document_id, full_text, n_pages, tables_md)
      │
      ▼
[Classificador LLM] ──► classificacoes.parquet (document_id, dimensão, score, justificativa, modelo, timestamp)
      │
      ▼
[Amostragem estratificada] ──► subset_validacao.parquet
      │
      ▼
[Interface de anotação humana] ──► validacoes_humanas.parquet
      │
      ▼
[Análise: concordância LLM↔humano, isotonic calibration, análise longitudinal]
```

Cada seta é um script idempotente que lê um parquet e escreve outro. Nada de banco vetorial. Parquet + pandas/polars em disco são suficientes para o volume esperado (estimativa: 3.000–8.000 documentos, dependendo dos tipos filtrados).

### 4.1. Por que Parquet e não SQLite

- Colunas como `full_text` (texto longo) e `justificativa` (JSON estruturado) lidam-se melhor em Parquet.
- Compressão nativa.
- Leitura em colunas (para análises agregadas, não precisa carregar o texto completo).
- Compatível com pandas/polars/duckdb sem overhead de driver.

SQLite permanece útil **apenas** para o estado operacional da ingestão (controle de documentos pendentes/processados/erro), replicando o que o IpeaPub já faz em `banco_metadados.py`.

---

## 5. Plano de implementação em fases

### Fase 0 — Setup (0.5 dia)

- Criar repositório `barzelay-ipea` (sugestão de nome).
- Ambiente Python 3.12, `uv` ou `poetry`.
- Dependências mínimas: `requests`, `beautifulsoup4`, `docling`, `pymupdf`, `pandas`, `pyarrow`, `anthropic` (ou `openai`), `tqdm`, `python-dotenv`.
- Estrutura de pastas:
  ```
  barzelay-ipea/
  ├── src/
  │   ├── scraping/
  │   ├── extracao/
  │   ├── classificacao/
  │   ├── validacao/
  │   └── analise/
  ├── data/
  │   ├── raw/
  │   ├── interim/
  │   └── processed/
  ├── cache/pdfs/
  ├── prompts/
  ├── notebooks/
  └── tests/
  ```

### Fase 1 — Scraping completo do repositório (1–2 dias)

Reaproveitar `ingestao/scraper.py` e `ingestao/utils/clean_itens.py` do IpeaPub como referência, reescrever como módulo limpo.

**Tarefas:**
1. Implementar loop paginado completo (o scraper atual do IpeaPub processa uma página por vez, mas não itera).
2. Salvar metadados brutos em `data/raw/metadados_raw_{timestamp}.parquet`.
3. Aplicar `clean_item` e salvar `data/interim/metadados.parquet`.
4. Gerar relatório de cobertura: quantos documentos por tipo (`dc.type`), por ano, com/sem handle válido.

**Critério de sucesso:** tabela com N total de documentos, N por tipo, cobertura temporal (min/max de ano).

### Fase 2 — Download e extração de texto (3–5 dias, dependendo de volume)

1. Reaproveitar `baixar_pdf_real()` de `clean_itens.py` — já resolve o padrão do DSpace.
2. Adicionar cache por SHA256 do URL **e** verificação de integridade.
3. Docling com configuração mínima para este projeto: `do_ocr=True`, `do_table_structure=True`, `generate_page_images=False`, `generate_picture_images=False` — não precisamos de imagens.
4. Salvar texto completo + metadados extraídos em `data/interim/textos.parquet`.
5. Logar falhas por documento (como IpeaPub já faz em `logs/`).

**Decisão a tomar no início da fase:** filtrar tipos documentais? Recomendação: começar por **Textos para Discussão (TDs)** — corpus mais coeso, ~2.500 documentos estimados, representa a produção analítica mais longeva (desde 1990). Deixar livros, capítulos e Nota Técnica para fase posterior.

### Fase 3 — Classificação LLM (2–4 dias + custo de API)

1. Definir o prompt de classificação (ver seção 6).
2. Para cada documento, submeter `(título, resumo, texto_completo_truncado)` ao LLM.
3. Estratégia para documentos longos (>32k tokens):
   - Opção A: truncar para primeiro N tokens (introdução + metodologia + conclusão).
   - Opção B: segmentar em janelas e agregar por média ponderada.
   - **Recomendação inicial:** Opção A (primeiro ~25k tokens), avaliar perda contra amostra validada.
4. Salvar em `data/processed/classificacoes.parquet` com schema:
   ```
   document_id: str
   dimensao: str  # D1-D5
   score: int     # 1-5
   justificativa: str
   modelo: str    # claude-opus-4-7, claude-sonnet-4-6, etc.
   prompt_version: str
   timestamp: datetime
   tokens_input: int
   tokens_output: int
   ```

**Modelo recomendado:** Claude Sonnet 4.6 para o grosso, Opus 4.7 para amostra de validação e casos difíceis. Gemini 2.5 Pro ou GPT como segunda opinião em amostra.

### Fase 4 — Validação humana (2–3 semanas com equipe)

1. Amostragem estratificada: ~300 documentos, com estratificação por década, tipo documental, e pontuação LLM (oversample de extremos e casos de discordância entre modelos).
2. Interface simples de anotação: Streamlit com uma tela por documento, mostra título/resumo/trechos identificados, e cinco sliders Likert 1–5 + campo de justificativa humana.
3. Cada documento anotado por pelo menos 2 pesquisadores, com terceiro em caso de divergência >1 ponto.
4. Calcular concordância interanotadores (Krippendorff's alpha) antes de comparar com LLM.

### Fase 5 — Calibração e análise (2 semanas)

1. **Calibração isotônica** dos scores LLM contra o padrão-ouro humano (já havíamos decidido isso em ago/2025 para o NATO-TAGGER — mesmo princípio aqui).
2. Análise de concordância: Cohen's kappa LLM↔humano por dimensão.
3. Análise longitudinal: trajetória dos scores médios por dimensão ao longo das décadas.
4. Análise por tipo documental: TDs vs. livros vs. notas técnicas.
5. Análise por autor/área temática (usando `palavras_chave`).

---

## 6. Especificação do prompt de classificação

### 6.1. Princípios de desenho

- Um prompt por dimensão (total: 5 chamadas por documento). Reduz confusão conceitual, melhora calibração.
- Saída em JSON estruturado para parsing robusto.
- Incluir 2–3 exemplos few-shot de cada polo.
- Pedir justificativa **com citação de trecho do documento**, para auditoria posterior.
- Temperatura 0.

### 6.2. Template (a adaptar por dimensão)

```
Você é um analista de textos de pesquisa em políticas públicas. Sua tarefa é
avaliar um documento do IPEA quanto a UMA dimensão específica, baseando-se
no framework de Michael Barzelay sobre idealização em disciplinas
profissionais.

DIMENSÃO: {nome_dimensao}
DEFINIÇÃO: {definicao_dimensao}

POLO 1 (score 1-2): {descricao_polo_descritivo}
POLO 5 (score 4-5): {descricao_polo_design}
Score 3: caso ambíguo, misto, ou não aplicável claramente.

EXEMPLOS:
[exemplo score 1] ...
[exemplo score 3] ...
[exemplo score 5] ...

DOCUMENTO A AVALIAR:
Título: {titulo}
Autores: {autores}
Ano: {ano}
Tipo: {tipo}

Texto (truncado em {n_tokens} tokens):
{texto}

INSTRUÇÕES:
1. Leia com atenção à linguagem usada, não ao conteúdo substantivo.
2. Identifique 2-3 trechos (3-15 palavras cada) que exemplifiquem a posição do
   documento nesta dimensão.
3. Atribua um score 1-5.
4. Justifique em 2-3 frases, referenciando os trechos.

Responda APENAS em JSON válido, sem preâmbulo:
{
  "dimensao": "{nome_dimensao}",
  "score": <int>,
  "trechos_evidencia": ["...", "...", "..."],
  "justificativa": "..."
}
```

### 6.3. Marcadores linguísticos por dimensão (heurística inicial, refinar empiricamente)

**D1 — Orientação temporal**
- Design: "deveria", "se fosse implementado", "recomenda-se", "proposta", "desenho de", "a ser construído"
- Descritivo: "foi observado", "houve", "resultou em", "verificou-se", "constatou-se"

**D2 — Posição epistemológica**
- Design: "recomendamos", "a política deve", "é preciso", "sugerimos", "intervenção adequada"
- Descritivo: "o estudo mostra", "os dados indicam", "a análise revela", "correlaciona-se"

**D3 — Relação teoria-prática**
- Design: teoria citada **e** aplicada a desenho concreto (ex. usa Beer/VSM para propor arquitetura específica de coordenação)
- Descritivo: teoria citada como moldura genérica ou revisão de literatura, sem aplicação a desenho

**D4 — Linguagem de agência**
- Design: "o Ministério X deve", "os gestores do programa Y podem", "a CGU teria condições de"
- Descritivo: "observa-se", "nota-se", "tende-se a", passiva difusa

**D5 — Foco analítico**
- Design: articula múltiplas dimensões (financiamento + capacidades + governança + arranjo federativo) em proposta coerente
- Descritivo: analisa uma variável por vez, sem síntese

---

## 7. Decisões técnicas já tomadas

| Decisão | Escolha | Justificativa |
|---|---|---|
| Indexação vetorial | Dispensar | Projeto é classificação, não recuperação |
| Formato de persistência | Parquet | Colunas longas, compressão, análises agregadas |
| Controle de ingestão | SQLite (reaproveitar `MetadataDB`) | Atomicidade, status de processamento |
| Extrator de PDF | Docling | IpeaPub já validou; alternativa: PyMuPDF4LLM se Docling for lento |
| Modelo LLM principal | Claude Sonnet 4.6 (varredura) + Opus 4.7 (validação) | Custo/qualidade |
| Segunda opinião | Gemini 2.5 Pro ou GPT em amostra | Detectar viés de modelo único |
| Calibração | Isotonic regression contra validação humana | Método já decidido para NATO-TAGGER |
| Prompt por dimensão | Um prompt por D1-D5 | Reduz confusão conceitual |
| Saída LLM | JSON estruturado, temperatura 0 | Parsing determinístico |

---

## 8. Riscos e armadilhas identificados

1. **Truncamento de documentos longos.** TDs de 80+ páginas podem ter a seção mais prescritiva no final. A opção A (truncar no início) pode subestimar design. Validar empiricamente antes de escalar.

2. **Viés por década.** Textos mais antigos tendem a usar linguagem mais descritiva por convenção estilística da época, não por falta de idealização. A análise longitudinal precisa de controle por estilo de época.

3. **Heterogeneidade por tipo documental.** TDs são mais analíticos, Nota Técnica é mais prescritiva por design. Análise cruzada tipo×dimensão é obrigatória.

4. **Viés do próprio LLM.** LLMs atuais foram treinados com literatura recente e podem ter viés que associa "design" com termos contemporâneos de política pública. Validação humana blindada (anotador não vê score LLM) é essencial.

5. **Circularidade Barzelay.** O próprio Barzelay publica em inglês e o corpus é majoritariamente em português. Os marcadores linguísticos precisam ser adaptados — não é tradução literal. Validar com amostra bilíngue.

6. **Direitos autorais e uso institucional.** Documentos do IPEA são públicos (Lei 12.527/2011 + política institucional), mas enviar texto completo a APIs externas de LLM precisa ser comunicado à DIEST/COGIT. Considerar LLM local para fases posteriores (Llama 3.3 70B ou Qwen em infra interna) se houver restrição.

---

## 9. Interface com o projeto Barzelay de ago/2025 (Complexity Science)

Este projeto é o **braço empírico** da crítica teórica desenvolvida em ago/2025 sobre idealização inadequada na ciência da complexidade. A estrutura metodológica é a mesma:

- Mesmas cinco dimensões.
- Mesma estratégia de LLM + validação humana.
- Mesmo método de calibração.

A diferença é o corpus: lá, literatura internacional de complexity science; aqui, repositório IPEA. Rodar os dois com o mesmo protocolo permite comparação direta **"o Estado brasileiro exibe idealização comparável à da ciência da complexidade?"** — esta é uma pergunta genuinamente original e publicável.

---

## 10. Próximos passos concretos para Claude Code

1. **Criar o esqueleto do repositório** conforme seção 4 e 5 (Fase 0).
2. **Implementar o scraper paginado completo** a partir do `scraper.py` do IpeaPub, com loop e relatório de cobertura (Fase 1).
3. **Antes de baixar PDFs**, gerar um relatório em notebook com distribuição de documentos por ano, tipo, palavras-chave. Lucas decide o corpus-alvo a partir desse relatório.
4. **Implementar o pipeline de extração** com Docling, partindo do corpus-alvo decidido.
5. **Implementar e testar o prompt D1** (Orientação temporal) em amostra de 20 documentos manualmente escolhidos (10 claramente design, 10 claramente descritivos). Calibrar antes de escalar.
6. Depois disso, rodar D2–D5.

---

## 11. Arquivos de referência no IpeaPub (a reaproveitar como bibliotecas)

| Arquivo IpeaPub | Função | Reaproveitamento |
|---|---|---|
| `ingestao/scraper.py` | Scraper DSpace | Reescrever como módulo com loop paginado; manter lógica de extração de campos |
| `ingestao/utils/clean_itens.py` | Normalização + `baixar_pdf_real()` | Reusar praticamente inteiro; destacar `baixar_pdf_real` como função pura |
| `ingestao/db/banco_metadados.py` | SQLite de controle | Reusar classe `MetadataDB` inteira |
| `ingestao/create_ingestion.py` | Pipeline Docling | Usar apenas `ler_pdf_com_docling()` e `split_pdf_em_blocos()`; descartar parte de embeddings |
| `ingestao/utils/semantic_chunker.py` | Chunker HDBSCAN | **Não usar** (órfão + `.lower()` destrutivo) |

---

## 12. Referências bibliográficas essenciais

- Barzelay, M. (2019). *Public Management as a Design-Oriented Professional Discipline*. Edward Elgar.
- Barzelay, M. (2007). Learning from second-hand experience: methodology for extrapolation-oriented case research. *Governance*, 20(3).
- Simon, H. A. (1996). *The Sciences of the Artificial* (3rd ed.). MIT Press.
- Conant, R. C., & Ashby, W. R. (1970). Every good regulator of a system must be a model of that system. *International Journal of Systems Science*, 1(2).

(Expandir conforme a fundamentação de agosto/2025 com citações completas — a conversa "Complexity Science and Cybernetic Idealness" tem o material já fundamentado.)

---

**Fim do briefing.**
