"""
Build estático do site em `docs/` para GitHub Pages.

Lê:
- `data/interim/metadados.parquet` (Fase 1 — obrigatório)
- `data/interim/textos.parquet` ou `textos_amostra10.parquet` (Fase 2 — opcional)
- `data/processed/classificacoes.parquet` preferido, senão
  `data/processed/classificacoes_synth.parquet` (Fase 3 — um dos dois)
- `data/processed/validacoes_humanas.parquet` (Fase 4 — opcional)

Escreve em `docs/`:
- `index.html` — dashboard single-page
- `style.css` — estilo mínimo
- `figures/*.png` — renderizadas via src.analise.viz
- `.nojekyll` — desliga Jekyll p/ o GitHub Pages servir arquivos as-is

Uso:
    uv run python -m src.analise.build_site
"""
from __future__ import annotations

import argparse
import html as html_mod
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.analise.longitudinal import (
    media_por_decada,
    media_por_decada_e_tipo,
    top_palavras_por_polo,
)
from src.analise.viz import (
    plot_distribuicao_scores,
    plot_heatmap_decada_dimensao,
    plot_media_por_decada,
    plot_media_por_tipo,
    plot_top_palavras_polo,
)

log = logging.getLogger(__name__)

# `ROOT` aponta para a raiz do projeto (barzelay-ipea/); `REPO_ROOT` para a
# raiz do repositório git (um nível acima), onde o GitHub Pages procura /docs.
ROOT = Path(__file__).resolve().parent.parent.parent
REPO_ROOT = ROOT.parent
DOCS = REPO_ROOT / "docs"
FIG_DIR = DOCS / "figures"


STYLE_CSS = """\
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  font-size: 16px;
  line-height: 1.55;
  color: #1a1a1a;
  background: #fafafa;
}
.wrap { max-width: 1040px; margin: 0 auto; padding: 24px; }
header { border-bottom: 1px solid #ddd; padding-bottom: 16px; margin-bottom: 24px; }
h1 { font-size: 28px; margin: 0 0 6px; letter-spacing: -0.02em; }
h2 { font-size: 22px; margin-top: 36px; border-bottom: 1px solid #eee; padding-bottom: 6px; }
h3 { font-size: 17px; margin-top: 22px; }
.subtitle { color: #666; margin: 0; }
.meta { color: #888; font-size: 13px; margin-top: 8px; }
.banner {
  background: #fff5d7; border-left: 4px solid #e6a700;
  padding: 12px 16px; margin: 20px 0;
  font-size: 14px; border-radius: 4px;
}
.banner.ok { background: #e7f4e4; border-left-color: #38a169; }
.banner.err { background: #fde5e5; border-left-color: #c53030; }
code { background: #f1f1f1; padding: 2px 5px; border-radius: 3px; font-size: 0.9em; }
pre {
  background: #f6f8fa; padding: 12px 16px;
  overflow-x: auto; font-size: 13px; border-radius: 4px;
}
table { border-collapse: collapse; margin: 16px 0; width: 100%; font-size: 14px; }
th, td { border: 1px solid #ddd; padding: 6px 10px; text-align: left; }
th { background: #f0f0f0; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
.kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px; margin: 20px 0; }
.kpi {
  background: white; border: 1px solid #e5e5e5; border-radius: 6px;
  padding: 14px 16px;
}
.kpi .label { color: #666; font-size: 13px; }
.kpi .value { font-size: 22px; font-weight: 600; margin-top: 4px; font-variant-numeric: tabular-nums; }
figure { margin: 20px 0; }
figure img { max-width: 100%; height: auto; border: 1px solid #eee; border-radius: 4px; }
figcaption { color: #666; font-size: 13px; margin-top: 6px; }
.pipeline { list-style: none; padding: 0; }
.pipeline li { padding: 8px 0; border-bottom: 1px dashed #e5e5e5; }
.pipeline .tag { display: inline-block; min-width: 72px; font-weight: 600; }
.pipeline .ok { color: #38a169; }
.pipeline .todo { color: #b58500; }
.dims td:first-child { font-weight: 600; white-space: nowrap; }
footer { color: #666; font-size: 13px; margin-top: 48px; padding-top: 16px; border-top: 1px solid #eee; }
a { color: #0366d6; }
"""


def _fmt_int(n: int | float) -> str:
    return f"{int(n):,}".replace(",", ".")


def _fmt_pct(n: float) -> str:
    return f"{n:.1%}"


def _kpi(label: str, value: str) -> str:
    return f'<div class="kpi"><div class="label">{html_mod.escape(label)}</div><div class="value">{value}</div></div>'


def build_site(
    meta_path: Path = ROOT / "data/interim/metadados.parquet",
    textos_paths: tuple[Path, ...] = (
        ROOT / "data/interim/textos_amostra10.parquet",
        ROOT / "data/interim/textos.parquet",
    ),
    cls_real: Path = ROOT / "data/processed/classificacoes.parquet",
    cls_demo: Path = ROOT / "data/processed/classificacoes_demo.parquet",
    cls_synth: Path = ROOT / "data/processed/classificacoes_synth.parquet",
    val_path: Path = ROOT / "data/processed/validacoes_humanas.parquet",
) -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS / ".nojekyll").write_text("")
    (DOCS / "style.css").write_text(STYLE_CSS)

    meta = pd.read_parquet(meta_path)
    textos = None
    textos_source = None
    for p in textos_paths:
        if p.exists():
            textos = pd.read_parquet(p)
            textos_source = p.name
            break

    # classificações: real > demo (mock features) > synth (só metadados)
    if cls_real.exists():
        cls = pd.read_parquet(cls_real)
        cls_source = "real"
    elif cls_demo.exists():
        cls = pd.read_parquet(cls_demo)
        cls_source = "demo"
    elif cls_synth.exists():
        cls = pd.read_parquet(cls_synth)
        cls_source = "synth"
    else:
        raise FileNotFoundError(
            "Sem parquet de classificações. Rode Fase 3 ou gere sintético."
        )
    cls_is_synth = cls_source == "synth"

    # gerar figuras
    log.info("Renderizando figuras para %s", FIG_DIR)
    if "score_calibrado" not in cls.columns:
        cls = cls.copy()
        cls["score_calibrado"] = cls["score"].astype(float)
    medias = media_por_decada(cls, meta)
    long_dt = media_por_decada_e_tipo(cls, meta)

    plot_media_por_decada(medias, save_to=FIG_DIR / "01_media_por_decada.png")
    plot_heatmap_decada_dimensao(medias, save_to=FIG_DIR / "02_heatmap_decada_dim.png")
    plot_distribuicao_scores(cls, save_to=FIG_DIR / "03_distribuicao_scores.png")
    for dim in ("D1", "D2", "D3", "D4", "D5"):
        plot_media_por_tipo(long_dt, dimensao=dim, save_to=FIG_DIR / f"04_tipo_{dim}.png")
    tops = {d: top_palavras_por_polo(cls, meta, dimensao=d) for d in ("D1", "D2", "D3", "D4", "D5")}
    plot_top_palavras_polo(tops, top_n=12, save_to=FIG_DIR / "05_palavras_polo.png")

    # HTML
    html_text = _render_html(meta=meta, textos=textos, textos_source=textos_source,
                             cls=cls, cls_source=cls_source, medias=medias)
    (DOCS / "index.html").write_text(html_text, encoding="utf-8")
    log.info("Site gerado: %s", DOCS / "index.html")


def _render_html(*, meta, textos, textos_source, cls, cls_source, medias) -> str:
    cls_is_synth = cls_source == "synth"
    cls_is_demo = cls_source == "demo"
    n_docs = len(meta)
    anos = meta["ano"].dropna().astype(int)
    ano_min, ano_max = int(anos.min()), int(anos.max())
    com_handle = int(meta["handle"].notna().sum())

    if textos is not None:
        textos_ok = int((textos["full_text"] != "").sum())
        textos_total = int(len(textos))
        chars_total = int(textos["full_text"].str.len().sum())
    else:
        textos_ok = textos_total = chars_total = 0

    n_cls = len(cls)
    n_cls_docs = cls["document_id"].nunique()

    if cls_is_synth:
        banner_cls = (
            '<div class="banner">⚠ <strong>Classificações sintéticas (placeholder).</strong> '
            'Geradas por <code>src.classificacao.synth</code> apenas a partir de metadados '
            '(ano, tipo) — não leem o texto. Rode '
            '<code>python -m src.classificacao --mock</code> para ver o demo que usa '
            'features textuais, ou configure <code>ANTHROPIC_API_KEY</code> para Fase 3 real.</div>'
        )
    elif cls_is_demo:
        banner_cls = (
            '<div class="banner">🧪 <strong>Demo via mock_caller (features textuais).</strong> '
            'Scores derivados de contagem de marcadores linguísticos no texto '
            '(modais deônticos, atores institucionais, etc), <em>não</em> de LLM. '
            'Substitua por Fase 3 real rodando <code>python -m src.classificacao</code> '
            'com <code>ANTHROPIC_API_KEY</code> — o site re-renderiza automaticamente.</div>'
        )
    else:
        banner_cls = '<div class="banner ok">✓ Classificações reais de LLM.</div>'

    tipo_counts = meta["tipo"].fillna("(sem tipo)").value_counts().head(10)
    tipo_rows = "\n".join(
        f"<tr><td>{html_mod.escape(str(t))}</td><td class='num'>{_fmt_int(n)}</td></tr>"
        for t, n in tipo_counts.items()
    )

    dec_rows = "\n".join(
        f"<tr><td>{int(dec)}</td><td class='num'>{_fmt_int(n)}</td></tr>"
        for dec, n in (anos // 10 * 10).value_counts().sort_index().items()
    )

    medias_rows = ""
    for dec, row in medias.round(2).iterrows():
        cells = "".join(
            f"<td class='num'>{row[d]:.2f}</td>" if pd.notna(row[d]) else "<td>—</td>"
            for d in ("D1", "D2", "D3", "D4", "D5")
        )
        medias_rows += f"<tr><td>{int(dec)}</td>{cells}</tr>\n"

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    kpi_grid = (
        _kpi("Documentos", _fmt_int(n_docs))
        + _kpi("Anos (min–max)", f"{ano_min}–{ano_max}")
        + _kpi("Com handle", f"{_fmt_pct(com_handle / n_docs)}")
        + _kpi("Textos extraídos", f"{_fmt_int(textos_ok)} / {_fmt_int(textos_total)}" if textos_total else "—")
        + _kpi("Chars extraídos", _fmt_int(chars_total) if chars_total else "—")
        + _kpi("Classificações", f"{_fmt_int(n_cls)} (≈ {_fmt_int(n_cls_docs)} docs × 5 dim)")
    )

    dim_table = """
    <table class="dims">
      <thead><tr><th>Dim</th><th>Polo 1–2 (descritivo)</th><th>Polo 4–5 (design)</th></tr></thead>
      <tbody>
        <tr><td>D1 Temporal</td><td>Backward-looking ("foi observado")</td><td>Forward-looking ("deveria")</td></tr>
        <tr><td>D2 Epistemológica</td><td>Descritiva ("os dados mostram")</td><td>Normativa ("recomenda-se")</td></tr>
        <tr><td>D3 Teoria-prática</td><td>Segregada (revisão isolada)</td><td>Integrada (teoria → desenho)</td></tr>
        <tr><td>D4 Agência</td><td>Passiva/estrutural</td><td>Atores institucionais nomeados</td></tr>
        <tr><td>D5 Foco analítico</td><td>Fragmentado (variável a variável)</td><td>Sintético (design coerente)</td></tr>
      </tbody>
    </table>
    """

    pipeline_html = f"""
    <ul class="pipeline">
      <li><span class="tag ok">Fase 1 ✓</span> Scraping paginado do DSpace IPEA — {_fmt_int(n_docs)} docs.</li>
      <li><span class="tag {'ok' if textos_total else 'todo'}">Fase 2 {'✓' if textos_total else '…'}</span>
        Download + extração PyMuPDF — {_fmt_int(textos_ok)}/{_fmt_int(textos_total)} docs {('(' + textos_source + ')') if textos_source else 'pendente'}.</li>
      <li><span class="tag {'ok' if cls_source == 'real' else 'todo'}">Fase 3 {'✓ real' if cls_source == 'real' else ('🧪 demo (mock)' if cls_is_demo else 'sintético')}</span>
        Classificação D1–D5 via LLM — precisa <code>ANTHROPIC_API_KEY</code>. Demo via <code>--mock</code> usa features textuais (custo zero).</li>
      <li><span class="tag todo">Fase 4 …</span> Validação humana via Streamlit — pesquisadores anotam ~300 docs.</li>
      <li><span class="tag todo">Fase 5 …</span> Calibração isotônica + Cohen's κ + análise longitudinal — roda quando Fase 4 existir.</li>
    </ul>
    """

    return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Barzelay-IPEA — Classificação de marcadores de idealização</title>
<link rel="stylesheet" href="style.css">
</head>
<body>
<div class="wrap">
<header>
  <h1>Barzelay-IPEA</h1>
  <p class="subtitle">Classificação sistemática do repositório do IPEA quanto a marcadores de <em>idealização</em> (Barzelay 2019) — via LLM com validação humana.</p>
  <p class="meta">Autor: Lucas Freire Silva (DIEST/COGIT/IPEA). Gerado em {now}.</p>
</header>

<section id="sobre">
  <h2>O projeto</h2>
  <p>O conhecimento produzido historicamente pelo IPEA exibe <strong>orientação para design</strong> (idealização forte) ou predominantemente marcadores descritivos (idealização fraca)? O corpus do repositório institucional é classificado em cinco dimensões operacionais derivadas de Michael Barzelay, escala Likert 1–5, via LLM com validação humana e calibração isotônica.</p>
  {dim_table}
</section>

<section id="pipeline">
  <h2>Pipeline</h2>
  {pipeline_html}
</section>

<section id="kpis">
  <h2>Estatísticas</h2>
  <div class="kpi-grid">{kpi_grid}</div>
</section>

<section id="classificacoes">
  <h2>Resultados das classificações</h2>
  {banner_cls}
  <figure>
    <img src="figures/01_media_por_decada.png" alt="Média por década × dimensão">
    <figcaption>Score médio por década × dimensão (1=descritivo, 5=design).</figcaption>
  </figure>
  <figure>
    <img src="figures/02_heatmap_decada_dim.png" alt="Heatmap década × dimensão">
    <figcaption>Heatmap dos mesmos dados. Cores vermelhas tendem ao polo descritivo, azuis ao design.</figcaption>
  </figure>
  <figure>
    <img src="figures/03_distribuicao_scores.png" alt="Distribuição de scores">
    <figcaption>Distribuição de cada nota (1–5) por dimensão no corpus todo.</figcaption>
  </figure>
  <h3>Evolução por tipo documental</h3>
  <figure><img src="figures/04_tipo_D1.png" alt="D1 por tipo"><figcaption>D1 Temporal.</figcaption></figure>
  <figure><img src="figures/04_tipo_D2.png" alt="D2 por tipo"><figcaption>D2 Epistemológica.</figcaption></figure>
  <figure><img src="figures/04_tipo_D3.png" alt="D3 por tipo"><figcaption>D3 Teoria-prática.</figcaption></figure>
  <figure><img src="figures/04_tipo_D4.png" alt="D4 por tipo"><figcaption>D4 Agência.</figcaption></figure>
  <figure><img src="figures/04_tipo_D5.png" alt="D5 por tipo"><figcaption>D5 Foco analítico.</figcaption></figure>
  <h3>Top palavras-chave por polo</h3>
  <figure>
    <img src="figures/05_palavras_polo.png" alt="Palavras-chave por polo">
    <figcaption>Palavras-chave mais frequentes nos documentos de score ≥4 (design) vs ≤2 (descritivo).</figcaption>
  </figure>
</section>

<section id="medias">
  <h2>Médias por década</h2>
  <table>
    <thead><tr><th>Década</th><th>D1</th><th>D2</th><th>D3</th><th>D4</th><th>D5</th></tr></thead>
    <tbody>
    {medias_rows}
    </tbody>
  </table>
</section>

<section id="corpus">
  <h2>Corpus (Fase 1)</h2>
  <div class="kpi-grid">
    <div class="kpi"><div class="label">Décadas cobertas</div>
      <table><thead><tr><th>Década</th><th>N</th></tr></thead><tbody>{dec_rows}</tbody></table>
    </div>
    <div class="kpi"><div class="label">Top tipos documentais</div>
      <table><thead><tr><th>Tipo</th><th>N</th></tr></thead><tbody>{tipo_rows}</tbody></table>
    </div>
  </div>
</section>

<section id="como-rodar">
  <h2>Como regerar com dados reais</h2>
  <pre><code># 1. Rodar Fase 2 completa (apenas a amostra 10% leva ~3h; todo corpus ~30h)
uv run python -m src.extracao --input data/interim/metadados_amostra10.parquet --engine pymupdf

# 2. Rodar Fase 3 real (precisa chave API — custo ~$30-80 no 10%)
export ANTHROPIC_API_KEY=sk-ant-...
uv run python -m src.classificacao --input data/interim/textos_amostra10.parquet

# 3. Regenerar o site
uv run python -m src.analise.build_site</code></pre>
  <p>GitHub Pages: em <em>Settings → Pages</em>, escolha <code>Deploy from a branch</code>, <code>main</code> / <code>/docs</code>.</p>
</section>

<footer>
  <p>Código: <a href="https://github.com/freirelucas/IDEA">github.com/freirelucas/IDEA</a> — branch <code>claude/analyze-codebase-CEDYD</code>.</p>
  <p>Briefing completo do projeto: <code>PROJETO_BARZELAY_IPEA.md</code>.</p>
</footer>

</div>
</body>
</html>
"""


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s [%(levelname)s] %(message)s",
                        stream=sys.stderr)
    build_site()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
