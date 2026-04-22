"""
Visualizações para os resultados das Fases 3–5.

Cada função aceita DataFrames já processados pelas rotinas de `longitudinal`,
`concordancia` e `calibracao`, e devolve um `matplotlib.figure.Figure`.
Todas aceitam `save_to: Path | None` para gravar PNG/PDF.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analise.concordancia import DIMENSOES

NOMES_DIM = {
    "D1": "D1 Temporal",
    "D2": "D2 Epistemológica",
    "D3": "D3 Teoria-Prática",
    "D4": "D4 Agência",
    "D5": "D5 Foco Analítico",
}


# ─── helper ────────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, save_to: Path | None) -> plt.Figure:
    if save_to is not None:
        save_to.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, dpi=150, bbox_inches="tight")
    return fig


def _dim_ticks(ax: plt.Axes, dims: list[str] | None = None) -> None:
    dims = dims or list(DIMENSOES)
    ax.set_xticks(range(len(dims)))
    ax.set_xticklabels([NOMES_DIM[d] for d in dims], rotation=20, ha="right")


# ─── 1. média por década × dimensão ────────────────────────────────────────────

def plot_media_por_decada(
    media: pd.DataFrame,
    *,
    title: str = "Score médio por década × dimensão (Barzelay-IPEA)",
    save_to: Path | None = None,
) -> plt.Figure:
    """`media` é o output de `longitudinal.media_por_decada`: índice=década,
    colunas=D1..D5."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for dim in DIMENSOES:
        if dim in media.columns:
            ax.plot(media.index, media[dim], marker="o", label=NOMES_DIM[dim])
    ax.axhline(3.0, color="grey", linestyle=":", linewidth=1, label="ponto neutro")
    ax.set_xlabel("Década")
    ax.set_ylabel("Score médio (1=descritivo, 5=design)")
    ax.set_ylim(1, 5)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, save_to)


# ─── 2. heatmap década × dimensão ──────────────────────────────────────────────

def plot_heatmap_decada_dimensao(
    media: pd.DataFrame,
    *,
    title: str = "Heatmap década × dimensão",
    save_to: Path | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(media))))
    data = media.reindex(columns=list(DIMENSOES)).values
    im = ax.imshow(data, cmap="RdYlBu_r", vmin=1, vmax=5, aspect="auto")
    ax.set_xticks(range(len(DIMENSOES)))
    ax.set_xticklabels([NOMES_DIM[d] for d in DIMENSOES], rotation=20, ha="right")
    ax.set_yticks(range(len(media.index)))
    ax.set_yticklabels(media.index)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color="white" if abs(val - 3) > 1.2 else "black", fontsize=9)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.7, label="score médio")
    fig.tight_layout()
    return _save(fig, save_to)


# ─── 3. distribuição por dimensão (stacked counts 1-5) ─────────────────────────

def plot_distribuicao_scores(
    classificacoes: pd.DataFrame,
    *,
    score_col: str = "score",
    title: str = "Distribuição de scores por dimensão",
    save_to: Path | None = None,
) -> plt.Figure:
    df = classificacoes.dropna(subset=[score_col]).copy()
    df[score_col] = df[score_col].astype(float).round().astype(int)
    counts = (
        df.groupby(["dimensao", score_col]).size().unstack(fill_value=0)
    )
    counts = counts.reindex(index=list(DIMENSOES), columns=[1, 2, 3, 4, 5], fill_value=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    # cores do vermelho (1=descritivo) ao azul (5=design)
    cmap = plt.cm.RdYlBu
    colors = [cmap(0.1), cmap(0.3), cmap(0.5), cmap(0.7), cmap(0.9)]
    bottoms = np.zeros(len(DIMENSOES))
    x = np.arange(len(DIMENSOES))
    for s, color in zip([1, 2, 3, 4, 5], colors):
        vals = counts[s].values
        ax.bar(x, vals, bottom=bottoms, color=color, label=f"score {s}", edgecolor="white")
        bottoms += vals
    _dim_ticks(ax, list(DIMENSOES))
    ax.set_ylabel("# documentos")
    ax.set_title(title)
    ax.legend(title="Score", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    return _save(fig, save_to)


# ─── 4. Cohen's kappa bar por dimensão ────────────────────────────────────────

def plot_cohen_kappa(
    kappas: list,  # list[KappaPorDim]
    *,
    title: str = "Cohen's κ (LLM↔mediana humana)",
    save_to: Path | None = None,
) -> plt.Figure:
    labels = [NOMES_DIM[k.dimensao] for k in kappas]
    valores = [float(k.kappa) if k.kappa == k.kappa else 0 for k in kappas]
    n_pares = [k.n_pares for k in kappas]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    cores = ["#d73027" if v < 0.2 else "#fdae61" if v < 0.4 else
             "#abd9e9" if v < 0.6 else "#74add1" if v < 0.8 else "#313695"
             for v in valores]
    bars = ax.bar(labels, valores, color=cores, edgecolor="white")
    for b, v, n in zip(bars, valores, n_pares):
        ax.text(b.get_x() + b.get_width()/2, v + 0.02, f"{v:.2f}\n(n={n})",
                ha="center", va="bottom", fontsize=9)
    # faixas de interpretação (Landis & Koch)
    for y, txt in [(0.2, "slight"), (0.4, "fair"), (0.6, "moderate"),
                   (0.8, "substantial")]:
        ax.axhline(y, color="grey", linestyle=":", linewidth=0.5)
        ax.text(-0.45, y, txt, fontsize=7, color="grey", va="bottom")
    ax.set_ylim(-0.1, 1.1)
    ax.set_ylabel("Cohen's κ (weighted linear)")
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    return _save(fig, save_to)


# ─── 5. Scatter LLM vs humano com identidade ──────────────────────────────────

def plot_scatter_llm_vs_humano(
    classificacoes: pd.DataFrame,
    validacoes: pd.DataFrame,
    *,
    save_to: Path | None = None,
) -> plt.Figure:
    valid = validacoes.dropna(subset=["score_humano"]).copy()
    valid["score_humano"] = valid["score_humano"].astype(float)
    humano = (
        valid.groupby(["document_id", "dimensao"])["score_humano"]
        .median().reset_index()
    )
    cls = classificacoes.dropna(subset=["score"]).copy()
    cls["score"] = cls["score"].astype(float)
    cls = cls.sort_values("timestamp").drop_duplicates(
        subset=["document_id", "dimensao"], keep="last"
    )
    merged = cls.merge(humano, on=["document_id", "dimensao"], how="inner")

    n = len(DIMENSOES)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=True)
    rng = np.random.default_rng(42)
    for ax, dim in zip(axes, DIMENSOES):
        sub = merged[merged["dimensao"] == dim]
        if sub.empty:
            ax.text(0.5, 0.5, "sem dados", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(NOMES_DIM[dim])
            continue
        jitter_x = rng.normal(0, 0.08, size=len(sub))
        jitter_y = rng.normal(0, 0.08, size=len(sub))
        ax.scatter(sub["score"] + jitter_x, sub["score_humano"] + jitter_y,
                   alpha=0.5, s=30)
        ax.plot([0.5, 5.5], [0.5, 5.5], "k--", linewidth=0.8, label="y=x")
        ax.set_xlim(0.5, 5.5); ax.set_ylim(0.5, 5.5)
        ax.set_xlabel("Score LLM")
        ax.set_title(f"{NOMES_DIM[dim]} (n={len(sub)})")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Score humano (mediana)")
    fig.suptitle("Concordância LLM vs humano por dimensão", y=1.02)
    fig.tight_layout()
    return _save(fig, save_to)


# ─── 6. top palavras por polo ──────────────────────────────────────────────────

def plot_top_palavras_polo(
    top_palavras: dict,  # {dimensao: {"design": Series, "descritivo": Series}}
    *,
    top_n: int = 15,
    save_to: Path | None = None,
) -> plt.Figure:
    fig, axes = plt.subplots(len(top_palavras), 2, figsize=(12, 3 * len(top_palavras)))
    if len(top_palavras) == 1:
        axes = axes.reshape(1, 2)
    for row, (dim, polos) in enumerate(top_palavras.items()):
        for col, (polo, cor) in enumerate(zip(["descritivo", "design"], ["#d73027", "#313695"])):
            s = polos.get(polo, pd.Series(dtype=int))
            s = s.head(top_n)[::-1]
            ax = axes[row, col]
            if s.empty:
                ax.text(0.5, 0.5, "sem dados", ha="center", va="center", transform=ax.transAxes)
            else:
                ax.barh(s.index, s.values, color=cor)
                ax.set_xlabel("# docs")
            ax.set_title(f"{NOMES_DIM.get(dim, dim)} — polo {polo}", fontsize=10)
    fig.tight_layout()
    return _save(fig, save_to)


# ─── 7. linhas por tipo documental ────────────────────────────────────────────

def plot_media_por_tipo(
    long_df: pd.DataFrame,  # output de media_por_decada_e_tipo
    *,
    dimensao: str = "D1",
    tipos: list[str] | None = None,
    save_to: Path | None = None,
) -> plt.Figure:
    df = long_df[long_df["dimensao"] == dimensao]
    if tipos is None:
        top_tipos = df.groupby("tipo").size().sort_values(ascending=False).head(5).index.tolist()
        tipos = top_tipos
    df = df[df["tipo"].isin(tipos)]

    fig, ax = plt.subplots(figsize=(10, 5))
    for tipo in tipos:
        sub = df[df["tipo"] == tipo].sort_values("decada")
        if not sub.empty:
            ax.plot(sub["decada"], sub["score_medio"], marker="o", label=tipo)
    ax.axhline(3.0, color="grey", linestyle=":", linewidth=1)
    ax.set_xlabel("Década")
    ax.set_ylabel("Score médio")
    ax.set_ylim(1, 5)
    ax.set_title(f"{NOMES_DIM.get(dimensao, dimensao)} — evolução por tipo documental")
    ax.legend(fontsize=9, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, save_to)


# ─── 8. calibração: curva LLM → humano ────────────────────────────────────────

def plot_curvas_calibracao(
    calibradores: dict,  # {dim: CalibradorPorDim}
    *,
    save_to: Path | None = None,
) -> plt.Figure:
    xs = np.linspace(1, 5, 50)
    n = len(calibradores) or 1
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3.5), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, (dim, cal) in zip(axes, calibradores.items()):
        ys = cal.modelo.predict(xs)
        ax.plot(xs, ys, marker=".", color="#313695", label="calibrador")
        ax.plot([1, 5], [1, 5], "k--", linewidth=0.8, label="identidade")
        ax.set_xlim(1, 5); ax.set_ylim(1, 5)
        ax.set_title(f"{NOMES_DIM[dim]} (n={cal.n_pares_treino})")
        ax.set_xlabel("score LLM")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("score calibrado")
    if calibradores:
        axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Curvas de calibração isotônica por dimensão", y=1.02)
    fig.tight_layout()
    return _save(fig, save_to)
