"""
LLMCaller mock baseado em features textuais simples.

Custo zero. Não substitui a Fase 3 real, mas produz respostas JSON no mesmo
formato de um LLM verdadeiro, com scores que correlacionam com marcadores
linguísticos observáveis. Útil para:
- rodar o pipeline end-to-end sem chave API;
- demo de visualização com dados "realistas";
- testes de integração do classificador.

Para cada dimensão, conta ocorrências de marcadores linguísticos opostos no
texto (normalizado por comprimento) e mapeia para score 1–5. Trechos de
evidência são os marcadores efetivamente encontrados.
"""
from __future__ import annotations

import json
import re

import numpy as np

from src.classificacao.classifier import CallStats, LLMCaller


# ─── marcadores por dimensão ──────────────────────────────────────────────────

MARKERS: dict[str, dict[str, list[str]]] = {
    "D1": {  # Temporal: forward vs backward
        "design": [
            r"\bdever[íi]a\b", r"\bdevem?\b", r"\bprecisa\w*\b", r"\bser[ií]a\b",
            r"recomenda[\-\s]se\b", r"\bproposta\b", r"\bse\s+implementad[oa]\b",
            r"\bdesenho de\b", r"\bseria\w*\b", r"\ba\s+ser\s+\w+",
            r"\bsugere\-?se\b", r"\bser[ií]am\b",
        ],
        "descritivo": [
            r"\bfoi\s+\w+", r"\bhouve\b", r"\bresultou\b", r"\bocorreu\b",
            r"verificou\-?se\b", r"constatou\-?se\b", r"observou\-?se\b",
            r"\bapresentou\b", r"\bforam\b", r"\bdemonstrou\w*\b",
        ],
    },
    "D2": {  # Epistemological: normative vs descriptive
        "design": [
            r"\brecomenda\w*\b", r"\bsugere\w*\b", r"\bsugerimos\b",
            r"\brecomendamos\b", r"\ba\s+pol[íi]tica\s+deve\b",
            r"\b[ée]\s+necess[áa]rio\b", r"\b[ée]\s+preciso\b",
            r"\bdeve\s+ser\b", r"\bprecisa\s+ser\b", r"\bdevemos\b",
            r"\bpropomos\b", r"\bdefendemos\b",
        ],
        "descritivo": [
            r"\bos\s+dados\b", r"\bos\s+resultados\b", r"\ba\s+an[áa]lise\s+\w+",
            r"\beste\s+(?:estudo|artigo|trabalho)\b", r"\bdescreve\w*\b",
            r"documenta\w*\b", r"correlaciona\w*\b", r"\bassocia\w+\s+se\b",
            r"\bindica\w*\b", r"\bcaracteri\w+\b",
        ],
    },
    "D3": {  # Teoria-prática
        "design": [
            r"\baplicand[oa]\s+o\s+\w+",
            r"\bcom\s+base\s+(?:em|no|na)\s+[A-Z]\w+",
            r"\bderiva\w*\s+do\s+modelo\b", r"\btrazido\w*\s+de\s+[A-Z]\w+",
            r"\bframework\s+de\b", r"\bmodelo\s+de\s+[A-Z]\w+",
            r"\barquitetura\s+propost\w+\b", r"\bmobiliz\w+\s+(?:a\s+teoria|o\s+conceito)",
        ],
        "descritivo": [
            r"\brevis[ãa]o\s+(?:da\s+)?literatura\b",
            r"\breferencial\s+te[óo]rico\b",
            r"\bcap[íi]tulo\s+\d\b", r"\bcomo\s+apontam?\b",
            r"\bsegundo\s+[A-Z]\w+", r"\bconforme\s+[A-Z]\w+",
            r"\b[àa]\s+luz\s+d[aeo]s?\b",
        ],
    },
    "D4": {  # Agência
        "design": [
            r"\bo?\s*Minist[ée]rio\s+d[aeo]\s+\w+",
            r"\bSecretaria\s+(?:Executiva\s+|Nacional\s+|de\s+)\w+",
            r"\bBanco\s+(?:Central|Nacional|do\s+Brasil)\b",
            r"\bCongresso\s+Nacional\b", r"\bTribunal\s+de\s+Contas\b",
            r"\bReceita\s+Federal\b", r"\bCGU\b", r"\bENAP\b",
            r"\bCasa\s+Civil\b", r"\bIPEA\b",
            r"\b(?:o|a)\s+(?:diretor|secret[áa]rio|presidente|gestor)\s*\w+",
            r"\bgestor(?:es)?\s+(?:do|da)\b", r"\bcabe\s+(?:a|ao|à)\b",
        ],
        "descritivo": [
            r"\bobserva\-?se\b", r"\bverifica\-?se\b", r"\bconstata\-?se\b",
            r"\btende\-?se\s+a\b", r"\bpredomina\b", r"\breproduz\w+\b",
            r"\bestrutur\w+\s+reproduz\b", r"\bo\s+sistema\s+(?:apresenta|tende)\b",
            r"\bo\s+estado\s+brasileiro\b",
        ],
    },
    "D5": {  # Foco: sintético vs fragmentado
        "design": [
            r"\barticul\w+\s+(?:as|os)\s+\w+", r"\barquitetura\s+integrada\b",
            r"\bcomplementari\w+\s+institucional\b",
            r"\bdesenho\s+coerente\b", r"\bmodelo\s+integrado\b",
            r"\btrade\-?offs?\b", r"\bsistema\s+de\s+\w+\s+e\s+\w+",
            r"\bcombina\s+\w+\s+(?:com|e)\s+\w+",
        ],
        "descritivo": [
            r"\bcap[íi]tulo\s+\d\s+analisa\b", r"\btabela\s+\d\s*:\s*regress[ãa]o\b",
            r"\bvari[áa]veis?\s+(?:explicativas?|independente)\b",
            r"\bmodelo\s+(?:logit|probit|ols|linear)\b",
            r"\bestimativas?\s+do\s+coeficiente\b",
            r"\brobusto\s+a\b",
        ],
    },
}


def _detect_dimensao(system_prompt: str) -> str:
    """Extrai a dimensão D1-D5 do system prompt (template tem `DIMENSÃO: X`)."""
    # o template tem formato '**Dimensão:** D1 — Orientação temporal'
    m = re.search(r"\*\*Dimens[ãa]o:\*\*\s*(D[1-5])", system_prompt, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # fallback: primeiro D[1-5] encontrado no system
    m = re.search(r"\b(D[1-5])\b", system_prompt)
    if m:
        return m.group(1).upper()
    return "D1"


def _extract_doc_text(user_prompt: str) -> str:
    """Extrai o texto do documento de dentro do user prompt.

    Template usa o delimitador `Texto (truncado em N tokens):\n\"\"\"\n{texto}\n\"\"\"`.
    """
    m = re.search(r'Texto.*?tokens\):\s*"""(.+?)"""', user_prompt, re.DOTALL)
    if m:
        return m.group(1).strip()
    # fallback: tudo após 'Texto'
    idx = user_prompt.find("Texto")
    return user_prompt[idx:] if idx >= 0 else user_prompt


def _count_matches(text: str, patterns: list[str]) -> tuple[int, list[str]]:
    """Conta matches totais de patterns em text e devolve até 4 exemplos."""
    total = 0
    exemplos: list[str] = []
    for pat in patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            total += 1
            if len(exemplos) < 4:
                # captura ~10 palavras em volta do match
                start = max(0, m.start() - 20)
                end = min(len(text), m.end() + 20)
                snippet = re.sub(r"\s+", " ", text[start:end]).strip()
                if snippet and snippet not in exemplos:
                    exemplos.append(snippet[:120])
    return total, exemplos


def _score_from_ratio(
    n_design: int, n_descritivo: int, n_chars: int, rng: np.random.Generator
) -> int:
    """Mapeia counts para score 1-5.

    - Normaliza por `n_chars/10000` (densidade de marcadores).
    - Se zero marcadores de ambos lados → score 3 (neutro).
    - Ratio design / (design + descritivo) com ruído → faixa 1-5.
    """
    total = n_design + n_descritivo
    if total == 0 or n_chars < 500:
        # sem evidência textual → neutro com leve ruído
        return int(np.clip(round(3 + rng.normal(0, 0.6)), 1, 5))
    ratio = n_design / total
    # magnitude (quantos marcadores por 10k chars) influencia confiança
    densidade = total / max(1, n_chars / 10000)
    # centralizar em 3, puxar para extremos conforme ratio * densidade
    delta = (ratio - 0.5) * 4 * min(1.0, densidade / 3)
    score = 3 + delta + rng.normal(0, 0.4)
    return int(np.clip(round(score), 1, 5))


def make_mock_caller(seed: int = 42) -> LLMCaller:
    """Constrói um LLMCaller mock que deriva scores de features textuais."""
    rng = np.random.default_rng(seed)

    def call(system: str, user: str, max_tokens: int, temperature: float) -> tuple[str, CallStats]:
        dim = _detect_dimensao(system)
        text = _extract_doc_text(user)

        markers = MARKERS.get(dim, {})
        n_design, ex_design = _count_matches(text, markers.get("design", []))
        n_descr, ex_descr = _count_matches(text, markers.get("descritivo", []))

        score = _score_from_ratio(n_design, n_descr, len(text), rng)

        # trechos: mistura dos encontrados, privilegiando o polo do score
        if score >= 4:
            trechos = ex_design[:3] or ex_descr[:2] or ["[mock: sem marcadores claros]"]
        elif score <= 2:
            trechos = ex_descr[:3] or ex_design[:2] or ["[mock: sem marcadores claros]"]
        else:
            trechos = (ex_design[:2] + ex_descr[:2])[:3] or ["[mock: sem marcadores claros]"]

        justificativa = (
            f"[mock/features] {dim}: {n_design} marcadores de design e {n_descr} "
            f"descritivos em ~{len(text)//1000}k chars; "
            f"score={score} via ratio+densidade. Substituir por LLM real p/ qualidade."
        )

        response = json.dumps({
            "dimensao": dim,
            "score": score,
            "trechos_evidencia": trechos,
            "justificativa": justificativa,
        })

        # estatísticas pseudo-realistas em tokens
        stats = CallStats(
            tokens_input=max(100, len(text) // 4),
            tokens_output=len(response) // 4,
        )
        return response, stats

    return call
