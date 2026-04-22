"""
Classificador LLM por dimensão.

Cada chamada submete um documento ao LLM com o prompt da dimensão D_i e
parseia a resposta JSON em `ClassificacaoLLM`. Múltiplas tentativas com
backoff caso o JSON volte malformado.

Briefing §6.1: temperatura 0, JSON estruturado, 1 prompt por dimensão.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable

from pydantic import ValidationError

from src.classificacao.prompts import PromptTemplate, load_prompt
from src.classificacao.schema import ClassificacaoLLM
from src.classificacao.truncate import truncate_to_tokens

log = logging.getLogger(__name__)

DEFAULT_MAX_INPUT_TOKENS = 25_000  # briefing §5 Fase 3 Opção A
DEFAULT_MAX_OUTPUT_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.0


@dataclass
class CallStats:
    tokens_input: int = 0
    tokens_output: int = 0


@dataclass
class CallResult:
    classificacao: ClassificacaoLLM
    stats: CallStats
    raw_response: str


# Tipo de uma função que envia uma mensagem ao LLM e devolve (texto, stats).
# Permite injetar mocks nos testes; em produção usar `make_anthropic_caller`.
LLMCaller = Callable[[str, str, int, float], tuple[str, CallStats]]


def make_anthropic_caller(model: str) -> LLMCaller:
    """Constrói um caller que usa o SDK oficial da Anthropic."""
    import anthropic
    client = anthropic.Anthropic()  # lê ANTHROPIC_API_KEY do env

    def call(system: str, user: str, max_tokens: int, temperature: float) -> tuple[str, CallStats]:
        msg = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = msg.content[0].text  # type: ignore[union-attr]
        stats = CallStats(
            tokens_input=getattr(msg.usage, "input_tokens", 0),
            tokens_output=getattr(msg.usage, "output_tokens", 0),
        )
        return text, stats

    return call


def _parse_json(text: str) -> dict[str, Any]:
    """Extrai um objeto JSON de uma resposta LLM, tolerando blocos ```json."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text
    candidate = candidate.strip()
    if not candidate.startswith("{"):
        m = re.search(r"\{.*\}", candidate, re.DOTALL)
        if m:
            candidate = m.group(0)
    return json.loads(candidate)


def classificar(
    *,
    doc_id: str,
    titulo: str,
    autores: str,
    ano: int | str | None,
    tipo: str,
    texto: str,
    dimensao: str,
    caller: LLMCaller,
    template: PromptTemplate | None = None,
    max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
) -> CallResult:
    """
    Classifica um documento numa dimensão D_i.

    `caller` é injetado para permitir mocks; produção usa `make_anthropic_caller`.
    """
    if template is None:
        template = load_prompt(dimensao)

    truncated, was_truncated = truncate_to_tokens(texto, max_input_tokens)
    if was_truncated:
        log.info("doc=%s D=%s: texto truncado (%d → ~%d tokens)",
                 doc_id, dimensao, len(texto), max_input_tokens)

    rendered = template.render(
        titulo=titulo, autores=autores, ano=ano, tipo=tipo,
        texto=truncated, n_tokens=max_input_tokens,
    )
    # split em "system" (tudo até a seção "## Documento") e "user" (resto)
    system, user = _split_system_user(rendered)

    raw, stats = caller(system, user, max_output_tokens, temperature)
    parsed = _parse_json(raw)
    try:
        clf = ClassificacaoLLM.model_validate(parsed)
    except ValidationError as e:
        raise ValueError(f"resposta LLM inválida para {doc_id}/{dimensao}: {e}\nbruto: {raw[:300]}") from e

    if clf.dimensao != dimensao:
        raise ValueError(
            f"dimensão na resposta ({clf.dimensao}) ≠ esperada ({dimensao}) para {doc_id}"
        )

    return CallResult(classificacao=clf, stats=stats, raw_response=raw)


def _split_system_user(rendered: str) -> tuple[str, str]:
    """Separa o template renderizado em system prompt + user prompt.

    A convenção dos templates D1–D5: tudo até `## Documento` é system;
    desse marcador em diante é user.
    """
    marker = "\n## Documento"
    if marker not in rendered:
        return "", rendered
    head, tail = rendered.split(marker, 1)
    return head.strip(), (marker.lstrip("\n") + tail).strip()
