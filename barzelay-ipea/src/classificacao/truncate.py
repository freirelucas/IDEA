"""
Truncamento de textos longos para enquadrar no contexto do LLM.

Briefing §5 Fase 3: Opção A — manter primeiros N tokens (introdução +
metodologia + parte da discussão). Validar empiricamente contra Opção B
(janelas + agregação) numa amostra antes de escalar.
"""
from __future__ import annotations


def truncate_to_chars(text: str, max_chars: int) -> tuple[str, bool]:
    """Truncamento simples por caracteres (proxy de token, ~4 chars/token)."""
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars], True


def truncate_to_tokens(
    text: str,
    max_tokens: int,
    chars_per_token: float = 4.0,
) -> tuple[str, bool]:
    """
    Trunca texto a `max_tokens` aproximados, usando chars_per_token como
    fator de conversão. Para textos em português científico, o ratio típico
    é 4–5 chars/token (BPE multilingual). Default conservador: 4.
    """
    max_chars = int(max_tokens * chars_per_token)
    return truncate_to_chars(text, max_chars)
