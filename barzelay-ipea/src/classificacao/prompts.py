"""
Carregamento dos templates de prompt versionados em `prompts/D{1..5}.md`.

Cada template tem placeholders `{titulo}`, `{autores}`, `{ano}`, `{tipo}`,
`{n_tokens}`, `{texto}`. Versão é extraída da linha `**Versão:** X.Y`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"

DIMENSOES = ("D1", "D2", "D3", "D4", "D5")
_VERSION_RE = re.compile(r"\*\*Vers[ãa]o:\*\*\s*([0-9]+\.[0-9]+)", re.IGNORECASE)


@dataclass(frozen=True)
class PromptTemplate:
    dimensao: str
    versao: str
    template: str  # contém placeholders {titulo}, {autores}, etc.

    def render(
        self,
        titulo: str,
        autores: str,
        ano: int | str | None,
        tipo: str,
        texto: str,
        n_tokens: int,
    ) -> str:
        return self.template.format(
            titulo=titulo or "(sem título)",
            autores=autores or "(sem autores)",
            ano=ano if ano is not None else "(sem ano)",
            tipo=tipo or "(sem tipo)",
            texto=texto,
            n_tokens=n_tokens,
        )


@lru_cache(maxsize=8)
def load_prompt(dimensao: str, prompts_dir: Path = PROMPTS_DIR) -> PromptTemplate:
    if dimensao not in DIMENSOES:
        raise ValueError(f"Dimensão inválida: {dimensao}. Use uma de {DIMENSOES}.")
    path = prompts_dir / f"{dimensao}.md"
    raw = path.read_text(encoding="utf-8")
    m = _VERSION_RE.search(raw)
    versao = m.group(1) if m else "0.0"
    return PromptTemplate(dimensao=dimensao, versao=versao, template=raw)


def load_all() -> dict[str, PromptTemplate]:
    return {d: load_prompt(d) for d in DIMENSOES}
