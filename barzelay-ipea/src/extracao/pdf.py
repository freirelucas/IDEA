"""
Extração de texto de PDFs.

Duas engines:
- `pymupdf`: rápida (<1s/PDF), usa apenas texto nativo/pré-OCR do PDF. Default.
- `docling`: pesada (~30s-2min/PDF), faz OCR novo com EasyOCR + extrai
  estrutura de tabelas. Precisa `easyocr` instalado.

Portado de `ipeadata-lab/IpeaPub: ingestao/create_ingestion.py`. Descartadas
as partes de embedding (Qdrant/FastEmbed/ColBERT) e chunker semântico —
Barzelay precisa só do texto completo para classificação.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pymupdf

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExtractedText:
    full_text: str
    n_pages: int
    tables_md: list[str]
    engine: str
    error: str | None = None


def extrair_com_pymupdf(pdf_path: Path) -> ExtractedText:
    """Extrai texto de um PDF via PyMuPDF — rápido, usa OCR pré-existente."""
    try:
        with pymupdf.open(pdf_path) as doc:
            n_pages = len(doc)
            paginas = [p.get_text() for p in doc]
    except Exception as e:
        log.error("PyMuPDF falhou em %s: %s", pdf_path.name, e)
        return ExtractedText("", 0, [], engine="pymupdf", error=f"pymupdf_err:{e}")

    texto = "\n\n".join(paginas).strip()
    if not texto:
        return ExtractedText("", n_pages, [], engine="pymupdf", error="texto_vazio")
    return ExtractedText(texto, n_pages, [], engine="pymupdf", error=None)


def split_pdf_em_blocos(
    pdf_path: Path,
    temp_dir: Path,
    pages_per_chunk: int = 5,
) -> list[Path]:
    """Fragmenta um PDF longo em blocos de N páginas para evitar estouro de
    memória do Docling. Devolve os paths dos PDFs temporários gerados."""
    temp_dir.mkdir(parents=True, exist_ok=True)
    blocos: list[Path] = []

    with pymupdf.open(pdf_path) as doc:
        total = len(doc)
        for start in range(0, total, pages_per_chunk):
            end = min(start + pages_per_chunk - 1, total - 1)
            new_doc = pymupdf.open()
            new_doc.insert_pdf(doc, from_page=start, to_page=end)
            block_n = (start // pages_per_chunk) + 1
            bloco_path = temp_dir / f"{pdf_path.stem}_bloco_{block_n}.pdf"
            new_doc.save(bloco_path)
            new_doc.close()
            blocos.append(bloco_path)

    return blocos


def ler_pdf_com_docling(
    pdf_path: Path,
    ocr: bool = True,
    table_structure: bool = True,
    page_threshold: int = 15,
    pages_per_chunk: int = 5,
) -> ExtractedText:
    """
    Extrai texto + tabelas de um PDF com Docling.

    Configuração mínima para Barzelay (briefing §5, Fase 2):
    - `do_ocr=True` — corpus tem PDFs antigos escaneados.
    - `do_table_structure=True` — tabelas fazem parte do texto classificado.
    - `generate_page_images=False`, `generate_picture_images=False` —
      dispensável para classificação de texto.

    PDFs com mais que `page_threshold` páginas são fragmentados em blocos
    de `pages_per_chunk` páginas para evitar OOM, seguindo a mesma
    estratégia do IpeaPub.
    """
    # Import tardio — Docling tem carregamento pesado (torch, easyocr)
    import torch
    from docling.datamodel.accelerator_options import (
        AcceleratorDevice,
        AcceleratorOptions,
    )
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import EasyOcrOptions, PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    accelerator = AcceleratorOptions(
        device=AcceleratorDevice.CUDA if torch.cuda.is_available() else AcceleratorDevice.CPU
    )
    pdf_options = PdfPipelineOptions(
        do_ocr=ocr,
        do_table_structure=table_structure,
        generate_page_images=False,
        generate_picture_images=False,
        accelerator_options=accelerator,
        ocr_options=EasyOcrOptions(lang=["pt", "en"]),
    )
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options)}
    )

    with pymupdf.open(pdf_path) as doc:
        total_pages = len(doc)

    if total_pages > page_threshold:
        temp_dir = Path("temp_pages") / pdf_path.stem
        paginas = split_pdf_em_blocos(pdf_path, temp_dir, pages_per_chunk)
    else:
        temp_dir = None
        paginas = [pdf_path]

    docs_parciais = []
    for pagina in paginas:
        try:
            result = converter.convert(pagina)
            docs_parciais.append(result.document)
        except Exception as e:
            log.warning("Bloco %s falhou: %s", pagina.name, e)

    if not docs_parciais:
        if temp_dir and temp_dir.exists():
            _cleanup(temp_dir)
        return ExtractedText("", total_pages, [], engine="docling", error="docling_empty")

    textos: list[str] = []
    tabelas: list[str] = []
    for doc in docs_parciais:
        textos.append(doc.export_to_markdown())
        for table in getattr(doc, "tables", []) or []:
            try:
                tabelas.append(table.export_to_markdown())
            except Exception:
                pass

    if temp_dir and temp_dir.exists():
        _cleanup(temp_dir)

    return ExtractedText(
        full_text="\n\n".join(textos),
        n_pages=total_pages,
        tables_md=tabelas,
        engine="docling",
        error=None,
    )


def _cleanup(path: Path) -> None:
    import shutil
    shutil.rmtree(path, ignore_errors=True)
