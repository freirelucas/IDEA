"""Testes do split_pdf_em_blocos (sem invocar Docling pesado)."""
from __future__ import annotations

import pymupdf

from src.extracao.pdf import split_pdf_em_blocos


def _make_pdf(path, n_pages: int) -> None:
    doc = pymupdf.open()
    for i in range(n_pages):
        page = doc.new_page()
        page.insert_text((72, 72), f"Página {i+1}")
    doc.save(path)
    doc.close()


def test_split_um_bloco_quando_pequeno(tmp_path):
    pdf = tmp_path / "small.pdf"
    _make_pdf(pdf, n_pages=3)
    blocos = split_pdf_em_blocos(pdf, tmp_path / "blocos", pages_per_chunk=5)
    assert len(blocos) == 1
    assert blocos[0].exists()
    with pymupdf.open(blocos[0]) as d:
        assert len(d) == 3


def test_split_multiplos_blocos(tmp_path):
    pdf = tmp_path / "big.pdf"
    _make_pdf(pdf, n_pages=12)
    blocos = split_pdf_em_blocos(pdf, tmp_path / "blocos", pages_per_chunk=5)
    assert len(blocos) == 3
    sizes = []
    for b in blocos:
        with pymupdf.open(b) as d:
            sizes.append(len(d))
    assert sizes == [5, 5, 2]


def test_split_exato_no_chunk(tmp_path):
    pdf = tmp_path / "exact.pdf"
    _make_pdf(pdf, n_pages=10)
    blocos = split_pdf_em_blocos(pdf, tmp_path / "blocos", pages_per_chunk=5)
    assert len(blocos) == 2
    for b in blocos:
        with pymupdf.open(b) as d:
            assert len(d) == 5
