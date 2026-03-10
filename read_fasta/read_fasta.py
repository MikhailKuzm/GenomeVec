"""
Чтение FASTA: контиги, метаданные, извлечение названия организма из заголовков.
Заголовки вида: >ACCESSION Organism_name strain XXX ... или ... , details
"""

from __future__ import annotations

import re
from pathlib import Path


def _parse_organism_from_header(header: str) -> str:
    """Из строки заголовка (>...) извлекает название организма (до 'strain' или запятой)."""
    s = header.strip()
    if not s.startswith(">"):
        return "unknown"
    s = s[1:].strip()
    # Убираем ведущий accession (часто начинается с букв и цифр с точкой: NZ_XXX.N или подобное)
    m = re.match(r"^[\w.]+\s+", s)
    if m:
        s = s[m.end() :].strip()
    # Берём подстроку до "strain" или до запятой
    for sep in (" strain ", " Strain ", ","):
        i = s.find(sep)
        if i != -1:
            s = s[:i]
    return " ".join(s.split()).strip() or "unknown"


def read_fasta(path: str | Path) -> tuple[list[str], dict]:
    """
    Читает FASTA-файл, разбивает на контиги, извлекает организм из первого заголовка.

    Returns:
        (contigs, meta): contigs — список строк нуклеотидов (по одной на контиг),
        meta — dict с num_contigs, total_nucleotides, organism_name, source_file.
    """
    path = Path(path)
    contigs: list[str] = []
    current_seq: list[str] = []
    organism_name = "unknown"

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n\r")
            if not line:
                continue
            if line.startswith(">"):
                # Сохраняем предыдущий контиг
                if current_seq:
                    contigs.append("".join(current_seq))
                    current_seq = []
                # Из первого заголовка достаём организм
                if organism_name == "unknown":
                    organism_name = _parse_organism_from_header(line)
            else:
                current_seq.append(line.upper())

    if current_seq:
        contigs.append("".join(current_seq))

    total_nt = sum(len(c) for c in contigs)
    meta = {
        "num_contigs": len(contigs),
        "total_nucleotides": total_nt,
        "organism_name": organism_name,
        "source_file": str(path),
    }
    return contigs, meta


class FastaLoader:
    """Обёртка для загрузки FASTA: метод load(path) возвращает (contigs, meta)."""

    def load(self, path: str | Path) -> tuple[list[str], dict]:
        """Читает FASTA по пути, возвращает (contigs, meta)."""
        return read_fasta(path)
