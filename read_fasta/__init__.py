"""
Пакет read_fasta: чтение FASTA-файлов, контиги и метаданные (в т.ч. organism_name).
Импорт: from read_fasta import read_fasta
"""

from .read_fasta import read_fasta, FastaLoader

__all__ = ["read_fasta", "FastaLoader"]
