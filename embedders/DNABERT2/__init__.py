"""
Пакет DNABERT2: эмбеддинги по FASTA, скользящее окно.
Импорт: from embedders.DNABERT2 import FastaToEmbeddings
"""

from .launch_main import FastaToEmbeddings, _fasta_string_to_contigs

__all__ = ["FastaToEmbeddings", "_fasta_string_to_contigs"]
