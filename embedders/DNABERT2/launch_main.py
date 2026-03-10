"""
Скрипт/модуль: скользящее окно по FASTA (только нуклеотидные строки), матрица эмбеддингов.
Класс FastaToEmbeddings принимает сырую строку FASTA и возвращает тензор (num_windows, hidden_size)
в порядке следования окон сверху вниз по файлу.
"""

from __future__ import annotations

from pathlib import Path

# Импорты из того же пакета (относительные для работы на Linux и из train)
from .tokenize_genome import tokenize_contigs
from .embeddings import _load_model_and_tokenizer, embeddings_matrix_for_windows


def _fasta_string_to_contigs(fasta_str: str) -> list[str]:
    """
    Из сырой строки FASTA извлекает контиги: только строки с нуклеотидами,
    строки с мета-информацией (начинающиеся с '>') пропускаются.
    Порядок контигов сохраняется (сверху вниз по тексту).
    """
    contigs: list[str] = []
    current_seq: list[str] = []
    for line in fasta_str.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current_seq:
                contigs.append("".join(current_seq).upper())
                current_seq = []
        else:
            current_seq.append(line.upper())
    if current_seq:
        contigs.append("".join(current_seq).upper())
    return contigs


class FastaToEmbeddings:
    """
    Класс для получения матрицы эмбеддингов по FASTA.
    Принимает сырую строку FASTA (или путь к файлу), скользит окном только по нуклеотидным
    строкам (заголовки контигов пропускаются), возвращает тензор (num_windows, hidden_size)
    в том же порядке, в каком окна идут в исходном FASTA сверху вниз.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        window_size: int = 512,
        overlap_tokens: int = 0,
        device: str | None = None,
        batch_size: int = 1,
    ):
        self.model_path = model_path
        self.window_size = window_size
        self.overlap_tokens = overlap_tokens
        self.device = device
        self.batch_size = batch_size
        self._model = None
        self._tokenizer = None

    def _ensure_model(self):
        """Ленивая загрузка модели и токенизатора."""
        if self._model is None:
            self._model, self._tokenizer, self._device = _load_model_and_tokenizer(
                self.model_path, self.device
            )
            self.device = self.device or self._device

    def __call__(self, fasta_content: str | Path) -> __import__("torch").Tensor:
        """
        Принимает сырую строку FASTA или путь к файлу. Возвращает тензор эмбеддингов
        формы (num_windows, hidden_size), порядок строк — как порядок окон в FASTA.
        """
        # Путь к файлу — только Path или короткая строка без переносов (не содержимое FASTA)
        if isinstance(fasta_content, Path) and fasta_content.is_file():
            fasta_content = fasta_content.read_text(encoding="utf-8", errors="replace")
        elif (
            isinstance(fasta_content, str)
            and "\n" not in fasta_content
            and len(fasta_content) < 4096
            and Path(fasta_content).is_file()
        ):
            fasta_content = Path(fasta_content).read_text(encoding="utf-8", errors="replace")
        contigs = _fasta_string_to_contigs(fasta_content)
        if not contigs:
            raise ValueError("В FASTA не найдено ни одной последовательности нуклеотидов")

        self._ensure_model()
        tokens_per_contig = tokenize_contigs(contigs, tokenizer=self._tokenizer)
        matrix, _ = embeddings_matrix_for_windows(
            tokens_per_contig,
            model=self._model,
            tokenizer=self._tokenizer,
            window_size=self.window_size,
            overlap_tokens=self.overlap_tokens,
            device=self._device,
            batch_size=self.batch_size,
        )
        return matrix
