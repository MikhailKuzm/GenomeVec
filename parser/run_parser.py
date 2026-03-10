#!/usr/bin/env python3
"""
Парсер референсных геномов прокариот с NCBI Datasets.
Скачивает пакет по таксону 2 (Bacteria), раскладывает FASTA в genoms/fasta,
формирует genoms/statistic.log по фенотипам (организмам).
Требует установленный NCBI Datasets CLI: datasets download genome taxon ...
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from collections import defaultdict

try:
    import yaml
except ImportError:
    yaml = None


# Корень проекта (родитель каталога parser)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config() -> dict:
    config_path = Path(__file__).resolve().parent / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Конфиг не найден: {config_path}")
    if yaml is None:
        raise ImportError("Установите PyYAML: pip install pyyaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_organism_from_report_line(obj: dict) -> str:
    """Из одной строки assembly_data_report.jsonl извлекает название организма (фенотип)."""
    for key in ("organism", "organismName", "organism_name", "organism-name", "species"):
        if key in obj and obj[key]:
            return str(obj[key]).strip()
    if "assembly" in obj and isinstance(obj["assembly"], dict):
        a = obj["assembly"]
        for k in ("organism", "organismName", "organism_name"):
            if k in a and a[k]:
                return str(a[k]).strip()
    return "unknown"


def _run_datasets_cmd(cmd: list[str], config: dict, cwd: str, env: dict) -> subprocess.CompletedProcess:
    """Запуск datasets CLI с повторными попытками при обрыве."""
    max_attempts = config.get("download_max_attempts", 3)
    last_error = None
    result = None
    for attempt in range(1, max_attempts + 1):
        if attempt > 1:
            delay = config.get("download_retry_delay_sec", 30) * (attempt - 1)
            print(f"Повторная попытка {attempt}/{max_attempts} через {delay} с...")
            time.sleep(delay)
        result = subprocess.run(cmd, cwd=cwd, env=env)
        if result.returncode == 0:
            return result
        last_error = RuntimeError(f"datasets завершился с кодом {result.returncode}")
    return result if result is not None else subprocess.CompletedProcess(cmd, -1)


def _resolve_datasets_exe(config: dict) -> str:
    """Возвращает путь к исполняемому файлу datasets из конфига или PATH (Linux: datasets, Windows: datasets.exe)."""
    datasets_exe = config.get("datasets_cli_path")
    if datasets_exe:
        datasets_exe = PROJECT_ROOT / datasets_exe
        if sys.platform == "win32":
            # Windows: ищем datasets.exe
            if datasets_exe.is_dir():
                candidates = [
                    datasets_exe / "datasets.exe",
                    datasets_exe / "bin" / "datasets.exe",
                ]
                for sub in datasets_exe.iterdir():
                    if sub.is_dir():
                        candidates.append(sub / "datasets.exe")
                        candidates.append(sub / "bin" / "datasets.exe")
                for candidate in candidates:
                    if candidate.is_file():
                        datasets_exe = candidate
                        break
                else:
                    raise FileNotFoundError(
                        f"В config.yaml указана папка {datasets_exe}, но в ней не найден datasets.exe.\n"
                        "Укажите полный путь к файлу datasets.exe (например: parser/tools/datasets.exe)."
                    )
            elif not datasets_exe.suffix.lower() == ".exe" and datasets_exe.is_file():
                exe_path = datasets_exe.with_suffix(".exe")
                if exe_path.is_file():
                    datasets_exe = exe_path
            elif not datasets_exe.is_file():
                search_dir = datasets_exe.parent if datasets_exe.parent.is_dir() else None
                if search_dir:
                    candidates = [search_dir / "datasets.exe", search_dir / "bin" / "datasets.exe"]
                    for sub in search_dir.iterdir():
                        if sub.is_dir():
                            candidates.append(sub / "datasets.exe")
                            candidates.append(sub / "bin" / "datasets.exe")
                    for candidate in candidates:
                        if candidate.is_file():
                            datasets_exe = candidate
                            break
                if not datasets_exe.is_file():
                    exe_path = datasets_exe.with_suffix(".exe") if datasets_exe.suffix else Path(str(datasets_exe) + ".exe")
                    if exe_path.is_file():
                        datasets_exe = exe_path
            elif datasets_exe.suffix.lower() != ".exe" and datasets_exe.is_file():
                exe_path = datasets_exe.with_suffix(".exe")
                if exe_path.is_file():
                    datasets_exe = exe_path
        else:
            # Linux и др.: ищем бинарник datasets (без .exe)
            if datasets_exe.is_dir():
                candidates = [
                    datasets_exe / "datasets",
                    datasets_exe / "bin" / "datasets",
                ]
                for sub in datasets_exe.iterdir():
                    if sub.is_dir():
                        candidates.append(sub / "datasets")
                        candidates.append(sub / "bin" / "datasets")
                for candidate in candidates:
                    if candidate.is_file() and os.access(candidate, os.X_OK):
                        datasets_exe = candidate
                        break
                else:
                    raise FileNotFoundError(
                        f"В config.yaml указана папка {datasets_exe}, но в ней не найден исполняемый файл 'datasets'.\n"
                        "Укажите путь к бинарнику datasets (например: parser/tools/datasets)."
                    )
        if not datasets_exe.is_file():
            raise FileNotFoundError(
                f"В config.yaml указан datasets_cli_path, но файл не найден: {datasets_exe}\n"
                "Linux: установите ncbi-datasets-cli (conda/pip) или укажите путь к бинарнику 'datasets'.\n"
                "Windows: скачайте windows-amd64.cli.package.zip и укажите путь к datasets.exe."
            )
        return str(datasets_exe)
    exe = shutil.which("datasets")
    if not exe:
        raise FileNotFoundError(
            "Не найден исполняемый файл 'datasets' (NCBI Datasets CLI).\n"
            "Варианты:\n"
            "  1) Установить в PATH: conda install -c conda-forge ncbi-datasets-cli (Linux)\n"
            "  2) Скачать с https://github.com/ncbi/datasets/releases и в config.yaml\n"
            "     указать путь: datasets_cli_path: \"parser/tools/datasets\" (Linux) или .../datasets.exe (Windows)"
        )
    return exe


def _get_accession_from_report_line(obj: dict) -> str | None:
    """Извлекает полный accession (с версией) из строки отчёта."""
    acc = obj.get("accession") or (obj.get("assembly", {}) or {}).get("accession")
    if not acc:
        return None
    if isinstance(acc, dict):
        acc = acc.get("accession") or acc.get("genbank") or ""
    return str(acc).strip()


def _sanitize_filename_part(s: str, max_len: int = 120) -> str:
    """Делает строку пригодной для имени файла: пробелы и недопустимые символы → подчёркивание."""
    bad = ' \\/:*?"<>|'
    out = "".join(c if c not in bad and ord(c) >= 32 else "_" for c in s)
    out = "_".join(out.split())  # схлопнуть подряд идущие _
    if len(out) > max_len:
        out = out[:max_len]
    return out.strip("_") or "unknown"


def run_datasets_download(config: dict) -> Path | list[Path]:
    """Скачивает пакет геномов: либо по таксону (полный/ограниченный), либо по списку accession.
    Если taxon_name_for_download — список названий, для каждого скачивается до max_genomes_to_download геномов."""
    import zipfile

    download_dir = PROJECT_ROOT / config["paths"]["download_dir"]
    zip_name = config["paths"]["zip_filename"]
    zip_path = download_dir / zip_name
    download_dir.mkdir(parents=True, exist_ok=True)

    # Таксон: одно имя, список имён или числовой ID
    taxon_raw = config.get("taxon_name_for_download")
    if isinstance(taxon_raw, list):
        taxon_list = [str(x).strip() for x in taxon_raw if str(x).strip()]
    elif taxon_raw:
        taxon_list = [str(taxon_raw).strip()]
    else:
        taxon_list = []

    ref_only = config.get("reference_only", True)
    include = config.get("include", "genome")
    assembly_source = config.get("assembly_source", "RefSeq")
    max_to_download = config.get("max_genomes_to_download")

    env = os.environ.copy()
    cur = env.get("GODEBUG", "")
    env["GODEBUG"] = f"{cur},http2client=0" if cur else "http2client=0"
    datasets_exe = _resolve_datasets_exe(config)

    if max_to_download is not None and max_to_download > 0:
        # Режим ограниченной загрузки: для каждого таксона — метаданные, затем N геномов по accession
        all_accessions: list[str] = []
        taxon_for_cmd = config.get("taxon_id", 2)  # fallback, если список пустой

        for taxon_name in (taxon_list if taxon_list else [None]):
            taxon = taxon_name if taxon_name else taxon_for_cmd
            metadata_zip = download_dir / "ncbi_metadata_temp.zip"
            cmd_meta = [
                datasets_exe, "download", "genome", "taxon", str(taxon),
                "--include", "none", "--filename", str(metadata_zip),
            ]
            if ref_only:
                cmd_meta.append("--reference")
            if assembly_source and assembly_source.lower() != "all":
                cmd_meta.extend(["--assembly-source", assembly_source])
            print("Загрузка списка геномов (метаданные):", " ".join(cmd_meta))
            result = _run_datasets_cmd(cmd_meta, config, str(PROJECT_ROOT), env)
            if result.returncode != 0:
                raise RuntimeError(f"Загрузка метаданных для таксона '{taxon}' завершилась с кодом {result.returncode}")
            if not metadata_zip.exists():
                raise FileNotFoundError("Архив метаданных не создан")

            extract_meta = download_dir / "ncbi_meta_extract"
            if extract_meta.exists():
                shutil.rmtree(extract_meta)
            extract_meta.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(metadata_zip, "r") as zf:
                zf.extractall(extract_meta)

            n_for_this_taxon = 0
            for report_path in extract_meta.rglob("assembly_data_report.jsonl"):
                with open(report_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            acc = _get_accession_from_report_line(obj)
                            if acc and acc not in all_accessions:
                                all_accessions.append(acc)
                                n_for_this_taxon += 1
                                if n_for_this_taxon >= max_to_download:
                                    break
                        except (json.JSONDecodeError, TypeError):
                            continue
                if n_for_this_taxon >= max_to_download:
                    break
            print(f"  Таксон '{taxon}': найдено {n_for_this_taxon} геномов")
            try:
                shutil.rmtree(extract_meta)
            except OSError:
                pass
            metadata_zip.unlink(missing_ok=True)

        if not all_accessions:
            raise RuntimeError("В метаданных не найдено ни одного accession")

        list_file = download_dir / "accession_list.txt"
        list_file.write_text("\n".join(all_accessions), encoding="utf-8")
        print(f"Скачивание {len(all_accessions)} геномов по списку accession...")
        cmd_acc = [
            datasets_exe, "download", "genome", "accession",
            "--inputfile", str(list_file),
            "--include", "genome",
            "--filename", str(zip_path),
        ]
        print("Запуск:", " ".join(cmd_acc))
        result = _run_datasets_cmd(cmd_acc, config, str(PROJECT_ROOT), env)
        list_file.unlink(missing_ok=True)
        if result.returncode != 0:
            raise RuntimeError(f"datasets download accession завершился с кодом {result.returncode}")
    else:
        # Полная загрузка по таксону (один таксон или несколько — по очереди, несколько zip'ов)
        if not taxon_list:
            taxon = config.get("taxon_id", 2)
            taxon_list = [None]  # один проход с taxon_id
        zips: list[Path] = []
        for taxon_name in taxon_list:
            taxon = taxon_name if taxon_name else config.get("taxon_id", 2)
            current_zip = zip_path
            if len(taxon_list) > 1 and taxon_name:
                base = zip_path.stem + "_" + _sanitize_filename_part(taxon_name)
                current_zip = download_dir / (base + zip_path.suffix)
            cmd = [
                datasets_exe, "download", "genome", "taxon", str(taxon),
                "--include", include, "--filename", str(current_zip),
            ]
            if ref_only:
                cmd.append("--reference")
            if assembly_source and assembly_source.lower() != "all":
                cmd.extend(["--assembly-source", assembly_source])
            print("Запуск:", " ".join(cmd))
            result = _run_datasets_cmd(cmd, config, str(PROJECT_ROOT), env)
            if result.returncode != 0:
                raise RuntimeError(f"datasets download для таксона '{taxon}' завершился с кодом {result.returncode}")
            if current_zip.exists():
                zips.append(current_zip)
        if len(zips) == 1:
            zip_path = zips[0]
        else:
            if not zips:
                raise FileNotFoundError("Ни один архив не был создан")
            # Вернём список zip'ов для извлечения по очереди
            if not zip_path.exists() and zips:
                zip_path = zips[0]
            for z in zips:
                if not z.exists():
                    raise FileNotFoundError(f"Архив не создан: {z}")
            return zips

    if not zip_path.exists():
        raise FileNotFoundError(f"Архив не создан: {zip_path}")
    return zip_path


def extract_fasta_and_collect_stats(
    zip_path: Path,
    config: dict,
) -> tuple[list[Path], dict[str, int]]:
    """
    Распаковывает архив, копирует *_genomic.fna в genoms/fasta,
    парсит assembly_data_report.jsonl. Возвращает список скопированных FASTA
    и счётчики по фенотипам (организм = одна сборка на строку в отчёте).
    """
    import zipfile

    fasta_dir = PROJECT_ROOT / config["paths"]["fasta_dir"]
    fasta_dir.mkdir(parents=True, exist_ok=True)
    max_genomes = config.get("max_genomes")  # None = без лимита

    extract_dir = zip_path.parent / "ncbi_dataset_extract"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    phenotype_counts: dict[str, int] = defaultdict(int)  # организм -> число сборок (геномов)
    copied: list[Path] = []
    report_path = None

    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if name.endswith("/"):
                continue
            if "assembly_data_report.jsonl" in name:
                report_path = extract_dir / name
            zf.extract(name, extract_dir)

    data_dir = extract_dir / "ncbi_dataset" / "data"
    if not data_dir.exists():
        data_dir = extract_dir / "data"
    if not data_dir.exists():
        data_dir = extract_dir
        for d in extract_dir.iterdir():
            if d.is_dir() and (d / "data").exists():
                data_dir = d / "data"
                break

    for p in data_dir.rglob("assembly_data_report.jsonl"):
        report_path = p
        break
    if report_path is None:
        report_path = data_dir / "assembly_data_report.jsonl"

    # Сопоставление accession -> organism и счётчик сборок по фенотипам (по отчёту: 1 строка = 1 сборка)
    accession_to_organism: dict[str, str] = {}
    if report_path.exists():
        with open(report_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    acc = obj.get("accession") or (obj.get("assembly", {}) or {}).get("accession")
                    if not acc:
                        continue
                    if isinstance(acc, dict):
                        acc = acc.get("accession") or acc.get("genbank") or ""
                    acc = str(acc).split(".")[0]
                    org = get_organism_from_report_line(obj)
                    accession_to_organism[acc] = org
                    phenotype_counts[org] += 1
                except (json.JSONDecodeError, TypeError):
                    continue

    # Копирование FASTA с маркировкой по фенотипу (организму)
    n = 0
    for fpath in data_dir.rglob("*_genomic.fna"):
        if max_genomes is not None and n >= max_genomes:
            break
        acc = fpath.parent.name.split(".")[0]
        organism = accession_to_organism.get(acc, "unknown")
        safe_org = _sanitize_filename_part(organism)
        dest_name = f"{safe_org}_{acc}_genomic.fna"
        dest = fasta_dir / dest_name
        if dest.exists():
            dest = fasta_dir / f"{safe_org}_{acc}_{n}_genomic.fna"
        shutil.copy2(fpath, dest)
        copied.append(dest)
        n += 1

    try:
        shutil.rmtree(extract_dir)
    except OSError:
        pass

    return copied, dict(phenotype_counts)


def write_statistic_log(config: dict, copied: list[Path], phenotype_counts: dict[str, int]) -> None:
    log_path = PROJECT_ROOT / config["paths"]["statistic_log"]
    log_path.parent.mkdir(parents=True, exist_ok=True)

    total_fasta = len(copied)
    num_phenotypes = len(phenotype_counts)

    lines = [
        "=== Статистика парсера референсных геномов прокариот (NCBI Datasets) ===",
        "",
        f"Общее число FASTA файлов: {total_fasta}",
        f"Количество фенотипов (организмов/видов): {num_phenotypes}",
        "",
        "--- Количество геномов по фенотипам (организмам) ---",
        "",
    ]
    for organism in sorted(phenotype_counts.keys(), key=lambda x: (-phenotype_counts[x], x)):
        lines.append(f"  {organism}: {phenotype_counts[organism]}")
    lines.append("")
    text = "\n".join(lines)

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Статистика записана: {log_path}")


def main() -> int:
    os.chdir(PROJECT_ROOT)
    config = load_config()
    zip_result = run_datasets_download(config)
    zip_paths: list[Path] = [zip_result] if isinstance(zip_result, Path) else zip_result
    all_copied: list[Path] = []
    all_phenotype_counts: dict[str, int] = defaultdict(int)
    for zp in zip_paths:
        copied, phenotype_counts = extract_fasta_and_collect_stats(zp, config)
        all_copied.extend(copied)
        for org, count in phenotype_counts.items():
            all_phenotype_counts[org] += count
    write_statistic_log(config, all_copied, all_phenotype_counts)
    print(f"Скопировано FASTA: {len(all_copied)} в {config['paths']['fasta_dir']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
