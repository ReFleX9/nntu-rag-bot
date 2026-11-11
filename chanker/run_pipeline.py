"""Настройки запуска подготовки корпуса. Запуск: python run_pipeline.py"""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Корпус и результаты.
INPUT_PATH = "corpus"
OUTPUT_DIR = "prepared"
CORPUS_ROOT = None
CORPUS_ID = "ai-impulse-docs"

# OCR.
TESSERACT_CMD = None
TESSDATA_DIR = None
OCR_LANGUAGES = "rus+eng"
OCR_ENABLED = True
DPI = 250
FULL_PAGE_OCR = False
RASTER_TABLES = False
NATIVE_TABLE_STRATEGY = "lines_strict"

# Чанкинг. Небольшие соседние блоки объединяются до нарезки; короткий хвост
# стараемся присоединять к предыдущему чанку.
CHUNK_SIZE = 1800
CHUNK_OVERLAP = 200
MIN_CHUNK_SIZE = 350
REMOVE_BOILERPLATE = True
FILTER_IMAGE_TEXT = True

# Для обычного PDF-текста склеиваем переносы строк внутри абзацев.
TEXT_KIND = "prose"
JOIN_WRAPPED_LINES = True

INCLUDE_PARTIAL = False

# Локальный токенизатор embedding-модели.
TOKENIZER_PATH = r"C:\Users\kiril\.cache\huggingface\hub\models--Qwen--Qwen3-Embedding-0.6B\snapshots\97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
MAX_CHUNK_TOKENS = 1600


def local_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else BASE_DIR / path).resolve()


def check_pipeline_compatibility() -> None:
    path = BASE_DIR / "document_pipeline.py"
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    version = None
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Constant)
            and any(isinstance(target, ast.Name) and target.id == "PIPELINE_VERSION" for target in node.targets)
        ):
            version = node.value.value
    options = {
        argument.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        for argument in node.args
        if isinstance(argument, ast.Constant) and isinstance(argument.value, str)
    }
    required = {
        "--corpus-id", "--tokenizer-path", "--max-chunk-tokens",
        "--min-chunk-size", "--keep-boilerplate", "--keep-noisy-image-text",
        "--native-table-strategy",
    }
    missing = sorted(required - options)
    if not isinstance(version, str) or not version.startswith("3.2") or missing:
        detail = f" Нет параметров: {', '.join(missing)}." if missing else ""
        raise RuntimeError(
            f"Несовместимая версия document_pipeline.py: {version or 'не указана'}."
            f"{detail} Нужна версия 3.2.x из этого комплекта файлов."
        )


def build_command() -> list[str]:
    for name in ("document_pipeline.py", "pipeline_metadata.py", "pdf_media.py", "text_cleaner.py"):
        if not (BASE_DIR / name).is_file():
            raise FileNotFoundError(f"Не найден обязательный файл: {BASE_DIR / name}")

    check_pipeline_compatibility()
    source = local_path(INPUT_PATH)
    if not source.exists():
        raise FileNotFoundError(f"Не найден входной путь: {source}. Измените INPUT_PATH.")

    command = [
        sys.executable, "-u", str(BASE_DIR / "document_pipeline.py"), str(source),
        "--output", str(local_path(OUTPUT_DIR)),
        "--corpus-id", CORPUS_ID,
        "--languages", OCR_LANGUAGES,
        "--dpi", str(DPI),
        "--chunk-size", str(CHUNK_SIZE),
        "--chunk-overlap", str(CHUNK_OVERLAP),
        "--min-chunk-size", str(MIN_CHUNK_SIZE),
        "--text-kind", TEXT_KIND,
        "--native-table-strategy", NATIVE_TABLE_STRATEGY,
    ]
    if CORPUS_ROOT is not None:
        command.extend(["--corpus-root", str(local_path(CORPUS_ROOT))])
    if TESSERACT_CMD is not None:
        command.extend(["--tesseract-cmd", str(TESSERACT_CMD)])
    if TESSDATA_DIR is not None:
        command.extend(["--tessdata-dir", str(local_path(TESSDATA_DIR))])
    if TOKENIZER_PATH is not None:
        command.extend(["--tokenizer-path", str(local_path(TOKENIZER_PATH))])
    if MAX_CHUNK_TOKENS is not None:
        if TOKENIZER_PATH is None:
            raise ValueError("Для MAX_CHUNK_TOKENS сначала задайте TOKENIZER_PATH")
        command.extend(["--max-chunk-tokens", str(MAX_CHUNK_TOKENS)])

    if not OCR_ENABLED:
        command.append("--no-ocr")
    if not RASTER_TABLES:
        command.append("--no-raster-tables")
    if FULL_PAGE_OCR:
        command.append("--full-page-ocr")
    if not JOIN_WRAPPED_LINES:
        command.append("--no-join-wrapped-lines")
    if not REMOVE_BOILERPLATE:
        command.append("--keep-boilerplate")
    if not FILTER_IMAGE_TEXT:
        command.append("--keep-noisy-image-text")
    if INCLUDE_PARTIAL:
        command.append("--include-partial")
    return command


def main() -> int:
    try:
        command = build_command()
        print(f"Пайплайн: {BASE_DIR / 'document_pipeline.py'}", flush=True)
        print(f"Вход: {local_path(INPUT_PATH)}", flush=True)
        print(f"Результаты: {local_path(OUTPUT_DIR)}", flush=True)
        result = subprocess.run(command, cwd=BASE_DIR, check=False)
        return result.returncode
    except KeyboardInterrupt:
        print("Обработка остановлена. Незавершённый запуск не используйте для индексации.", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"Ошибка запуска: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
