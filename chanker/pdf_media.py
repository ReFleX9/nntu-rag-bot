"""Таблицы и изображения PDF для корпуса документации.

Обычные таблицы: PyMuPDF. Таблицы на картинках: img2table + Tesseract.
Текст изображений: pytesseract. Очистка: соседний text_cleaner.py.
Файл PDF на диске не изменяется. Страницы в результате начинаются с 1.
Координаты bbox — PDF points относительно видимой страницы (после поворота).
Интерфейс пайплайна: create_ocr_engine, on_page, ocr_engine, rag_word_indices.

CLI:
    python pdf_media.py corpus/rosa/manual.pdf --corpus-root corpus --output out

Это модуль подготовки данных, а не реализация HTTP API. Он не извлекает
аббревиатуры и не интерпретирует смысл схем, диаграмм или фотографий.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import re
import shutil
import subprocess
import sys
import unicodedata
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Callable

import pymupdf
from PIL import Image

from text_cleaner import clean_text

BBox = tuple[float, float, float, float]
MEDIA_API_VERSION = 2


@dataclass
class MediaConfig:
    languages: str = "rus+eng"
    dpi: int = 250
    ocr_enabled: bool = True
    raster_tables: bool = True
    borderless_tables: bool = False
    native_table_strategy: str = "lines_strict"
    full_page_ocr: bool = False
    image_ocr_timeout: float = 60.0
    max_render_pixels: int = 20_000_000
    table_min_confidence: int = 40
    tesseract_cmd: str | None = None
    tessdata_dir: str | None = None

    def validate(self) -> None:
        if self.dpi < 72 or self.max_render_pixels < 100_000:
            raise ValueError("Некорректные dpi или max_render_pixels")
        if self.image_ocr_timeout <= 0 or not 0 <= self.table_min_confidence <= 99:
            raise ValueError("Некорректные параметры OCR")
        if self.native_table_strategy not in {"lines", "lines_strict", "text"}:
            raise ValueError("Неизвестная стратегия поиска обычных таблиц")
        if not re.fullmatch(r"[A-Za-z0-9_]+(?:\+[A-Za-z0-9_]+)*", self.languages):
            raise ValueError("Языки задаются как rus+eng")


@dataclass
class TableData:
    table_id: str
    page: int
    bbox: BBox
    method: str
    raw_rows: list[list[str | None]]
    rows: list[list[str | None]]
    cells: list[dict[str, Any]]
    markdown: str
    header_names: list[str | None] = field(default_factory=list)
    header_bbox: BBox | None = None
    header_external: bool = False
    image_id: str | None = None
    duplicate_of: str | None = None


@dataclass
class ImageData:
    image_id: str
    page: int
    bbox: BBox
    image_path: str
    method: str
    effective_dpi: float
    ocr_text: str = ""
    text_clean: str = ""
    # В rag_text исключены подтверждённые совпадения с ячейками и PDF-слоем.
    rag_text: str = ""
    rag_word_indices: list[int] = field(default_factory=list)
    words: list[dict[str, Any]] = field(default_factory=list)
    mean_confidence: float | None = None
    status: str = "pending"
    table_status: str = "pending"
    errors: list[str] = field(default_factory=list)


@dataclass
class PageMedia:
    page: int
    width: float
    height: float
    original_rotation: int
    tables: list[TableData] = field(default_factory=list)
    images: list[ImageData] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class PDFMedia:
    document_id: str
    sha256: str
    page_count: int
    asset_directory: str
    config: dict[str, Any]
    versions: dict[str, str]
    pages: list[PageMedia] = field(default_factory=list)
    coordinate_system: str = "displayed_page_points"

    @property
    def has_errors(self) -> bool:
        return any(p.errors or any(i.errors for i in p.images) for p in self.pages)

    def save_json(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        payload["has_errors"] = self.has_errors
        # Сначала целиком записываем временный файл рядом с итоговым.
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                             encoding="utf-8")
        temporary.replace(target)


def _box(rect: Any) -> BBox:
    return tuple(round(float(value), 3) for value in rect)


def _area(rect: BBox) -> float:
    return max(0.0, rect[2] - rect[0]) * max(0.0, rect[3] - rect[1])


def _overlap(first: BBox, second: BBox) -> float:
    intersection = (
        max(first[0], second[0]), max(first[1], second[1]),
        min(first[2], second[2]), min(first[3], second[3]),
    )
    return _area(intersection) / max(1e-9, min(_area(first), _area(second)))


def _inside(word_box: BBox, region: BBox) -> bool:
    cx = (word_box[0] + word_box[2]) / 2
    cy = (word_box[1] + word_box[3]) / 2
    return region[0] <= cx <= region[2] and region[1] <= cy <= region[3]


def _token_key(text: str) -> str:
    return unicodedata.normalize("NFC", text).casefold().strip()


def _cell_text(value: str | None) -> str | None:
    return None if value is None else clean_text(str(value), kind="table").text


def _markdown(rows: list[list[str | None]], external_header: list[str | None]) -> str:
    """Не заполняем объединённые/пустые ячейки соседними значениями."""
    width = max([len(external_header)] + [len(row) for row in rows], default=0)
    if not width:
        return ""
    # Искусственные подписи не объявляют первую строку данных заголовком.
    header = external_header or [f"Столбец {i + 1}" for i in range(width)]

    def render(row: list[str | None]) -> str:
        padded = row + [None] * (width - len(row))
        values = [(value or "").replace("\\", "\\\\").replace("|", "\\|")
                  .replace("\n", "<br>") for value in padded]
        return "| " + " | ".join(values) + " |"

    return "\n".join([render(header), render(["---"] * width)]
                      + [render(row) for row in rows])


def _resolve_tesseract(command: str | None) -> Path:
    """Явный путь/имя, затем PATH, затем стандартные установки Windows."""
    if command:
        command = os.path.expandvars(command.strip().strip('"'))
        candidate = Path(command).expanduser()
        if candidate.is_dir():
            raise RuntimeError(f"Указана папка {candidate}. Нужен полный путь к tesseract.exe")
        found = str(candidate) if candidate.is_file() else shutil.which(command)
        if not found:
            raise RuntimeError(f"Не найден указанный Tesseract: {command}. Проверьте --tesseract-cmd")
    else:
        found = shutil.which("tesseract")
        if not found and sys.platform == "win32":
            candidates = [
                Path(os.environ.get("ProgramFiles", r"C:\Program Files"))/"Tesseract-OCR/tesseract.exe",
                Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))/"Tesseract-OCR/tesseract.exe",
            ]
            if os.environ.get("LOCALAPPDATA"):
                local = Path(os.environ["LOCALAPPDATA"])
                candidates += [local/"Programs/Tesseract-OCR/tesseract.exe", local/"Tesseract-OCR/tesseract.exe"]
            found = next((str(path) for path in candidates if path.is_file()), None)
        if not found:
            raise RuntimeError(
                "Tesseract не найден текущим Python-процессом. Укажите --tesseract-cmd "
                '"C:\\Program Files\\Tesseract-OCR\\tesseract.exe" или свой путь установки. '
                "После изменения PATH перезапустите терминал/IDE."
            )
    executable = Path(found).resolve()
    if executable.name.lower() not in {"tesseract", "tesseract.exe"}:
        raise ValueError("img2table ожидает executable с именем tesseract или tesseract.exe")
    return executable


def _tesseract_info(executable: Path, option: str) -> str:
    """Прямой запуск сохраняет исходную причину ошибки, включая ошибки ОС."""
    try:
        completed = subprocess.run([str(executable), option], capture_output=True,
                                   timeout=15, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"Не удалось выполнить {executable} {option}: "
                           f"{type(error).__name__}: {error}") from error
    output = (completed.stdout + b"\n" + completed.stderr).decode("utf-8", errors="replace").strip()
    if completed.returncode:
        raise RuntimeError(f"Tesseract {option} завершился с кодом {completed.returncode}. "
                           f"Путь: {executable}. Вывод: {output[:2000] or '(пусто)'}")
    return output


class _OCREngine:
    def __init__(self, config: MediaConfig):
        import pytesseract

        self.pytesseract = pytesseract
        self.config = config
        command = config.tesseract_cmd or os.environ.get("TESSERACT_CMD")
        existing = pytesseract.pytesseract.tesseract_cmd
        if not command and existing not in {"tesseract", "tesseract.exe"}:
            command = existing  # Учитываем и путь, ранее настроенный пользователем.
        executable = _resolve_tesseract(command)
        self.executable = str(executable)
        pytesseract.pytesseract.tesseract_cmd = self.executable
        # img2table вызывает tesseract по имени; ему нужен тот же каталог в PATH.
        paths = os.environ.get("PATH", "").split(os.pathsep)
        os.environ["PATH"] = os.pathsep.join([str(executable.parent)] +
                                            [path for path in paths if path != str(executable.parent)])
        if config.tessdata_dir:
            folder = Path(os.path.expandvars(config.tessdata_dir.strip().strip('"'))).expanduser().resolve()
            if not folder.is_dir():
                raise RuntimeError(f"Не найден каталог tessdata: {folder}")
            os.environ["TESSDATA_PREFIX"] = str(folder)
        elif not os.environ.get("TESSDATA_PREFIX") and (executable.parent/"tessdata").is_dir():
            os.environ["TESSDATA_PREFIX"] = str(executable.parent/"tessdata")
        self.version = _tesseract_info(executable, "--version").splitlines()[0]
        language_output = _tesseract_info(executable, "--list-langs")
        available = {line.strip() for line in language_output.splitlines()
                     if re.fullmatch(r"[A-Za-z0-9_]+(?:/[A-Za-z0-9_]+)*", line.strip())}
        self.available_languages = sorted(available)
        self.tessdata_dir = os.environ.get("TESSDATA_PREFIX")
        missing = set(config.languages.split("+")) - available
        if missing:
            raise RuntimeError(
                f"Tesseract запускается: {executable}. Недоступны языки: {', '.join(sorted(missing))}. "
                f"Доступны: {', '.join(sorted(available)) or '(нет)'}. "
                f"TESSDATA_PREFIX={self.tessdata_dir or '(не задан)'}. "
                "Укажите --tessdata-dir с нужными .traineddata либо установите языковые данные. "
                f"Вывод --list-langs: {language_output[:2000]}"
            )

        self.table_ocr = None
        if config.raster_tables:
            from img2table.ocr import TesseractOCR
            try:
                self.table_ocr = TesseractOCR(n_threads=1, lang=config.languages, psm=11,
                                             tessdata_dir=self.tessdata_dir)
            except Exception as error:
                raise RuntimeError(
                    f"Tesseract по пути {executable} запускается, но img2table не инициализировался: "
                    f"{type(error).__name__}: {error}. "
                    "Для проверки остального OCR можно запустить --no-raster-tables."
                ) from error

    def words(self, image: Image.Image) -> list[dict[str, Any]]:
        data = self.pytesseract.image_to_data(
            image, lang=self.config.languages, config="--psm 11",
            output_type=self.pytesseract.Output.DICT,
            timeout=self.config.image_ocr_timeout,
        )
        result = []
        for i, value in enumerate(data["text"]):
            value = str(value).strip()
            if int(data["level"][i]) != 5 or not value:
                continue
            confidence = float(data["conf"][i])
            x, y = int(data["left"][i]), int(data["top"][i])
            result.append({
                "text": value,
                "bbox": (x, y, x + int(data["width"][i]), y + int(data["height"][i])),
                "confidence": confidence,
                "line": (int(data["block_num"][i]), int(data["par_num"][i]), int(data["line_num"][i])),
            })
        return result

    def tables(self, image_path: Path):
        from img2table.document import Image as TableImage
        document = TableImage(src=str(image_path), detect_rotation=False)
        return document.extract_tables(
            ocr=self.table_ocr,
            implicit_rows=False,
            implicit_columns=False,
            borderless_tables=self.config.borderless_tables,
            min_confidence=self.config.table_min_confidence,
        )


def _words_to_text(words: list[dict[str, Any]]) -> str:
    lines: dict[tuple, list[str]] = {}
    for word in words:
        lines.setdefault(tuple(word["line"]), []).append(word["text"])
    return "\n".join(" ".join(values) for values in lines.values())


def _native_tables(page: pymupdf.Page, strategy: str) -> list[TableData]:
    result = []
    finder = page.find_tables(strategy=strategy)
    for number, table in enumerate(finder.tables, 1):
        raw_rows = table.extract()
        if not any(value and value.strip() for row in raw_rows for value in row):
            continue
        rows = [[_cell_text(value) for value in row] for row in raw_rows]
        cells = []
        for r, table_row in enumerate(table.rows):
            for c, rect in enumerate(table_row.cells):
                value = raw_rows[r][c] if c < len(raw_rows[r]) else None
                cells.append({"row": r, "column": c,
                              "bbox": _box(rect) if rect is not None else None,
                              "raw_value": value, "value": _cell_text(value)})
        header = table.header
        names = [str(value) if value is not None else None for value in header.names]
        result.append(TableData(
            table_id=f"p{page.number + 1:04d}_table_{number:03d}",
            page=page.number + 1, bbox=_box(table.bbox), method="pdf_text",
            raw_rows=raw_rows, rows=rows, cells=cells,
            markdown=_markdown(rows, names if header.external else []),
            header_names=names, header_bbox=_box(header.bbox) if header.bbox else None,
            header_external=bool(header.external),
        ))
    # Объекты Table не переживают страницу: выше всё скопировано в обычные данные.
    return result


def _render(page: pymupdf.Page, region: BBox, config: MediaConfig):
    clip = pymupdf.Rect(region) & page.rect
    if clip.is_empty or clip.is_infinite:
        raise ValueError("Пустая или некорректная область изображения")
    scale = min(config.dpi / 72.0, math.sqrt(config.max_render_pixels / clip.get_area()))
    pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), clip=clip,
                          colorspace=pymupdf.csRGB, alpha=False, annots=False)
    image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    # Pixmap округляет границы в пикселях. Учитываем его действительное начало.
    origin_x, origin_y = pix.x / scale, pix.y / scale

    def to_page(box) -> BBox:
        return _box((origin_x + box[0] / scale, origin_y + box[1] / scale,
                     origin_x + box[2] / scale, origin_y + box[3] / scale))

    return image, to_page, scale * 72


def _raster_table(table, image: ImageData, number: int, to_page) -> TableData:
    raw_rows, cells = [], []
    for r, row in enumerate(table.content.values()):
        values = []
        for c, cell in enumerate(row):
            value = None if cell.value is None else str(cell.value)
            values.append(value)
            box = cell.bbox
            cells.append({"row": r, "column": c,
                          "bbox": to_page((box.x1, box.y1, box.x2, box.y2)),
                          "raw_value": value, "value": _cell_text(value)})
        raw_rows.append(values)
    rows = [[_cell_text(value) for value in row] for row in raw_rows]
    box = table.bbox
    return TableData(
        table_id=f"{image.image_id}_table_{number:03d}", page=image.page,
        bbox=to_page((box.x1, box.y1, box.x2, box.y2)), method="image_ocr",
        raw_rows=raw_rows, rows=rows, cells=cells, markdown=_markdown(rows, []),
        image_id=image.image_id,
    )


def _table_key(table: TableData) -> str:
    return " ".join(_token_key(value) for row in table.rows for value in row if value).strip()


def _filter_ocr_for_rag(page_result: PageMedia, native_words: list) -> None:
    represented_regions = []
    for table in page_result.tables:
        if not table.duplicate_of:
            for cell in table.cells:
                if cell["bbox"] and cell["raw_value"]:
                    tokens = {_token_key(value) for value in cell["raw_value"].split()}
                    represented_regions.append((cell["bbox"], tokens))
            if table.header_external and table.header_bbox:
                tokens = {_token_key(value) for name in table.header_names
                          if name for value in name.split()}
                represented_regions.append((table.header_bbox, tokens))
    native_index = defaultdict(list)
    for word in native_words:
        native_index[_token_key(word[4])].append(_box(word[:4]))
    for image in page_result.images:
        selected = []
        image.rag_word_indices = []
        for word_index, word in enumerate(image.words):
            # Не удаляем весь прямоугольник таблицы: её OCR мог пропустить слово.
            if any(_inside(word["bbox"], region) and _token_key(word["text"]) in tokens
                   for region, tokens in represented_regions):
                continue
            if any(_overlap(word["bbox"], region) >= 0.5
                   for region in native_index.get(_token_key(word["text"]), [])):
                continue
            selected.append(word)
            image.rag_word_indices.append(word_index)
        image.rag_text = clean_text(_words_to_text(selected), kind="unknown").text


def create_ocr_engine(config: MediaConfig) -> _OCREngine | None:
    """Проверить окружение и подготовить OCR один раз для последовательной пачки."""
    config.validate()
    return _OCREngine(config) if config.ocr_enabled else None


def extract_pdf_media(
    pdf_path: str | Path,
    *,
    output_dir: str | Path,
    corpus_root: str | Path | None = None,
    document_id: str | None = None,
    config: MediaConfig | None = None,
    on_page: Callable[[pymupdf.Page, PageMedia], None] | None = None,
    ocr_engine: Any = None,
) -> PDFMedia:
    """Обработать все страницы одного PDF. Ошибки сохраняются явно.

    Задайте corpus_root, чтобы получить правильный относительный document_id,
    либо передайте document_id явно (например, для загруженного файла).
    Не смешивайте разные настройки Tesseract в потоках одного процесса.
    on_page вызывается после извлечения media, пока нормализованная страница
    ещё открыта. Ошибка обработчика записывается как ошибка страницы.
    """
    config = config or MediaConfig()
    config.validate()
    source = Path(pdf_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if document_id is None:
        if corpus_root is None:
            raise ValueError("Задайте corpus_root или document_id")
        document_id = source.relative_to(Path(corpus_root).expanduser().resolve()).as_posix()
    parsed_id = PurePosixPath(document_id)
    if (not parsed_id.name or parsed_id.suffix.lower() != ".pdf"
            or parsed_id.is_absolute() or ".." in parsed_id.parts or "\\" in document_id):
        raise ValueError("document_id должен быть относительным POSIX-путём")

    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(data)
    checksum = digest.hexdigest()
    identity = hashlib.sha256(document_id.encode("utf-8")).hexdigest()[:8]
    asset_directory = Path(output_dir).expanduser().resolve() / f"{checksum[:12]}-{identity}"
    asset_directory.mkdir(parents=True, exist_ok=True)
    if ocr_engine is not None and ocr_engine.config != config:
        raise ValueError("Настройки повторно используемого OCR не совпадают")
    engine = ocr_engine if ocr_engine is not None else create_ocr_engine(config)
    versions = {"pymupdf": pymupdf.VersionBind}
    for package in ("pytesseract", "img2table", "Pillow"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    if engine:
        versions["tesseract"] = engine.version

    with pymupdf.open(source) as pdf:
        if not pdf.is_pdf or pdf.needs_pass:
            raise ValueError("Нужен PDF без пароля")
        result = PDFMedia(document_id, checksum, len(pdf), str(asset_directory),
                          asdict(config), versions)
        for index in range(len(pdf)):
            page_result = PageMedia(index + 1, 0, 0, 0)
            result.pages.append(page_result)
            try:
                page = pdf[index]
                page_result.original_rotation = page.rotation
                if page.rotation:
                    # Только в памяти: вид страницы сохраняется, оси унифицируются.
                    page.remove_rotation()
                page_result.width, page_result.height = page.rect.width, page.rect.height
                native_words = page.get_text("words", sort=True)
                try:
                    page_result.tables.extend(_native_tables(page, config.native_table_strategy))
                except Exception as error:
                    page_result.errors.append(f"native_tables: {type(error).__name__}: {error}")

                image_info = page.get_image_info()
                # Если текст нарисован контурами или скан разбит на плитки,
                # рендер целой страницы сохраняет общую структуру таблиц.
                full_page = config.full_page_ocr or (
                    not native_words and bool(image_info or page.get_drawings())
                )
                if full_page:
                    regions = [(_box(page.rect), "page_render")]
                else:
                    regions = []
                    seen = set()
                    for item in image_info:
                        rect = pymupdf.Rect(item["bbox"]) & page.rect
                        if rect.is_empty or rect.is_infinite:
                            continue
                        box = _box(rect)
                        if box not in seen:
                            seen.add(box)
                            regions.append((box, "image_region"))

                for image_number, (region, method) in enumerate(regions, 1):
                    image_id = f"p{index + 1:04d}_image_{image_number:03d}"
                    path = asset_directory / f"{image_id}.png"
                    entry = ImageData(image_id, index + 1, region, str(path), method, 0)
                    page_result.images.append(entry)
                    try:
                        rendered, to_page, entry.effective_dpi = _render(page, region, config)
                        try:
                            rendered.save(path, format="PNG")
                        except Exception:
                            rendered.close()
                            raise
                        if engine is None:
                            entry.status, entry.table_status = "ocr_disabled", "ocr_disabled"
                            rendered.close()
                            continue
                        try:
                            pixel_words = engine.words(rendered)
                            entry.words = [dict(word, bbox=to_page(word["bbox"])) for word in pixel_words]
                            entry.ocr_text = _words_to_text(entry.words)
                            entry.text_clean = clean_text(entry.ocr_text, kind="unknown").text
                            entry.mean_confidence = (
                                sum(word["confidence"] for word in entry.words) / len(entry.words)
                                if entry.words else None
                            )
                            entry.status = "ok" if entry.words else "no_text_detected"
                        except Exception as error:
                            entry.status = "error"
                            entry.errors.append(f"image_ocr: {type(error).__name__}: {error}")
                        finally:
                            rendered.close()

                        if not config.raster_tables:
                            entry.table_status = "disabled"
                        else:
                            try:
                                for number, table in enumerate(engine.tables(path), 1):
                                    extracted = _raster_table(table, entry, number, to_page)
                                    for previous in page_result.tables:
                                        if (_overlap(extracted.bbox, previous.bbox) >= 0.9
                                                and _table_key(extracted) == _table_key(previous)):
                                            extracted.duplicate_of = previous.table_id
                                            break
                                    page_result.tables.append(extracted)
                                entry.table_status = "ok"
                            except Exception as error:
                                entry.table_status = "error"
                                entry.errors.append(f"raster_tables: {type(error).__name__}: {error}")
                    except Exception as error:
                        entry.status = "error"
                        entry.errors.append(f"render: {type(error).__name__}: {error}")
                _filter_ocr_for_rag(page_result, native_words)
                if on_page is not None:
                    on_page(page, page_result)
            except Exception as error:
                page_result.errors.append(f"page: {type(error).__name__}: {error}")
    return result


def to_langchain_documents(result: PDFMedia):
    """Только таблицы и новый текст изображений; обычный текст PDF добавляется отдельно."""
    from langchain_core.documents import Document

    documents = []
    for page in result.pages:
        base = {"document_id": result.document_id, "page": page.page,
                "coordinate_system": result.coordinate_system}
        for table in page.tables:
            if table.duplicate_of or not table.markdown:
                continue
            documents.append(Document(page_content=table.markdown, metadata={
                **base, "block_id": table.table_id, "kind": "table", "bbox": table.bbox,
                "extraction_method": table.method,
            }))
        for image in page.images:
            if not image.rag_text.strip():
                continue
            documents.append(Document(page_content=image.rag_text, metadata={
                **base, "block_id": image.image_id, "kind": "image_text", "bbox": image.bbox,
                "image_path": image.image_path, "extraction_method": "tesseract",
            }))
    return documents


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("processed/media"))
    parser.add_argument("--languages", default="rus+eng")
    parser.add_argument("--tesseract-cmd")
    parser.add_argument("--tessdata-dir")
    parser.add_argument("--dpi", type=int, default=250)
    parser.add_argument("--no-ocr", action="store_true")
    parser.add_argument("--no-raster-tables", action="store_true")
    parser.add_argument("--borderless-tables", action="store_true")
    parser.add_argument("--full-page-ocr", action="store_true")
    parser.add_argument("--native-table-strategy", choices=["lines", "lines_strict", "text"], default="lines_strict")
    args = parser.parse_args()
    try:
        result = extract_pdf_media(
            args.pdf, corpus_root=args.corpus_root, output_dir=args.output,
            config=MediaConfig(languages=args.languages, dpi=args.dpi,
                ocr_enabled=not args.no_ocr, raster_tables=not args.no_raster_tables,
                borderless_tables=args.borderless_tables,
                native_table_strategy=args.native_table_strategy,
                full_page_ocr=args.full_page_ocr, tesseract_cmd=args.tesseract_cmd,
                tessdata_dir=args.tessdata_dir),
        )
        target = Path(result.asset_directory) / "media.json"
        result.save_json(target)
        print(json.dumps({"document_id": result.document_id, "pages": result.page_count,
                          "tables": sum(len(p.tables) for p in result.pages),
                          "images": sum(len(p.images) for p in result.pages),
                          "has_errors": result.has_errors, "result": str(target)},
                         ensure_ascii=False, indent=2))
        return 2 if result.has_errors else 0
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
