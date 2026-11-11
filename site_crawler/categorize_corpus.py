#!/usr/bin/env python3
"""Build a provisional topical view of the NNTU crawl without moving raw files.

Each category directory contains NTFS hard links to source files, so this is a
reversible browsing view and does not duplicate the corpus bytes. Classification
uses crawl metadata only; low-confidence records are intentionally quarantined.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path


RULES = {
    "01_поступающим": [
        "абитури", "abitur", "приемная комиссия", "приёмная комиссия", "поступлен",
        "поступающ", "поступление", "вступительн", "ege", "егэ", "целевая квота",
        "прием 202", "приём 202", "день открытых дверей", "прием документов",
    ],
    "02_студентам": [
        "studentam", "студент", "расписан", "raspisan", "общежит", "стипенд",
        "академическ", "социальная поддерж", "эиос", "eios", "moodle",
        "электронная образовательная среда", "дистанционн", "сессия", "пересдач",
        "студгород", "профком", "практик", "личный кабинет студента",
    ],
    "03_образовательные_программы": [
        "учебный план", "учебн_plan", "ucheb_plan", "fgos", "опво", "opvo",
        "rpp_", "рабочая программа", "программа гиа", "программы гиа", "gia_",
        "/sveden/", "образовательн", "направлени подготовки", "календарный учебный",
        "практик_", "дисциплин",
    ],
    "04_кампусная_жизнь": [
        "жизнь", "культур", "спорт", "студенческ", "клуб", "отряд", "волонтер",
        "волонтёр", "мероприят", "творческ", "здоров", "сок-", "лагер",
    ],
    "05_преподавателям_и_сотрудникам": [
        "преподавател", "сотрудник", "работник", "кадров", "персонал",
        "ваканси", "оплате труда", "сотрудникам", "преподавателям", "должност",
        "справочник", "аттестаци", "повышени квалификаци",
    ],
    "06_наука_и_публикации": [
        "наука", "nauka", "исследован", "научн", "журнал", "conference",
        "конференц", "патент", "полезн_модел", "изобретен", "trudy-ngtu",
        "труды нгту", "диссертац", "аспирант",
    ],
    "07_об_университете": [
        "об университете", "университет", "university", "история нгту", "история университета",
        "структура университета", "руководство университета", "контакт", "лиценз",
        "аккредитац", "устав", "символик", "корпус", "реквизит",
    ],
    "08_нормативные_и_административные": [
        "приказ", "положение", "регламент", "правила", "нормативн", "федеральн закон",
        "законодательств", "договор", "заявлени", "бланк", "локальный акт",
    ],
}

IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp", "image/svg+xml"}
PRIORITY = list(RULES)


def classify(rows: list[dict]) -> dict:
    mime = (rows[0].get("content_type") or "").casefold()
    if mime.startswith("image/"):
        return {"category": "00_медиа_не_для_текста", "confidence": "high", "signals": ["image MIME"], "secondary": []}
    text = "\n".join(" ".join(str(row.get(k) or "") for k in (
        "url", "filename", "source_page", "link_text", "link_context", "topics"
    )) for row in rows).casefold().replace("ё", "е")
    scores = {name: [term for term in terms if term.casefold().replace("ё", "е") in text]
              for name, terms in RULES.items()}
    ranked = sorted(((len(hits), PRIORITY.index(name), name, hits)
                     for name, hits in scores.items() if hits), reverse=True)
    if not ranked:
        return {"category": "99_неопределено_на_проверку", "confidence": "low", "signals": [], "secondary": []}
    top_score, _, category, signals = ranked[0]
    confidence = "high" if top_score >= 3 else "medium" if top_score == 2 else "low"
    secondary = [name for score, _, name, _ in ranked[1:] if score and score >= top_score - 1]
    return {"category": category, "confidence": confidence,
            "signals": sorted(set(signals)), "secondary": secondary}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", nargs="?", default="data/nntu_site")
    parser.add_argument("--materialize-links", action="store_true",
                        help="создать обратимые hard-link папки organized_view")
    args = parser.parse_args()
    root = Path(args.corpus).resolve()
    view = root / "organized_view"
    grouped: dict[str, list[dict]] = defaultdict(list)
    for line in (root / "files.jsonl").open(encoding="utf-8"):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("stored_path"):
            source_path = (root / row["stored_path"]).resolve()
            # Collapse Windows path aliases that differ only by letter case.
            grouped[os.path.normcase(str(source_path))].append(row)

    records, counts, confidence_counts = [], Counter(), defaultdict(Counter)
    for _, rows in grouped.items():
        rel = next((row["stored_path"] for row in rows if (root / row["stored_path"]).is_file()), rows[0]["stored_path"])
        source = (root / rel).resolve()
        if not source.is_file():
            continue
        result = classify(rows)
        item = {
            "stored_path": rel,
            "filename": source.name,
            "size_bytes": source.stat().st_size,
            "sha256": next((x.get("sha256") for x in rows if x.get("sha256")), None),
            "category": result["category"],
            "confidence": result["confidence"],
            "signals": result["signals"],
            "secondary_categories": result["secondary"],
            "source_urls": sorted({x.get("url", "") for x in rows if x.get("url")}),
            "source_pages": sorted({x.get("source_page", "") for x in rows if x.get("source_page")}),
        }
        records.append(item)
        counts[item["category"]] += 1
        confidence_counts[item["category"]][item["confidence"]] += 1

    root.joinpath("classification.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8"
    )
    with root.joinpath("classification.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["category", "confidence", "filename", "size_bytes", "sha256", "stored_path", "signals", "secondary_categories", "source_urls"])
        writer.writeheader()
        for row in records:
            writer.writerow({
                "category": row["category"], "confidence": row["confidence"],
                "filename": row["filename"], "size_bytes": row["size_bytes"],
                "sha256": row["sha256"], "stored_path": row["stored_path"],
                "signals": "; ".join(row["signals"]),
                "secondary_categories": "; ".join(row["secondary_categories"]),
                "source_urls": " | ".join(row["source_urls"]),
            })

    if args.materialize_links:
        for row in records:
            suffix = Path(row["filename"]).suffix.lower()
            stable_name = (row.get("sha256") or re.sub(r"[^0-9a-f]", "", row["stored_path"].casefold()))[:24]
            destination = view / row["category"] / f"{stable_name}{suffix}"
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                try:
                    os.link(root / row["stored_path"], destination)
                except OSError as exc:
                    raise RuntimeError(f"Не удалось создать hard link {destination}: {exc}") from exc

    hardlink_count = sum(1 for p in view.rglob("*") if p.is_file()) if view.exists() else 0
    lines = ["# Предварительная тематическая группировка корпуса НГТУ", "",
             "Группировка построена по имени, URL, источнику, подписи и контексту ссылки; содержимое документов и OCR пока не анализировались.",
             "Папки `organized_view/` (если созданы) содержат hard links: они не копируют гигабайты и не меняют исходные файлы в `files/`.", "",
             f"Файлов с существующим путем в манифесте: {len(records):,}.", "",
             "| Категория | Файлов | Низкая уверенность | Средняя | Высокая |", "|---|---:|---:|---:|---:|"]
    if hardlink_count:
        lines.insert(5, f"Создано файловых записей-витрин: {hardlink_count:,}; одинаковое содержимое в одной категории сведено к одной ссылке.")
    for category, count in sorted(counts.items()):
        c = confidence_counts[category]
        lines.append(f"| `{category}` | {count:,} | {c['low']:,} | {c['medium']:,} | {c['high']:,} |")
    lines += ["", "Низкая уверенность означает, что найден только один слабый тематический сигнал; такие файлы следует проверить после извлечения текста.", ""]
    root.joinpath("classification_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
