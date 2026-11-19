"""Annotate reviewed section ranges and reconcile excluded chunks by category."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

# Boundaries reviewed in extracted text. Skip mixed boundary chunks and appendices.
SECTIONS = {
    "Pravila_priema.pdf": [
        (11, 19, "III. Вступительные испытания"),
        (21, 24, "IV. Учёт индивидуальных достижений поступающих"),
        (26, 37, "V. Особые права при приёме"),
        (38, 51, "VI. Приём заявлений и документов"),
        (52, 58, "VII. Списки подавших заявление и конкурсные списки"),
        (59, 64, "VIII. Зачисление"),
        (66, 70, "IX. Зачисление на места в рамках контрольных цифр приёма"),
        (72, 73, "X. Информирование о приёме"),
        (75, 76, "XI. Сроки приёма"),
        (78, 81, "XII. Вступительные испытания для инвалидов и лиц с ОВЗ"),
        (83, 85, "XIII. Приём на целевое обучение"),
        (86, 87, "XIV. Дополнительный приём"),
        (89, 91, "XV. Приём иностранных граждан и лиц без гражданства"),
    ],
    "polog-porjdok-perevod-otchisl-vosstanovl.pdf": [
        (5, 6, "1. Область применения"),
        (7, 7, "2. Нормативные ссылки"),
        (15, 24, "5. Порядок и основания перевода обучающихся"),
        (25, 25, "5.11. Перевод внутри университета"),
        (26, 29, "5. Порядок и основания перевода обучающихся"),
        (31, 47, "6. Порядок отчисления обучающихся"),
        (49, 49, "7. Восстановление"),
    ],
    "polozhenie_stipendii.pdf": [
        (2, 2, "2. Нормативные ссылки"),
        (4, 5, "4. Общие положения"),
    ],
}


def records(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=Path("work"))
    args = parser.parse_args()
    work = args.work
    output = work / "v3_candidate_r3"
    output.mkdir()
    baseline_ids = {row["chunk_id"] for row in records(work / "v2/core_relinked.jsonl")}
    core_ids = set()
    sections = Counter()
    with (output / "core.jsonl").open("w", encoding="utf-8") as stream:
        for row in records(work / "v3_candidate_r2/core.jsonl"):
            core_ids.add(row["chunk_id"])
            if not row.get("section_title") and not row.get("section_path"):
                for start, end, title in SECTIONS.get(row["title"], ()):
                    if start <= row["chunk_index"] <= end:
                        row["section_title"] = title
                        row["section_path"] = [title]
                        sections[row["title"]] += 1
                        break
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts = defaultdict(Counter)
    documents = defaultdict(set)
    retained_docs = set()
    for row in records(work / "v2/chunks.jsonl"):
        category = row["category"]
        identifier = row["chunk_id"]
        if identifier in core_ids:
            group = "core_r3"
            retained_docs.add(row["document_id"])
        elif identifier in baseline_ids:
            group = "supplemental_v3"
        else:
            group = "excluded_v2"
        counts[category][group] += 1
        documents[category, group].add(row["document_id"])

    report = {
        "annotated_chunks_by_title": sections,
        "categories": {
            category: {
                group: {
                    "chunks": counts[category][group],
                    "documents_with_chunks": len(documents[category, group]),
                    "documents_absent_from_core_r3": len(documents[category, group] - retained_docs),
                }
                for group in ("core_r3", "supplemental_v3", "excluded_v2")
            }
            for category in sorted(counts)
        },
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
