"""Build a reviewable core and supplemental corpus from the frozen v2 selection."""
import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def route(document):
    title = document[0]["title"].casefold()
    if title == "osobennosy_priema_v_porydke_perevoda_v_ngtu_v_2023.pdf":
        return "special_transfer_2023"
    if title == "prikaz_stipendiya_covid19.pdf":
        return "historical_scholarship_covid"
    if re.match(r"^prikaz_o_naznach_stip.*(?:24-25|gor24)\.pdf$", title):
        return "historical_scholarship_award"
    if title in {
        "rasp_zanyatij_2_semestr_25-26.xlsx",
        "rasp_exam_2_semestr-4bak_25-26.pdf",
        "rasp_exam_2_semestr-2mag_25-26.pdf",
    }:
        return "schedule_2025_2026_second_semester"
    if title.startswith("anketa") and all(
        "/norm_doc_i_loc_akty/" in url.casefold()
        for row in document for url in row["source_urls"]
    ):
        return "survey_form" if any(row["source_urls"] for row in document) else "core"
    if title in {"защита диссертаций", "вакансии университета"}:
        return "specialized_directory"
    return "core"


def review_flags(row):
    flags = []
    if re.search(r"лист регистрации изменений|лист ознакомления", row["text"], re.I):
        flags.append("administrative_phrase")
    if row.get("is_ocr"):
        flags.append("ocr")
    if re.match(r"^\d{2}\.\d{2}\.\d{2}_20\d{2}", row["title"]):
        flags.append("program_code_and_year_in_title")
    return flags


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    with args.source.open("rb") as stream:
        source_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        stream.seek(0)
        rows = [json.loads(line) for line in stream]
    documents = defaultdict(list)
    for row in rows:
        documents[row["document_id"]].append(row)
    routes = {key: route(document) for key, document in documents.items()}

    args.output_dir.mkdir(parents=True, exist_ok=False)
    counts = Counter(routes.values())
    chunks = Counter()
    categories = defaultdict(Counter)
    flags = defaultdict(Counter)
    with (
        (args.output_dir / "core.jsonl").open("w", encoding="utf-8", newline="\n") as core,
        (args.output_dir / "supplemental.jsonl").open("w", encoding="utf-8", newline="\n") as supplemental,
        (args.output_dir / "document_inventory.jsonl").open("w", encoding="utf-8", newline="\n") as inventory,
    ):
        for key, document in documents.items():
            first = document[0]
            record = {"document_id": key, "title": first["title"],
                      "source_urls": sorted({url for row in document for url in row["source_urls"]}),
                      "category": first["category"], "chunks": len(document),
                      "route": routes[key]}
            inventory.write(json.dumps(record, ensure_ascii=False) + "\n")
        for row in rows:
            reason = routes[row["document_id"]]
            target = core if reason == "core" else supplemental
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
            chunks[reason] += 1
            categories[reason][row["category"]] += 1
            flags[reason].update(review_flags(row))

    files = {}
    for path in sorted(args.output_dir.glob("*.jsonl")):
        with path.open("rb") as stream:
            files[path.name] = hashlib.file_digest(stream, "sha256").hexdigest()
    report = {"source": str(args.source), "source_sha256": source_hash,
              "source_chunks": len(rows), "source_documents": len(documents),
              "chunks_by_route": chunks, "documents_by_route": counts,
              "categories_by_route": categories, "review_flags_by_route": flags,
              "output_sha256": files}
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
