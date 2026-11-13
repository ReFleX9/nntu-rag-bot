#!/usr/bin/env python3
"""Select student/applicant sources for the next index and report every exclusion."""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import unquote, urlsplit

MEDIA = ("/nomera/", "/politehnik/", "/radio+/", "/smi-o-", "/news/")
UNRELATED = ("/vakansii/", "/nauka/", "/prioritet2030/", "/upr_person/",
             "/kadr-rezerv/", "/blanks/", "/ottiis/", "/zakupki/")
NORMATIVE = ("/norm_docs_ngtu/", "/sveden/document", "/studentam/", "/abiturientam/",
             "/studgorodok/docs/", "/obrazovanie/", "/uchebno-metodicheskoe-upravlenie")


def reason(row: dict) -> str | None:
    if not row.get("indexable") or row.get("indexing_exclusions"):
        return "not_indexable"
    if len(row.get("text") or "") < 80:
        return "too_short"
    url = unquote((row.get("source_urls") or [""])[0]).lower()
    path = urlsplit(url).path
    if row.get("source_type") == "html" and path in {"", "/"}:
        return "home_navigation"
    if row.get("source_type") == "html" and row.get("document_length_chars", 0) > 100_000:
        return "oversized_directory_separate_index"
    if any(x in path for x in MEDIA):
        return "media_archive"
    if any(x in path for x in UNRELATED):
        return "unrelated_section"
    if "/otsenka_kach_od/vnutr_sistema_ok_obr/rezultaty/" in path:
        return "survey_reports_separate_index"
    if any(x in path for x in ("/security/gochs/posobiya/", "/security/antinark/")):
        return "general_safety_manuals"
    if "/universitet/security/" in path:
        return "general_security_archive"
    if "/otsenka_kach_od/nok_obr/" in path:
        return "quality_monitoring_reports"
    if any(x in path for x in ("plan_vosp_rab", "calendarn_plan_vosp_rab", "/plan_year/")):
        return "annual_plans_separate_index"
    admission_year = re.search(r"/abiturientam/(20\d\d)/", path)
    if admission_year and int(admission_year.group(1)) < datetime.date.today().year:
        return "archived_admissions_year"
    category = row.get("category") or ""
    if category.startswith("03_"):
        return "program_materials_separate_index"
    if category.startswith(("00_", "05_", "06_")):
        return "outside_audience"
    if category.startswith("99_"):
        return "unclassified"
    if category.startswith("08_") and not any(x in path for x in NORMATIVE):
        return "unrelated_administration"
    if category.startswith("07_") and row.get("source_type") != "html" and not any(x in path for x in NORMATIVE):
        return "unrelated_university_file"
    if row.get("is_ocr") and row.get("ocr_confidence") is not None and row["ocr_confidence"] < 60:
        return "low_ocr_confidence"
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; use a new path to preserve the previous corpus")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix + ".partial")
    counts, examples, by_category = Counter(), defaultdict(list), Counter()
    seen_text = set()
    with args.source.open(encoding="utf-8") as source, temp.open("w", encoding="utf-8", newline="\n") as output:
        for line in source:
            row = json.loads(line)
            why = reason(row)
            if not why:
                digest = row.get("content_sha256")
                if digest and digest in seen_text:
                    why = "duplicate_text"
                elif digest:
                    seen_text.add(digest)
            counts[why or "selected"] += 1
            if why:
                if len(examples[why]) < 5:
                    examples[why].append({"title": row.get("title"), "url": (row.get("source_urls") or [""])[0]})
                continue
            by_category[row.get("category")] += 1
            output.write(line)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temp, args.output)
    report = {"counts": counts, "selected_by_category": by_category, "excluded_examples": examples}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"counts": counts, "selected_by_category": by_category}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
