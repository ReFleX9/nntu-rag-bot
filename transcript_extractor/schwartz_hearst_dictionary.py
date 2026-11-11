"""
python .\swartz\schwartz_hearst_dictionary.py `
  --dictionary .\glasary\dictionary.csv `
  --abbreviations .\glasary\abbreviations.csv `
  --chunks .\glasary\chunks.csv `
  --documents .\glasary\documents.csv `
  --output-dictionary .\glasary\dictionary_schwartz_hearst.csv `
  --output-pairs .\glasary\schwartz_hearst_pairs.csv `
  --output-summary .\glasary\schwartz_hearst_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

# The NER export contains large multi-line contexts.
try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2_147_483_647)

WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+")
PAREN_RE = re.compile(r"\(([^()\n]{1,120})\)")
PRECEDING_SHORT_RE = re.compile(
    r"([A-Za-zА-Яа-яЁё0-9][A-Za-zА-Яа-яЁё0-9.+/_-]{1,19})\s*$"
)
MARKER_RE = re.compile(
    r"^(?:"
    r"далее(?:\s+по\s+тексту|\s+в\s+тексте)?"
    r"|hereinafter(?:\s+referred\s+to\s+as)?"
    r")\s*[-—–:]*\s*",
    re.IGNORECASE,
)

# Conservative stop list only for filtering obvious code/noise.
CODE_WORDS = {
    "select", "insert", "update", "delete", "values", "from", "where", "table",
    "varchar", "null", "return", "def", "class", "import", "const", "let", "var",
}
BAD_LF_RE = re.compile(
    r"https?://|www\.|\\\\|[<>{}=\[\]]|(?:\.[A-Za-z0-9]{1,4}\b)",
    re.IGNORECASE,
)


def normalize_abbreviation(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "").strip()
    value = re.sub(r"\s+", " ", value)
    return value.casefold()


def normalize_expansion_key(value: str) -> str:
    """Normalize only punctuation/case, not morphology or semantics."""
    value = unicodedata.normalize("NFKC", value or "").casefold()
    return " ".join(WORD_RE.findall(value))


def clean_parenthesized_short(content: str) -> tuple[str, bool]:
    raw = unicodedata.normalize("NFKC", content).strip()
    marker = bool(MARKER_RE.match(raw))
    short = MARKER_RE.sub("", raw).strip(" \t\r\n,;:.")
    return short, marker


def short_form_chars(short: str) -> list[str]:
    return [c.lower() for c in short if c.isalnum()]


def is_valid_short(short: str, allowed: set[str]) -> bool:
    short = short.strip()
    if not (2 <= len(short) <= 20) or " " in short:
        return False
    if not re.fullmatch(
        r"[A-Za-zА-Яа-яЁё0-9][A-Za-zА-Яа-яЁё0-9.+/_-]*", short
    ):
        return False
    if normalize_abbreviation(short) not in allowed:
        return False

    letters = [c for c in short if c.isalpha()]
    if len(letters) < 2:
        return False

    # High precision is more important here than recall:
    # ordinary lowercase words from NER should not become "confirmed" abbreviations.
    if sum(c.isupper() for c in letters) < 2:
        return False

    return True


def is_valid_long(expansion: str, short: str) -> bool:
    if not expansion or len(expansion) > 240:
        return False
    if BAD_LF_RE.search(expansion):
        return False

    words = WORD_RE.findall(expansion)
    if len(words) < 2:
        return False

    # Very long glued PDF/OCR tokens are almost always extraction noise.
    if any(len(word) > 36 for word in words):
        return False

    # A one-letter token is allowed only for common function words.
    one_letter_ok = {"a", "i", "v", "k", "s", "и", "в", "к", "с", "о", "у"}
    if any(len(word) == 1 and word.casefold() not in one_letter_ok for word in words):
        return False

    sf_key = normalize_abbreviation(short)
    if any(normalize_abbreviation(word) == sf_key for word in words):
        return False

    lower_words = {word.casefold() for word in words}
    if lower_words & CODE_WORDS:
        return False

    if re.search(r"\b(?:стр\.?|рисунок|таблица|example)\b", expansion, re.I):
        return False

    return True


def initial_word_match(
    short: str,
    long_candidate: str,
    *,
    anchored_end: bool,
    require_full_span: bool = False,
) -> str | None:
    """
    Conservative Schwartz-Hearst adaptation.

    Instead of allowing a short-form character to match anywhere inside a word,
    this confirmation stage requires letters to align with word initials.
    This sharply reduces false positives in technical manuals and OCR text.

    anchored_end=True:
        final short-form character must match the final lexical word.
    anchored_end=False:
        used only for explicit '(далее — SF)' markers; a trailing product name
        may be ignored.
    """
    chars = short_form_chars(short)
    if len(chars) < 2:
        return None

    words = list(WORD_RE.finditer(long_candidate))
    if len(words) < 2:
        return None

    wi = len(words) - 1
    if anchored_end:
        if words[wi].group(0)[0].lower() != chars[-1]:
            return None
        last = wi
    else:
        while wi >= 0 and words[wi].group(0)[0].lower() != chars[-1]:
            wi -= 1
        if wi < 0:
            return None
        last = wi

    matched = [last]
    wi = last - 1
    for char in reversed(chars[:-1]):
        while wi >= 0 and words[wi].group(0)[0].lower() != char:
            wi -= 1
        if wi < 0:
            return None
        matched.append(wi)
        wi -= 1

    first = matched[-1]
    span_words = last - first + 1
    max_words = min(len(chars) + 5, len(chars) * 2)
    if span_words > max_words:
        return None

    if require_full_span and (first != 0 or last != len(words) - 1):
        return None

    start_pos = words[first].start()
    end_pos = words[last].end()

    # Keep a closing quote that directly belongs to the final term token.
    tail = long_candidate[end_pos:].strip()
    if tail and all(char in "»”’\"" for char in tail):
        end_pos = len(long_candidate.rstrip())

    expansion = long_candidate[start_pos:end_pos].strip(
        " \t\r\n,;:-—–\"'«"
    )

    if not is_valid_long(expansion, short):
        return None
    return expansion


def left_context_variants(
    text_before_parenthesis: str,
    short: str,
    explicit_marker: bool,
) -> list[tuple[str, bool]]:
    """
    Build a small set of plausible long-form candidates immediately to the left.

    For ordinary LF(SF), matching is anchored to the last lexical word.
    For explicit '(далее — SF)' markers, one non-anchored pass is allowed so
    product/version qualifiers after the generic term can be ignored.
    """
    left = text_before_parenthesis[-700:].rstrip()

    # Do not jump through a previous sentence / paragraph / closed parenthesis.
    boundaries = [
        match.end()
        for match in re.finditer(r"(?:[.!?;]\s+|\n\s*\n|\)\s+)", left)
    ]
    if boundaries:
        left = left[boundaries[-1] :].strip()

    chars = short_form_chars(short)
    max_words = (
        min(len(chars) + 7, len(chars) * 3 + 3)
        if explicit_marker
        else min(len(chars) + 5, len(chars) * 2 + 2)
    )
    tokens = list(re.finditer(r"\S+", left))
    if len(tokens) > max_words:
        left = left[tokens[-max_words].start() :]

    candidates: list[tuple[str, bool]] = [(left, not explicit_marker)]

    # Generic term followed by a quoted product name:
    # "прикладного программного обеспечения «Аврора Центр» (ППО)".
    for quote in ("«", '"'):
        index = left.rfind(quote)
        if index > 0:
            candidates.append((left[:index].rstrip(" ,:-—–"), True))

    # Bilingual definitions:
    # "Pluggable Authentication Modules – подключаемые модули ... (PAM)".
    for separator in (" — ", " – ", " - "):
        if separator in left:
            for part in left.split(separator):
                part = part.strip()
                if part:
                    candidates.append((part, True))

    if explicit_marker:
        candidates.append((left, False))

    seen: set[tuple[str, bool]] = set()
    result: list[tuple[str, bool]] = []
    for item in candidates:
        if item not in seen and item[0]:
            seen.add(item)
            result.append(item)
    return result


@dataclass
class Evidence:
    document_id: str
    title: str
    page: str
    chunk_id: str
    pattern: str
    context: str

    def key(self) -> tuple[str, str, str]:
        return self.document_id, self.page, self.chunk_id

    def as_dict(self) -> dict:
        return {
            "document_id": self.document_id,
            "title": self.title,
            "page": self.page,
            "chunk_id": self.chunk_id,
            "pattern": self.pattern,
            "reference": (
                f"{self.title} | PDF page {self.page} | chunk {self.chunk_id}"
                if self.title
                else f"{self.document_id} | PDF page {self.page} | chunk {self.chunk_id}"
            ),
            "context": self.context,
        }


@dataclass
class ExpansionRecord:
    # Surface forms are first merged by punctuation/case normalization.
    variants: Counter = field(default_factory=Counter)
    evidence: dict[tuple[str, str, str], Evidence] = field(default_factory=dict)
    evidence_keys: set[tuple[str, str, str]] = field(default_factory=set)
    patterns: Counter = field(default_factory=Counter)

    def add(
        self,
        expansion: str,
        evidence: Evidence,
        *,
        max_evidence: int | None = None,
    ) -> None:
        self.variants[expansion] += 1
        self.patterns[evidence.pattern] += 1
        key = evidence.key()
        self.evidence_keys.add(key)
        if max_evidence is None or len(self.evidence) < max_evidence:
            self.evidence.setdefault(key, evidence)

    def merge_from(self, other: "ExpansionRecord", *, max_evidence: int) -> None:
        self.variants.update(other.variants)
        self.patterns.update(other.patterns)
        self.evidence_keys.update(other.evidence_keys)
        for key, evidence in other.evidence.items():
            if len(self.evidence) >= max_evidence:
                break
            self.evidence.setdefault(key, evidence)

    @property
    def display_expansion(self) -> str:
        return sorted(
            self.variants,
            key=lambda value: (-self.variants[value], len(value), value.casefold()),
        )[0]

    @property
    def evidence_count(self) -> int:
        return len(self.evidence_keys)


def extract_from_context(
    text: str,
    allowed: set[str],
) -> list[tuple[str, str, str, int, int]]:
    results: list[tuple[str, str, str, int, int]] = []

    for match in PAREN_RE.finditer(text):
        content = match.group(1).strip()

        # Canonical Schwartz-Hearst direction: LF (SF)
        short, explicit_marker = clean_parenthesized_short(content)
        if is_valid_short(short, allowed):
            for candidate, anchored in left_context_variants(
                text[: match.start()], short, explicit_marker
            ):
                expansion = initial_word_match(
                    short,
                    candidate,
                    anchored_end=anchored,
                )
                if expansion:
                    pattern = "LF(marker-SF)" if explicit_marker else "LF(SF)"
                    results.append(
                        (short, expansion, pattern, match.start(), match.end())
                    )
                    break

        # Useful bidirectional extension: SF (LF)
        inside_words = WORD_RE.findall(content)
        if 2 <= len(inside_words) <= 12:
            before = text[: match.start()].rstrip()
            preceding = PRECEDING_SHORT_RE.search(before)
            if preceding:
                short2 = preceding.group(1)
                if is_valid_short(short2, allowed):
                    chars = short_form_chars(short2)
                    # Keep reverse definitions tight; explanatory parentheticals
                    # such as "ППО (установочные файлы ...)" must not be promoted.
                    if len(inside_words) <= len(chars) + 2:
                        expansion2 = initial_word_match(
                            short2,
                            content,
                            anchored_end=True,
                            require_full_span=True,
                        )
                        if expansion2:
                            results.append(
                                (
                                    short2,
                                    expansion2,
                                    "SF(LF)",
                                    preceding.start(1),
                                    match.end(),
                                )
                            )

    # A chunk can contain overlapping NER contexts; de-duplicate exact pairs.
    seen = set()
    unique = []
    for row in results:
        key = (
            normalize_abbreviation(row[0]),
            normalize_expansion_key(row[1]),
            row[2],
        )
        if key not in seen:
            seen.add(key)
            unique.append(row)
    return unique


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as stream:
        reader = csv.DictReader(stream, delimiter=";")
        if not reader.fieldnames:
            return [], []
        return list(reader.fieldnames), list(reader)


def load_base_dictionary(path: Path) -> tuple[list[str], list[dict[str, str]], set[str]]:
    headers, rows = read_csv_rows(path)
    allowed: set[str] = set()
    for row in rows:
        key = normalize_abbreviation(
            row.get("canonical_key") or row.get("abbreviation") or ""
        )
        if key:
            allowed.add(key)
    return headers, rows, allowed


def load_chunk_index(path: Path) -> dict[str, dict[str, str]]:
    """chunks.csv stores metadata; input_line is a source line number, not chunk text."""
    _, rows = read_csv_rows(path)
    return {
        row["chunk_id"]: {
            "document_id": row.get("document_id", ""),
            "title": row.get("title", ""),
            "page": row.get("page", ""),
            "status": row.get("status", ""),
        }
        for row in rows
        if row.get("chunk_id")
    }


def load_document_index(path: Path) -> dict[str, dict[str, str]]:
    _, rows = read_csv_rows(path)
    return {row["document_id"]: row for row in rows if row.get("document_id")}


def harvest(
    abbreviations_path: Path,
    allowed: set[str],
    chunk_index: dict[str, dict[str, str]],
    document_index: dict[str, dict[str, str]],
    *,
    max_evidence_per_expansion: int,
) -> tuple[
    dict[str, dict[str, ExpansionRecord]],
    dict[str, int],
]:
    found: dict[str, dict[str, ExpansionRecord]] = defaultdict(dict)
    stats = Counter()

    # The NER export can have several mentions from the same chunk.
    # Process each unique chunk context once.
    seen_chunks: set[str] = set()

    with abbreviations_path.open(
        "r", encoding="utf-8-sig", errors="replace", newline=""
    ) as stream:
        reader = csv.DictReader(stream, delimiter=";")
        for row in reader:
            stats["ner_rows"] += 1
            chunk_id = (row.get("chunk_id") or "").strip()
            if not chunk_id or chunk_id in seen_chunks:
                continue
            seen_chunks.add(chunk_id)
            stats["unique_context_chunks"] += 1

            context = row.get("context") or ""
            if not context.strip():
                stats["empty_contexts"] += 1
                continue

            chunk_meta = chunk_index.get(chunk_id, {})
            document_id = (
                row.get("document_id")
                or chunk_meta.get("document_id")
                or ""
            )
            document_meta = document_index.get(document_id, {})

            # Evidence is accepted only from known successful chunks/documents.
            if chunk_meta and chunk_meta.get("status") not in {"", "ok"}:
                stats["skipped_non_ok_chunks"] += 1
                continue
            if document_meta and document_meta.get("status") not in {"", "complete"}:
                stats["skipped_non_complete_documents"] += 1
                continue

            title = (
                row.get("title")
                or chunk_meta.get("title")
                or document_meta.get("title")
                or ""
            )
            page = row.get("page") or chunk_meta.get("page") or ""

            extracted = extract_from_context(context, allowed)
            stats["raw_pair_hits"] += len(extracted)

            for short, expansion, pattern, start, end in extracted:
                short_key = normalize_abbreviation(short)
                expansion_key = normalize_expansion_key(expansion)
                if not short_key or not expansion_key:
                    continue

                snippet_start = max(0, start - 180)
                snippet_end = min(len(context), end + 180)
                snippet = re.sub(
                    r"\s+",
                    " ",
                    context[snippet_start:snippet_end],
                ).strip()

                evidence = Evidence(
                    document_id=document_id,
                    title=title,
                    page=str(page),
                    chunk_id=chunk_id,
                    pattern=pattern,
                    context=snippet,
                )

                record = found[short_key].setdefault(
                    expansion_key, ExpansionRecord()
                )
                record.add(
                    expansion,
                    evidence,
                    max_evidence=max_evidence_per_expansion,
                )

    stats["confirmed_abbreviations"] = len(found)
    stats["confirmed_expansions"] = sum(len(records) for records in found.values())
    return found, dict(stats)



def variant_signature(value: str) -> tuple[str, ...]:
    """
    Conservative surface-variant grouping.

    Russian inflection is approximated by a 3-letter stem prefix per token;
    Latin tokens use a 5-letter prefix. This groups forms such as
    'база/базы/базой данных' and 'module/modules' without merging clearly
    different expansions such as 'Common Name' and 'Commit Number'.
    """
    tokens = [token.casefold() for token in WORD_RE.findall(value)]
    signature = []
    for token in tokens:
        if re.search(r"[А-Яа-яЁё]", token):
            signature.append(token[:3])
        elif token.endswith("s") and len(token) > 5:
            signature.append(token[:-1][:5])
        else:
            signature.append(token[:5])
    return tuple(signature)


def group_meanings(
    records: dict[str, ExpansionRecord],
    *,
    max_evidence: int,
) -> list[ExpansionRecord]:
    grouped: dict[tuple[str, ...], ExpansionRecord] = {}
    for record in records.values():
        signature = variant_signature(record.display_expansion)
        target = grouped.setdefault(signature, ExpansionRecord())
        target.merge_from(record, max_evidence=max_evidence)
    return sorted(
        grouped.values(),
        key=lambda record: (
            -sum(record.variants.values()),
            record.display_expansion.casefold(),
        ),
    )

def write_outputs(
    output_dictionary: Path,
    output_pairs: Path,
    output_summary: Path,
    base_headers: list[str],
    base_rows: list[dict[str, str]],
    found: dict[str, dict[str, ExpansionRecord]],
    stats: dict[str, int],
    *,
    max_evidence_per_expansion: int,
) -> None:
    output_dictionary.parent.mkdir(parents=True, exist_ok=True)
    output_pairs.parent.mkdir(parents=True, exist_ok=True)
    output_summary.parent.mkdir(parents=True, exist_ok=True)

    extra_headers = [
        "status",
        "expansion_count",
        "expansions_json",
        "evidence_count",
        "evidence_refs_json",
        "confirmation_method",
    ]
    dictionary_headers = base_headers + [
        name for name in extra_headers if name not in base_headers
    ]

    confirmed_rows = 0
    candidate_rows = 0

    with output_dictionary.open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=dictionary_headers,
            delimiter=";",
            extrasaction="ignore",
        )
        writer.writeheader()

        for base in base_rows:
            key = normalize_abbreviation(
                base.get("canonical_key") or base.get("abbreviation") or ""
            )
            records = found.get(key, {})
            meanings = group_meanings(
                records,
                max_evidence=max_evidence_per_expansion,
            )

            expansion_objects = []
            evidence_refs = []
            for record in meanings:
                evidences = [ev.as_dict() for ev in record.evidence.values()]
                expansion_objects.append(
                    {
                        "expansion": record.display_expansion,
                        "variants": [
                            {
                                "text": value,
                                "hits": record.variants[value],
                            }
                            for value in sorted(
                                record.variants,
                                key=lambda value: (
                                    -record.variants[value],
                                    len(value),
                                    value.casefold(),
                                ),
                            )
                        ],
                        "evidence_count": record.evidence_count,
                        "patterns": dict(record.patterns),
                        "evidence": evidences,
                    }
                )
                evidence_refs.extend(ev["reference"] for ev in evidences)

            row = dict(base)
            if expansion_objects:
                row["status"] = "confirmed"
                row["expansion_count"] = str(len(meanings))
                row["expansions_json"] = json.dumps(
                    expansion_objects, ensure_ascii=False
                )
                row["evidence_count"] = str(
                    sum(item["evidence_count"] for item in expansion_objects)
                )
                row["evidence_refs_json"] = json.dumps(
                    evidence_refs, ensure_ascii=False
                )
                row["confirmation_method"] = "schwartz_hearst_adapted"
                confirmed_rows += 1
            else:
                row["status"] = "candidate"
                row["expansion_count"] = "0"
                row["expansions_json"] = "[]"
                row["evidence_count"] = "0"
                row["evidence_refs_json"] = "[]"
                row["confirmation_method"] = ""
                candidate_rows += 1

            writer.writerow(row)

    pair_headers = [
        "abbreviation",
        "canonical_key",
        "expansion",
        "expansion_variants_json",
        "status",
        "evidence_count",
        "patterns_json",
        "evidence_json",
    ]
    base_by_key = {
        normalize_abbreviation(
            row.get("canonical_key") or row.get("abbreviation") or ""
        ): row
        for row in base_rows
        if normalize_abbreviation(
            row.get("canonical_key") or row.get("abbreviation") or ""
        )
    }

    with output_pairs.open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=pair_headers, delimiter=";")
        writer.writeheader()
        for key in sorted(found):
            base = base_by_key.get(key, {})
            abbreviation = base.get("abbreviation") or key
            canonical = base.get("canonical_key") or key
            for record in group_meanings(
                found[key],
                max_evidence=max_evidence_per_expansion,
            ):
                writer.writerow(
                    {
                        "abbreviation": abbreviation,
                        "canonical_key": canonical,
                        "expansion": record.display_expansion,
                        "expansion_variants_json": json.dumps(
                            dict(record.variants), ensure_ascii=False
                        ),
                        "status": "confirmed",
                        "evidence_count": record.evidence_count,
                        "patterns_json": json.dumps(
                            dict(record.patterns), ensure_ascii=False
                        ),
                        "evidence_json": json.dumps(
                            [ev.as_dict() for ev in record.evidence.values()],
                            ensure_ascii=False,
                        ),
                    }
                )

    grouped_meaning_count = sum(
        len(group_meanings(records, max_evidence=max_evidence_per_expansion))
        for records in found.values()
    )

    summary = {
        **stats,
        "confirmed_meanings": grouped_meaning_count,
        "base_dictionary_rows": len(base_rows),
        "confirmed_dictionary_rows": confirmed_rows,
        "candidate_dictionary_rows": candidate_rows,
        "coverage": (
            confirmed_rows / len(base_rows)
            if base_rows
            else 0.0
        ),
        "outputs": {
            "dictionary": str(output_dictionary),
            "pairs": str(output_pairs),
        },
        "notes": [
            "status=confirmed only when an explicit Schwartz-Hearst-style pair is found.",
            "status=candidate means the NER abbreviation remains unresolved after this stage.",
            "Multiple expansions for one abbreviation are preserved.",
            "Evidence points to document title, physical PDF page and chunk_id.",
            "chunks.csv is used for provenance validation; its input_line column is not treated as text.",
        ],
    }
    output_summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> None:
    dictionary_path = Path(args.dictionary)
    abbreviations_path = Path(args.abbreviations)
    chunks_path = Path(args.chunks)
    documents_path = Path(args.documents)

    base_headers, base_rows, allowed = load_base_dictionary(dictionary_path)
    chunk_index = load_chunk_index(chunks_path)
    document_index = load_document_index(documents_path)

    found, stats = harvest(
        abbreviations_path,
        allowed,
        chunk_index,
        document_index,
        max_evidence_per_expansion=args.max_evidence,
    )

    write_outputs(
        Path(args.output_dictionary),
        Path(args.output_pairs),
        Path(args.output_summary),
        base_headers,
        base_rows,
        found,
        stats,
        max_evidence_per_expansion=args.max_evidence,
    )

    print(json.dumps({
        "base_dictionary_rows": len(base_rows),
        "confirmed_abbreviations": stats.get("confirmed_abbreviations", 0),
        "confirmed_expansions_raw": stats.get("confirmed_expansions", 0),
        "confirmed_meanings": sum(
            len(group_meanings(records, max_evidence=args.max_evidence))
            for records in found.values()
        ),
        "candidate_abbreviations": len(base_rows) - stats.get("confirmed_abbreviations", 0),
        "output_dictionary": args.output_dictionary,
        "output_pairs": args.output_pairs,
        "output_summary": args.output_summary,
    }, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Apply a conservative Russian/English Schwartz-Hearst adaptation "
            "to an NER abbreviation dictionary and preserve exact provenance."
        )
    )
    parser.add_argument("--dictionary", required=True)
    parser.add_argument("--abbreviations", required=True)
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--documents", required=True)
    parser.add_argument(
        "--output-dictionary",
        default="dictionary_schwartz_hearst.csv",
    )
    parser.add_argument(
        "--output-pairs",
        default="schwartz_hearst_pairs.csv",
    )
    parser.add_argument(
        "--output-summary",
        default="schwartz_hearst_summary.json",
    )
    parser.add_argument(
        "--max-evidence",
        type=int,
        default=5,
        help="Maximum stored evidence references per expansion.",
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
