"""Conservative cleaning of extracted PDF text for RAG and term extraction.

Python 3.10+, standard library only. Pass ONE page or ONE block per call.
Keep document_id, physical PDF page and block_id in the surrounding pipeline.

Use kind="unknown" for a flat page whose layout has not been classified.
Use kind="prose" only for a known prose block. Line wrapping is opt-in.
Tables and code are kept verbatim except for CRLF/CR -> LF and explicit drops.
Repeated margin lines are proposed for review, never removed automatically.

Quotes must use raw_text/source_for_line(), not offsets into cleaned text.
Source line numbers below are text-line numbers, NOT physical PDF pages.

Example:
    result = clean_text(
        "Настройте програм-\nмное обеспечение.",
        kind="prose", known_words={"программное"},
    )
    assert result.text == "Настройте программное обеспечение."
    assert result.source_for_line(1) == result.raw_text
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

BlockKind = Literal["unknown", "prose", "heading", "list", "table", "code"]

_NEWLINE = re.compile(r"\r\n|\r|\n")
_LETTERS = "A-Za-zА-Яа-яЁё"
_LOWER = "a-zа-яё"
_HYPHEN_END = re.compile(rf"(?P<word>[{_LETTERS}]+)(?P<mark>[-\u2010\u00ad])$")
_WORD_START = re.compile(rf"^(?P<word>[{_LOWER}]+)(?![\w-])")
_LIST_OR_HEADING = re.compile(
    r"^\s*(?:#{1,6}\s|[-*+•‣▪●]\s|\d+(?:\.\d+)*[.)]?\s|[A-Za-zА-Яа-я][.)]\s)"
)
_TECHNICAL = re.compile(
    r"https?://|www\.|(?:[A-Za-z]:[\\/])|(?:\S+[\\/]\S+)"
    r"|(?<!\w)--[A-Za-z]|`|\||\t"
    r"|^\s*(?:\$\s|>>>\s|#|[{}\[\]])"
    r"|^\s*(?:SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP|GRANT|WITH|FROM|WHERE)\b"
    r"|^\s*[A-Za-z_][\w.-]*\s*="
)
_PAGE_NUMBER = re.compile(
    r"(?:(?:стр\.?|страница|page)\s*)?\d+"
    r"(?:\s*(?:из|of|/)\s*\d+)?\.?",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CleaningResult:
    raw_text: str
    text: str
    # One tuple per output line, with 1-based line numbers in raw_text.
    line_sources: tuple[tuple[int, ...], ...]
    changes: dict[str, int]
    warnings: tuple[str, ...]
    removed_lines: tuple[tuple[int, str], ...]

    def source_for_line(self, line_number: int) -> str:
        """Return the exact raw slice underlying an output line (1-based)."""
        if not 1 <= line_number <= len(self.line_sources):
            raise IndexError("Output line number is out of range")
        source = self.line_sources[line_number - 1]
        if not source:
            return ""
        spans = _raw_line_spans(self.raw_text)
        return self.raw_text[spans[source[0] - 1][0]:spans[source[-1] - 1][1]]


@dataclass(frozen=True)
class MarginOccurrence:
    # Position in the supplied document's page sequence, 1-based.
    page: int
    line_number: int
    raw_line: str


@dataclass(frozen=True)
class MarginCandidate:
    side: Literal["top", "bottom"]
    signature: str
    occurrences: tuple[MarginOccurrence, ...]


@dataclass
class _Line:
    text: str
    sources: tuple[int, ...]
    protected: bool = False
    boundary: bool = False
    hard_break: bool = False


def _raw_line_spans(text: str) -> list[tuple[int, int]]:
    spans = []
    start = 0
    for match in _NEWLINE.finditer(text):
        spans.append((start, match.start()))
        start = match.end()
    spans.append((start, len(text)))
    return spans


def _key(text: str) -> str:
    # Only comparison keys are case-folded; output text keeps its case.
    return unicodedata.normalize("NFC", text).casefold()


def _normalize_line(text: str, changes: Counter) -> str:
    normalized = unicodedata.normalize("NFC", text)
    if normalized != text:
        changes["nfc_lines"] += 1
    chars = []
    for char in normalized:
        if char in ("\u200b", "\ufeff"):
            changes["removed_zero_width_or_bom"] += 1
        elif char == "\x00":
            changes["removed_nul"] += 1
        elif unicodedata.category(char) == "Zs" and char != " ":
            chars.append(" ")
            changes["unicode_spaces"] += 1
        elif unicodedata.category(char) == "Co":
            # Многие PDF-шрифты кодируют маркеры списков/иконки через Private Use Area
            # (например, \uf02d, \uf0fc). Для полнотекстового поиска эти глифы
            # не несут лексической информации и только создают микрочанки.
            chars.append(" ")
            changes["private_use_to_space"] += 1
        elif unicodedata.category(char) == "Cc" and char not in "\t\f\v":
            # A separator becomes a space, not silent concatenation of words.
            chars.append(" ")
            changes["control_to_space"] += 1
        else:
            chars.append(char)
    return "".join(chars)


def _try_join(
    left: _Line,
    right: _Line,
    known_words: set[str],
    join_wrapped_lines: bool,
    changes: Counter,
    warnings: set[str],
) -> _Line | None:
    if (
        left.protected or right.protected or left.boundary or right.boundary
        or left.hard_break or not left.text.strip() or not right.text.strip()
        or left.sources[-1] + 1 != right.sources[0]
    ):
        return None

    first = left.text.rstrip()
    second = right.text.lstrip()
    end = _HYPHEN_END.search(first)
    beginning = _WORD_START.match(second)

    if end and beginning and not end.group("word").isupper():
        joined_word = _key(end.group("word") + beginning.group("word"))
        hyphenated_word = _key(end.group("word") + "-" + beginning.group("word"))
        if end.group("mark") == "\u00ad":
            changes["soft_hyphen_line_joins"] += 1
        elif joined_word in known_words and hyphenated_word not in known_words:
            changes["confirmed_hyphen_line_joins"] += 1
        else:
            warnings.add("Ambiguous line-end hyphen retained; supply confirmed known_words.")
            return None
        return _Line(first[:-1] + second, left.sources + right.sources,
                     hard_break=right.hard_break)

    if not join_wrapped_lines:
        return None
    # This optional heuristic is for classified prose paragraphs only.
    if (
        first.endswith((".", "!", "?", ":", ";", "…", "-", "\u2010", "\u00ad"))
        or not second[0].islower()
    ):
        return None
    changes["wrapped_line_joins"] += 1
    return _Line(first + " " + second, left.sources + right.sources,
                 hard_break=right.hard_break)


def clean_text(
    text: str,
    *,
    kind: BlockKind = "unknown",
    known_words: Iterable[str] = (),
    join_wrapped_lines: bool = False,
    drop_lines: Iterable[int] = (),
) -> CleaningResult:
    """Clean one extracted page/block, preserving provenance and structure.

    kind="unknown": basic Unicode/space cleanup, no space collapsing or joining.
    kind="prose": normalize spaces, remove soft hyphens, repair confirmed wraps.
    kind="heading"/"list": normalize spaces without joining lines.
    kind="code"/"table": preserve content except line-ending normalization.

    join_wrapped_lines is an optional heuristic, only valid for prose.
    known_words must contain trusted complete spellings, including genuine
    hyphenated forms when present. Ordinary hyphens are not guessed away.
    drop_lines explicitly approves deletion of 1-based RAW text lines. Supply
    only indices checked against PDF layout or reviewed margin candidates.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if kind not in {"unknown", "prose", "heading", "list", "table", "code"}:
        raise ValueError("Unsupported block kind")
    if join_wrapped_lines and kind != "prose":
        raise ValueError("Line joining requires kind='prose'")
    if isinstance(known_words, str):
        raise TypeError("known_words must be an iterable of words, not one string")

    changes: Counter = Counter()
    warnings: set[str] = set()
    spans = _raw_line_spans(text)
    raw_lines = [text[start:end] for start, end in spans]
    drops = set(drop_lines)
    if any(type(n) is not int or not 1 <= n <= len(raw_lines) for n in drops):
        raise ValueError("drop_lines must contain valid 1-based raw line numbers")
    vocabulary = {_key(word) for word in known_words}

    if "\ufffd" in text:
        warnings.add("Replacement character U+FFFD retained; check PDF extraction/OCR.")
    if any(char in text for char in "\f\v\u2028\u2029"):
        warnings.add("Unusual text/page separators retained; inspect block boundaries.")
    if any(char in text for char in "\u200c\u200d\u200e\u200f\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069"):
        warnings.add("Joiner or directional controls retained for review.")
    newline_changes = len(re.findall(r"\r\n|\r", text))
    if newline_changes:
        changes["line_endings"] = newline_changes

    lines: list[_Line] = []
    removed = []
    fence_char = ""
    fence_length = 0
    for number, raw in enumerate(raw_lines, 1):
        fence = re.match(r"^\s*(`{3,}|~{3,})", raw)
        inside_fence = bool(fence_char)
        if fence:
            marker = fence.group(1)
            if not fence_char:
                fence_char, fence_length = marker[0], len(marker)
            elif marker[0] == fence_char and len(marker) >= fence_length:
                fence_char = ""

        if number in drops:
            removed.append((number, raw))
            changes["approved_line_removals"] += 1
            continue

        protected = (
            kind in {"code", "table"} or inside_fence or bool(fence)
            or raw.startswith(("    ", "\t"))
            or bool(_TECHNICAL.search(raw))
            or any(char in raw for char in "\f\v\u2028\u2029")
        )
        hard_break = raw.endswith("  ")
        cleaned = raw if protected else _normalize_line(raw, changes)
        if not protected and kind in {"prose", "heading", "list"}:
            indent = re.match(r"^ *", cleaned).group()
            body = cleaned[len(indent):]
            body, count = re.subn(r" {2,}", " ", body)
            if count:
                changes["collapsed_space_runs"] += count
            updated = indent + body.rstrip(" ")
            if hard_break and updated.strip():
                updated += "  "  # Preserve an explicit Markdown hard break.
            if updated != cleaned and not count:
                changes["trailing_space_lines"] += 1
            cleaned = updated
        boundary = bool(_LIST_OR_HEADING.match(cleaned)) or cleaned.strip().isupper()
        line = _Line(cleaned, (number,), protected, boundary, hard_break)
        if lines and kind == "prose":
            joined = _try_join(lines[-1], line, vocabulary, join_wrapped_lines,
                               changes, warnings)
            if joined is not None:
                lines[-1] = joined
                continue
        lines.append(line)

    for line in lines:
        if not line.protected and kind in {"prose", "heading", "list"}:
            count = line.text.count("\u00ad")
            if count:
                line.text = line.text.replace("\u00ad", "")
                changes["removed_soft_hyphens"] += count

    if kind in {"prose", "heading", "list"}:
        compact = []
        for line in lines:
            if not line.protected and not line.text.strip():
                line.text = ""
                if compact and compact[-1].text == "" and not compact[-1].protected:
                    changes["extra_blank_lines"] += 1
                    continue
            compact.append(line)
        while compact and not compact[0].protected and compact[0].text == "":
            compact.pop(0)
            changes["edge_blank_lines"] += 1
        while compact and not compact[-1].protected and compact[-1].text == "":
            compact.pop()
            changes["edge_blank_lines"] += 1
        lines = compact

    if not lines:
        lines = [_Line("", ())]
    return CleaningResult(
        raw_text=text,
        text="\n".join(line.text for line in lines),
        line_sources=tuple(line.sources for line in lines),
        changes=dict(changes),
        warnings=tuple(sorted(warnings)),
        removed_lines=tuple(removed),
    )


def find_repeated_margins(
    pages: Sequence[str],
    *,
    margin_lines: int = 2,
    min_pages: int = 3,
    min_fraction: float = 0.6,
) -> list[MarginCandidate]:
    """Suggest repeated top/bottom lines from ONE document, without deleting.

    These are candidates, not proof of a header/footer: PDF bounding boxes
    or human review must confirm margin placement before using drop_lines.
    Supply pages in physical order, including empty pages and the cover.
    Exact titles/versions stay distinct; only standalone page labels/numbers
    are grouped under a shared signature.
    """
    if margin_lines < 1 or min_pages < 2 or not 0 < min_fraction <= 1:
        raise ValueError("Invalid candidate detection thresholds")
    found: dict[tuple[str, str], list[MarginOccurrence]] = defaultdict(list)
    for page_number, text in enumerate(pages, 1):
        raw_lines = _NEWLINE.split(text)
        nonempty = [(n, raw) for n, raw in enumerate(raw_lines, 1) if raw.strip()]
        # Very short pages provide too little context to distinguish margins.
        if len(nonempty) < 2 * margin_lines + 1:
            continue
        for side, edge in (("top", nonempty[:margin_lines]),
                           ("bottom", nonempty[-margin_lines:])):
            for line_number, raw in edge:
                signature = " ".join(unicodedata.normalize("NFC", raw).split())
                if _PAGE_NUMBER.fullmatch(signature):
                    signature = "<PAGE_NUMBER>"
                found[(side, signature)].append(
                    MarginOccurrence(page_number, line_number, raw)
                )
    required = max(min_pages, math.ceil(len(pages) * min_fraction))
    return [
        MarginCandidate(side, signature, tuple(occurrences))
        for (side, signature), occurrences in sorted(found.items())
        if len({item.page for item in occurrences}) >= required
    ]
