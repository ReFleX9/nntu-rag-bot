"""Проверки автоматических базовых метаданных и разделов без metadata.json."""
from __future__ import annotations

import unittest

from pipeline_metadata import make_sections, resolve_document_metadata, stable_hash


class MetadataTests(unittest.TestCase):
    def test_pdf_title_is_used(self):
        metadata = resolve_document_metadata(
            "product/manual.pdf", {"title": "  Руководство  "}, "corpus"
        )
        self.assertEqual(metadata["title"], "Руководство")

    def test_filename_is_title_fallback(self):
        metadata = resolve_document_metadata("product/manual.pdf", {}, "corpus")
        self.assertEqual(metadata["title"], "manual")

    def test_document_key_is_stable_and_corpus_specific(self):
        first = resolve_document_metadata("product/manual.pdf", {}, "corpus")
        second = resolve_document_metadata("product/manual.pdf", {}, "corpus")
        other = resolve_document_metadata("product/manual.pdf", {}, "other")
        self.assertEqual(first["document_key"], second["document_key"])
        self.assertNotEqual(first["document_key"], other["document_key"])

    def test_sections_come_only_from_pdf_outline(self):
        key = stable_hash(["corpus", "manual.pdf"])
        sections, ambiguous, warnings = make_sections(
            key,
            [[1, "Intro", 1], [2, "Install", 2], [1, "Appendix", 3]],
            3,
        )
        self.assertFalse(ambiguous)
        self.assertFalse(warnings)
        self.assertEqual(sections[1]["section_path"], ["Intro", "Install"])
        self.assertEqual(sections[1]["section_source"], "pdf_outline")

    def test_same_page_bookmarks_are_marked_ambiguous(self):
        key = stable_hash(["corpus", "manual.pdf"])
        _, ambiguous, warnings = make_sections(
            key, [[1, "A", 1], [1, "B", 1]], 2
        )
        self.assertEqual(ambiguous, {1})
        self.assertTrue(warnings)


if __name__ == "__main__":
    unittest.main()
