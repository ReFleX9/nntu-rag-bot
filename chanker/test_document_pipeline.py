"""Быстрые тесты пайплайна без внешнего Tesseract."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pymupdf

import document_pipeline as pipeline
from document_pipeline import PipelineConfig, prepare_document
from pdf_media import MediaConfig


class PreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.corpus = cls.root / "corpus"
        (cls.corpus / "product").mkdir(parents=True)
        cls.pdf_path = cls.corpus / "product" / "manual.pdf"

        with pymupdf.open() as doc:
            # Несколько страниц с повторяющимся верхним колонтитулом и номером страницы.
            for page_no in range(1, 7):
                page = doc.new_page(width=600, height=800)
                page.insert_text((40, 30), "REPEATED MANUAL HEADER", fontsize=10)
                if page_no == 1:
                    page.insert_text((40, 80), "1. Installation", fontsize=14)
                page.insert_text((40, 120), f"Paragraph {page_no} begins with useful content.", fontsize=11)
                page.insert_text((40, 145), "This is a second nearby text block that must be merged.", fontsize=11)
                page.insert_text((40, 170), "More technical explanation follows on the same physical page.", fontsize=11)
                page.insert_text((290, 780), str(page_no), fontsize=9)
            doc.set_metadata({"title": "PDF title"})
            doc.set_toc([[1, "Installation", 1]])
            doc.save(cls.pdf_path)

        cls.config = PipelineConfig(
            media=MediaConfig(ocr_enabled=False, raster_tables=False),
            chunk_size_chars=500,
            chunk_overlap_chars=60,
            min_chunk_chars=180,
            text_kind="prose",
            join_wrapped_lines=True,
        )

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def prepare(self):
        return prepare_document(
            self.pdf_path,
            corpus_root=self.corpus,
            output_dir=self.root / "prepared",
            config=self.config,
        )

    def test_boilerplate_and_page_numbers_are_removed(self):
        prepared = self.prepare()
        corpus_text = "\n".join(chunk["text"] for chunk in prepared.chunks)
        self.assertNotIn("REPEATED MANUAL HEADER", corpus_text)
        self.assertGreaterEqual(prepared.document["noise_removed"].get("repeated_boilerplate", 0), 6)
        self.assertGreaterEqual(prepared.document["noise_removed"].get("standalone_page_number", 0), 6)

    def test_small_text_blocks_are_merged(self):
        prepared = self.prepare()
        page_two = [c for c in prepared.chunks if c["page"] == 2 and c["kind"] == "text"]
        self.assertEqual(len(page_two), 1)
        self.assertIn("Paragraph 2 begins", page_two[0]["text"])
        self.assertIn("second nearby text block", page_two[0]["text"])
        self.assertIn("technical explanation", page_two[0]["text"])

    def test_public_chunks_have_only_compact_metadata(self):
        prepared = self.prepare()
        exported = [json.loads(line) for line in prepared.chunks_path.read_text(encoding="utf-8").splitlines()]
        self.assertTrue(exported)
        chunk = exported[0]
        self.assertEqual(chunk["title"], "PDF title")
        self.assertNotIn("product", chunk)
        self.assertNotIn("product_version", chunk)
        self.assertNotIn("language", chunk)
        self.assertNotIn("config_hash", chunk)
        self.assertNotIn("metadata_hash", chunk)
        self.assertNotIn("processed_at_utc", chunk)
        self.assertNotIn("page_width", chunk)
        self.assertLessEqual(len(chunk), 30)
        self.assertTrue(all("source_spans" not in ref for ref in chunk["sources"]))

    def test_chunk_ids_are_stable_between_runs(self):
        first = self.prepare()
        second = prepare_document(
            self.pdf_path,
            corpus_root=self.corpus,
            output_dir=self.root / "prepared2",
            config=self.config,
        )
        self.assertEqual([c["chunk_id"] for c in first.chunks], [c["chunk_id"] for c in second.chunks])

    def test_balanced_tail(self):
        text = "word " * 500
        windows = pipeline._text_windows(text, 500, 60, 180)
        self.assertGreaterEqual(windows[-1][1] - windows[-1][0], 180)

class ImageTextFilterTests(unittest.TestCase):
    def test_obvious_ocr_fragments_are_filtered(self):
        noisy = {"text": "DN", "mean_confidence": 80.0}
        fragmented = {"text": "Г ] a", "mean_confidence": 85.0}
        useful = {"text": "Bluetooth", "mean_confidence": 92.0}
        useful_ui = {"text": "Сменить пользователя\nОтмена", "mean_confidence": 95.0}

        self.assertEqual(pipeline._image_text_noise_reason(noisy), "image_text_too_short")
        self.assertIsNotNone(pipeline._image_text_noise_reason(fragmented))
        self.assertIsNone(pipeline._image_text_noise_reason(useful))
        self.assertIsNone(pipeline._image_text_noise_reason(useful_ui))


if __name__ == "__main__":
    unittest.main()
