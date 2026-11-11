import tempfile
import unittest
from pathlib import Path

from extract_abbreviations import canonical, clean_abbreviation_span, deduplicate, join_subword_entities, open_db, parse_pages, windows


class CharacterTokenizer:
    def __call__(self, text, **kwargs):
        return {"offset_mapping": [(i, i + 1) for i in range(len(text))]}


class Tests(unittest.TestCase):
    def test_boundaries_and_mixed_case(self):
        self.assertIsNone(clean_abbreviation_span("GUC", 0, 2))
        self.assertIsNone(clean_abbreviation_span("Postgres", 0, 8))
        self.assertEqual(clean_abbreviation_span("GiST", 0, 4), (0, 4))
        self.assertEqual(clean_abbreviation_span("SQL-команд", 0, 10), (0, 3))
        self.assertEqual(clean_abbreviation_span("SQL-оператор", 0, 4), (0, 3))

    def test_subwords_join_but_separate_acronyms_do_not(self):
        text = "MVCC SQL XML"
        entities = [{"start": a, "end": b, "score": .9, "label": "AC"}
                    for a, b in [(0, 2), (2, 4), (5, 8), (9, 12)]]
        result = join_subword_entities(entities, text)
        self.assertEqual([text[e["start"]:e["end"]] for e in result], ["MVCC", "SQL", "XML"])

    def test_overlap_preserves_boundary_abbreviation(self):
        text = "Текст перед границей " * 12 + "WAL MVCC GiST СУБД " + "Ещё текст. " * 12
        chunks = list(windows(text, CharacterTokenizer(), budget=70, overlap=25))
        for term in ["WAL", "MVCC", "GiST", "СУБД"]:
            self.assertTrue(any(term in chunk for _, chunk in chunks))
        for start, chunk in chunks:
            self.assertEqual(text[start:start + len(chunk)], chunk)
            self.assertLessEqual(len(chunk), 70)

    def test_duplicates_and_separate_mentions(self):
        def e(start, end, score):
            return {"start": start, "end": end, "score": score}
        result = deduplicate([e(0, 3, .8), e(0, 3, .9), e(0, 2, .7), e(10, 13, .6)])
        self.assertEqual([(x["start"], x["score"]) for x in result], [(0, .9), (10, .6)])

    def test_case_without_cross_script_conflation(self):
        self.assertEqual(canonical("GiST"), canonical("GIST"))
        self.assertNotEqual(canonical("С"), canonical("C"))

    def test_physical_page_ranges(self):
        self.assertEqual(parse_pages("428,2730-2734"), {428, 2730, 2731, 2732, 2733, 2734})
        for bad in ["0", "5-2", "1-2-3"]:
            with self.assertRaises(ValueError):
                parse_pages(bad)

    def test_config_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "progress.sqlite3"
            open_db(path, {"model": "one"}).close()
            open_db(path, {"model": "one"}).close()
            with self.assertRaises(ValueError):
                open_db(path, {"model": "two"})


if __name__ == "__main__":
    unittest.main()
