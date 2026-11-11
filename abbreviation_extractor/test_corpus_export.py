import csv
import io
import json
import re
import tempfile
import unittest
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from corpus_export import DocumentContext
import extract_abbreviations as app


def page(number, text, status='ok'):
    return dict(page=number, text=text, status=status, printed_page='', section='')


class ContextTests(unittest.TestCase):
    def test_bullet_does_not_merge_two_sentences(self):
        text = 'Первое предложение. Второе предложение. Третье предложение.\n3.1.\nВведение\nWAL хранит данные.\n• Затем идёт проверка. Следом идёт запись. В конце идёт чтение.'
        a = text.index('WAL')
        result = DocumentContext([page(1, text)]).get(1, a, a + 3)
        self.assertNotIn('Затем', result['sentence'])
        self.assertTrue(result['context'].startswith('Первое предложение.'))
        self.assertEqual((result['sentences_before'], result['sentences_after']), (3, 3))

    def test_glossary_entries_are_short_structured_context(self):
        terms = ['AAA', 'BBB', 'CCC', 'DDD', 'WAL', 'EEE', 'FFF', 'GGG', 'HHH']
        text = '\n'.join(f'{term}\nРасшифровка термина {i}' for i, term in enumerate(terms))
        a = text.index('WAL')
        result = DocumentContext([page(1, text)]).get(1, a, a + 3)
        self.assertEqual(result['sentence'], 'WAL Расшифровка термина 4')
        self.assertEqual(result['context_status'], 'structured_text')
        self.assertNotIn('AAA', result['context'])
        self.assertNotIn('HHH', result['context'])
        self.assertEqual((result['sentences_before'], result['sentences_after']), (0, 0))

    def test_three_sentences_each_side_and_cross_page_sentence(self):
        texts = ['Первое предложение. Второе предложение. Третье предложение. Система WAL хранит',
                 'журнал изменений. Четвёртое предложение. Пятое предложение. Шестое предложение. Лишнее предложение.']
        ctx = DocumentContext([page(i + 1, t) for i, t in enumerate(texts)])
        a = texts[0].index('WAL')
        result = ctx.get(1, a, a + 3)
        self.assertEqual(result['sentences_before'], 3)
        self.assertEqual(result['sentences_after'], 3)
        self.assertEqual(result['sentence'], 'Система WAL хранит журнал изменений.')
        self.assertEqual(result['context'], ' '.join(texts).split(' Лишнее')[0])
        self.assertEqual(result['context_status'], 'full_window')
        self.assertEqual((result['context_start_page'], result['context_end_page']), (1, 2))
        for f in result['context_fragments']:
            self.assertEqual(f['raw_text'], texts[f['physical_page'] - 1][f['start']:f['end']])

    def test_short_document_and_repeated_abbreviation(self):
        text = 'WAL сохраняет данные. Затем WAL помогает восстановлению.'
        ctx = DocumentContext([page(1, text)])
        for match in re.finditer('WAL', text):
            result = ctx.get(1, match.start(), match.end())
            self.assertEqual(result['context_status'], 'limited')
            self.assertEqual(result['context'], text)
        self.assertEqual(ctx.get(1, 0, 3)['sentences_before'], 0)

    def test_never_bridge_scanned_page_or_missing_page(self):
        for middle in [[], [page(2, '', 'needs_ocr')]]:
            ctx = DocumentContext([page(1, 'Далёкое предложение.')] + middle +
                                  [page(3, 'WAL хранит журнал. После него идёт текст.')])
            result = ctx.get(3, 0, 3)
            self.assertEqual(result['sentences_before'], 0)
            self.assertNotIn('Далёкое', result['context'])

    def test_mixed_language_dotted_abbreviation_and_decimal(self):
        text = 'Это т. е. пример версии 2.5. SQL stores data. После него идёт описание.'
        a = text.index('SQL')
        result = DocumentContext([page(1, text)]).get(1, a, a + 3)
        self.assertEqual(result['sentence'], 'SQL stores data.')
        self.assertEqual(result['sentences_before'], 1)
        self.assertEqual(result['sentences_after'], 1)

    def test_page_number_and_repeated_header_mask_preserves_offsets(self):
        texts = ['Руководство\nПервое предложение. Второе предложение. Третье предложение. Система\n1\n',
                 'Руководство\nWAL хранит журнал. Четвёртое предложение. Пятое предложение. Шестое предложение.\n2\n',
                 'Руководство\nКонец документа.\n3\n']
        ctx = DocumentContext([page(i + 1, t) for i, t in enumerate(texts)])
        a = texts[1].index('WAL')
        result = ctx.get(2, a, a + 3)
        self.assertEqual(result['sentence'], 'Система WAL хранит журнал.')
        self.assertNotIn('Руководство', result['context'])
        self.assertNotIn(' 1 ', result['context'])


class FakeExtractor:
    calls = 0
    def __init__(self, args):
        pass

    def extract(self, text):
        type(self).calls += 1
        return [dict(start=m.start(), end=m.end(), abbreviation=m.group(),
                     canonical_key=app.canonical(m.group()), score=.9, label='AC')
                for m in re.finditer(r'\b(?:WAL|SQL)\b', text)]


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f, delimiter=';'))


def make_pdf(path, text):
    import pymupdf
    path.parent.mkdir(parents=True, exist_ok=True)
    with pymupdf.open() as pdf:
        p = pdf.new_page()
        p.insert_text((72, 72), text)
        pdf.save(path)


class PipelineTests(unittest.TestCase):
    def run_app(self, argv):
        with patch.object(app, 'Extractor', FakeExtractor), redirect_stdout(io.StringIO()):
            return app.main([str(x) for x in argv])

    def test_139_files_counts_recursive_resume_changed_deleted_and_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, out = Path(tmp) / 'pdfs', Path(tmp) / 'results'
            for i in range(139):
                # Duplicate basenames in product folders and uppercase extension.
                filename = 'book.PDF' if i % 2 else 'book.pdf'
                make_pdf(root / f'product_{i:03}' / filename,
                         'Start of the manual. WAL writes. WAL restores. SQL queries. End.' if i < 138
                         else 'This document contains no selected abbreviations.')
            FakeExtractor.calls = 0
            with self.assertLogs('abbreviations', level='INFO'):
                self.assertEqual(self.run_app([root, '--output', out]), 0)
            self.assertEqual(FakeExtractor.calls, 139)
            details = read_csv(out / 'abbreviations.csv')
            dictionary = {r['abbreviation']: r for r in read_csv(out / 'dictionary.csv')}
            self.assertEqual(len(details), 414)
            self.assertEqual(int(dictionary['WAL']['occurrences']), 276)
            self.assertEqual(int(dictionary['WAL']['document_count']), 138)
            self.assertEqual(len(read_csv(out / 'documents.csv')), 139)
            self.assertEqual(read_csv(out / 'documents.csv')[-1]['mentions'], '0')
            self.assertTrue(all(r['physical_page'] == '1' for r in details))
            self.assertTrue(all(r['occurrences_in_document'] == '2' for r in details if r['abbreviation'] == 'WAL'))
            with self.assertLogs('abbreviations', level='INFO'):
                self.assertEqual(self.run_app([root, '--output', out]), 0)
            self.assertEqual(FakeExtractor.calls, 139)
            self.assertEqual(read_csv(out / 'abbreviations.csv'), details)
            changed = root / 'product_000' / 'book.pdf'
            changed.unlink()
            make_pdf(changed, 'New text. SQL remains available. End of this example.')
            (root / 'product_001' / 'book.PDF').unlink()
            with self.assertLogs('abbreviations', level='INFO'):
                self.assertEqual(self.run_app([root, '--output', out]), 0)
            dictionary = {r['abbreviation']: r for r in read_csv(out / 'dictionary.csv')}
            self.assertEqual(int(dictionary['WAL']['occurrences']), 272)
            self.assertEqual(int(dictionary['SQL']['occurrences']), 137)
            self.assertEqual(FakeExtractor.calls, 140)
            # A damaged replacement must not retain stale detections.
            changed.write_bytes(b'not a PDF')
            with self.assertLogs('abbreviations', level='INFO'):
                self.assertEqual(self.run_app([root, '--output', out]), 1)
            docs = read_csv(out / 'documents.csv')
            self.assertEqual(len(docs), 138)
            self.assertEqual(docs[0]['status'], 'error')
            self.assertEqual(json.loads((out / 'run.json').read_text())['status'], 'partial')
            dictionary = {r['abbreviation']: r for r in read_csv(out / 'dictionary.csv')}
            self.assertEqual(int(dictionary['SQL']['occurrences']), 136)

    def test_interrupt_and_resume_preserves_finished_pages(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, out = Path(tmp) / 'pdfs', Path(tmp) / 'results'
            for i in range(2):
                make_pdf(root / f'{i}.pdf', 'WAL writes information. This is a test document.')
            original = FakeExtractor.extract
            calls = 0
            def stop_second(extractor, text):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise KeyboardInterrupt
                return original(extractor, text)
            with patch.object(FakeExtractor, 'extract', stop_second), self.assertLogs('abbreviations', level='INFO'):
                self.assertEqual(self.run_app([root, '--output', out]), 130)
            self.assertEqual(len(read_csv(out / 'abbreviations.csv')), 1)
            self.assertEqual(json.loads((out / 'run.json').read_text())['status'], 'interrupted')
            before = FakeExtractor.calls
            with self.assertLogs('abbreviations', level='INFO'):
                self.assertEqual(self.run_app([root, '--output', out]), 0)
            self.assertEqual(FakeExtractor.calls, before + 1)
            self.assertEqual(len(read_csv(out / 'abbreviations.csv')), 2)


if __name__ == '__main__':
    unittest.main()
