"""Corpus CSV export and sentence context. Does not load the NER model."""
from __future__ import annotations

import bisect
from collections import Counter, defaultdict
import csv
import importlib.metadata
import json
import logging
from pathlib import Path
import re
from datetime import datetime, timezone

LOG = logging.getLogger('abbreviations')
FIELDS = {
    'abbreviation': 'Написание аббревиатуры в этом месте документа.',
    'occurrences': 'Всего найденных моделью упоминаний этого ключа во всём корпусе. Повторяется в каждой строке: не суммировать эту колонку.',
    'document_count': 'Число PDF, в которых модель нашла этот ключ. Разные пути считаются разными документами.',
    'physical_page': 'Страница PDF, начиная с 1, включая обложку и оглавление.',
    'document_name': 'Имя PDF-файла с расширением.',
    'context': 'Предложение с находкой, до 3 полных предложений перед ним и до 3 после. Может переходить через страницы того же PDF.',
    'sentence': 'Предложение или фрагмент, содержащий конкретную находку.',
    'occurrences_in_document': 'Всего найденных упоминаний этого ключа в данном PDF.',
    'source_file': 'Путь к PDF относительно входной папки. Различает одноимённые файлы.',
    'document_title': 'Название из метаданных PDF. При отсутствии используется имя файла без расширения.',
    'product_family': 'Первая вложенная папка внутри pdfs. Если PDF лежит прямо в pdfs, поле пустое.',
    'section': 'Ближайший начавшийся раздел по закладкам PDF, с родительскими разделами. Точность до страницы; без закладок поле пустое.',
    'printed_page': 'Метка страницы из PDF, если издатель её задал. Это не OCR номера, напечатанного внизу страницы.',
    'score': 'Оценка NER-модели от 0 до 1; не является проверенной вероятностью правильности.',
    'sentences_before': 'Сколько полных предложений удалось взять перед находкой: 0–3.',
    'sentences_after': 'Сколько полных предложений удалось взять после находки: 0–3.',
    'context_start_page': 'Первая физическая страница контекста.',
    'context_end_page': 'Последняя физическая страница контекста.',
    'context_status': 'full_window: получено 3+1+3 предложений; limited: край документа/разрыв текста; structured_text: вместо предложений запись словаря/фрагмент и до 3 соседних текстовых блоков с каждой стороны.',
    'canonical_key': 'Ключ для группировки: NFKC, без пробелов и различий регистра. Кириллица и латиница не смешиваются. Это написание, не значение.',
}
DICT_FIELDS = {
    'abbreviation': 'Самое частое написание среди найденных вариантов этого ключа.',
    'occurrences': 'Всего найденных упоминаний во всём корпусе.',
    'document_count': 'Число PDF с найденными упоминаниями.',
    'variants': 'Найденные варианты написания этого ключа, разделённые |.',
    'example_document': 'Относительный путь к одному документу с примером.',
    'example_page': 'Физическая страница примера, начиная с 1.',
    'example_context': 'Один контекст: предпочтение полному окну, затем большей оценке модели. Остальные контексты в abbreviations.csv.',
    'canonical_key': FIELDS['canonical_key'],
}
DOC_FIELDS = {
    'source_file': 'Путь к входному PDF. Одна строка на каждый найденный файл, включая повреждённые.',
    'document_title': FIELDS['document_title'],
    'product_family': FIELDS['product_family'],
    'total_pages': 'Всего физических страниц; пусто, если PDF не удалось открыть.',
    'pages_with_ner': 'Страницы, на которых NER успешно завершилась, включая страницы без находок.',
    'pages_for_ocr_review': 'Страницы с почти отсутствующим текстом: возможны сканы, пустые страницы и обложки.',
    'failed_pages': 'Ошибки чтения или NER.',
    'pending_pages': 'Страницы, на которых NER ещё не выполнена.',
    'mentions': 'Количество найденных упоминаний в документе.',
    'unique_abbreviations': 'Количество различных ключей в документе.',
    'status': 'complete / complete_with_ocr_review / partial / error / not_processed.',
    'error': 'Описание ошибки файла, если есть.',
    'source_sha256': 'Хеш содержимого файла. Одинаковые хеши помогают заметить копии PDF под разными именами.',
}


def readable(text):
    return re.sub(r'\s+', ' ', text).strip()


def complete_sentence(text):
    return sum(c.isalpha() for c in text) >= 2 and bool(re.search(r'[.!?…][»”"\'\)\]]*$', text.strip()))


def edge_lines(text):
    lines, offset = [], 0
    for line in text.splitlines(keepends=True):
        if line.strip():
            lines.append((offset, offset + len(line), line.strip()))
        offset += len(line)
    # Only outermost lines: do not erase headings just because they recur.
    return [lines[i] for i in sorted({0, len(lines) - 1})] if lines else []


class DocumentContext:
    """Offset-preserving text and sentence indexes, one document at a time.

    Never bridge an unreadable/scanned page. Whitespace normalization only affects
    the readable quote; raw page offsets are supplied in context_fragments.
    """
    def __init__(self, pages, protected=None):
        from razdel import sentenize
        self.pages = {p['page']: p for p in pages}
        self.locations = {}
        self.groups = []
        protected = protected or {}
        headers = Counter()
        for page in pages:
            headers.update({line for _, _, line in edge_lines(page['text']) if len(line) < 160})
        pending = []
        previous = None

        def flush():
            if not pending:
                return
            pieces, starts, numbers, length, cuts = [], [], [], 0, {0: False}
            for p, masked in pending:
                self.locations[p['page']] = (len(self.groups), length)
                starts.append(length)
                numbers.append(p['page'])
                pieces.append(masked)
                # A glossary lists short forms on separate lines. Treat entries
                # as structured blocks instead of one sentence spanning pages.
                entries, strong_entries = [], 0
                for m in re.finditer(r'(?m)^[ \t]*([A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё0-9/+_-]{1,24})[ \t]*$', masked):
                    term = m.group(1)
                    following = masked[m.end():].lstrip('\r\n').split('\n', 1)[0]
                    if term[0].isupper() and any(c.isalpha() for c in following):
                        entries.append(m.start())
                        strong_entries += int(sum(c.isupper() for c in term) >= 2 and len(following.split()) >= 2)
                if strong_entries >= 3:
                    for offset in entries:
                        cuts[length + offset] = True
                    cuts[length + len(masked)] = False
                for m in re.finditer(r'(?m)^[ \t]*(?:[•●▪][ \t]+|(?:Глава|Приложение|Библиография)\b)', masked):
                    cuts.setdefault(length + m.start(), False)
                length += len(masked) + 1
            text = '\n'.join(pieces)
            # A PDF line break is not a sentence boundary; retain character offsets.
            flat = re.sub(r'\s', ' ', text)
            cuts[len(text)] = False
            boundaries = sorted(cuts)
            spans, structured = [], []
            for a, b in zip(boundaries, boundaries[1:]):
                if cuts[a]:
                    # Keep the complete glossary entry, even with internal dots.
                    left = a + len(flat[a:b]) - len(flat[a:b].lstrip())
                    right = b - (len(flat[a:b]) - len(flat[a:b].rstrip()))
                    if left < right:
                        spans.append((left, right))
                        structured.append(True)
                else:
                    for s in sentenize(flat[a:b]):
                        spans.append((a + s.start, a + s.stop))
                        structured.append(False)
            self.groups.append({'text': text, 'spans': spans, 'ends': [b for a, b in spans],
                                'starts': starts, 'numbers': numbers, 'structured': structured})
            pending.clear()

        for p in pages:
            if previous is not None and p['page'] != previous + 1:
                flush()
            previous = p['page']
            if p['status'] in {'text_error', 'needs_ocr'} or not p['text'].strip():
                flush()
                continue
            masked = p['text']
            for a, b, line in edge_lines(masked):
                page_number = bool(re.fullmatch(r'(?:\d{1,6}|[ivxlcdmIVXLCDM]{1,12})', line))
                repeated = len(line) < 160 and headers[line] >= 3
                intersects = any(a < end and start < b for start, end in protected.get(p['page'], []))
                if (page_number or repeated) and not intersects:
                    masked = masked[:a] + ' ' * (b - a) + masked[b:]
            pending.append((p, masked))
        flush()

    def get(self, page, start, end):
        group_i, base = self.locations[page]
        group = self.groups[group_i]
        a, b = base + start, base + end
        spans = group['spans']
        first = min(bisect.bisect_right(group['ends'], a), len(spans) - 1)
        last = min(bisect.bisect_right(group['ends'], b - 1), len(spans) - 1)
        if first < 0 or not (spans[first][0] <= a and b <= spans[last][1]):
            raise ValueError(f'Sentence boundaries do not contain mention on page {page}')
        left, right = first, last
        before = after = 0
        target = group['text'][spans[first][0]:spans[last][1]]
        is_structured = any(group['structured'][first:last + 1]) or not complete_sentence(target)
        def natural(i):
            return not group['structured'][i] and complete_sentence(group['text'][slice(*spans[i])])
        if is_structured:
            left, right = max(0, first - 3), min(len(spans) - 1, last + 3)
            before = sum(natural(i) for i in range(left, first))
            after = sum(natural(i) for i in range(last + 1, right + 1))
        else:
            while left > 0 and before < 3:
                left -= 1
                before += int(natural(left))
            while right + 1 < len(spans) and after < 3:
                right += 1
                after += int(natural(right))
        ca, cb = spans[left][0], spans[right][1]
        starts, numbers = group['starts'], group['numbers']
        p0 = max(0, bisect.bisect_right(starts, ca) - 1)
        p1 = max(0, bisect.bisect_right(starts, cb - 1) - 1)
        fragments = []
        for i in range(p0, p1 + 1):
            number = numbers[i]
            raw = self.pages[number]['text']
            fstart, fend = max(0, ca - starts[i]), min(len(raw), cb - starts[i])
            if fend > fstart:
                fragments.append({'physical_page': number, 'start': fstart, 'end': fend,
                                  'raw_text': raw[fstart:fend]})
        return {'sentence': readable(target), 'context': readable(group['text'][ca:cb]),
                'sentences_before': before, 'sentences_after': after,
                'context_start_page': numbers[p0], 'context_end_page': numbers[p1],
                'context_status': 'structured_text' if is_structured else 'full_window' if before == after == 3 else 'limited',
                'context_fragments': fragments}


def description():
    text = '''Описание результатов поиска аббревиатур

abbreviations.csv — основная подробная таблица. Одна строка = одно найденное
упоминание в конкретном месте PDF. Повторы аббревиатуры сохраняются, потому что
страницы, контексты и возможные значения различаются. occurrences — готовый итог
по всему корпусу, его нельзя суммировать по строкам этой таблицы.

dictionary.csv — краткий словарь: одна строка = один canonical_key. Именно здесь
удобно сортировать аббревиатуры по частоте и количеству документов.

documents.csv — проверка охвата всех входных PDF, включая файлы без находок и ошибки.

Счётчики относятся к найденным моделью упоминаниям. Модель может пропускать
сокращения и давать ложные находки. Расшифровки не извлекаются на этом этапе.
Группировка написаний не означает совпадение значений. Копии PDF по разным путям
учитываются отдельно; их можно обнаружить по source_sha256 в documents.csv.

Контекст
Предложение с находкой + до 3 полных предложений до и после. Разбиение выполняет
razdel. Текст соседних страниц берётся только из того же документа. На краю PDF,
возле сканов или ошибок текста предложений может быть меньше — это показывают
sentences_before / sentences_after и context_status. Разрывы строк заменены
пробелами. Крайние строки с номерами страниц и повторяющиеся колонтитулы
маскируются, но строки с найденными упоминаниями сохраняются. Исходные цитаты
и точные смещения по страницам остаются в context_fragments внутри mentions.jsonl.
В таблицах, коде и перечнях может не быть обычных предложений; автоматическое
разбиение не гарантирует правильных границ для любой PDF-вёрстки.
Если найдена запись словаря или незавершённый текстовый блок, сохраняются сам блок
и до 3 соседних блоков до/после; context_status=structured_text. Эти блоки не
выдаются за полные предложения. Колонки sentences_before/after считают только
распознанные полные предложения, не строки словаря.

CSV: UTF-8 с BOM, разделитель — точка с запятой. Если Excel открыл всё в одной
колонке: Данные → Из текста/CSV → UTF-8 → разделитель «;». В Python:
pd.read_csv("results_all/abbreviations.csv", sep=";", encoding="utf-8-sig")
CSV не обрезается по лимитам Excel: для очень больших файлов используйте Python
или SQLite. В Excel длинные контексты могут отображаться не полностью.

'''
    for filename, fields in [('abbreviations.csv', FIELDS), ('dictionary.csv', DICT_FIELDS), ('documents.csv', DOC_FIELDS)]:
        text += filename + '\n' + '\n'.join(f'{k}: {v}' for k, v in fields.items()) + '\n\n'
    return text + '''Служебные файлы
mentions.jsonl: те же упоминания, дополнительно точные позиции, хеш PDF, модель,
статус candidate и expansion=null. Одна JSON-запись на строку.
progress.sqlite3: исходный текст и прогресс; не удаляйте для продолжения обработки.
issues.json: страницы для OCR/проверки и ошибки.
run.json: настройки, версии, охват, дата экспорта, статус полного/частичного результата.
RUN_STATUS.txt: короткий итог. Если запуск прерван до экспорта, здесь остаётся
running. CSV предыдущего запуска в таком случае не считаются обновлёнными.
'''


def csv_value(value):
    # Protect text cells opened by spreadsheet programs. JSONL retains exact text.
    if isinstance(value, str) and value.lstrip().startswith(('=', '+', '-', '@')):
        return "'" + value
    return value


def write_row(writer, row):
    writer.writerow({k: csv_value(row.get(k, '')) for k in writer.fieldnames})


def atomic_json(path, data):
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def export_results(db, output, config, errors, expected_files=None, interrupted=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summaries, doc_counts = {}, Counter()
    doc_mentions, doc_unique = Counter(), defaultdict(set)
    for doc_id, data in db.execute('SELECT doc_id,data FROM mentions'):
        e = json.loads(data)
        key = e['canonical_key']
        item = summaries.setdefault(key, {'occurrences': 0, 'files': set(), 'variants': Counter(), 'best': None})
        item['occurrences'] += 1
        item['files'].add(doc_id)
        item['variants'][e['abbreviation']] += 1
        doc_counts[doc_id, key] += 1
        doc_mentions[doc_id] += 1
        doc_unique[doc_id].add(key)
    temp_csv = output / 'abbreviations.csv.tmp'
    temp_jsonl = output / 'mentions.jsonl.tmp'
    documents = db.execute('SELECT id,path,sha256,title,family,page_count FROM documents ORDER BY path').fetchall()
    with temp_csv.open('w', encoding='utf-8-sig', newline='') as cf, temp_jsonl.open('w', encoding='utf-8') as jf:
        writer = csv.DictWriter(cf, fieldnames=list(FIELDS), delimiter=';')
        writer.writeheader()
        for doc_id, path, sha, title, family, total in documents:
            if not doc_mentions[doc_id]:
                continue
            LOG.info('Сохранение контекстов: %s', path)
            pages = [dict(zip(['page', 'printed_page', 'status', 'text', 'section'], row)) for row in db.execute(
                'SELECT page,printed_page,status,text,section FROM pages WHERE doc_id=? ORDER BY page', (doc_id,))]
            protected = defaultdict(list)
            for page, a, b in db.execute('SELECT page,start,end FROM mentions WHERE doc_id=?', (doc_id,)):
                protected[page].append((a, b))
            contexts = DocumentContext(pages, protected)
            page_meta = {p['page']: p for p in pages}
            for page, data in db.execute('SELECT page,data FROM mentions WHERE doc_id=? ORDER BY page,start', (doc_id,)):
                e = json.loads(data)
                if page_meta[page]['text'][e['start']:e['end']] != e['abbreviation']:
                    raise ValueError(f'Abbreviation offsets do not match saved page: {path}, {page}')
                e.update(contexts.get(page, e['start'], e['end']))
                key = e['canonical_key']
                e.update(occurrences=summaries[key]['occurrences'], document_count=len(summaries[key]['files']),
                         occurrences_in_document=doc_counts[doc_id, key], physical_page=page,
                         document_name=Path(path).name, source_file=path, source_sha256=sha,
                         document_title=title or Path(path).stem, product_family=family,
                         section=page_meta[page]['section'], printed_page=page_meta[page]['printed_page'],
                         status='candidate', expansion=None, backend=config['backend'], model=config['model'])
                write_row(writer, e)
                jf.write(json.dumps(e, ensure_ascii=False) + '\n')
                rank = (e['context_status'] == 'full_window', e['sentences_before'] + e['sentences_after'], e['score'])
                best = summaries[key]['best']
                if best is None or rank > best[0]:
                    summaries[key]['best'] = (rank, path, page, e['context'])
            del contexts
    temp_csv.replace(output / 'abbreviations.csv')
    temp_jsonl.replace(output / 'mentions.jsonl')
    temp = output / 'dictionary.csv.tmp'
    with temp.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(DICT_FIELDS), delimiter=';')
        writer.writeheader()
        for key, item in sorted(summaries.items(), key=lambda kv: (-kv[1]['occurrences'], kv[0])):
            variants = sorted(item['variants'], key=lambda v: (-item['variants'][v], v))
            _, path, page, context = item['best']
            write_row(writer, dict(abbreviation=variants[0], occurrences=item['occurrences'],
                                  document_count=len(item['files']), variants=' | '.join(variants),
                                  example_document=path, example_page=page, example_context=context, canonical_key=key))
    temp.replace(output / 'dictionary.csv')
    issues = [{'source_file': p, 'physical_page': n, 'status': s, 'error': e}
              for p, n, s, e in db.execute("SELECT d.path,p.page,p.status,p.error FROM pages p JOIN documents d ON d.id=p.doc_id WHERE p.status IN ('needs_ocr','error','text_error') ORDER BY d.path,p.page")]
    issues.extend(errors)
    expected_files = sorted(expected_files if expected_files is not None else [d[1] for d in documents])
    by_path = {d[1]: d for d in documents}
    file_errors = {e['source_file']: e.get('error', '') for e in errors if e.get('source_file')}
    coverage = Counter({key: 0 for key in ['documents', 'pages_in_documents', 'pages_with_ner',
                                          'pages_for_ocr_review', 'failed_pages', 'pending_pages',
                                          'completed_documents']})
    temp = output / 'documents.csv.tmp'
    with temp.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(DOC_FIELDS), delimiter=';')
        writer.writeheader()
        for path in expected_files:
            if path not in by_path:
                write_row(writer, {'source_file': path, 'status': 'error' if path in file_errors else 'not_processed', 'error': file_errors.get(path, '')})
                continue
            doc_id, _, sha, title, family, total = by_path[path]
            counts = Counter(dict(db.execute('SELECT status,COUNT(*) FROM pages WHERE doc_id=? GROUP BY status', (doc_id,))))
            failed = counts['error'] + counts['text_error']
            pending = total - counts['ok'] - counts['needs_ocr'] - failed
            status = 'error' if failed or path in file_errors else 'partial' if pending else 'complete_with_ocr_review' if counts['needs_ocr'] else 'complete'
            write_row(writer, dict(source_file=path, document_title=title or Path(path).stem, product_family=family,
                                  total_pages=total, pages_with_ner=counts['ok'], pages_for_ocr_review=counts['needs_ocr'],
                                  failed_pages=failed, pending_pages=pending, mentions=doc_mentions[doc_id],
                                  unique_abbreviations=len(doc_unique[doc_id]), status=status,
                                  error=file_errors.get(path, ''), source_sha256=sha))
            coverage.update(documents=1, pages_in_documents=total, pages_with_ner=counts['ok'],
                            pages_for_ocr_review=counts['needs_ocr'], failed_pages=failed, pending_pages=pending)
            if status in {'complete', 'complete_with_ocr_review'}:
                coverage['completed_documents'] += 1
    temp.replace(output / 'documents.csv')
    coverage['input_documents'] = len(expected_files)
    coverage['unopened_documents'] = len(set(expected_files) - set(by_path))
    versions = {}
    for name in ['torch', 'transformers', 'gliner', 'pymupdf', 'tokenizers', 'razdel']:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    status = 'interrupted' if interrupted else 'partial' if errors or coverage['failed_pages'] or coverage['pending_pages'] or coverage['unopened_documents'] else 'complete_with_ocr_review' if coverage['pages_for_ocr_review'] else 'complete'
    report = dict(config=config, package_versions=versions, coverage=dict(coverage), mentions=sum(doc_mentions.values()),
                  unique_surface_forms=len(summaries), issues=len(issues), status=status,
                  exported_at=datetime.now(timezone.utc).isoformat(),
                  context_method='razdel 0.5.0; 3 before + target sentence + 3 after; within one PDF',
                  csv_encoding='utf-8-sig', csv_delimiter=';',
                  note='NER candidates only. Counts describe model detections, not all actual occurrences. Expansions are not inferred.')
    atomic_json(output / 'issues.json', issues)
    atomic_json(output / 'run.json', report)
    (output / 'TABLE_DESCRIPTION.txt').write_text(description(), encoding='utf-8-sig')
    (output / 'RUN_STATUS.txt').write_text(f"{status}\nPDF во входной папке: {len(expected_files)}\nPDF с завершённой обработкой: {coverage['completed_documents']}\nУпоминаний: {report['mentions']}\nРазных сокращений: {len(summaries)}\nСтраниц для OCR/проверки: {coverage['pages_for_ocr_review']}\nОшибок страниц: {coverage['failed_pages']}\n", encoding='utf-8-sig')
    return report
