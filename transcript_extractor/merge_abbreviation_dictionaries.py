"""Объединение полных словарей Шварца, правил и LLM. Python 3.10+, без зависимостей.

Объединяет уже подтверждённые результаты, не проверяет их правильность заново.
Кандидатские гипотезы не становятся подтверждёнными при объединении.
В dictionary_merged.csv сохраняется до --max-evidence ссылок на расшифровку;
полные основания и исходные объекты — в merged_evidence.jsonl.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import tempfile
from collections import Counter

csv.field_size_limit(100_000_000)


def js(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def norm(value):
    return ' '.join(str(value or '').split()).casefold()


def unique(items):
    return list({js(x): x for x in items}.values())


def array(raw, location):
    try:
        value = json.loads(raw) if raw else []
    except (ValueError, TypeError) as exc:
        raise ValueError(f'{location}: повреждён JSON: {exc}') from exc
    if not isinstance(value, list):
        raise ValueError(f'{location}: ожидается JSON-массив')
    return value


def read_dictionary(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        first = f.readline(); f.seek(0)
        delimiter = max(';\t,', key=first.count)
        reader = csv.DictReader(f, delimiter=delimiter)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)) or not {'abbreviation', 'status', 'expansions_json'} <= set(fields):
            raise ValueError(f'{path}: нужен полный словарь с abbreviation, status, expansions_json, не new_pairs.csv')
        rows = list(reader)
    seen = set()
    for number, row in enumerate(rows, 2):
        if None in row or any(x is None for x in row.values()):
            raise ValueError(f'{path}, запись {number}: повреждён CSV')
        key = norm(row.get('canonical_key') or row['abbreviation'])
        if not key or key in seen:
            raise ValueError(f'{path}, запись {number}: пустой или повторный canonical_key={key!r}')
        seen.add(key)
        row['canonical_key'] = key
    return fields, rows


def aliases(obj):
    result = {norm(obj['expansion'])}
    for item in obj.get('variants', []):
        text = item.get('text') if isinstance(item, dict) else item
        if isinstance(text, str) and text.strip():
            result.add(norm(text))
    return result


def group_expansions(records):
    # Only exact normalized names or explicitly recorded variants connect groups.
    # No stemming/fuzzy matching: different meanings must survive.
    groups = []
    for rec in records:
        names = aliases(rec['object'])
        hit = [g for g in groups if names & g['names']]
        combined = {'names': set(names), 'records': [rec]}
        for g in hit:
            combined['names'].update(g['names']); combined['records'].extend(g['records'])
            groups.remove(g)
        groups.append(combined)
    return groups


def build_merge(paths, max_evidence=3):
    headers, entries, inputs, review = [], {}, [], []
    for path in paths:
        fields, rows = read_dictionary(path)
        headers.extend(x for x in fields if x not in headers)
        inputs.append({'path': str(path.resolve()), 'rows': len(rows),
                       'confirmed': sum(norm(r['status']) == 'confirmed' for r in rows)})
        for row in rows:
            key = row['canonical_key']; source = str(path.resolve())
            slot = entries.setdefault(key, {'rows': [], 'records': []})
            slot['rows'].append((source, row))
            objects = array(row['expansions_json'], f'{path}: {key}')
            if norm(row['status']) != 'confirmed':
                if objects:
                    review.append({'canonical_key': key, 'reason': 'unconfirmed_hypotheses_excluded',
                                   'details_json': js({'source': source, 'objects': objects})})
                continue
            if not objects:
                raise ValueError(f'{path}: {key}: confirmed без расшифровок; исправьте входной словарь')
            for obj in objects:
                if not isinstance(obj, dict) or not isinstance(obj.get('expansion'), str) or not obj['expansion'].strip():
                    raise ValueError(f'{path}: {key}: неверный объект расшифровки')
                if not isinstance(obj.get('evidence', []), list) or not all(isinstance(e, dict) for e in obj.get('evidence', [])):
                    raise ValueError(f'{path}: {key}: evidence должен быть массивом объектов')
                if not isinstance(obj.get('variants', []), list):
                    raise ValueError(f'{path}: {key}: variants должен быть массивом')
                slot['records'].append({'source_dictionary': source,
                                        'row_confirmation_method': row.get('confirmation_method', ''),
                                        'row_evidence_refs': array(row.get('evidence_refs_json'), f'{path}: {key}: refs'),
                                        'object': obj})
    output, pairs, evidence_archive = [], [], []
    metadata_fields = [h for h in headers if h not in {'status', 'expansion_count', 'expansions_json', 'evidence_count',
                      'evidence_refs_json', 'confirmation_method'} and not h.startswith('merge_')]
    for key, slot in entries.items():
        row = dict(slot['rows'][0][1])
        # Inputs refer to the same corpus: occurrence counts are not added together.
        for field in metadata_fields:
            values = unique([r[field] for _, r in slot['rows'] if r.get(field)])
            if not row.get(field) and values:
                row[field] = values[0]
            if len(values) > 1:
                review.append({'canonical_key': key, 'reason': 'metadata_differs:' + field,
                               'details_json': js([{'source': src, 'value': r.get(field)} for src, r in slot['rows']])})
        merged = []
        for group in group_expansions(slot['records']):
            records = group['records']; originals = [r['object'] for r in records]
            name = originals[0]['expansion'].strip()
            texts = [name]
            for obj in originals:
                texts.append(obj['expansion'])
                texts.extend(x.get('text', '') if isinstance(x, dict) else x for x in obj.get('variants', []))
            methods = sorted({str(r['object'].get('confirmation_method') or r['row_confirmation_method'])
                              for r in records if r['object'].get('confirmation_method') or r['row_confirmation_method']})
            all_evidence = unique([e for obj in originals for e in obj.get('evidence', [])])
            # Prefer different documents, then fill remaining slots deterministically.
            diverse, rest, docs = [], [], set()
            for ev in all_evidence:
                doc = ev.get('document_id') or ev.get('title') or ev.get('reference')
                if doc and doc not in docs:
                    diverse.append(ev); docs.add(doc)
                else:
                    rest.append(ev)
            all_evidence = diverse + rest
            kept = all_evidence[:max_evidence] if max_evidence else all_evidence
            relations = sorted({str(rel) for obj in originals for rel in obj.get('relations', [])})
            merged_obj = {'expansion': name, 'variants': [{'text': t} for t in unique(texts) if isinstance(t, str) and t.strip()],
                          'evidence_count': len(kept), 'evidence': kept,
                          'confirmation_method': ';'.join(methods), 'relations': relations,
                          'merge_available_evidence_count': len(all_evidence),
                          'merge_source_dictionaries': sorted({r['source_dictionary'] for r in records})}
            merged.append(merged_obj)
            evidence_archive.append({'canonical_key': key, 'abbreviation': row['abbreviation'], 'expansion': name,
                                     'evidence': all_evidence, 'source_records': records})
            pairs.append({'abbreviation': row['abbreviation'], 'canonical_key': key, 'expansion': name,
                          'confirmation_method': ';'.join(methods), 'evidence_count': len(kept),
                          'evidence_json': js(kept), 'source_dictionaries_json': js(merged_obj['merge_source_dictionaries'])})
            if not all_evidence:
                review.append({'canonical_key': key, 'reason': 'confirmed_pair_without_structured_evidence',
                               'details_json': js({'expansion': name, 'sources': merged_obj['merge_source_dictionaries']})})
        if len(merged) > 1:
            review.append({'canonical_key': key, 'reason': 'multiple_expansions_check_meanings_or_translations',
                           'details_json': js([x['expansion'] for x in merged])})
        statuses = {norm(r['status']) for _, r in slot['rows']}
        status = 'confirmed' if merged else ('candidate' if 'candidate' in statuses or '' in statuses else norm(row['status']))
        refs = unique([e['reference'] for obj in merged for e in obj['evidence'] if e.get('reference')])
        row.update(canonical_key=key, status=status, expansion_count=str(len(merged)), expansions_json=js(merged),
                   evidence_count=str(sum(len(x['evidence']) for x in merged)), evidence_refs_json=js(refs),
                   confirmation_method=';'.join(sorted({x['confirmation_method'] for x in merged if x['confirmation_method']})),
                   merge_sources_json=js([src for src, _ in slot['rows']]))
        output.append(row)
    extra = ['canonical_key','status','expansion_count','expansions_json','evidence_count','evidence_refs_json',
             'confirmation_method','merge_sources_json']
    headers.extend(x for x in extra if x not in headers)
    summary = {'input_files': inputs, 'rows': len(output), 'statuses': dict(Counter(r['status'] for r in output)),
               'unique_confirmed_pairs': len(pairs), 'review_items': len(review), 'max_evidence_per_pair': max_evidence,
               'note': 'Union of input confirmations; no new semantic verification. evidence_count counts stored evidence objects, not total corpus occurrences. Full original counts and metadata remain in merged_evidence.jsonl.'}
    return headers, output, pairs, review, evidence_archive, summary


def atomic_write(path, writer):
    fd, name = tempfile.mkstemp(prefix=path.name, suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8-sig' if path.suffix == '.csv' else 'utf-8', newline='') as f:
            writer(f)
        os.replace(name, path)
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise


def write_csv(path, fields, rows):
    def write(f):
        w = csv.DictWriter(f, fieldnames=fields, delimiter=';'); w.writeheader(); w.writerows(rows)
    atomic_write(path, write)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, nargs='+', required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--max-evidence', type=int, default=3, help='Ссылок на расшифровку в основном словаре; 0 = все')
    args = p.parse_args(argv)
    try:
        if args.max_evidence < 0: raise ValueError('max-evidence должен быть >= 0')
        paths = [x.resolve() for x in args.inputs]
        if len(paths) != len(set(paths)): raise ValueError('Один входной файл указан несколько раз')
        names = ['dictionary_merged.csv','pairs_merged.csv','merge_review.csv','merged_evidence.jsonl','merge_summary.json']
        out = args.output_dir.resolve()
        if any(out / name in paths for name in names): raise ValueError('Нельзя перезаписывать входной словарь')
        headers, rows, pairs, review, archive, summary = build_merge(paths, args.max_evidence)
        out.mkdir(parents=True, exist_ok=True)
        write_csv(out/names[0], headers, rows)
        write_csv(out/names[1], ['abbreviation','canonical_key','expansion','confirmation_method','evidence_count','evidence_json','source_dictionaries_json'], pairs)
        write_csv(out/names[2], ['canonical_key','reason','details_json'], review)
        def write_archive(f):
            for item in archive: f.write(js(item)+'\n')
        atomic_write(out/names[3], write_archive)
        atomic_write(out/names[4], lambda f: json.dump(summary, f, ensure_ascii=False, indent=2))
        print(json.dumps(summary, ensure_ascii=False, indent=2)); return 0
    except (ValueError, OSError, csv.Error) as exc:
        print(f'Ошибка: {exc}', file=sys.stderr); return 1


if __name__ == '__main__':
    raise SystemExit(main())
