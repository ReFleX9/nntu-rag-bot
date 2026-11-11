"""Выгрузка выдачи по evaluation__train.xlsx; метрики — только при наличии qrels."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from kb import add_search_args, load_kb, retrieve


def read_questions(path: str, sheet_name: str | None = None):
    from openpyxl import load_workbook
    workbook = load_workbook(path, read_only=True, data_only=True)
    sheet = workbook[sheet_name] if sheet_name else workbook.active
    rows = sheet.iter_rows(values_only=True)
    headers = next(rows)
    mapping = {str(value).strip(): i for i, value in enumerate(headers) if value is not None}
    if "Вопрос" not in mapping:
        raise ValueError("В первой строке Excel нужна колонка 'Вопрос'")
    try:
        for row_number, values in enumerate(rows, 2):
            question = values[mapping["Вопрос"]]
            if question is None or not str(question).strip():
                continue
            extra = {name: values[mapping[name]] for name in ("Аббревиатуры", "Эталонный ответ", "Обязательные факты")
                     if name in mapping}
            yield {"row": row_number, "question": str(question).strip(), "reference": extra}
    finally:
        workbook.close()


def metric_row(retrieved: list[str], relevant: set[str], k: int) -> dict:
    if not relevant:
        raise ValueError("Для Recall/MRR нужен хотя бы один размеченный релевантный чанк")
    ranked = retrieved[:k]
    matches = set(ranked) & relevant
    first = next((i for i, cid in enumerate(ranked, 1) if cid in relevant), None)
    return {"hit": float(bool(matches)), "recall": len(matches)/len(relevant),
            "mrr": 1/first if first else 0.0}


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    add_search_args(parser)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--sheet")
    parser.add_argument("--limit", type=int, default=0, help="0: все вопросы")
    parser.add_argument("--qrels", help="JSONL: question и relevant_chunk_ids (список)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("limit должен быть неотрицательным")
    kb = load_kb(args)
    qrels = {}
    if args.qrels:
        with Path(args.qrels).open(encoding="utf-8-sig") as stream:
            for line in stream:
                if not line.strip():
                    continue
                record = json.loads(line)
                question = record["question"].strip()
                if question in qrels:
                    raise ValueError(f"Повторный вопрос в qrels: {question}")
                values = record["relevant_chunk_ids"]
                if not isinstance(values, list) or not all(isinstance(x, str) for x in values):
                    raise ValueError("relevant_chunk_ids должен быть списком строк")
                relevant = set(values)
                unknown = relevant - kb.by_id.keys()
                if unknown:
                    raise ValueError(f"Разметка ссылается на отсутствующие chunk_id: {sorted(unknown)[:5]}")
                qrels[question] = relevant
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Результат уже существует: {output}. Укажите другой --output.")
    metrics, count = [], 0
    with output.open("w", encoding="utf-8") as stream:
        for question in read_questions(args.questions, args.sheet):
            if args.limit and count >= args.limit:
                break
            # Answers, mandatory facts and acronyms are retained for review only.
            # They are never supplied to the search query or indexed as knowledge.
            result = retrieve(kb, question["question"], args)
            record = {**question, **result}
            relevant = qrels.get(question["question"])
            if relevant:
                score = metric_row([h["chunk_id"] for h in result["hits"]], relevant, args.top_k)
                record["metrics"] = score
                metrics.append(score)
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            count += 1
            logging.info("Обработано вопросов: %s", count)
    summary = {"questions": count, "judged_questions": len(metrics), "top_k": args.top_k, "mode": args.mode}
    if metrics:
        summary["metrics"] = {f"{name}@{args.top_k}": sum(r[name] for r in metrics)/len(metrics)
                              for name in ("hit", "recall", "mrr")}
    else:
        summary["note"] = ("Без relevant_chunk_ids Recall/Hit/MRR не вычисляются. "
                           "Эталонный ответ сам по себе не является разметкой релевантных чанков.")
    output.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
