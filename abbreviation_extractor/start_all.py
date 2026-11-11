"""Jupyter: %run start_all.py — all PDFs under the adjacent pdfs folder."""
from pathlib import Path
from extract_abbreviations import main

if __name__ == '__main__':
    folder = Path(__file__).resolve().parent
    code = main([str(folder / 'pdfs'), '--output', str(folder / 'results_corpus')])
    if code:
        print('Обработка завершилась не полностью. Подробности: results_corpus/RUN_STATUS.txt и issues.json.')
