import json
from pathlib import Path


INPUT_FILE = Path("prepared/20260922T173154Z_2b40f93c/chunks.jsonl")

OCR_FILE = Path("prepared/chunks_ocr.jsonl")
NO_OCR_FILE = Path("prepared/chunks_no_ocr.jsonl")


def split_chunks():
    ocr_count = 0
    no_ocr_count = 0

    with (
        INPUT_FILE.open("r", encoding="utf-8") as source,
        OCR_FILE.open("w", encoding="utf-8") as ocr_out,
        NO_OCR_FILE.open("w", encoding="utf-8") as no_ocr_out,
    ):
        for line in source:
            line = line.strip()

            if not line:
                continue

            chunk = json.loads(line)

            if chunk.get("is_ocr", False):
                ocr_out.write(
                    json.dumps(chunk, ensure_ascii=False) + "\n"
                )
                ocr_count += 1
            else:
                no_ocr_out.write(
                    json.dumps(chunk, ensure_ascii=False) + "\n"
                )
                no_ocr_count += 1

    print(f"OCR чанков:     {ocr_count}")
    print(f"Без OCR:        {no_ocr_count}")
    print(f"Всего:          {ocr_count + no_ocr_count}")
    print()
    print(f"OCR:     {OCR_FILE.resolve()}")
    print(f"Без OCR: {NO_OCR_FILE.resolve()}")


if __name__ == "__main__":
    split_chunks()