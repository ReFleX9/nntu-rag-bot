"""Select original university files for document requests."""
import re
from urllib.parse import urlsplit

GROUP = re.compile(r"(?<!\w)\d{2}\s*[-–—]\s*[а-яёa-z]{2,8}(?:[-–—]\d{1,2})?(?!\w)", re.I)
PROGRAM = re.compile(r"(?<!\d)\d{2}\.\d{2}\.\d{2}(?!\d)")
YEAR = re.compile(r"(?<!\d)(20\d{2}|2\d)\s*[/–—-]\s*(20\d{2}|2\d)(?!\d)")
TYPE_PATTERNS = {
    "calendar": r"календарн.{0,35}(?:график|учебн)|учебн.{0,20}график|\bkug[_-]",
    "exam": r"расписани.{0,30}(?:экзамен|сесси|вступительн)|rasp[_-]?(?:exam|ekz|sess|vstup|gia)",
    "schedule": r"расписани|raspisanie|rasp[_-](?:zanyat|mag)",
    "tariffs": r"(?:тариф|плат[аыу]|стоимост).{0,35}(?:общежит|проживан)|plata_za_prozhivanie",
    "study_plan": r"учебн(?:ый|ого|ые) план|uchebn.{0,5}plan|uchebnyj_plan",
    "practice_form": r"договор.{0,40}практик|dogovor_praktiki",
    "practice_catalog": r"баз[аыу]\s+практик|baza-praktik|bazy_praktik",
    "syllabus": r"рабочи[ея]\s+программ|\bрп[дп]\b",
    "educational_material": r"учебн\w+\s+пособи|методич\w+\s+(?:указани|рекомендац)|\bметодичк",
    "program_description": r"описани\w+\s+(?:образовательн\w+\s+)?программ|\bопоп\b",
    "assessment_material": r"оценочн\w+\s+(?:материал|средств)|\bфос\b",
    "practice_policy": r"положени\w+.{0,40}практик",
}


def groups(text):
    return {re.sub(r"\s+", "", match[0]).lower().replace("–", "-").replace("—", "-")
            for match in GROUP.finditer(text)}


def years(text):
    result = set()
    for first, second in YEAR.findall(text):
        start, end = (int(n) + (2000 if len(n) == 2 else 0) for n in (first, second))
        if end == start + 1:
            result.add(f"{start}/{end}")
    return result


def document_type(text):
    return next((kind for kind, pattern in TYPE_PATTERNS.items()
                 if re.search(pattern, text, re.I)), None)


def official_file(url):
    parts = urlsplit(url)
    host = parts.hostname or ""
    return (parts.scheme == "https" and (host == "nntu.ru" or host.endswith(".nntu.ru"))
            and re.search(r"\.(?:pdf|xlsx?|docx?|ods|csv|rtf)$", parts.path, re.I))


def find_file(question, catalog, scope):
    kind = document_type(question)
    exact_files = [row for row in catalog if len(row["title"]) > 5
                   and row["title"].casefold() in question.casefold()
                   and official_file(row["url"])]
    if exact_files and len({row["url"] for row in exact_files}) == 1:
        return "Откройте полный файл НГТУ.", exact_files[:1]
    if not kind:
        return None
    wanted_groups, wanted_years = groups(question), years(question)
    programs = set(PROGRAM.findall(question))
    # A generic navigation question can use the normal university web index.
    if not (wanted_groups or wanted_years or programs or scope == "supplemental"
            or re.search(r"файл|скача|пришли|отправь|pdf|xlsx", question, re.I)):
        return None
    candidates = []
    for row in catalog:
        if row["type"] != kind or not official_file(row["url"]):
            continue
        if not programs.issubset(row.get("program_codes", [])):
            continue
        if scope == "core" and row["scope"] != "core" and not wanted_years:
            continue
        if scope == "supplemental" and row["scope"] == "core" and not wanted_years:
            continue
        if not wanted_groups.issubset(row["groups"]) or not wanted_years.issubset(row["years"]):
            continue
        candidates.append(row)
    by_url = {row["url"]: row for row in candidates}
    if not by_url:
        return "Не нашёл файл с подтверждённым совпадением по запросу. Уточните институт, группу и учебный год.", []
    if len(by_url) > 1:
        names = ", ".join(dict.fromkeys(row["title"] for row in list(by_url.values())[:5]))
        return f"Подходит несколько файлов ({len(by_url)}). Укажите точное имя нужного файла. Варианты: {names}.", []
    row = next(iter(by_url.values()))
    return "Откройте полный файл НГТУ. Данные сверяйте по таблице в оригинале.", [row]
