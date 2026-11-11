# База знаний по chunks(4).jsonl: LangChain, BM25, FAISS

Готовые чанки загружаются без повторного разбиения. Один исходный чанк соответствует
одному `Document` LangChain и одному вектору FAISS. `text`, `chunk_id`, `document_id`,
PDF-страница, раздел, тип, OCR-признаки и исходные `sources` сохраняются.

Источник для этой версии: полный `chunks(4).jsonl`, 39 082 чанка из 139 документов.
Отдельный `chunks_ocr.jsonl` добавлять не нужно: OCR уже входит в полный файл.
Результаты проверки находятся в `AUDIT.md` и `checks/chunks_audit.json`.

## Как работает поиск

Для корпуса НГТУ используйте модель `Qwen/Qwen3-Embedding-0.6B` из
`vendor/nntu-models/qwen3-embedding-0.6b` и локальную Ollama с моделью
`nntu-qwen:4b`. Команда `ask` по умолчанию определяет язык вопроса,
переводит запрос на русский для поиска и отвечает на исходном языке.
Язык можно задать явно через `--language ru|en|tk|uz`. Для `ru` перевод пропускается.

Пример из корня проекта после установки зависимостей в `vendor/nntu-deps`:

```powershell
$env:PYTHONPATH = (Resolve-Path vendor/nntu-deps).Path
$env:OLLAMA_MODELS = Join-Path (Get-Location) 'vendor/nntu-ollama/models'
$env:OLLAMA_HOST = '127.0.0.1:11434'
$env:OLLAMA_NO_CLOUD = '1'
# Запустить в отдельном терминале, если Ollama ещё не работает:
.\vendor\nntu-ollama\ollama.exe serve
```

В терминале поиска из корня проекта:

```powershell
$env:PYTHONPATH = (Resolve-Path vendor/nntu-deps).Path
$env:USE_TF = '0'
$rag = 'rag_hybrid_langchain/rag_hybrid'
$corpus = 'data/nntu_site/extracted'
$model = 'vendor/nntu-models/qwen3-embedding-0.6b'
python -S -X utf8 "$rag/nntu_token_audit.py" "$corpus/chunks.jsonl" --tokenizer-path $model --output "$corpus/qwen_token_audit.json"
python -S -X utf8 "$rag/kb.py" build --chunks "$corpus/chunks.jsonl" --out "$corpus/knowledge_base" --model $model --revision 97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3 --local-files-only --device cpu --max-length 2400 --instruction 'Retrieve the passage that directly answers the question about Nizhny Novgorod State Technical University.'
python -S -X utf8 "$rag/kb.py" ask --index "$corpus/knowledge_base" --local-files-only --llm-model nntu-qwen:4b --num-ctx 4096 --max-context-chars 5000 --query 'Where can I find the class schedule?'
```

Аудит токенов требует Python 3.12 или новее. На текущем корпусе максимальная
длина входа составляет 2 310 токенов, поэтому для сборки выбран лимит 2 400.

1. Для индексации к тексту добавляются имя PDF, папка продукта и известный раздел.
2. `Qwen/Qwen3-Embedding-0.6B` строит нормализованные векторы. Инструкция добавляется
   только к вопросу, согласно формату модели.
3. FAISS `IndexFlatIP` ищет по скалярному произведению. Для нормализованных векторов
   это косинусное сходство. Этот индекс выполняет точный поиск.
4. `BM25Retriever` LangChain ищет по словам. Используются приведение регистра,
   `ё → е`, русский Snowball-стемминг и сохранение технических обозначений:
   `pg_hba.conf`, `--restore-data`, `TLS`, `1.3`. Сетевые загрузки словарей NLTK не нужны.
5. Weighted Reciprocal Rank Fusion объединяет первые 40 результатов каждого метода:
   `score(d) = sum(weight_i / (60 + rank_i(d)))`, ранги начинаются с 1.
   Веса нормализуются, стартовые значения — 0.5/0.5. Это начальная настройка.
6. Возвращаются до 8 чанков. Полностью одинаковые тексты внутри одного документа
   объединяются в выдаче с сохранением дополнительных ссылок в `duplicate_citations`.
   В базе остаются все исходные записи. Разные документы и версии не объединяются.
7. Для контекста добавляется по одному соседу в каждую сторону по исходным ID.
   Сосед должен принадлежать тому же документу, быть на той же или соседней странице
   и не переходить в другой известный раздел. Фильтры распространяются и на соседей.

RRF-score и BM25-score не являются вероятностью правильного ответа. Оценки двух
методов не складываются напрямую. У запроса без лексических совпадений BM25
возвращает пустую выдачу; FAISS продолжает искать по смыслу.

## Установка

Рекомендуемая среда: Python 3.11 или 3.12. Распакуй папку `rag_hybrid` в проект.
Команды ниже выполняются из неё. Укажи свой путь к полному файлу чанков.

```powershell
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Если у тебя уже работает PyTorch с CUDA, используй это окружение. `faiss-cpu`
выполняет поиск на CPU, а Qwen при `--device cuda` вычисляет эмбеддинги на GPU.
Для 39 тысяч векторов такой вариант подходит как начальная реализация.
Зависимости поиска закреплены на версиях, использованных при проверке.
`sentence-transformers` ограничен веткой 5.x, `transformers` — 4.x для согласованных API.

## 1. Проверка входного файла

```powershell
python kb.py audit --chunks "F:\Project\Track_1-Tensor-Kb1zw\prepared\chunks.jsonl" --output audit.json
```

Путь `prepared\chunks.jsonl` — пример. Можно передать исходное имя `chunks(4).jsonl`.
Команда `audit` использует стандартную библиотеку Python и не загружает модель.

## 2. Построение индекса

```powershell
python kb.py build --chunks "F:\Project\Track_1-Tensor-Kb1zw\prepared\chunks.jsonl" --out knowledge_base --device cuda --batch-size 8 --max-length 2048
```

Для твоей RTX 4070 SUPER 12 GB `batch-size=8` — консервативная начальная настройка,
но фактическое потребление зависит от остальных процессов. Если возникает CUDA OOM,
уменьши batch до 4 или 2. Лимит длины сохраняй, чтобы не обрезать таблицы.
В этой среде GPU-запуск не проверялся.

Первый запуск загружает модель. Уже скачанную полную модель можно указать локально:

```powershell
python kb.py build --chunks ".\chunks(4).jsonl" --out knowledge_base --device cuda --model "C:\models\Qwen3-Embedding-0.6B" --local-files-only
```

В локальном каталоге нужны веса и конфигурация Sentence Transformers, включая
pooling; одного токенизатора недостаточно. На CPU используй `--device cpu`.

У модели размерность 1024 и предельное окно 32K. Здесь `2048` — настроенный предел
одного входа при индексации. Настоящий токенизатор проверяет текст вместе с
метаданными перед вычислением вектора. Если предел превышен, программа выдаёт
ошибку вместо скрытого обрезания. На приложенном полном файле максимум — 1 680.

Создаются:

| Файл | Содержимое |
|---|---|
| `knowledge_base/index.faiss` | Векторы; около 153 MiB для 39 082 × 1024 float32 |
| `knowledge_base/documents.jsonl` | Исходные тексты и полные метаданные |
| `knowledge_base/manifest.json` | Параметры модели, ревизия при наличии, контрольные суммы и аудит |

Это постоянная база на диске. BM25 восстанавливается из сохранённых документов
при загрузке процесса. Его нужно загрузить один раз при старте приложения.
Файлы pickle не используются. При загрузке проверяются контрольные суммы и
согласованность числа/размерности векторов. Сохраняется именно модель построения.

Повторное построение выполняй в новый каталог, например `knowledge_base_v2`.
Смена модели, правил подготовки индексируемого текста или состава корпуса требует
перестроения обоих поисковых представлений. Обновление «на месте» не реализовано.

По умолчанию сохраняются все `indexable=true` без `indexing_exclusions`.
`needs_review` не служит автоматическим запретом. Дополнительно доступны
`--min-ocr-confidence 60` и `--exclude-needs-review`; эти флаги влияют на полноту базы.
Для начального сравнения качества оставь их выключенными.

## 3. Поиск и контекст

```powershell
python kb.py search --index knowledge_base --query "Как в документации eXpress расшифровывается CTS?" --top-k 8 --output search_result.json
```

В JSON есть найденные исходные тексты, ранги каждого метода, итоговый RRF-score,
собранный `context` и нумерованные `sources`. `--show-context` выводит контекст
в терминал. `--neighbors 0` отключает соседей. PDF-страницы остаются 1-based,
как в исходном JSONL. Ссылки на локальные PDF приложение может строить по
`document_id` относительно своего каталога `corpus`.

Бюджет контекста — `--max-context-chars 24000`. Это количество символов,
не токенов LLM. Целые чанки не режутся; не вместившиеся пропускаются.
`context_candidates` и `context_included` показывают, сколько блоков поместилось.
Для иной LLM подстрой этот бюджет по её токенизатору и контекстному окну.

Фильтр папки продукта применяется к BM25 и FAISS до выбора итоговых кандидатов:

```powershell
python kb.py search --index knowledge_base --query "Настройка pg_hba.conf" --source-folder postgres_pro
```

Поддерживаются `--source-folder`, `--document-id` (точное значение) и `--kind`.
Без явно указанного фильтра поиск идёт по всему корпусу. Автоматического определения
продукта и версии в этой версии нет.

Сравнивать методы можно через `--mode bm25`, `--mode faiss`, `--mode hybrid`.
Чтобы проверить BM25 ещё до загрузки Qwen:

```powershell
python -m pip install -r requirements-search.txt
python kb.py search --chunks ".\chunks(4).jsonl" --mode bm25 --query "Настройка pg_hba.conf"
```

## 4. Использование из Python / FastAPI

```python
from retrieval import KnowledgeBase, format_context

# Один раз при старте приложения.
kb = KnowledgeBase.load("knowledge_base", device="cuda")
retriever = kb.as_retriever(top_k=8, candidate_k=40)

question = "Как настроить резервное копирование?"
hits = retriever.invoke(question)  # Стандартный LangChain BaseRetriever.
documents = kb.expand_context(hits, neighbors=1)
context, sources = format_context(documents, max_chars=24000)
# Передай question + context в LLM, sources сохрани для интерфейса.
```

`retriever` поддерживает стандартный Runnable-интерфейс LangChain. Состояние
параметров поиска не меняется между запросами. При использовании GPU в веб-приложении
ограничь параллельные вызовы модели, например одним рабочим процессом/очередью.
Этот комплект предоставляет модуль для приложения и CLI, без отдельного HTTP-сервиса.

## 5. Необязательное получение ответа через Ollama

После проверки поиска можно подключить уже установленную локальную модель:

```powershell
python -m pip install -r requirements-llm.txt
python kb.py ask --index knowledge_base --query "Как в eXpress расшифровывается CTS?" --llm-model qwen3:8b --output answer.json
```

Должны работать Ollama и указанная модель. Команда не скачивает LLM автоматически.
Модель генерации отдельная от Qwen3-Embedding. `--num-ctx 16384` — исходная настройка
окна Ollama. Ответ формируется через `ChatPromptTemplate | ChatOllama | StrOutputParser`,
с указанием источников `[1]`, `[2]`. Промпт требует отвечать по найденным данным;
сам по себе он не гарантирует фактическую корректность ответа и цитат.

## 6. Проверка на evaluation__train.xlsx

В приложенном Excel пять непустых вопросов. Есть эталонные ответы и обязательные
факты, но нет релевантных `chunk_id`/PDF-страниц. Поэтому код выгружает выдачу,
а Recall@k и MRR@k считает только после добавления разметки источников.

```powershell
python evaluate.py --index knowledge_base --questions "..\evaluation__train.xlsx" --mode bm25 --output results_bm25.jsonl
python evaluate.py --index knowledge_base --questions "..\evaluation__train.xlsx" --mode faiss --output results_faiss.jsonl
python evaluate.py --index knowledge_base --questions "..\evaluation__train.xlsx" --mode hybrid --output results_hybrid.jsonl
```

`--limit 20` позволяет обрабатывать часть более крупного набора.
Вопрос передаётся в поиск в исходном виде. Эталонный ответ, обязательные факты
и аббревиатуры из столбцов не используются для запроса, индекса или подсказки модели.

Для каждого вопроса вручную укажи релевантные чанки в `qrels.jsonl`:

```json
{"question":"Точный текст вопроса из Excel","relevant_chunk_ids":["существующий_chunk_id_1","существующий_chunk_id_2"]}
```

Повтори запуск с `--qrels qrels.jsonl`. Появятся Hit@k, Recall@k и MRR@k по размеченным
вопросам; метрики относятся к основным найденным чанкам, до добавления соседей.
Разметка должна перечислять допустимые альтернативные источники, включая повторы,
если они тоже подходят. Для вопросов без ответа нужны отдельные оценки отказа.
Выходной `.summary.json` показывает число обработанных и размеченных вопросов.

Следующий этап: расширить вопросы по продуктам и типам задач, разметить источники,
сравнить три режима. Затем подобрать веса RRF/число кандидатов на отдельной части
разметки. Реранкер имеет смысл добавлять после этого сравнения. Пять примеров
не дают надёжной общей оценки. Для диалога с вопросами «а как это отключить?»
понадобится преобразование истории и нового вопроса в самостоятельный поисковый запрос.

## Проверки и ограничения

```powershell
python -m pip install -r requirements-test.txt
python -m pytest tests -q
```

Автотесты проверяют сохранение/загрузку реального FAISS с небольшими тестовыми
векторами, фильтрацию обеих веток, ранги, сохранность источников, OCR-фильтры,
соседние чанки, лимит контекста и обнаружение повреждённого индекса.
Отдельно проверен полный входной корпус и его длины настоящим токенизатором Qwen.
Подробности фактически выполненных запусков приведены в `AUDIT.md`.

## Документация компонентов

- [Qwen3-Embedding-0.6B: формат вопроса и параметры модели](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
- [LangChain BM25Retriever](https://reference.langchain.com/python/langchain-community/retrievers/bm25/BM25Retriever)
- [LangChain BaseRetriever](https://reference.langchain.com/python/langchain-core/retrievers/BaseRetriever)
- [FAISS: IndexFlatIP и нормализация](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)
