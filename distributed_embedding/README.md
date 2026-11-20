# Расчёт эмбеддингов r23 на двух компьютерах

Папка `r23/` содержит две последовательные части кандидата r23: 8 086 и
8 087 чанков. `plan.json` фиксирует SHA-256 исходного корпуса и каждой части.
Разделение не меняет текст, порядок и `chunk_id`. Оригинальный r23 остаётся
в локальном `work/v3_candidate_r23/core.jsonl`.

На обоих компьютерах нужны Python, зависимости
`rag_hybrid_langchain/rag_hybrid/requirements.txt` и одинаковая локальная
копия `Qwen3-Embedding-0.6B`, включая `model.safetensors` и `tokenizer.json`.
Скрипт сверяет хэши этих двух файлов перед объединением. Модель (~1,2 ГБ)
в Git не загружена. Её можно скопировать с основного ПК из
`vendor/nntu-models/qwen3-embedding-0.6b` на ноутбук. Пути к модели на двух
компьютерах могут различаться.

## На основном ПК

```powershell
python -m pip install -r rag_hybrid_langchain/rag_hybrid/requirements.txt
python distributed_embedding/two_pc.py worker `
  --folder distributed_embedding/r23 --part 0 `
  --model-dir 'C:\Users\lisit\OneDrive\Desktop\rag_gazprom\vendor\nntu-models\qwen3-embedding-0.6b' `
  --device cpu --batch-size 4 --out work/two_pc_part0
```

## На ноутбуке

Клонируйте репозиторий с пропуском старых больших файлов Git LFS:

```powershell
$env:GIT_LFS_SKIP_SMUDGE='1'
git clone https://github.com/ReFleX9/nntu-rag-bot.git
cd nntu-rag-bot
python -m pip install -r rag_hybrid_langchain/rag_hybrid/requirements.txt
python distributed_embedding/two_pc.py worker `
  --folder distributed_embedding/r23 --part 1 `
  --model-dir C:\path\to\qwen3-embedding-0.6b `
  --device auto --batch-size 4 --out work/two_pc_part1
```

`--device auto` выберет CUDA при наличии совместимой NVIDIA и сборки PyTorch
с CUDA, иначе CPU. Для GPU можно увеличить `--batch-size` до 8 или 16 при
достаточном объёме видеопамяти. Обе задачи работают независимо. После каждого
шарда сохраняется контрольная точка; повтор той же команды продолжает работу.
Не меняйте файлы `r23/` и параметры модели во время расчёта.

## Объединение на основном ПК

Перенесите папку `work/two_pc_part1` с ноутбука на основной ПК целиком, включая
`index/`, `two_pc_job.json`, `job.json` и `shard_*`. Для объединения обязательны
`index/` и `two_pc_job.json`; остальные файлы сохраняют возможность продолжения.

```powershell
python distributed_embedding/two_pc.py merge `
  --folder distributed_embedding/r23 `
  --first work/two_pc_part0 --second work/two_pc_part1 `
  --out work/kb-r23-two-pc/index
```

На выходе получится один обычный FAISS индекс проекта: `index.faiss`,
`documents.jsonl`, `manifest.json`. Скрипт проверяет SHA частей и модели,
число записей, порядок `chunk_id` и совпадение параметров эмбеддинга. BM25
чат восстановит из `documents.jsonl` при загрузке. Подключение индекса к
работающему боту и проверка ответов выполняются отдельным шагом.

Если скорость обоих ПК равна измеренной на основном ПК (~0,057 чанка/с),
параллельная обработка займёт около 40 часов плюс перенос и объединение.
Реальное время определяется более медленной половиной. Деление поровну по
количеству чанков может дать разную нагрузку из-за длины текстов.

Повторное создание входных частей из сохранённого исходника:

```powershell
python distributed_embedding/two_pc.py prepare `
  --chunks work/v3_candidate_r23/core.jsonl `
  --folder path/to/new_folder
```
