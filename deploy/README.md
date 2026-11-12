# Векторизация НГТУ на Ubuntu с NVIDIA

Используйте Ubuntu 26.04 x86-64 GPU-образ Selectel с драйвером NVIDIA 595.
Подойдет предложенный Selectel вариант с Docker, хотя скрипты выполняются
в Python venv на хосте. Скрипт установки рассчитан на пользователя с sudo и
рабочий драйвер NVIDIA. Для PyTorch используется готовая сборка CUDA 12.8,
отдельно ставить CUDA Toolkit не требуется.
Серверу нужен доступ к GitHub и Git LFS, PyPI, репозиторию PyTorch и Hugging Face.

## Перенос

На сервере выполните:

```bash
sudo apt-get update
sudo apt-get install -y git
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/ReFleX9/nntu-rag-bot.git ~/nntu-rag-bot
```

На сервере установите зависимости и скачайте готовые чанки:

```bash
cd ~/nntu-rag-bot
bash deploy/setup_gpu.sh
mkdir -p work
tmux new -s nntu-index
```

## Замер на GPU

В tmux создайте случайную выборку из 2 048 чанков с фиксированным seed.
Первый запуск скачает модель Qwen3-Embedding-0.6B с зафиксированной ревизией.
По завершении `time -v` покажет общее время и пик RAM. В другом SSH-сеансе
можно наблюдать загрузку и память GPU через `nvidia-smi -l 2`.

```bash
set -o pipefail
source_corpus=data/nntu_site/extracted/chunks.jsonl
.venv/bin/python deploy/select_corpus.py "$source_corpus" work/sample.jsonl --sample 2048
BATCH_SIZE=16 bash deploy/build_gpu.sh work/sample.jsonl work/kb-sample 2>&1 | tee work/sample.log
```

Для оценки скорости используйте интервалы между сообщениями «Индексировано»
после первых батчей, чтобы скачивание и загрузка модели не исказили замер.
Разделите 99 267 (основной набор) или 258 769 (полный корпус) на измеренное
число чанков в секунду. Добавьте время на подготовку и запись индекса.
Для полного корпуса это приблизительная оценка: распределение длин текстов
может отличаться от основной выборки.
Сравнивайте размер батча 16 и 32 с отдельным выходным каталогом.
При CUDA OOM попробуйте `BATCH_SIZE=8`. Из tmux можно выйти через Ctrl+B, D
и вернуться командой `tmux attach -t nntu-index`.

## Построение рабочего индекса

```bash
source_corpus=data/nntu_site/extracted/chunks.jsonl
.venv/bin/python deploy/select_corpus.py "$source_corpus" work/core.jsonl
BATCH_SIZE=16 bash deploy/build_gpu.sh work/core.jsonl work/kb-core 2>&1 | tee work/core.log
```

В основной набор входят пять тематических категорий. Для добавления
проверенных нормативных документов передайте селектору
`--include-documents work/extra.txt` и укажите в файле по одному точному
`document_id` на строку. Используйте новое имя выходного файла.
Для полного корпуса передайте исходный JSONL напрямую:

```bash
BATCH_SIZE=16 bash deploy/build_gpu.sh "$source_corpus" work/kb-full 2>&1 | tee work/full.log
```

После каждых 256 чанков создается самостоятельный FAISS-индекс в каталоге
`work/kb-core/shard_XXXXX`. На сетевом диске он остается после прерывания
сервера. Запустите ту же команду снова после восстановления ВМ, и уже готовые
шарды будут проверены по SHA-256 и пропущены. Готовый объединенный индекс
появится в `work/kb-core/index`. CUDA используется для эмбеддингов, а FAISS
и BM25 работают на CPU.

Для автоматического продолжения после восстановления прерываемого сервера
запустите службу вместо команды из tmux:

```bash
bash deploy/install_service.sh
journalctl -u nntu-index -f
```

Служба выбирает тематический корпус, строит индекс и запускается после каждого
старта Ubuntu. Готовые шарды остаются на сетевом загрузочном диске. Статус
проверяется командой `systemctl status nntu-index --no-pager`.

## Ответ на вопрос и вывоз индекса

Для генерации ответов установите [Ollama для Linux](https://ollama.com/download/linux)
и выполните `ollama pull qwen3:4b`. Из корня репозитория:

```bash
.venv/bin/python rag_hybrid_langchain/rag_hybrid/kb.py ask \
  --index work/kb-core/index --device cuda --llm-model qwen3:4b \
  --language ru --num-ctx 4096 --max-context-chars 5000 \
  --query 'Где найти расписание занятий?'
```

Для переноса готовой базы упакуйте весь каталог. На машине, где будут
выполняться запросы, понадобится та же модель эмбеддингов из манифеста.

```bash
tar -czf work/kb-core.tar.gz -C work/kb-core index
```

Скачайте архив на ПК перед удалением временной ВМ:

```powershell
scp USER@SERVER:nntu-rag-bot/work/kb-core.tar.gz .
```
