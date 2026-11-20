"""Private Telegram supervisor for one Windows embedding worker (stdlib only)."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import logging
import os
import secrets
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger(__name__)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


class Bot:
    def __init__(self, args):
        self.args = args
        self.token = os.environ['NNTU_TELEGRAM_TOKEN'].strip()
        self.owner = int(os.environ['NNTU_TELEGRAM_USER_ID'])
        self.out = args.out.resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.stop = self.out / 'stop.request'
        if self.stop.exists():
            raise ValueError(f'Remove {self.stop} explicitly to resume this job')
        self.process = None
        self.mode = None
        self.pending = None
        self.offset = None
        self.finished = False
        self.shutdown_at = None

    def api(self, method, **payload):
        request = urllib.request.Request(
            f'https://api.telegram.org/bot{self.token}/{method}',
            data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
        # Never log exceptions containing request URLs: the URL includes a token.
        with urllib.request.urlopen(request, timeout=40) as response:
            data = json.load(response)
        if not data.get('ok'):
            raise RuntimeError('Telegram API rejected the request')
        return data['result']

    def send(self, text, buttons=None):
        payload = dict(chat_id=self.owner, text=text[:4000])
        if buttons:
            payload['reply_markup'] = {'inline_keyboard': buttons}
        return self.api('sendMessage', **payload)

    def menu(self):
        return [[{'text': 'Прогресс', 'callback_data': 'status'}],
                [{'text': 'Сохранить порцию и выключить', 'callback_data': 'pause'}],
                [{'text': 'Выключить после проверки результата', 'callback_data': 'finish'}],
                [{'text': 'Отменить выключение', 'callback_data': 'cancel'}]]

    def status(self):
        count = 0
        for folder in self.out.glob('shard_*'):
            try:
                count += json.loads((folder / 'manifest.json').read_text())['count']
            except (OSError, ValueError, KeyError):
                pass
        plan = json.loads((self.args.folder / 'plan.json').read_text())
        total = plan['parts'][self.args.part]['rows']
        with (self.out / 'worker.log').open('rb') as log:
            log.seek(max(0, log.seek(0, 2) - 3000))
            tail = log.read().decode('utf-8', errors='replace').strip().splitlines()[-4:]
        running = self.process.poll() is None
        return (f'Сохранено: {count}/{total} ({100*count/total:.1f}%).\n'
                f'Процесс: {"работает" if running else "завершён"}.\n'
                f'Выключение: {self.mode or "не запланировано"}.\n'
                'Прогресс обновляется после сохранения шарда (до 256 чанков).\n\n'
                + '\n'.join(tail))

    def verify(self, complete):
        # Independent disk verification in the same Python environment as the worker.
        from two_pc import digest, load_plan, unpack
        import faiss
        import numpy as np
        plan = load_plan(self.args.folder)
        part = plan['parts'][self.args.part]
        raw = unpack(self.args.folder, part)
        job = json.loads((self.out / 'job.json').read_text())
        if job['source_sha256'] != part['sha256']:
            raise ValueError('Wrong source')
        folders = sorted(self.out.glob('shard_*'))
        if not folders:
            raise ValueError('No saved checkpoints')
        count = 0
        with raw.open('rb') as source:
            for number, folder in enumerate(folders):
                if folder.name != f'shard_{number:05d}':
                    raise ValueError('Missing checkpoint')
                lines = [source.readline() for _ in range(min(job['shard_size'], part['rows']-count))]
                manifest = json.loads((folder / 'manifest.json').read_text())
                if hashlib.sha256(b''.join(lines)).hexdigest() != manifest['source_sha256']:
                    raise ValueError('Checkpoint source mismatch')
                self.verify_index(folder, manifest, lines, faiss, np)
                count += len(lines)
        if complete:
            if count != part['rows']:
                raise ValueError('Incomplete result')
            metadata = json.loads((self.out / 'two_pc_job.json').read_text())
            if metadata['plan_sha256'] != digest(self.args.folder / 'plan.json') or metadata['part'] != self.args.part:
                raise ValueError('Wrong result metadata')
            manifest = json.loads((self.out / 'index/manifest.json').read_text())
            if manifest['source_sha256'] != part['sha256']:
                raise ValueError('Wrong merged source')
            with raw.open('rb') as source:
                self.verify_index(self.out / 'index', manifest, source, faiss, np)
        return count

    @staticmethod
    def verify_index(folder, manifest, source, faiss, np):
        for name in ('index.faiss', 'documents.jsonl'):
            if sha(folder / name) != manifest['files'][name]:
                raise ValueError('Checksum mismatch')
        index = faiss.deserialize_index(np.frombuffer((folder / 'index.faiss').read_bytes(), dtype=np.uint8).copy())
        count = 0
        with (folder / 'documents.jsonl').open(encoding='utf-8') as docs:
            for line in source:
                result = docs.readline()
                if not result or json.loads(line)['chunk_id'] != json.loads(result)['metadata']['chunk_id']:
                    raise ValueError('Document order mismatch')
                count += 1
            if docs.readline():
                raise ValueError('Extra documents')
        if count != manifest['count'] or index.ntotal != count or index.d != 1024:
            raise ValueError('Vector count/dimension mismatch')
        if index.metric_type != faiss.METRIC_INNER_PRODUCT:
            raise ValueError('Wrong index metric')
        vectors = index.reconstruct_n(0, index.ntotal)
        if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=0.002):
            raise ValueError('Invalid vector values or normalization')

    def action(self, action):
        if action == 'status':
            self.send(self.status(), self.menu())
        elif action == 'cancel':
            self.mode = self.pending = self.shutdown_at = None
            self.stop.unlink(missing_ok=True)
            self.send('Выключение отменено. Если расчёт уже остановился, его нужно запустить заново.', self.menu())
        elif action in ('pause', 'finish'):
            nonce = secrets.token_hex(8)
            self.pending = (nonce, action, time.monotonic() + 120)
            text = ('Дождаться сохранения текущего шарда, проверить сохранённые части и выключить ПК? '
                    'Ожидание шарда на CPU может занять больше часа.' if action == 'pause' else
                    'Разрешить автоматическое выключение после завершения этой половины корпуса и проверки файлов? '
                    'Повторного подтверждения после завершения не потребуется.')
            self.send(text, [[{'text': 'Подтверждаю', 'callback_data': 'confirm:'+nonce}],
                             [{'text': 'Отмена', 'callback_data': 'cancel'}]])
        elif action.startswith('confirm:'):
            if not self.pending or self.pending[0] != action[8:] or time.monotonic() > self.pending[2]:
                return self.send('Подтверждение устарело. Выберите действие заново.', self.menu())
            self.mode = self.pending[1]
            self.pending = None
            if self.mode == 'pause' and self.process.poll() is None:
                self.stop.touch()
            self.send('Режим принят. Можно отменить выключение кнопкой ниже.', self.menu())
        else:
            self.send(self.status(), self.menu())

    def tick(self):
        code = self.process.poll()
        if code is not None and not self.finished:
            self.finished = True
            self.send('Расчёт завершился.\n' + self.status(), self.menu())
        if self.mode and code is not None and self.shutdown_at is None:
            if code not in (0, 42) or (self.mode == 'finish' and code != 0):
                self.mode = None
                return self.send('Ошибка или остановка расчёта: автоматическое выключение отменено.', self.menu())
            try:
                count = self.verify(complete=(code == 0))
            except Exception:
                self.mode = None
                return self.send('Проверка результата не пройдена. ПК остаётся включённым; проверьте файлы и журнал.', self.menu())
            self.send(f'Проверено {count} сохранённых чанков. Выключу ПК через 120 секунд. '
                      'Сохраните документы в других приложениях.', self.menu())
            self.shutdown_at = time.monotonic() + 120
        if self.shutdown_at is not None and time.monotonic() >= self.shutdown_at:
            # No /f: do not forcibly close applications with unsaved documents.
            subprocess.run(['shutdown.exe', '/s', '/t', '0'], check=True)
            self.shutdown_at = None
            self.mode = None

    def run(self):
        if os.name != 'nt':
            raise RuntimeError('Windows only')
        import msvcrt
        self.lockfile = (self.out / 'supervisor.lock').open('a+b')
        self.lockfile.seek(0)
        if not self.lockfile.read(1):
            self.lockfile.write(b'0')
            self.lockfile.flush()
        self.lockfile.seek(0)
        msvcrt.locking(self.lockfile.fileno(), msvcrt.LK_NBLCK, 1)
        me = self.api('getMe')
        if self.api('getWebhookInfo').get('url'):
            raise RuntimeError('Use a dedicated bot without a webhook')
        # Ignore stale commands from earlier runs, including old confirmations.
        updates = self.api('getUpdates', offset=-1, timeout=0)
        if updates:
            self.offset = updates[-1]['update_id'] + 1
        self.send('Управление расчётом запускается. ' + me['username'], self.menu())
        log = (self.out / 'worker.log').open('ab', buffering=0)
        command = [sys.executable, '-u', str(ROOT / 'distributed_embedding/two_pc.py'), 'worker',
                   '--folder', str(self.args.folder), '--part', str(self.args.part),
                   '--model-dir', str(self.args.model_dir.resolve()), '--out', str(self.out),
                   '--device', self.args.device, '--batch-size', str(self.args.batch_size),
                   '--stop-file', str(self.stop)]
        # Keep the CPU awake while the supervisor runs; display may turn off.
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        self.process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                        cwd=ROOT, creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            while True:
                try:
                    updates = self.api('getUpdates', offset=self.offset, timeout=10,
                                       allowed_updates=['message', 'callback_query'])
                    for update in updates:
                        self.offset = update['update_id'] + 1
                        callback = update.get('callback_query')
                        message = callback.get('message', {}) if callback else update.get('message', {})
                        actor = callback.get('from', {}) if callback else message.get('from', {})
                        if actor.get('id') != self.owner or message.get('chat', {}).get('id') != self.owner:
                            continue
                        if callback:
                            self.api('answerCallbackQuery', callback_query_id=callback['id'])
                        self.action(callback['data'] if callback else message.get('text', 'status'))
                    self.tick()
                except Exception:
                    LOG.warning('Supervisor iteration failed; retrying (details suppressed to protect token)')
                    time.sleep(5)
        finally:
            # Never power off if the supervisor itself is interrupted.
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
            log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, default=ROOT / 'distributed_embedding/r23')
    parser.add_argument('--part', type=int, required=True, choices=(0, 1))
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda', 'auto'), default='auto')
    parser.add_argument('--batch-size', type=int, default=4)
    args = parser.parse_args()
    args.folder = args.folder.resolve()
    logging.basicConfig(level=logging.INFO)
    try:
        Bot(args).run()
    except KeyboardInterrupt:
        print('Supervisor stopped. Check worker status before closing this session.')
    except Exception:
        raise SystemExit('Cannot start supervisor. Check token/user ID, output lock, dependencies and network; secret details suppressed.') from None


if __name__ == '__main__':
    main()
