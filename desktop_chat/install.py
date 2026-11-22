"""Install the minimal Windows desktop distribution in its extracted folder."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVISION = '97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3'
OLLAMA_URL = 'https://github.com/ollama/ollama/releases/download/v0.40.16/ollama-windows-amd64.zip'
OLLAMA_SHA = '50fcf5ad54277f2fa5d173ebfc93bcb017adbc0c9ef2314daabc96ca9e6aa64e'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_payload():
    manifest = json.loads((ROOT / 'desktop_chat/payload.json').read_text(encoding='utf-8'))
    for name, sha in manifest['files'].items():
        path = ROOT / name
        if not path.is_file() or digest(path) != sha:
            raise ValueError(f'Повреждён или отсутствует файл: {name}')
    return manifest


def download(url, target, sha):
    if target.exists() and digest(target) == sha: return
    temp = target.with_suffix('.partial')
    print('Скачивание:', url, flush=True)
    with urllib.request.urlopen(url, timeout=120) as source, temp.open('wb') as output:
        shutil.copyfileobj(source, output)
    if digest(temp) != sha: raise ValueError('Не совпадает SHA-256 загрузки')
    os.replace(temp, target)


def shortcut(config):
    # Use structured environment values instead of interpolating filesystem paths in PS code.
    env = os.environ.copy()
    env.update(NNTU_PYTHONW=str(Path(config['python']).with_name('pythonw.exe')),
               NNTU_LAUNCHER=str(ROOT / 'desktop_chat/launcher.py'), NNTU_ROOT=str(ROOT),
               NNTU_ICON=str(ROOT / 'desktop_chat/nntu.ico'))
    script = """
$ErrorActionPreference='Stop'
$desktop=[Environment]::GetFolderPath('Desktop')
$link=(New-Object -ComObject WScript.Shell).CreateShortcut((Join-Path $desktop 'Чат НГТУ.lnk'))
$link.TargetPath=$env:NNTU_PYTHONW
$link.Arguments='"'+$env:NNTU_LAUNCHER+'"'
$link.WorkingDirectory=$env:NNTU_ROOT
$link.IconLocation=$env:NNTU_ICON+',0'
$link.Description='Локальный чат НГТУ. Закрытие окна освобождает память.'
$link.Save()
"""
    subprocess.run(['powershell.exe', '-NoProfile', '-Command', script], env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--shortcut-only', action='store_true')
    args = parser.parse_args()
    if args.verify_only:
        print('Файлы комплекта проверены:', verify_payload()['chunks']); return
    if os.name != 'nt' or platform.machine().lower() not in ('amd64', 'x86_64'):
        raise RuntimeError('Нужна Windows x64')
    if not (3, 11) <= sys.version_info[:2] <= (3, 13):
        raise RuntimeError('Установите Python 3.11–3.13 x64 с компонентом Tcl/Tk')
    import tkinter  # Require GUI support before a long installation.
    work = ROOT / 'work'; work.mkdir(exist_ok=True)
    config_path = work / 'desktop_chat.json'
    if args.shortcut_only:
        shortcut(json.loads(config_path.read_text(encoding='utf-8'))); return
    verify_payload()
    venv = ROOT / '.venv'
    python = venv / 'Scripts/python.exe'
    if not python.exists(): subprocess.run([sys.executable, '-m', 'venv', str(venv)], check=True)
    subprocess.run([str(python), '-m', 'pip', 'install', '--upgrade', 'pip'], check=True)
    subprocess.run([str(python), '-m', 'pip', 'install', 'torch>=2.6,<3',
                    '--index-url', 'https://download.pytorch.org/whl/cpu'], check=True)
    subprocess.run([str(python), '-m', 'pip', 'install', '-r',
                    str(ROOT / 'rag_hybrid_langchain/rag_hybrid/requirements.txt')], check=True)
    embedding = ROOT / 'vendor/models/qwen3-embedding-0.6b'
    subprocess.run([str(python), '-c',
                    'from huggingface_hub import snapshot_download; import sys; '
                    'snapshot_download("Qwen/Qwen3-Embedding-0.6B",revision=sys.argv[1],local_dir=sys.argv[2])',
                    REVISION, str(embedding)], check=True)
    expected = json.loads((ROOT / 'desktop_chat/payload.json').read_text(encoding='utf-8'))['embedding_files']
    for name, sha in expected.items():
        if digest(embedding / name) != sha: raise ValueError('Embedding модель не соответствует индексу')
    runtime = ROOT / 'vendor/ollama'; runtime.mkdir(parents=True, exist_ok=True)
    archive = work / 'ollama-windows-amd64.zip'
    if not (runtime / 'ollama.exe').exists():
        download(OLLAMA_URL, archive, OLLAMA_SHA)
        with zipfile.ZipFile(archive) as bundle:
            for item in bundle.infolist():
                if not (runtime / item.filename).resolve().is_relative_to(runtime.resolve()):
                    raise ValueError('Небезопасный путь в архиве')
            bundle.extractall(runtime)
    models = ROOT / 'vendor/ollama-models'; models.mkdir(parents=True, exist_ok=True)
    edge = next((path for path in [Path(os.environ.get('ProgramFiles(x86)', '')) / 'Microsoft/Edge/Application/msedge.exe',
                                   Path(os.environ.get('ProgramFiles', '')) / 'Microsoft/Edge/Application/msedge.exe',
                                   Path(os.environ['LOCALAPPDATA']) / 'Microsoft/Edge/Application/msedge.exe'] if path.is_file()), None)
    if not edge: raise FileNotFoundError('Установите Microsoft Edge')
    config = dict(python=str(python), dependencies='', embedding_model=str(embedding),
                  ollama=str(runtime / 'ollama.exe'), ollama_models=str(models), browser=str(edge),
                  index=str(ROOT / 'runtime/index'), catalog=str(ROOT / 'runtime/file_catalog.jsonl'),
                  llm_model='qwen3:4b', chat_port=18080, ollama_port=11435)
    env = os.environ.copy(); env.update(OLLAMA_MODELS=str(models), OLLAMA_HOST='127.0.0.1:11435')
    # Refuse a busy port: never download models into an unrelated Ollama service.
    import socket
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 11435))
    with (work / 'install_ollama.log').open('ab') as log:
        server = subprocess.Popen([config['ollama'], 'serve'], env=env, stdout=log, stderr=log,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            for _ in range(90):
                if server.poll() is not None: raise RuntimeError('Ollama не запустилась')
                try:
                    with urllib.request.urlopen('http://127.0.0.1:11435/api/tags', timeout=1): break
                except OSError: time.sleep(1)
            else: raise RuntimeError('Ollama не отвечает')
            subprocess.run([config['ollama'], 'pull', config['llm_model']], env=env, check=True)
        finally:
            if server.poll() is None:
                subprocess.run(['taskkill.exe', '/PID', str(server.pid), '/T', '/F'],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               creationflags=subprocess.CREATE_NO_WINDOW)
                server.wait(timeout=20)
    pending = config_path.with_suffix('.partial')
    pending.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(pending, config_path)
    shortcut(config)
    print('Готово. Запускайте ярлык «Чат НГТУ» на рабочем столе.')


if __name__ == '__main__':
    main()
