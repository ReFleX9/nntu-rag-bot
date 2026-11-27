"""Set up the consolidated source checkout, reusing its downloaded resources."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install', action='store_true', help='Install missing project libraries')
    parser.add_argument('--shortcut', action='store_true', help='Update the Windows desktop shortcut')
    args = parser.parse_args()
    embedding = ROOT / 'vendor/nntu-models/qwen3-embedding-0.6b'
    ollama = ROOT / 'vendor/nntu-ollama/ollama.exe'
    if not embedding.is_dir() or not ollama.is_file():
        # The portable ZIP retains its original downloader for a clean machine.
        subprocess.run([sys.executable, str(ROOT / 'desktop_chat/install.py')], check=True)
        return
    if not (3, 11) <= sys.version_info[:2] <= (3, 13):
        raise RuntimeError('Use Python 3.11–3.13 x64')
    python = ROOT / '.venv/Scripts/python.exe'
    if not python.exists():
        subprocess.run([sys.executable, '-m', 'venv', str(ROOT / '.venv')], check=True)
    site = ROOT / '.venv/Lib/site-packages'
    # Paths are relative to site-packages, so vendored libraries stay in this project.
    (site / 'project_vendor.pth').write_text(
        '../../../vendor/nntu-deps\n../../../vendor/python\n', encoding='utf-8')
    if args.install:
        subprocess.run([str(python), '-m', 'pip', 'install', '-r', str(ROOT / 'requirements.txt')], check=True)
        subprocess.run([str(python), '-m', 'pip', 'check'], check=True)
    index = ROOT / 'runtime/index'
    catalog = ROOT / 'runtime/file_catalog.jsonl'
    for path in (index / 'manifest.json', index / 'documents.jsonl', index / 'index.faiss', catalog):
        if not path.is_file():
            raise FileNotFoundError(f'Missing consolidated runtime resource: {path}')
    manifest = json.loads((index / 'manifest.json').read_text(encoding='utf-8'))
    from install import digest
    for name, expected in manifest['files'].items():
        if digest(index / name) != expected:
            raise ValueError('Index checksum mismatch: ' + name)
    expected_models = json.loads((ROOT / 'desktop_chat/payload.json').read_text(encoding='utf-8'))['embedding_files']
    for name, expected in expected_models.items():
        if digest(embedding / name) != expected:
            raise ValueError('Embedding model checksum mismatch: ' + name)
    browser = next((p for p in (
        Path(os.environ.get('ProgramFiles(x86)', '')) / 'Microsoft/Edge/Application/msedge.exe',
        Path(os.environ.get('ProgramFiles', '')) / 'Microsoft/Edge/Application/msedge.exe',
    ) if p.is_file()), None)
    if browser is None:
        raise FileNotFoundError('Microsoft Edge is not installed')
    config = dict(python='.venv/Scripts/python.exe', dependencies='',
                  embedding_model='vendor/nntu-models/qwen3-embedding-0.6b',
                  ollama='vendor/nntu-ollama/ollama.exe',
                  ollama_models='vendor/nntu-ollama/models', browser=str(browser),
                  index='runtime/index', catalog='runtime/file_catalog.jsonl',
                  llm_model='nntu-qwen:4b', chat_port=18080, ollama_port=11435)
    work = ROOT / 'work'
    work.mkdir(exist_ok=True)
    (work / 'desktop_chat.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    if args.shortcut:
        from install import shortcut
        from runtime import load_config
        shortcut(load_config())
    print(f'Configuration ready: {ROOT}; {manifest["count"]} chunks')


if __name__ == '__main__':
    main()
