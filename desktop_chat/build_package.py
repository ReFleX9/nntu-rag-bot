"""Build a minimal desktop ZIP from an explicit list; never include work secrets."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.index / 'manifest.json').read_text(encoding='utf-8'))
    for name, sha in manifest['files'].items():
        if digest(args.index / name) != sha: raise ValueError(f'Index checksum: {name}')
    files = {f'runtime/index/{name}': args.index / name
             for name in ('manifest.json', 'documents.jsonl', 'index.faiss')}
    files['runtime/file_catalog.jsonl'] = args.catalog
    for name in ('Install.cmd', 'Start.cmd', 'README.md'):
        files[name] = ROOT / name
    for name in ('launcher.py', 'server.py', 'install.py', 'setup_local.py', 'runtime.py', 'README.md', 'nntu.ico'):
        files['desktop_chat/' + name] = ROOT / 'desktop_chat' / name
    module = ROOT / 'rag_hybrid_langchain/rag_hybrid'
    for name in ('chat_web.py', 'kb.py', 'corpus.py', 'retrieval.py', 'embeddings.py',
                 'file_lookup.py', 'nntu_languages.py', 'requirements.txt', 'requirements-search.txt'):
        files['rag_hybrid_langchain/rag_hybrid/' + name] = module / name
    for name in ('index.html', 'nntu-mark.png'):
        files['rag_hybrid_langchain/rag_hybrid/static/' + name] = module / 'static' / name
    payload = {'version': 'r23-consolidated-1', 'chunks': manifest['count'],
               'source_sha256': manifest['source_sha256'],
               'files': {name: digest(path) for name, path in files.items()},
               'embedding_revision': manifest['embedding']['revision'],
               'embedding_files': {name: digest(args.model_dir / name)
                                   for name in ('model.safetensors', 'tokenizer.json')}}
    (ROOT / 'desktop_chat/payload.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    files['desktop_chat/payload.json'] = ROOT / 'desktop_chat/payload.json'
    output = ROOT / 'distribution/nntu-desktop-win.zip'
    output.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, path in sorted(files.items()): archive.write(path, 'nntu-chat/' + name)
    result = {'file': output.name, 'bytes': output.stat().st_size, 'sha256': digest(output),
              'version': payload['version'], 'chunks': manifest['count'], 'files': len(files)}
    (output.parent / 'manifest.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
