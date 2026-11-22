"""Package the computed checkpoints without temporary files or logs."""
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEST = Path(__file__).resolve().parent
SOURCES = {
    'part0_complete.zip': (Path('C:/rag/nntu-rag-bot/work/two_pc_part0'),
                           ['job.json', 'two_pc_job.json', 'index'] + [f'shard_{i:05d}' for i in range(32)]),
    'part1_remaining.zip': (ROOT / 'work/two_pc_part1_remaining',
                            ['job.json'] + [f'shard_{i:05d}' for i in range(5, 32)]),
}

manifest = {}
for archive_name, (source, entries) in SOURCES.items():
    target = DEST / archive_name
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for entry in entries:
            path = source / entry
            files = sorted(path.rglob('*')) if path.is_dir() else [path]
            for file in files:
                if file.is_file():
                    archive.write(file, f'{archive_name.removesuffix(".zip")}/{file.relative_to(source).as_posix()}')
    digest = hashlib.file_digest(target.open('rb'), 'sha256').hexdigest()
    manifest[archive_name] = {'bytes': target.stat().st_size, 'sha256': digest}
    print(archive_name, manifest[archive_name], flush=True)
(DEST / 'archives.sha256.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
