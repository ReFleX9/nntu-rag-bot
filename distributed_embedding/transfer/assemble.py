"""Assemble uploaded checkpoints with the first five shards from the other PC."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'distributed_embedding'))
from two_pc import digest, load_plan


def extract(name, destination):
    archive = HERE / name
    expected = json.loads((HERE / 'archives.sha256.json').read_text(encoding='utf-8'))[name]
    if digest(archive) != expected['sha256']:
        raise ValueError(f'Archive checksum mismatch: {archive}')
    with zipfile.ZipFile(archive) as source:
        for item in source.infolist():
            parts = Path(item.filename).parts
            if '..' in parts or not parts or parts[0] != name.removesuffix('.zip'):
                raise ValueError(f'Unsafe archive path: {item.filename}')
        source.extractall(destination)
    return destination / name.removesuffix('.zip')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--other-part1', type=Path, required=True,
                        help='Folder from the other PC containing shard_00000 through shard_00004')
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=ROOT / 'work/assembled_r23')
    args = parser.parse_args()
    target = args.out.resolve()
    if target.exists():
        raise FileExistsError(target)
    target.mkdir(parents=True)
    first = extract('part0_complete.zip', target)
    second = extract('part1_remaining.zip', target)
    source = args.other_part1.resolve()
    for n in range(5):
        path = source / f'shard_{n:05d}'
        if not path.is_dir():
            raise FileNotFoundError(path)
        shutil.copytree(path, second / path.name)

    plan = load_plan(ROOT / 'distributed_embedding/r23')
    local_job = json.loads((second / 'job.json').read_text(encoding='utf-8'))
    remote_job = json.loads((source / 'job.json').read_text(encoding='utf-8'))
    if local_job != remote_job or local_job['source_sha256'] != plan['parts'][1]['sha256']:
        raise ValueError('Part 1 source or embedding configuration differs between computers')
    model = args.model_dir.resolve()
    hashes = {name: digest(model / name) for name in ('model.safetensors', 'tokenizer.json')}
    first_job = json.loads((first / 'two_pc_job.json').read_text(encoding='utf-8'))
    if hashes['model.safetensors'] != first_job['model_sha256'] or hashes['tokenizer.json'] != first_job['tokenizer_sha256']:
        raise ValueError('Model differs from part 0')
    if hashes != local_job['model_hashes']:
        raise ValueError('Model differs from part 1 checkpoints')

    command = [sys.executable, str(ROOT / 'distributed_embedding/two_pc.py')]
    subprocess.run(command + ['worker', '--folder', str(ROOT / 'distributed_embedding/r23'),
                              '--part', '1', '--model-dir', str(model), '--device', 'cpu',
                              '--batch-size', '1', '--out', str(second)], check=True)
    subprocess.run(command + ['merge', '--folder', str(ROOT / 'distributed_embedding/r23'),
                              '--first', str(first), '--second', str(second),
                              '--out', str(target / 'index')], check=True)
    print(f'Complete merged index: {target / "index"}')


if __name__ == '__main__':
    main()
