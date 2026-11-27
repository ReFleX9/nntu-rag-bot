"""Resolve desktop resources relative to the single project directory."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH_KEYS = ('python', 'dependencies', 'embedding_model', 'ollama',
             'ollama_models', 'browser', 'index', 'catalog')


def load_config():
    config = json.loads((ROOT / 'work/desktop_chat.json').read_text(encoding='utf-8-sig'))
    for key in PATH_KEYS:
        if config.get(key):
            path = Path(config[key]).expanduser()
            config[key] = str((ROOT / path).resolve() if not path.is_absolute() else path)
    return config
