"""Local-only chat backend owned by the desktop launcher."""
import html
import json
import secrets
import sys
from pathlib import Path
from runtime import load_config

ROOT = Path(__file__).resolve().parents[1]
config = load_config()
sys.path.insert(0, str(ROOT / 'rag_hybrid_langchain/rag_hybrid'))
from embeddings import EmbeddingConfig, QwenEmbeddings
from retrieval import KnowledgeBase
import chat_web

index = Path(config['index'])
manifest = json.loads((index / 'manifest.json').read_text(encoding='utf-8'))
spec = {key: value for key, value in manifest['embedding'].items() if key != 'backend'}
spec.update(model=config['embedding_model'], revision=None)
encoder = QwenEmbeddings(EmbeddingConfig(**spec), device='cpu', batch_size=4, local_files_only=True)
kb = KnowledgeBase.load(index, embeddings=encoder, device='cpu', local_files_only=True)
chat_web.load_kb = lambda args: kb
token = secrets.token_urlsafe(32)
token_path = ROOT / 'work/desktop_chat.token'
token_path.write_text(token, encoding='utf-8')
# This process listens exclusively on loopback. Autofill avoids a separate login step.
chat_web.PAGE = chat_web.PAGE.replace('id="token" type="password"',
    'id="token" value="' + html.escape(token, quote=True) + '" type="password"')
chat_web.PAGE = chat_web.PAGE.replace("$('token').value = sessionStorage.getItem('nntu-chat-token') || '';",
                                     "$('token').value = $('token').defaultValue;")
sys.argv = ['chat_web.py', '--index', str(index), '--token-file', str(token_path),
            '--device', 'cpu', '--llm-model', config['llm_model'], '--host', '127.0.0.1',
            '--port', str(config['chat_port']), '--ollama-url', f"http://127.0.0.1:{config['ollama_port']}"]
if Path(config.get('catalog', '')).is_file():
    sys.argv += ['--file-catalog', config['catalog']]
chat_web.main()
