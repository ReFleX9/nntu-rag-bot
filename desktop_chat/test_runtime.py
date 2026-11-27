"""Regression checks for paths and Russian queries in the desktop application."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_runtime_resources_are_in_primary_project():
    runtime = module('desktop_runtime', 'desktop_chat/runtime.py')
    config = runtime.load_config()
    for key in ('python', 'embedding_model', 'ollama', 'ollama_models', 'index', 'catalog'):
        path = Path(config[key]).resolve()
        assert path.is_relative_to(ROOT)
        assert path.exists(), key
    assert config['dependencies'] == ''  # dependencies are provided by .venv


class NoTranslator:
    def with_structured_output(self, *args, **kwargs):
        raise AssertionError('Russian questions must not call a language model for translation')


def test_russian_compound_question_avoids_translation():
    languages = module('nntu_languages_test', 'rag_hybrid_langchain/rag_hybrid/nntu_languages.py')
    question = 'Номер телефона ИНЭУ и как зовут директора ИНЭУ?'
    assert languages.translate_query(NoTranslator(), question, 'auto') == ('ru', question)


def test_explicit_russian_is_not_reclassified():
    languages = module('nntu_languages_explicit', 'rag_hybrid_langchain/rag_hybrid/nntu_languages.py')
    assert languages.translate_query(NoTranslator(), 'ИНЭУ', 'ru') == ('ru', 'ИНЭУ')
