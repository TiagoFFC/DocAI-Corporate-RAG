"""Fixtures compartilhadas da suíte de testes do DocAI.

Os testes marcados com ``@pytest.mark.ollama`` exigem um servidor Ollama
acessível (variável ``OLLAMA_BASE_URL``, padrão ``http://localhost:11434``)
com os modelos ``nomic-embed-text``, ``llama3.1:8b`` e ``llama3.2:3b``.
Os demais rodam offline (usados na esteira de CI).
"""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag_engine import RAGEngine  # noqa: E402

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DOCS_DIR = str(ROOT / "documentos")
PRIMARY_MODEL = os.getenv("DOCAI_TEST_MODEL", "llama3.1:8b")
EXPECTED_DOCS = {
    "politica_de_ferias.txt",
    "guia_de_beneficios.txt",
    "manual_ti_e_seguranca.txt",
}


@pytest.fixture(scope="session")
def ollama_url():
    return OLLAMA_URL


@pytest.fixture(scope="session")
def docs_dir():
    return DOCS_DIR


@pytest.fixture(scope="session")
def persist_dir(tmp_path_factory):
    """Base vetorial isolada: os testes nunca tocam no ./chroma_db real."""
    return str(tmp_path_factory.mktemp("chroma_test"))


@pytest.fixture(scope="session")
def index_result(ollama_url, persist_dir, docs_dir):
    engine = RAGEngine(
        ollama_url=ollama_url, llm_model=PRIMARY_MODEL, persist_dir=persist_dir
    )
    result = engine.index_directory(docs_dir=docs_dir)
    return engine, result


@pytest.fixture(scope="session")
def engine(index_result):
    eng, result = index_result
    assert result["success"], result.get("message")
    return eng
