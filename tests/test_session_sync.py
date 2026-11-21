"""Testes offline da sincronização reativa de configuração do RAGEngine."""
from rag_engine import RAGEngine


def test_sync_config_atualiza_llm_preservando_vectorstore(tmp_path):
    engine = RAGEngine(persist_dir=str(tmp_path))
    sentinel_vs = object()
    engine.vectorstore = sentinel_vs

    invalidated = engine.sync_config(
        ollama_url="http://localhost:11434",
        llm_model="llama3.2:3b",
        embedding_model="nomic-embed-text",
    )

    assert invalidated is False
    assert engine.llm_model == "llama3.2:3b"
    assert engine.vectorstore is sentinel_vs


def test_sync_config_sem_mudancas_preserva_estado(tmp_path):
    engine = RAGEngine(persist_dir=str(tmp_path))
    sentinel_vs = object()
    engine.vectorstore = sentinel_vs

    invalidated = engine.sync_config(
        ollama_url="http://localhost:11434",
        llm_model="llama3.1:8b",
        embedding_model="nomic-embed-text",
    )

    assert invalidated is False
    assert engine.vectorstore is sentinel_vs


def test_sync_config_troca_ollama_url_invalida_vectorstore(tmp_path):
    engine = RAGEngine(persist_dir=str(tmp_path))
    engine.vectorstore = object()

    invalidated = engine.sync_config(
        ollama_url="http://127.0.0.1:11435",
        llm_model="llama3.1:8b",
        embedding_model="nomic-embed-text",
    )

    assert invalidated is True
    assert engine.ollama_url == "http://127.0.0.1:11435"
    assert engine.vectorstore is None


def test_sync_config_troca_embedding_model_invalida_vectorstore(tmp_path):
    engine = RAGEngine(persist_dir=str(tmp_path))
    engine.vectorstore = object()

    invalidated = engine.sync_config(
        ollama_url="http://localhost:11434",
        llm_model="llama3.1:8b",
        embedding_model="bge-m3",
    )

    assert invalidated is True
    assert engine.embedding_model == "bge-m3"
    assert engine.vectorstore is None
