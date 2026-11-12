"""Suíte ponta a ponta do pipeline RAG do DocAI.

Execução:
    python -m pytest tests/                 # tudo (exige Ollama no ar)
    python -m pytest tests/ -m "not ollama" # apenas testes offline (CI)
    python tests/test_rag_pipeline.py       # atalho equivalente ao primeiro
"""
import math
import sys

import pytest
import requests

from conftest import EXPECTED_DOCS, PRIMARY_MODEL
from rag_engine import (
    REFUSAL_MESSAGE,
    SYSTEM_PROMPT,
    RAGEngine,
    apply_refusal_guard,
)

REQUIRED_MODELS = ["nomic-embed-text", "llama3.1:8b", "llama3.2:3b"]

Q_FERIAS = "Qual a antecedência mínima para solicitar férias e posso vender 10 dias?"
Q_SAUDE = "Como funciona a coparticipação em consultas e exames no plano de saúde?"
Q_NOTEBOOK = "O que devo fazer caso meu notebook corporativo seja roubado?"
Q_PEGADINHA = "Qual é o valor do bônus de Natal para estagiários?"

EXPECTED_REFUSAL = (
    "Essa informação não consta nas políticas e manuais internos disponíveis "
    "na minha base. Recomendo abrir um chamado junto ao setor responsável "
    "(Gente & Gestão ou TI)."
)

ollama = pytest.mark.ollama


# ---------------------------------------------------------------------------
# Teste 1 - Conectividade Ollama
# ---------------------------------------------------------------------------
@ollama
def test_1_ollama_endpoint_responde(ollama_url):
    ok, msg = RAGEngine.test_connection(ollama_url)
    assert ok, msg


@ollama
@pytest.mark.parametrize("model", REQUIRED_MODELS)
def test_1_modelo_obrigatorio_instalado(ollama_url, model):
    resp = requests.get(f"{ollama_url.rstrip('/')}/api/tags", timeout=10)
    installed = [m["name"] for m in resp.json()["models"]]
    assert any(name == model or name.split(":")[0] == model for name in installed), (
        f"Modelo '{model}' ausente no Ollama. Instalados: {installed}. "
        f"Rode: ollama pull {model}"
    )


# ---------------------------------------------------------------------------
# Teste 2 - Ingestão e Vetorização
# ---------------------------------------------------------------------------
def test_2_leitura_dos_tres_documentos(docs_dir):
    chunks = RAGEngine().load_and_split(docs_dir)
    assert {c.metadata["source_file"] for c in chunks} == EXPECTED_DOCS


def test_2_chunks_respeitam_tamanho_maximo(docs_dir):
    chunks = RAGEngine().load_and_split(docs_dir)
    assert len(chunks) >= 9
    assert max(len(c.page_content) for c in chunks) <= 700


def test_2_chunking_preserva_clausulas_chave(docs_dir):
    chunks = RAGEngine().load_and_split(docs_dir)
    texto = "\n".join(c.page_content for c in chunks)
    for trecho in (
        "antecedência mínima de 30 (trinta) dias",
        "Consultas eletivas: R$ 35,00",
        "prazo máximo de 24 horas",
    ):
        assert trecho in texto


def test_2_chunking_de_pasta_vazia_retorna_lista_vazia(tmp_path):
    assert RAGEngine().load_and_split(str(tmp_path)) == []


@ollama
def test_2_indexacao_grava_todos_os_chunks(index_result, docs_dir):
    engine, result = index_result
    assert result["success"], result.get("message")
    assert set(result["files"]) == EXPECTED_DOCS
    assert result["chunks_count"] == len(engine.load_and_split(docs_dir))
    assert engine.vectorstore._collection.count() == result["chunks_count"]


@ollama
def test_2_base_persiste_e_recarrega_do_disco(engine, ollama_url, persist_dir):
    novo = RAGEngine(ollama_url=ollama_url, persist_dir=persist_dir)
    assert novo.load_existing_vectorstore() is True
    assert novo.vectorstore._collection.count() == engine.vectorstore._collection.count()


# ---------------------------------------------------------------------------
# Teste 3 - Recuperação Semântica (Top-k por similaridade de cossenos)
# ---------------------------------------------------------------------------
@ollama
@pytest.mark.parametrize(
    "pergunta, fonte, clausula",
    [
        (Q_FERIAS, "politica_de_ferias.txt", "3.1. Toda solicitação de férias"),
        (Q_SAUDE, "guia_de_beneficios.txt", "2.5. Coparticipação"),
        (Q_NOTEBOOK, "manual_ti_e_seguranca.txt", "2.3. Em caso de furto ou roubo"),
    ],
    ids=["ferias", "saude", "notebook"],
)
def test_3_top1_e_a_fonte_e_clausula_corretas(engine, pergunta, fonte, clausula):
    top1 = engine.retrieve(pergunta, k=5)[0]
    assert top1["doc"] == fonte
    assert clausula in top1["content"]


@ollama
def test_3_abono_pecuniario_esta_no_topk_da_pergunta_composta(engine):
    conteudos = [s["content"] for s in engine.retrieve(Q_FERIAS, k=5)]
    assert any("5.1. É facultado ao colaborador" in c for c in conteudos)


@ollama
def test_3_scores_sao_similaridade_decrescente(engine):
    scores = [s["score"] for s in engine.retrieve(Q_SAUDE, k=5)]
    assert len(scores) == 5
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 < s <= 1.0 for s in scores)


@ollama
def test_3_score_equivale_a_formula_do_cosseno(engine):
    """O score exibido deve ser S_C(q, d) = (q . d) / (||q|| ||d||) (Equação 1 do artigo)."""
    top1 = engine.retrieve(Q_NOTEBOOK, k=1)[0]
    emb = engine.get_embeddings()
    q = emb.embed_query(Q_NOTEBOOK)
    d = emb.embed_documents([top1["content"]])[0]
    dot = sum(a * b for a, b in zip(q, d))
    cosseno = dot / (math.sqrt(sum(a * a for a in q)) * math.sqrt(sum(b * b for b in d)))
    assert top1["score"] == pytest.approx(cosseno, abs=1e-3)


@ollama
def test_3_base_legada_sem_cosseno_exige_reindexacao(ollama_url, tmp_path, docs_dir):
    from langchain_community.vectorstores import Chroma

    legado = RAGEngine(ollama_url=ollama_url, persist_dir=str(tmp_path))
    Chroma.from_documents(
        documents=legado.load_and_split(docs_dir)[:2],
        embedding=legado.get_embeddings(),
        persist_directory=str(tmp_path),
    )
    assert legado.load_existing_vectorstore() is False


# ---------------------------------------------------------------------------
# Teste 4 - Geração de resposta funcional (3 perguntas centrais)
# ---------------------------------------------------------------------------
def _ask(engine, pergunta, model=PRIMARY_MODEL):
    engine.llm_model = model
    return engine.ask(pergunta, temperature=0.0, seed=42)


@ollama
def test_4_ferias_e_abono_citam_clausulas_3_1_e_5_1(engine):
    res = _ask(engine, Q_FERIAS)
    resposta = res["answer"]
    assert "3.1" in resposta and "5.1" in resposta
    assert "30" in resposta
    assert "politica_de_ferias.txt" in {s["doc"] for s in res["sources"]}


@ollama
def test_4_coparticipacao_informa_35_e_60_reais(engine):
    resposta = _ask(engine, Q_SAUDE)["answer"]
    assert "35,00" in resposta
    assert "60,00" in resposta


@ollama
def test_4_furto_de_notebook_orienta_bo_em_24h_e_acionar_ti(engine):
    resposta = _ask(engine, Q_NOTEBOOK)["answer"]
    assert "B.O." in resposta or "Boletim de Ocorrência" in resposta
    assert "24 horas" in resposta
    assert "security@empresa.com.br" in resposta or "time de TI" in resposta


@ollama
def test_4_respostas_fundamentadas_nao_sao_substituidas_pela_recusa(engine):
    assert EXPECTED_REFUSAL not in _ask(engine, Q_NOTEBOOK)["answer"]


# ---------------------------------------------------------------------------
# Teste 5 - Trava anti-alucinação (crucial)
# ---------------------------------------------------------------------------
def test_5_mensagem_de_recusa_e_a_padrao_oficial():
    assert REFUSAL_MESSAGE == EXPECTED_REFUSAL


def test_5_system_prompt_contem_a_recusa_padrao():
    assert EXPECTED_REFUSAL in SYSTEM_PROMPT
    assert "{context}" in SYSTEM_PROMPT and "{question}" in SYSTEM_PROMPT


def test_5_guarda_normaliza_recusa_com_preambulo():
    bruto = (
        "Infelizmente, não consta expressamente no contexto fornecido informações "
        "sobre o valor do bônus de Natal para estagiários.\n\n" + EXPECTED_REFUSAL
    )
    assert apply_refusal_guard(bruto, ["politica_de_ferias.txt"]) == EXPECTED_REFUSAL


def test_5_guarda_normaliza_negativa_sem_a_frase_padrao():
    bruto = "Não consta expressamente no contexto fornecido informações sobre o bônus de Natal."
    assert apply_refusal_guard(bruto, ["guia_de_beneficios.txt"]) == EXPECTED_REFUSAL


def test_5_guarda_preserva_resposta_fundamentada():
    resposta = (
        "Conforme a cláusula 3.1 do documento politica_de_ferias.txt, a antecedência "
        "mínima é de 30 dias corridos."
    )
    assert apply_refusal_guard(resposta, ["politica_de_ferias.txt"]) == resposta


def test_5_guarda_preserva_resposta_parcial_com_citacao():
    resposta = (
        "A antecedência mínima é de 30 dias (cláusula 3.1, politica_de_ferias.txt). "
        "Sobre o bônus: " + EXPECTED_REFUSAL
    )
    assert apply_refusal_guard(resposta, ["politica_de_ferias.txt"]) == resposta


@ollama
@pytest.mark.parametrize("model", ["llama3.1:8b", "llama3.2:3b"])
def test_5_pergunta_fora_da_base_retorna_estritamente_a_recusa(engine, model):
    resposta = _ask(engine, Q_PEGADINHA, model=model)["answer"]
    assert resposta == EXPECTED_REFUSAL


@ollama
def test_5_recusa_nao_inventa_valor_monetario(engine):
    assert "R$" not in _ask(engine, Q_PEGADINHA)["answer"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
