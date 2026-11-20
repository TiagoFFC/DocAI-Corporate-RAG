"""Suíte ponta a ponta do pipeline RAG do DocAI.

Execução:
    python -m pytest tests/                 # tudo (exige Ollama no ar)
    python -m pytest tests/ -m "not ollama" # apenas testes offline (CI)
    python tests/test_rag_pipeline.py       # atalho equivalente ao primeiro
"""
import math
import os
import sys

import pytest
import requests

from conftest import EXPECTED_DOCS, PRIMARY_MODEL
from rag_engine import (
    REFUSAL_MESSAGE,
    SYSTEM_PROMPT,
    RAGEngine,
    apply_refusal_guard,
    safe_document_path,
)
from ui_helpers import source_box_html

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


CTX_FERIAS = (
    "--- [Trecho 1 | Fonte: politica_de_ferias.txt] ---\n"
    "3. PRAZOS DE SOLICITAÇÃO E ANTECEDÊNCIA\n"
    "3.1. Toda solicitação de férias deve ser registrada com antecedência mínima "
    "de 30 (trinta) dias corridos.\n"
    "5.1. É facultado converter 1/3 (um terço) das férias em abono pecuniário "
    "(equivalente a 10 dias).\n\n"
    "--- [Trecho 2 | Fonte: guia_de_beneficios.txt] ---\n"
    "5.1. Reembolso de até 70% da mensalidade, limitado ao teto de R$ 800,00 mensais.\n"
)
FONTES_CTX = ["politica_de_ferias.txt", "guia_de_beneficios.txt"]


def _guarda(resposta, pergunta=""):
    return apply_refusal_guard(resposta, FONTES_CTX, context=CTX_FERIAS, question=pergunta)


@pytest.mark.parametrize(
    "resposta",
    [
        "Não consta no contexto o bônus, mas estimo cerca de 1.5 salários.",
        "Não encontrei isso, talvez no Windows 10.5.",
    ],
    ids=["decimal_1.5", "versao_10.5"],
)
def test_5_decimais_nao_contam_como_clausula(resposta):
    assert _guarda(resposta) == EXPECTED_REFUSAL


@pytest.mark.parametrize(
    "resposta",
    [
        "Essa regra inexiste nos documentos fornecidos.",
        "A informação está ausente do material.",
        "Há uma ausência de previsão sobre o tema.",
        "O documento não especifica esse valor.",
        "Sem previsão no material fornecido.",
    ],
)
def test_5_vocabulario_ampliado_de_negativa(resposta):
    assert _guarda(resposta) == EXPECTED_REFUSAL


def test_5_negativa_no_meio_sem_citacao_e_recusada():
    resposta = "A regra não consta para estagiários, mas vale para CLT: 30 dias."
    assert _guarda(resposta) == EXPECTED_REFUSAL


def test_5_clausula_inexistente_nos_trechos_e_recusada():
    resposta = "Conforme a cláusula 9.9 de politica_de_ferias.txt, o prazo é de 30 dias."
    assert _guarda(resposta) == EXPECTED_REFUSAL


def test_5_documento_fora_das_fontes_recuperadas_e_recusado():
    resposta = "Conforme a cláusula 3.1 de politica_inventada.txt, o prazo é de 30 dias."
    assert _guarda(resposta) == EXPECTED_REFUSAL


def test_5_valor_monetario_inventado_com_citacao_e_recusado():
    resposta = "Conforme guia_de_beneficios.txt, o bônus de Natal é R$ 500,00."
    assert _guarda(resposta) == EXPECTED_REFUSAL


def test_5_valor_monetario_inventado_sem_citacao_e_recusado():
    resposta = "O bônus de Natal para estagiários é de R$ 500,00."
    assert _guarda(resposta) == EXPECTED_REFUSAL


def test_5_prazo_e_percentual_inventados_sao_recusados():
    assert _guarda("Pela cláusula 3.1, a antecedência é de 45 dias.") == EXPECTED_REFUSAL
    assert _guarda("Pela cláusula 5.1, o reembolso é de 90%.") == EXPECTED_REFUSAL


def test_5_resposta_valida_com_valores_do_contexto_e_preservada():
    resposta = (
        "Pela cláusula 3.1 (politica_de_ferias.txt), a antecedência mínima é de 30 dias. "
        "O abono pecuniário (cláusula 5.1) equivale a 10 dias; no guia_de_beneficios.txt, "
        "o reembolso é de até 70%, com teto de R$ 800,00."
    )
    assert _guarda(resposta) == resposta


def test_5_valor_citado_na_pergunta_nao_conta_como_invencao():
    resposta = "Sim, cláusula 5.1: o abono pecuniário equivale a 10 dias."
    assert _guarda(resposta, "Posso vender 10 dias?") == resposta


def test_5_lista_de_clausulas_validas_e_preservada():
    resposta = "Conforme as cláusulas 3.1 e 5.1 de politica_de_ferias.txt, são 30 dias e 10 dias."
    assert _guarda(resposta) == resposta


def test_5_engine_repassa_contexto_e_pergunta_ao_guarda(monkeypatch):
    """ask() deve aplicar a guarda estrutural com o contexto realmente enviado ao modelo."""
    from langchain_core.runnables import RunnableLambda

    import rag_engine

    fontes = [
        {"doc": "politica_de_ferias.txt", "content": "3.1. Prazo de 30 dias.", "score": 0.9},
        {"doc": "guia_de_beneficios.txt", "content": "5.1. Teto de R$ 800,00.", "score": 0.8},
    ]
    chamadas = []

    def espiao(answer, source_names, context=None, question=""):
        chamadas.append(
            {"answer": answer, "source_names": source_names, "context": context, "question": question}
        )
        return "resposta final"

    eng = RAGEngine()
    eng.vectorstore = object()  # evita carregar o ChromaDB
    monkeypatch.setattr(eng, "retrieve", lambda q, k=5: fontes)
    monkeypatch.setattr(eng, "get_llm", lambda **kw: RunnableLambda(lambda _: "resposta bruta"))
    monkeypatch.setattr(rag_engine, "apply_refusal_guard", espiao)

    resultado = eng.ask("Qual o prazo?")

    assert resultado["answer"] == "resposta final"
    assert len(chamadas) == 1
    chamada = chamadas[0]
    assert chamada["answer"] == "resposta bruta"
    assert chamada["source_names"] == ["politica_de_ferias.txt", "guia_de_beneficios.txt"]
    assert chamada["question"] == "Qual o prazo?"
    assert chamada["context"] == resultado["context"]
    assert "3.1. Prazo de 30 dias." in chamada["context"]
    assert "5.1. Teto de R$ 800,00." in chamada["context"]


def test_5_nome_de_arquivo_com_espaco_ou_ponto_e_aceito_quando_e_fonte():
    fontes = ["Manual TI.PDF", "guia.v2.txt"]
    ctx = "--- [Trecho 1 | Fonte: Manual TI.PDF] ---\n2.1. Abra chamado no ServiceDesk.\n"
    resposta = "Conforme a cláusula 2.1 do Manual TI.PDF, abra um chamado no ServiceDesk."
    assert apply_refusal_guard(resposta, fontes, context=ctx) == resposta
    resposta2 = "Conforme a cláusula 2.1 do guia.v2.txt, abra um chamado no ServiceDesk."
    assert apply_refusal_guard(resposta2, fontes, context=ctx) == resposta2


def test_5_nome_de_arquivo_com_espaco_nao_fonte_continua_recusado():
    ctx = "--- [Trecho 1 | Fonte: Manual TI.PDF] ---\n2.1. Abra chamado no ServiceDesk.\n"
    resposta = "Conforme a cláusula 2.1 do Manual Falso.PDF, abra um chamado."
    assert apply_refusal_guard(resposta, ["Manual TI.PDF"], context=ctx) == EXPECTED_REFUSAL


def test_5_clausula_de_tres_niveis_no_contexto_valida_o_prefixo():
    ctx = "--- [Trecho 1 | Fonte: politica_de_ferias.txt] ---\n3.1.2 Prazo de 30 dias corridos.\n"
    fontes = ["politica_de_ferias.txt"]
    for resposta in (
        "Pela cláusula 3.1 de politica_de_ferias.txt, o prazo é de 30 dias.",
        "Pela cláusula 3.1.2 de politica_de_ferias.txt, o prazo é de 30 dias.",
    ):
        assert apply_refusal_guard(resposta, fontes, context=ctx) == resposta


def test_5_clausula_no_meio_da_linha_do_contexto_e_valida():
    ctx = (
        "--- [Trecho 1 | Fonte: politica_de_ferias.txt] ---\n"
        "o prazo segue o disposto em 5.1 abaixo, de 10 dias.\n"
    )
    resposta = "Conforme a cláusula 5.1 de politica_de_ferias.txt, o abono equivale a 10 dias."
    assert apply_refusal_guard(resposta, ["politica_de_ferias.txt"], context=ctx) == resposta


def test_5_clausula_inexistente_continua_recusada_com_tres_niveis():
    ctx = "--- [Trecho 1 | Fonte: politica_de_ferias.txt] ---\n3.1.2 Prazo de 30 dias corridos.\n"
    resposta = "Pela cláusula 4.1 de politica_de_ferias.txt, o prazo é de 30 dias."
    assert apply_refusal_guard(resposta, ["politica_de_ferias.txt"], context=ctx) == EXPECTED_REFUSAL


@pytest.mark.parametrize("valor", ["2 dias", "2 meses", "2 horas", "2 mês", "2 dia"])
def test_5_numeros_dos_cabecalhos_de_trecho_nao_validam_valores(valor):
    """'Trecho 2' do cabeçalho não é evidência de que '2 dias', '2 meses' constem (inclui singular)."""
    resposta = f"Pela cláusula 3.1 (politica_de_ferias.txt), o prazo é de {valor}."
    assert _guarda(resposta) == EXPECTED_REFUSAL


# ---------------------------------------------------------------------------
# Teste 6 - Segurança do upload e da renderização (offline)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "nome",
    [
        "../../etc/passwd.txt",
        r"..\..\Windows\win.txt",
        "/abs/caminho/politica.txt",
        r"C:\Users\x\politica.txt",
        "../fora.md",
    ],
)
def test_6_upload_nao_escapa_da_pasta_de_documentos(tmp_path, nome):
    destino = safe_document_path(str(tmp_path), nome)
    assert destino is not None
    assert os.path.dirname(os.path.abspath(destino)) == os.path.abspath(str(tmp_path))
    assert os.path.basename(destino) in {"passwd.txt", "win.txt", "politica.txt", "fora.md"}


@pytest.mark.parametrize("nome", ["", "   ", "..", ".", ".env", "script.exe", "doc.txt.exe", "sem_extensao", "dir/"])
def test_6_upload_rejeita_nomes_invalidos_ou_extensao_nao_suportada(tmp_path, nome):
    assert safe_document_path(str(tmp_path), nome) is None


@pytest.mark.parametrize("nome", ["politica.txt", "Manual TI.PDF", "notas.md"])
def test_6_upload_aceita_nomes_validos(tmp_path, nome):
    destino = safe_document_path(str(tmp_path), nome)
    assert destino == os.path.join(os.path.abspath(str(tmp_path)), nome)


def test_6_trecho_recuperado_e_escapado_antes_de_virar_html():
    html_saida = source_box_html('<script>alert(1)</script> <img src=x onerror=y> & "ok"')
    assert "<script>" not in html_saida and "<img" not in html_saida
    assert "&lt;script&gt;" in html_saida and "&amp;" in html_saida
    assert html_saida.startswith('<div class="source-box">')


def test_6_trecho_preserva_quebras_de_linha_do_texto():
    assert "<br>" in source_box_html("linha 1\nlinha 2")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
