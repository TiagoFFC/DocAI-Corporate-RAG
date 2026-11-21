"""Testes unitários e de integração do controle de acesso por setor."""
import pytest

from conftest import PRIMARY_MODEL
from rag_engine import REFUSAL_MESSAGE, RAGEngine, infer_sector

Q_FERIAS = "Qual a antecedência mínima para solicitar férias e posso vender 10 dias?"
Q_SAUDE = "Como funciona a coparticipação em consultas e exames no plano de saúde?"
Q_NOTEBOOK = "O que devo fazer caso meu notebook corporativo seja roubado?"

ollama = pytest.mark.ollama


# ---------------------------------------------------------------------------
# Testes offline (not ollama)
# ---------------------------------------------------------------------------
def test_infer_sector_arquivos_na_raiz():
    assert infer_sector("politica_de_ferias.txt") == "rh"
    assert infer_sector("documentos/guia_de_beneficios.txt") == "rh"
    assert infer_sector("documentos/manual_ti_e_seguranca.txt") == "ti"
    assert infer_sector("documentos/comunicado_geral.txt") == "geral"


def test_infer_sector_subpastas(tmp_path):
    assert infer_sector("documentos/rh/norma.txt") == "rh"
    assert infer_sector("documentos/TI/politica.pdf") == "ti"
    assert infer_sector("documentos/Juridico/termo.md") == "juridico"

    sub_arquivo = tmp_path / "financeiro" / "reembolso.txt"
    raiz_arquivo = tmp_path / "aviso.txt"
    assert infer_sector(str(sub_arquivo), docs_dir=str(tmp_path)) == "financeiro"
    assert infer_sector(str(raiz_arquivo), docs_dir=str(tmp_path)) == "geral"


def test_load_and_split_grava_metadado_sector_nos_documentos_padrao(docs_dir):
    chunks = RAGEngine().load_and_split(docs_dir)
    assert chunks

    expected = {
        "politica_de_ferias.txt": "rh",
        "guia_de_beneficios.txt": "rh",
        "manual_ti_e_seguranca.txt": "ti",
    }
    for chunk in chunks:
        assert "sector" in chunk.metadata
        assert chunk.metadata["sector"] == expected[chunk.metadata["source_file"]]


def test_load_and_split_busca_recursiva_em_subpastas(tmp_path):
    (tmp_path / "rh").mkdir()
    (tmp_path / "ti").mkdir()
    (tmp_path / "c.txt").write_text("Comunicado geral da empresa.", encoding="utf-8")
    (tmp_path / "rh" / "a.txt").write_text("Norma interna de RH.", encoding="utf-8")
    (tmp_path / "ti" / "b.txt").write_text("Norma interna de TI.", encoding="utf-8")

    chunks = RAGEngine().load_and_split(str(tmp_path))
    observados = [(c.metadata["source_file"], c.metadata["sector"]) for c in chunks]

    assert observados == [
        ("c.txt", "geral"),
        ("a.txt", "rh"),
        ("b.txt", "ti"),
    ]


def test_retrieve_com_lista_vazia_de_setores_retorna_vazio(tmp_path):
    engine = RAGEngine(persist_dir=str(tmp_path))
    engine.vectorstore = object()

    assert engine.retrieve("ferias", allowed_sectors=[]) == []


def test_ask_retorna_recusa_direta_sem_chamar_llm_quando_sem_trechos_autorizados(
    tmp_path, monkeypatch
):
    engine = RAGEngine(persist_dir=str(tmp_path))
    engine.vectorstore = object()

    monkeypatch.setattr(engine, "retrieve", lambda *args, **kwargs: [])

    def _fail_llm(*args, **kwargs):
        raise AssertionError("LLM não deveria ser chamado")

    monkeypatch.setattr(engine, "get_llm", _fail_llm)

    res = engine.ask("Qual o prazo de férias?", allowed_sectors=["geral"])
    assert res == {
        "answer": REFUSAL_MESSAGE,
        "sources": [],
        "context": "",
    }


# ---------------------------------------------------------------------------
# Testes de integração (@pytest.mark.ollama)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def sector_engine(ollama_url, tmp_path_factory, docs_dir):
    persist_dir = str(tmp_path_factory.mktemp("chroma_sector_test"))
    eng = RAGEngine(
        ollama_url=ollama_url,
        llm_model=PRIMARY_MODEL,
        persist_dir=persist_dir,
    )
    result = eng.index_directory(docs_dir=docs_dir)
    assert result["success"], result.get("message")
    yield eng
    eng.vectorstore = None


@ollama
def test_integracao_perfil_ti_recupera_manual_ti(sector_engine):
    sources = sector_engine.retrieve(Q_NOTEBOOK, k=5, allowed_sectors=["ti"])
    assert sources
    assert sources[0]["doc"] == "manual_ti_e_seguranca.txt"
    assert all(
        src["doc"] == "manual_ti_e_seguranca.txt" and src["sector"] == "ti"
        for src in sources
    )


@ollama
@pytest.mark.parametrize("pergunta", [Q_FERIAS, Q_SAUDE], ids=["ferias", "saude"])
def test_integracao_perfil_ti_bloqueia_ferias_e_beneficios_e_recusa(
    sector_engine, pergunta
):
    sources = sector_engine.retrieve(pergunta, k=5, allowed_sectors=["ti"])
    docs_recuperados = {src["doc"] for src in sources}
    assert "politica_de_ferias.txt" not in docs_recuperados
    assert "guia_de_beneficios.txt" not in docs_recuperados

    res = sector_engine.ask(
        pergunta, temperature=0.0, seed=42, allowed_sectors=["ti"]
    )
    assert res["answer"] == REFUSAL_MESSAGE


@ollama
def test_integracao_perfil_rh_recupera_rh_e_bloqueia_ti(sector_engine):
    src_ferias = sector_engine.retrieve(Q_FERIAS, k=5, allowed_sectors=["rh"])
    src_saude = sector_engine.retrieve(Q_SAUDE, k=5, allowed_sectors=["rh"])
    src_ti = sector_engine.retrieve(Q_NOTEBOOK, k=5, allowed_sectors=["rh"])

    assert src_ferias[0]["doc"] == "politica_de_ferias.txt"
    assert src_saude[0]["doc"] == "guia_de_beneficios.txt"
    assert "manual_ti_e_seguranca.txt" not in {s["doc"] for s in src_ti}

    res = sector_engine.ask(
        Q_NOTEBOOK, temperature=0.0, seed=42, allowed_sectors=["rh"]
    )
    assert res["answer"] == REFUSAL_MESSAGE


@ollama
def test_integracao_perfil_geral_sem_documentos_recusa(sector_engine):
    assert sector_engine.retrieve(Q_FERIAS, k=5, allowed_sectors=["geral"]) == []
    res = sector_engine.ask(
        Q_FERIAS, temperature=0.0, seed=42, allowed_sectors=["geral"]
    )
    assert res["answer"] == REFUSAL_MESSAGE
    assert res["sources"] == []


@ollama
def test_integracao_admin_acessa_todos_os_setores(sector_engine):
    assert (
        sector_engine.retrieve(Q_FERIAS, k=5, allowed_sectors=None)[0]["doc"]
        == "politica_de_ferias.txt"
    )
    assert (
        sector_engine.retrieve(Q_SAUDE, k=5, allowed_sectors=None)[0]["doc"]
        == "guia_de_beneficios.txt"
    )
    assert (
        sector_engine.retrieve(Q_NOTEBOOK, k=5, allowed_sectors=None)[0]["doc"]
        == "manual_ti_e_seguranca.txt"
    )
