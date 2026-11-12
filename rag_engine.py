import os
import re
import glob
import requests
from typing import List, Dict, Any, Optional, Tuple

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# Recusa padrão da trava anti-alucinação. NÃO altere este texto sem atualizar
# tests/test_rag_pipeline.py, README.md, AGENTS.md e o artigo (seção de resultados).
REFUSAL_MESSAGE = (
    "Essa informação não consta nas políticas e manuais internos disponíveis "
    "na minha base. Recomendo abrir um chamado junto ao setor responsável "
    "(Gente & Gestão ou TI)."
)

SYSTEM_PROMPT = """Você é o Assistente Virtual Corporativo (DocAI) de uma empresa.
Sua missão é responder à dúvida do colaborador baseando-se ESTRITAMENTE nos trechos de documentos fornecidos no contexto abaixo.

REGRAS OBRIGATÓRIAS DE RESPOSTA:
1. Responda APENAS com base nos trechos fornecidos no CONTEXTO. Se a pergunta contiver múltiplos itens (ex: prazo de antecedência e venda de dias), responda a cada item com base nas cláusulas correspondentes presentes no contexto.
2. Se a informação solicitada (ou parte dela) NÃO constar expressamente nos documentos fornecidos, indique com precisão o que consta e, para o que não constar, informe educadamente:
   "__REFUSAL__"
   Se NENHUMA parte da pergunta puder ser respondida com o contexto, responda SOMENTE com essa frase, sem acrescentar mais nada.
3. NUNCA invente, presuma, extrapole ou deduza regras que não estejam escritas no texto.
4. Sempre mencione o nome do documento fonte e o número da seção ou cláusula relevante onde a regra está escrita.
5. Mantenha um tom profissional, cordial, claro e objetivo em português do Brasil.

---
CONTEXTO DOS DOCUMENTOS INTERNOS:
{context}
---

PERGUNTA DO COLABORADOR:
{question}
""".replace("__REFUSAL__", REFUSAL_MESSAGE)

# Sinais textuais de que o modelo declarou não ter encontrado a informação.
_ABSENCE_PATTERN = re.compile(
    r"não\s+const[ae]m?\b|não\s+há\s+(?:nenhuma\s+)?(?:informaç|menç|regra|previs)"
    r"|não\s+(?:foi\s+possível\s+)?encontr|não\s+(?:é|está)\s+mencionad"
    r"|não\s+possuo\s+ess|não\s+tenho\s+ess",
    re.IGNORECASE,
)
# Citação de cláusula/seção numerada (ex.: 3.1, 2.5) — evidência de resposta ancorada.
_CLAUSE_PATTERN = re.compile(r"\b\d{1,2}\.\d{1,2}\b")


def apply_refusal_guard(answer: str, source_names: List[str]) -> str:
    """Guarda determinística pós-geração da trava anti-alucinação.

    Se o modelo sinalizou que a informação não consta na base e a resposta não
    cita nenhum documento-fonte nem cláusula numerada, devolve ESTRITAMENTE a
    recusa padrão (sem preâmbulos nem especulação). Respostas ancoradas — ainda
    que parciais — são preservadas sem alteração.
    """
    text = (answer or "").strip()
    signals_absence = REFUSAL_MESSAGE in text or bool(_ABSENCE_PATTERN.search(text))
    if not signals_absence:
        return answer

    body = text.replace(REFUSAL_MESSAGE, "")
    cites_source = any(name and name in body for name in source_names)
    cites_clause = bool(_CLAUSE_PATTERN.search(body))
    if cites_source or cites_clause:
        return answer
    return REFUSAL_MESSAGE


class RAGEngine:
    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        llm_model: str = "llama3.1:8b",
        embedding_model: str = "nomic-embed-text",
        persist_dir: str = "./chroma_db",
    ):
        self.ollama_url = ollama_url
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.persist_dir = persist_dir
        self.vectorstore = None

    @staticmethod
    def test_connection(ollama_url: str) -> Tuple[bool, str]:
        """Testa se o serviço do Ollama está acessível."""
        try:
            resp = requests.get(f"{ollama_url.rstrip('/')}/api/tags", timeout=3)
            if resp.status_code == 200:
                models = [m.get("name") for m in resp.json().get("models", [])]
                return True, f"Conectado ao Ollama! Modelos instalados: {', '.join(models) if models else 'Nenhum'}"
            return False, f"Ollama respondeu com código de erro {resp.status_code}"
        except Exception as e:
            return False, f"Falha ao conectar no Ollama em {ollama_url}: {str(e)}"

    def get_embeddings(self) -> OllamaEmbeddings:
        return OllamaEmbeddings(
            base_url=self.ollama_url,
            model=self.embedding_model,
        )

    def get_llm(self, temperature: float = 0.1, seed: Optional[int] = None) -> ChatOllama:
        return ChatOllama(
            base_url=self.ollama_url,
            model=self.llm_model,
            temperature=temperature,
            seed=seed,
        )

    def load_existing_vectorstore(self) -> bool:
        """Carrega a base vetorial do disco caso já exista."""
        if os.path.exists(self.persist_dir) and os.listdir(self.persist_dir):
            try:
                embeddings = self.get_embeddings()
                store = Chroma(
                    persist_directory=self.persist_dir,
                    embedding_function=embeddings,
                )
                # Bases antigas (distância L2) ou vazias exigem reindexação.
                metadata = store._collection.metadata or {}
                if metadata.get("hnsw:space") != "cosine" or store._collection.count() == 0:
                    return False
                self.vectorstore = store
                return True
            except Exception:
                return False
        return False

    def load_and_split(self, docs_dir: str = "./documentos") -> List[Any]:
        """Lê os arquivos (.txt, .md, .pdf) da pasta e devolve a lista de chunks."""
        files = sorted(glob.glob(os.path.join(docs_dir, "*.*")))
        supported_files = [f for f in files if f.endswith((".txt", ".md", ".pdf"))]

        documents = []
        for file_path in supported_files:
            file_name = os.path.basename(file_path)
            try:
                if file_path.endswith(".pdf"):
                    docs = PyPDFLoader(file_path).load()
                else:
                    docs = TextLoader(file_path, encoding="utf-8").load()
                for d in docs:
                    d.metadata["source_file"] = file_name
                documents.extend(docs)
            except Exception as e:
                print(f"Erro ao ler {file_name}: {e}")

        # Chunking: divisão em pedaços com sobreposição (overlap) para manter contexto
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=120,
            separators=["\n\n", "\n", " ", ""]
        )
        return text_splitter.split_documents(documents)

    def index_directory(self, docs_dir: str = "./documentos") -> Dict[str, Any]:
        """Lê os arquivos de texto e PDF da pasta, divide em chunks e salva no ChromaDB."""
        if not os.path.exists(docs_dir):
            os.makedirs(docs_dir, exist_ok=True)

        files = glob.glob(os.path.join(docs_dir, "*.*"))
        supported_files = [f for f in files if f.endswith((".txt", ".md", ".pdf"))]

        if not supported_files:
            return {
                "success": False,
                "message": f"Nenhum arquivo (.txt, .md, .pdf) encontrado na pasta '{docs_dir}'."
            }

        chunks = self.load_and_split(docs_dir)
        if not chunks:
            return {"success": False, "message": "Não foi possível extrair texto dos documentos."}

        # Vetorização e gravação no ChromaDB
        embeddings = self.get_embeddings()
        # Remove a coleção anterior (inclusive bases legadas) para não duplicar trechos
        try:
            Chroma(persist_directory=self.persist_dir, embedding_function=embeddings).delete_collection()
        except Exception:
            pass
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=self.persist_dir,
            collection_metadata={"hnsw:space": "cosine"},  # Equação (1) do artigo
        )

        return {
            "success": True,
            "files_count": len(supported_files),
            "chunks_count": len(chunks),
            "files": [os.path.basename(f) for f in supported_files],
            "message": f"{len(supported_files)} documentos indexados com sucesso em {len(chunks)} trechos vetoriais!"
        }

    def retrieve(self, question: str, k: int = 5) -> List[Dict[str, Any]]:
        """Busca top-k por similaridade de cossenos S_C(q, d), do mais para o menos similar."""
        if not self.vectorstore and not self.load_existing_vectorstore():
            return []

        # Busca candidatos e deduplica trechos redundantes para garantir diversidade de cláusulas
        raw_docs = self.vectorstore.similarity_search_with_score(question, k=max(k * 2, 8))
        seen_texts = set()
        sources = []
        for doc, distance in raw_docs:
            text_norm = " ".join(doc.page_content.split())[:120]
            if text_norm in seen_texts:
                continue
            seen_texts.add(text_norm)
            sources.append({
                "doc": doc.metadata.get("source_file", "Documento Interno"),
                "content": doc.page_content.strip(),
                # ChromaDB devolve distância de cosseno (1 - S_C); convertemos para similaridade
                "score": round(1.0 - float(distance), 4),
            })
            if len(sources) >= k:
                break
        return sources

    def ask(
        self,
        question: str,
        k: int = 5,
        temperature: float = 0.1,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Realiza a busca vetorial por similaridade e gera a resposta com citação."""
        if not self.vectorstore:
            loaded = self.load_existing_vectorstore()
            if not loaded:
                return {
                    "answer": "A base de documentos ainda não foi indexada. Por favor, clique no botão 'Indexar Documentos' na barra lateral.",
                    "sources": [],
                    "context": ""
                }

        sources = self.retrieve(question, k=k)

        # Montagem do contexto textual para o prompt
        full_context = "\n\n".join(
            f"--- [Trecho {idx} | Fonte: {src['doc']}] ---\n{src['content']}"
            for idx, src in enumerate(sources, 1)
        )

        # Invocação da LLM com prompt restrito
        prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)
        llm = self.get_llm(temperature=temperature, seed=seed)
        chain = prompt | llm | StrOutputParser()

        try:
            answer = chain.invoke({
                "context": full_context,
                "question": question
            })
            # Trava anti-alucinação determinística (pós-geração)
            answer = apply_refusal_guard(answer, [src["doc"] for src in sources])
        except Exception as e:
            answer = f"Ocorreu um erro ao consultar o modelo de linguagem no Ollama: {str(e)}"

        return {
            "answer": answer,
            "sources": sources,
            "context": full_context
        }
