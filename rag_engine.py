import os
import re
import glob
import requests
from decimal import Decimal, InvalidOperation
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
    r"não\s+const[ae]m?\b|não\s+há\s+(?:nenhum[ao]?\s+)?(?:informaç|menç|regra|previs|dados|registro|dispos)"
    r"|não\s+(?:foi\s+possível\s+)?encontr|não\s+(?:é|está)\s+mencionad"
    r"|não\s+possuo\s+ess|não\s+tenho\s+ess"
    r"|inexist|\baus[eê]ncia\b|\bausentes?\b|sem\s+previs"
    r"|não\s+(?:especific|detalh|aborda|trata|prev[eê]\b|estabelec|defin|cita)\w*"
    r"|não\s+exist[ei]m?\b",
    re.IGNORECASE,
)
# Citação de cláusula/seção numerada (ex.: "cláusula 3.1", "cláusulas 3.1 e 5.1", "(2.5)").
# Exige palavra-chave ou parênteses: números soltos ("1.5", "10.5") não são cláusulas.
_CLAUSE_REF_PATTERN = re.compile(
    r"(?:cl[áa]usulas?|se[çc](?:[ãa]o|[õo]es)|itens?|artigos?|art\.)\s*"
    r"((?:\d{1,2}\.\d{1,2}\.?(?:\s*(?:,|e|ou|a|/)\s*)?)+)"
    r"|\(\s*(\d{1,2}\.\d{1,2})\s*\)",
    re.IGNORECASE,
)
_CLAUSE_NUMBER = re.compile(r"\d{1,2}\.\d{1,2}")
# Numeração de cláusula presente nos trechos recuperados, em qualquer posição da linha.
# "3.1.2" valida também o prefixo "3.1"; o lookbehind evita casar "1.2" dentro de "3.1.2".
_CONTEXT_CLAUSE_PATTERN = re.compile(r"(?<![\d.])(\d{1,2}\.\d{1,2})(?!\d)")
# Cabeçalho "--- [Trecho N | Fonte: x] ---" montado por ask(): não é evidência de valores.
_CONTEXT_HEADER_PATTERN = re.compile(r"--- \[Trecho \d+ \| Fonte:[^\]]*\] ---")
# Nome de arquivo citado na resposta.
_DOC_NAME_PATTERN = re.compile(r"\b[\w\-]+\.(?:txt|md|pdf)\b", re.IGNORECASE)

# Valores verificáveis contra o contexto: dinheiro, percentual e prazos.
_NUMBER_PATTERN = re.compile(r"\d+(?:\.\d{3})*(?:,\d+)?")
_VALUE_PATTERNS = (
    re.compile(r"R\$\s*(\d+(?:\.\d{3})*(?:,\d+)?)"),
    re.compile(r"(\d+(?:,\d+)?)\s*%"),
    re.compile(r"(\d+)\s*(?:\(\s*[^)\d]{1,40}\)\s*)?(?:dias?|horas?|m[eê]s(?:es)?)\b", re.IGNORECASE),
)


SUPPORTED_EXTENSIONS = (".txt", ".md", ".pdf")


def safe_document_path(docs_dir: str, filename: str) -> Optional[str]:
    """Resolve com segurança o destino de um arquivo enviado pelo usuário.

    Descarta qualquer componente de diretório (``../``, caminhos absolutos,
    separadores de Windows ou Unix), rejeita nomes vazios, ocultos ou com
    extensão fora de ``SUPPORTED_EXTENSIONS`` e garante que o resultado fique
    dentro de ``docs_dir``. Devolve ``None`` quando o nome não é aceitável.
    """
    name = os.path.basename((filename or "").replace("\\", "/")).strip()
    if not name or name.startswith(".") or not name.lower().endswith(SUPPORTED_EXTENSIONS):
        return None
    base = os.path.abspath(docs_dir)
    destino = os.path.abspath(os.path.join(base, name))
    if os.path.dirname(destino) != base:
        return None
    return destino


def _to_decimal(token: str) -> Optional[Decimal]:
    """Converte '1.500,50' / '35,00' / '30' em Decimal normalizado (None se inválido)."""
    try:
        return Decimal(token.replace(".", "").replace(",", ".")).normalize()
    except InvalidOperation:
        return None


def _cited_clauses(text: str) -> set:
    """Cláusulas explicitamente citadas na resposta (com palavra-chave ou entre parênteses)."""
    cited = set()
    for match in _CLAUSE_REF_PATTERN.finditer(text):
        cited.update(_CLAUSE_NUMBER.findall(match.group(1) or match.group(2) or ""))
    return cited


def _strip_known_names(text: str, source_names: List[str]) -> str:
    """Remove da resposta os nomes das fontes recuperadas (com espaços ou pontos no nome).

    Sobra apenas o que parece nome de arquivo mas não é uma fonte conhecida.
    """
    for name in sorted({n for n in source_names if n}, key=len, reverse=True):
        text = re.sub(re.escape(name), " ", text, flags=re.IGNORECASE)
    return text


def _unsupported_values(body: str, evidence: str) -> bool:
    """True se a resposta traz R$, % ou prazo cujo número não aparece na evidência."""
    allowed = {d for d in map(_to_decimal, _NUMBER_PATTERN.findall(evidence)) if d is not None}
    for pattern in _VALUE_PATTERNS:
        for match in pattern.finditer(body):
            value = _to_decimal(match.group(1))
            if value is not None and value not in allowed:
                return True
    return False


def apply_refusal_guard(
    answer: str,
    source_names: List[str],
    context: Optional[str] = None,
    question: str = "",
) -> str:
    """Guarda determinística pós-geração da trava anti-alucinação.

    1. Se o modelo sinalizou que a informação não consta na base e a resposta não
       cita nenhum documento-fonte nem cláusula numerada, devolve ESTRITAMENTE a
       recusa padrão (sem preâmbulos nem especulação). Respostas ancoradas — ainda
       que parciais — são preservadas sem alteração.
    2. Quando ``context`` (trechos enviados ao modelo) é informado, aplica também
       a checagem estrutural, que devolve a recusa se a resposta: cita cláusula
       que não existe nos trechos; cita documento que não está em ``source_names``;
       ou traz valor (R$, %, prazo) ausente dos trechos e da pergunta.
    """
    text = (answer or "").strip()
    body = text.replace(REFUSAL_MESSAGE, "")
    cited_clauses = _cited_clauses(body)

    if context is not None:
        valid_clauses = set(_CONTEXT_CLAUSE_PATTERN.findall(context))
        if cited_clauses - valid_clauses:
            return REFUSAL_MESSAGE
        if _DOC_NAME_PATTERN.search(_strip_known_names(body, source_names)):
            return REFUSAL_MESSAGE
        evidence = _CONTEXT_HEADER_PATTERN.sub("", context)
        if _unsupported_values(body, f"{evidence}\n{question}"):
            return REFUSAL_MESSAGE

    signals_absence = REFUSAL_MESSAGE in text or bool(_ABSENCE_PATTERN.search(text))
    if not signals_absence:
        return answer

    cites_source = any(name and name in body for name in source_names)
    if cites_source or cited_clauses:
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
            answer = apply_refusal_guard(
                answer,
                [src["doc"] for src in sources],
                context=full_context,
                question=question,
            )
        except Exception as e:
            answer = f"Ocorreu um erro ao consultar o modelo de linguagem no Ollama: {str(e)}"

        return {
            "answer": answer,
            "sources": sources,
            "context": full_context
        }
