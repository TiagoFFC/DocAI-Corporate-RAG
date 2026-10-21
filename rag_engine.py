import os
import glob
import requests
from typing import List, Dict, Any, Tuple

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


SYSTEM_PROMPT = """Você é o Assistente Virtual Corporativo (DocAI) de uma empresa.
Sua missão é responder à dúvida do colaborador baseando-se ESTRITAMENTE nos trechos de documentos fornecidos no contexto abaixo.

REGRAS OBRIGATÓRIAS DE RESPOSTA:
1. Responda APENAS com base nos trechos fornecidos no CONTEXTO. Se a pergunta contiver múltiplos itens (ex: prazo de antecedência e venda de dias), responda a cada item com base nas cláusulas correspondentes presentes no contexto.
2. Se a informação solicitada (ou parte dela) NÃO constar expressamente nos documentos fornecidos, indique com precisão o que consta e, para o que não constar, informe educadamente:
   "Essa informação não consta nas políticas e manuais internos disponíveis na minha base. Recomendo abrir um chamado junto ao setor responsável (Gente & Gestão ou TI)."
3. NUNCA invente, presuma, extrapole ou deduza regras que não estejam escritas no texto.
4. Sempre mencione o nome do documento fonte e o número da seção ou cláusula relevante onde a regra está escrita.
5. Mantenha um tom profissional, cordial, claro e objetivo em português do Brasil.

---
CONTEXTO DOS DOCUMENTOS INTERNOS:
{context}
---

PERGUNTA DO COLABORADOR:
{question}
"""


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

    def get_llm(self, temperature: float = 0.1) -> ChatOllama:
        return ChatOllama(
            base_url=self.ollama_url,
            model=self.llm_model,
            temperature=temperature,
        )

    def load_existing_vectorstore(self) -> bool:
        """Carrega a base vetorial do disco caso já exista."""
        if os.path.exists(self.persist_dir) and os.listdir(self.persist_dir):
            try:
                embeddings = self.get_embeddings()
                self.vectorstore = Chroma(
                    persist_directory=self.persist_dir,
                    embedding_function=embeddings,
                )
                return True
            except Exception:
                return False
        return False

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

        documents = []
        for file_path in supported_files:
            file_name = os.path.basename(file_path)
            try:
                if file_path.endswith(".pdf"):
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                else:
                    loader = TextLoader(file_path, encoding="utf-8")
                    docs = loader.load()

                for d in docs:
                    d.metadata["source_file"] = file_name
                documents.extend(docs)
            except Exception as e:
                print(f"Erro ao ler {file_name}: {e}")

        if not documents:
            return {"success": False, "message": "Não foi possível extrair texto dos documentos."}

        # Chunking: divisão em pedaços com sobreposição (overlap) para manter contexto
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=120,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)

        # Vetorização e gravação no ChromaDB
        embeddings = self.get_embeddings()
        if self.vectorstore is not None:
            try:
                self.vectorstore.delete_collection()
            except Exception:
                pass
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=self.persist_dir,
        )

        return {
            "success": True,
            "files_count": len(supported_files),
            "chunks_count": len(chunks),
            "files": [os.path.basename(f) for f in supported_files],
            "message": f"{len(supported_files)} documentos indexados com sucesso em {len(chunks)} trechos vetoriais!"
        }

    def ask(self, question: str, k: int = 5, temperature: float = 0.1) -> Dict[str, Any]:
        """Realiza a busca vetorial por similaridade e gera a resposta com citação."""
        if not self.vectorstore:
            loaded = self.load_existing_vectorstore()
            if not loaded:
                return {
                    "answer": "A base de documentos ainda não foi indexada. Por favor, clique no botão 'Indexar Documentos' na barra lateral.",
                    "sources": [],
                    "context": ""
                }

        # Busca candidatos e deduplica trechos redundantes para garantir diversidade de cláusulas
        raw_docs = self.vectorstore.similarity_search_with_score(question, k=max(k * 2, 8))
        seen_texts = set()
        docs_with_scores = []
        for doc, score in raw_docs:
            text_norm = " ".join(doc.page_content.split())[:120]
            if text_norm not in seen_texts:
                seen_texts.add(text_norm)
                docs_with_scores.append((doc, score))
            if len(docs_with_scores) >= k:
                break

        # Montagem do contexto textual para o prompt
        context_parts = []
        sources = []
        for idx, (doc, score) in enumerate(docs_with_scores, 1):
            source_name = doc.metadata.get("source_file", "Documento Interno")
            snippet = doc.page_content.strip()
            context_parts.append(f"--- [Trecho {idx} | Fonte: {source_name}] ---\n{snippet}")
            sources.append({
                "doc": source_name,
                "content": snippet,
                "score": round(float(score), 4)
            })

        full_context = "\n\n".join(context_parts)

        # Invocação da LLM com prompt restrito
        prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)
        llm = self.get_llm(temperature=temperature)
        chain = prompt | llm | StrOutputParser()

        try:
            answer = chain.invoke({
                "context": full_context,
                "question": question
            })
        except Exception as e:
            answer = f"Ocorreu um erro ao consultar o modelo de linguagem no Ollama: {str(e)}"

        return {
            "answer": answer,
            "sources": sources,
            "context": full_context
        }
