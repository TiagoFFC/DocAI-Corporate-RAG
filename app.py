import os
import time
import streamlit as st
from rag_engine import RAGEngine

# Configuração da página
st.set_page_config(
    page_title="DocAI - Assistente Corporativo",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilização visual moderna e corporativa
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-badge {
        display: inline-block;
        background-color: #E0E7FF;
        color: #3730A3;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    .source-box {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 0.75rem;
        border-radius: 0.25rem;
        margin-top: 0.5rem;
        font-size: 0.9rem;
    }
    .stChatMessage {
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Detecta se está dentro do Docker ou no Host
DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Inicialização de variáveis de sessão
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Olá! Eu sou o **DocAI**, seu assistente para consulta de políticas internas, benefícios de RH e regras de TI. Como posso te ajudar hoje?",
            "sources": [],
            "latency": None
        }
    ]

if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = None

if "indexed" not in st.session_state:
    st.session_state.indexed = False


# BARRA LATERAL (CONFIGURAÇÕES E INDEXAÇÃO)
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712109.png", width=64)
    st.title("Painel de Controle")

    st.subheader("1. Conexão Ollama (Local)")
    ollama_url = st.text_input("URL do Ollama:", value=DEFAULT_OLLAMA_URL)
    
    col_conn1, col_conn2 = st.columns([1, 1])
    with col_conn1:
        if st.button("🔌 Testar"):
            ok, msg = RAGEngine.test_connection(ollama_url)
            if ok:
                st.success("Ollama Online!")
            else:
                st.error("Offline")
                st.caption(msg)
                
    with col_conn2:
        temperature = st.slider("Criatividade (Temp):", 0.0, 0.5, 0.1, 0.05)

    model_name = st.selectbox(
        "Modelo LLM:",
        ["llama3.1:8b", "llama3.2:3b", "qwen2.5:7b", "mistral:7b"],
        index=0
    )
    
    embedding_model = st.text_input("Modelo Embeddings:", value="nomic-embed-text")

    st.divider()

    st.subheader("2. Base de Documentos")
    docs_folder = "./documentos"
    
    if os.path.exists(docs_folder):
        files = os.listdir(docs_folder)
        st.caption(f"📁 **Arquivos disponíveis ({len(files)}):**")
        for f in files:
            st.markdown(f"- 📄 `{f}`")
    else:
        st.warning("Pasta ./documentos não encontrada.")

    # Upload dinâmico opcional
    uploaded_files = st.file_uploader("Adicionar novo PDF ou TXT:", accept_multiple_files=True, type=["txt", "pdf", "md"])
    if uploaded_files:
        for uf in uploaded_files:
            file_path = os.path.join(docs_folder, uf.name)
            with open(file_path, "wb") as f_out:
                f_out.write(uf.getbuffer())
        st.success(f"{len(uploaded_files)} arquivo(s) salvo(s)!")

    # Botão de Indexação
    if st.button("⚡ Indexar Documentos no ChromaDB", type="primary", use_container_width=True):
        with st.spinner("Lendo arquivos, criando chunks e gerando embeddings..."):
            engine = RAGEngine(
                ollama_url=ollama_url,
                llm_model=model_name,
                embedding_model=embedding_model
            )
            res = engine.index_directory(docs_dir=docs_folder)
            if res.get("success"):
                st.session_state.rag_engine = engine
                st.session_state.indexed = True
                st.success(res.get("message"))
            else:
                st.error(res.get("message"))

    st.divider()

    # Perguntas para a Demonstração na Apresentação
    st.subheader("🎯 Testes Prontos para Apresentação")
    st.caption("Clique para preencher uma pergunta de teste:")
    
    demo_q1 = "Qual a antecedência mínima para solicitar férias e posso vender 10 dias?"
    demo_q2 = "Como funciona a coparticipação em consultas e exames no plano de saúde?"
    demo_q3 = "O que devo fazer caso meu notebook corporativo seja roubado?"
    demo_q4 = "Qual é o valor do bônus de Natal pago para estagiários?"

    if st.button("🏖️ 1. Regra de Férias e Abono", use_container_width=True):
        st.session_state.selected_question = demo_q1
    if st.button("🏥 2. Coparticipação Saúde", use_container_width=True):
        st.session_state.selected_question = demo_q2
    if st.button("💻 3. Furto de Notebook / TI", use_container_width=True):
        st.session_state.selected_question = demo_q3
    if st.button("⚠️ 4. Pegadinha (Trava de Alucinação)", use_container_width=True):
        st.session_state.selected_question = demo_q4

    if st.button("🗑️ Limpar Conversa", use_container_width=True):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Conversa reiniciada. Como posso te ajudar?",
                "sources": [],
                "latency": None
            }
        ]
        st.rerun()


# ÁREA PRINCIPAL
st.markdown('<div class="main-title">💼 DocAI: Assistente Corporativo Inteligente</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Arquitetura RAG com modelo 100% local (Ollama + ChromaDB) e garantia de fidelidade aos manuais internos.</div>', unsafe_allow_html=True)

# Inicializa o motor RAG se ainda não estiver na memória
if st.session_state.rag_engine is None:
    st.session_state.rag_engine = RAGEngine(
        ollama_url=ollama_url,
        llm_model=model_name,
        embedding_model=embedding_model
    )
    # Tenta carregar base existente do disco
    st.session_state.rag_engine.load_existing_vectorstore()

# Renderiza histórico de mensagens
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Mostra métricas de tempo e fontes se disponíveis
        if msg.get("latency") is not None:
            st.markdown(
                f'<span class="metric-badge">⏱️ {msg["latency"]:.2f}s</span>'
                f'<span class="metric-badge">📚 {len(msg["sources"])} fontes citadas</span>',
                unsafe_allow_html=True
            )
            
        if msg.get("sources"):
            with st.expander("🔍 Ver trechos e fontes recuperadas da base"):
                for idx, src in enumerate(msg["sources"], 1):
                    st.markdown(f"**Fonte {idx}: `{src['doc']}`** *(Score de similaridade: {src['score']})*")
                    st.markdown(f'<div class="source-box">{src["content"]}</div>', unsafe_allow_html=True)

# Captura pergunta enviada via caixa de texto ou pelos botões rápidos
prompt_input = st.chat_input("Digite sua dúvida sobre as políticas internas...")
user_question = None

if "selected_question" in st.session_state and st.session_state.selected_question:
    user_question = st.session_state.selected_question
    st.session_state.selected_question = None
elif prompt_input:
    user_question = prompt_input

if user_question:
    # Registra pergunta do usuário
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    # Processamento com RAG
    with st.chat_message("assistant"):
        with st.spinner("Consultando políticas internas e gerando resposta..."):
            t_start = time.time()
            result = st.session_state.rag_engine.ask(
                question=user_question,
                temperature=temperature
            )
            t_end = time.time()
            latency = t_end - t_start

            answer = result.get("answer", "")
            sources = result.get("sources", [])

            st.markdown(answer)
            st.markdown(
                f'<span class="metric-badge">⏱️ {latency:.2f}s</span>'
                f'<span class="metric-badge">📚 {len(sources)} fontes citadas</span>',
                unsafe_allow_html=True
            )

            if sources:
                with st.expander("🔍 Ver trechos e fontes recuperadas da base"):
                    for idx, src in enumerate(sources, 1):
                        st.markdown(f"**Fonte {idx}: `{src['doc']}`** *(Score de similaridade: {src['score']})*")
                        st.markdown(f'<div class="source-box">{src["content"]}</div>', unsafe_allow_html=True)

            # Salva no histórico
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "latency": latency
            })
