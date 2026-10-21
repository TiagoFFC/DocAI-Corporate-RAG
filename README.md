# DocAI: Assistente Virtual Corporativo com IA Local (RAG)

Protótipo desenvolvido para validação de Prova de Conceito (PoC) de um assistente de Inteligência Artificial para consultas a políticas e documentos corporativos internos.

A solução opera **100% localmente**, utilizando o **Ollama** com aceleração por GPU (NVIDIA RTX 3060 12GB), garantindo privacidade total dos dados e custo zero de execução.

---

## 🚀 Arquitetura Técnica

* **LLM Local**: `llama3.1:8b` (Meta) ou `llama3.2:3b` rodando via Ollama.
* **Modelo de Embeddings**: `nomic-embed-text` (Ollama).
* **Banco de Vetores**: `ChromaDB` (armazenamento persistente local em `./chroma_db`).
* **Framework RAG**: `LangChain` com `RecursiveCharacterTextSplitter`.
* **Interface**: `Streamlit` (Dashboard com chat, métricas de latência e citação de fontes).
* **Containerização**: `Docker` e `Docker Compose`.

---

## 🛠️ Como rodar com ou sem Ollama instalado no Windows

Como você tem o **Docker**, você nem precisa instalar o Ollama no Windows! O próprio `docker-compose.yml` agora sobe o container oficial do **Ollama** e a aplicação juntos.

---

### Opção 1: Rodar o Ollama direto pelo Docker (Sem instalar nada no Windows)

1. Suba os containers (Ollama + Chatbot):
```powershell
docker compose up -d --build
```

2. Baixe os modelos direto dentro do container do Ollama:
```powershell
docker exec -it ollama ollama pull nomic-embed-text
docker exec -it ollama ollama pull llama3.1:8b
```

3. Acesse no navegador:
**`http://localhost:8501`**

---

### Opção 2: Se preferir reinstalar o Ollama no Windows
Se quiser instalar o Ollama nativo no Windows (para aproveitar a aceleração direta de GPU da RTX 3060):

No PowerShell:
```powershell
winget install Ollama.Ollama
```
Depois:
```powershell
ollama pull nomic-embed-text
ollama pull llama3.1:8b
```


#### Opção B: Rodando Direto em Python (Sem Docker)
Se preferir rodar direto no Windows:

```powershell
# Cria o ambiente virtual
python -m venv venv

# Ativa o ambiente virtual
.\venv\Scripts\activate

# Instala as dependências
pip install -r requirements.txt

# Executa o Streamlit
streamlit run app.py
```
Acesse no navegador: **`http://localhost:8501`**

---

## 🎬 Roteiro para a Apresentação (Live Demo para o Professor)

1. **Apresentar a Barra Lateral**:
   * Mostre que o Ollama está conectado (`🔌 Testar`).
   * Mostre os 3 arquivos de exemplo na pasta `documentos/`:
     * `politica_de_ferias.txt`
     * `guia_de_beneficios.txt`
     * `manual_ti_e_seguranca.txt`
2. **Demonstrar a Indexação RAG**:
   * Clique no botão **`⚡ Indexar Documentos no ChromaDB`**.
   * Explique ao professor: *"Aqui o sistema está lendo os documentos, quebrando em chunks de 700 caracteres e gerando os vetores semânticos no ChromaDB local."*
3. **Fazer as Consultas pelos Botões Rápidos**:
   * Clique em **`🏖️ 1. Regra de Férias e Abono`**:
     * Observe o tempo de resposta medido na tela (~1 a 2 segundos).
     * Abra o expander *"🔍 Ver trechos e fontes recuperadas da base"* para mostrar exatamente qual cláusula da política foi recuperada pelo RAG.
   * Clique em **`🏥 2. Coparticipação Saúde`**:
     * Veja o bot listar valores exatos de consultas (R$ 35) e exames complexos (R$ 60).
   * Clique em **`💻 3. Furto de Notebook / TI`**:
     * Veja o bot orientar o registro de B.O. em 24h e o e-mail de segurança.
4. **O Teste da Trava Anti-Alucinação (O Momento de Destaque)**:
   * Clique em **`⚠️ 4. Pegadinha (Trava de Alucinação)`** (*"Qual é o valor do bônus de Natal pago para estagiários?"*).
   * O bot responderá categoricamente: *"Essa informação não consta nas políticas e manuais internos disponíveis..."*.
   * Destaque ao professor: *"Ao contrário do ChatGPT comum que inventaria uma resposta genérica, nosso modelo tem guardrails estritos e nunca extrapola os documentos fornecidos."*

---

## 📂 Estrutura de Arquivos

```text
d:/Projetos/Chatbot/
├── app.py                     # Interface Web Streamlit com chat e métricas
├── rag_engine.py              # Motor RAG com LangChain, ChromaDB e Ollama
├── requirements.txt           # Dependências Python
├── Dockerfile                 # Configuração da imagem Docker
├── docker-compose.yml         # Orquestração do container
├── README.md                  # Este guia
└── documentos/                # Pasta onde ficam os manuais e políticas
    ├── politica_de_ferias.txt
    ├── guia_de_beneficios.txt
    └── manual_ti_e_seguranca.txt
```
