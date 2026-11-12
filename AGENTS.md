# AGENTS.md — Guia de contexto rápido para agentes de IA

Leia este arquivo antes de alterar qualquer coisa. Ele vale para qualquer agente (Claude, GPT, Gemini, Cursor, Codex).

## O que é este projeto

**DocAI** é um assistente corporativo com RAG 100% local: Streamlit → LangChain → ChromaDB + Ollama. Acompanha um artigo científico (Unisul, template SBC). Autores: Tiago A. Ventura e Jorge Werner (orientador).

É um projeto **Python 3.11**. Não há Node.js, Next.js nem banco relacional.

## Mapa de arquivos

| Caminho | Papel | Pode editar? |
| --- | --- | --- |
| `rag_engine.py` | Motor RAG: chunking, indexação, busca por cosseno, prompt, trava anti-alucinação | Sim, com testes |
| `app.py` | Interface Streamlit (sem regra de negócio) | Sim |
| `documentos/` | Corpus (`.txt`, `.md`, `.pdf`) que alimenta a base | Sim (ver "Adicionar documentos") |
| `chroma_db/` | Base vetorial **gerada**; fora do Git | Nunca à mão |
| `tests/test_rag_pipeline.py` | Suíte ponta a ponta (30 testes) | Sim |
| `scripts/verify_docx.py` | Auditoria do artigo + paridade Word × Markdown | Sim |
| `scripts/docx_to_md.py` | Gera `Artigo/ARTIGO_DOCAI.md` a partir do Word | Sim |
| `scripts/benchmark_latency.py` | Mede latência por modelo | Sim |
| `scripts/fill_docx.py` | Histórico: script que preencheu o template. **Não reexecute** (sobrescreve o artigo) | Não |
| `Artigo/Artigo - Chatbot.docx` | Artigo oficial — **fonte única da verdade** | Só a pedido dos autores |
| `Artigo/ARTIGO_DOCAI.md` | Transcrição gerada do Word | **Nunca à mão** |
| `Artigo/Artigo - Chatbot_original.docx` | Template original em branco | Não |
| `docs/HISTORICO_PROJETO.md` | Histórico executivo e changelog | Sim |

## Comandos

O Python do projeto roda dentro do Docker (o host pode não ter Python configurado).

```bash
# Subir tudo (Ollama + app em http://localhost:8501)
docker compose up -d --build

# Baixar os modelos (apenas na primeira vez)
docker exec ollama ollama pull nomic-embed-text
docker exec ollama ollama pull llama3.1:8b
docker exec ollama ollama pull llama3.2:3b

# Testes e auditoria dentro de um container com o repositório montado
docker run --rm --network chatbot_default -e OLLAMA_BASE_URL=http://ollama:11434 \
  -v "$PWD:/workspace" -w /workspace --entrypoint bash chatbot-docai-chatbot \
  -c "pip install -q -r requirements-dev.txt && python -m pytest tests/ && python scripts/verify_docx.py 'Artigo/Artigo - Chatbot.docx'"
```

O nome da rede (`chatbot_default`) e da imagem (`chatbot-docai-chatbot`) derivam do nome da pasta do projeto; confira com `docker network ls` e `docker images`.

Com Python local: `pip install -r requirements-dev.txt`, depois `python -m pytest tests/`.
Sem Ollama (CI): `python -m pytest tests/ -m "not ollama"`.

## Regras invioláveis

### 1. A trava anti-alucinação não pode ser enfraquecida

Ela tem três partes em `rag_engine.py`, e as três devem continuar existindo:

1. `REFUSAL_MESSAGE` — o texto exato da recusa:
   `Essa informação não consta nas políticas e manuais internos disponíveis na minha base. Recomendo abrir um chamado junto ao setor responsável (Gente & Gestão ou TI).`
2. `SYSTEM_PROMPT` — regras 1 a 5 (responder só com o contexto; recusar o que não consta; nunca inventar; citar documento e cláusula; português formal).
3. `apply_refusal_guard()` — guarda determinística pós-geração: se a resposta sinaliza ausência de informação e não cita nenhum documento-fonte nem cláusula numerada, ela é substituída por `REFUSAL_MESSAGE`.

Não altere o texto da recusa, não remova regras do prompt, não contorne a guarda e não eleve a temperatura padrão (0,1). Os testes `test_5_*` protegem esse comportamento: **nunca os afrouxe para fazer a suíte passar**. Se uma mudança legítima exigir novo texto de recusa, atualize junto: `rag_engine.py`, `tests/test_rag_pipeline.py`, `README.md`, este arquivo e o artigo.

### 2. O score exibido é similaridade de cossenos

A coleção é criada com `hnsw:space = cosine` e `retrieve()` converte distância em similaridade (`1 - distância`), conforme a Equação (1) do artigo. Não troque a métrica. Bases antigas (L2) são rejeitadas por `load_existing_vectorstore()` e exigem reindexação.

### 3. O Word manda no Markdown

Mudou o artigo? Edite o `.docx`, rode `python scripts/docx_to_md.py` e depois `python scripts/verify_docx.py "Artigo/Artigo - Chatbot.docx"` (deve sair com código 0). Não edite números do artigo (latências, tabela) sem autorização dos autores.

### 4. Nada sai da máquina

Não adicione chamadas a APIs de LLM em nuvem, telemetria ou upload de documentos. O argumento central do projeto é soberania de dados (LGPD).

### 5. Evidência antes de afirmar

Não declare "testes passando" sem rodar o comando e ler a saída.

## Tarefas comuns

### Adicionar um documento à base

1. Coloque o arquivo (`.txt`, `.md` ou `.pdf`, UTF-8) em `documentos/`. Numere as cláusulas (`3.1.`, `3.2.`) — o bot cita por número.
2. Reindexe (abaixo).
3. Acrescente em `tests/test_rag_pipeline.py` um caso de recuperação (`test_3_*`) e um de resposta (`test_4_*`) para o novo documento. Se o conjunto de arquivos mudar, atualize `EXPECTED_DOCS` em `tests/conftest.py`.

### Reindexar o ChromaDB

- Pela interface: botão **⚡ Indexar Documentos no ChromaDB** na barra lateral.
- Por linha de comando:
  ```bash
  docker exec docai-chatbot python -c "from rag_engine import RAGEngine; print(RAGEngine(ollama_url='http://ollama:11434').index_directory('./documentos'))"
  ```

A indexação apaga a coleção anterior e recria tudo (não é incremental). Se a base corromper, apague a pasta `chroma_db/` e reindexe.

### Trocar ou adicionar um modelo LLM

Adicione o nome à lista do `st.selectbox` em `app.py`, faça `ollama pull` e rode a suíte com `DOCAI_TEST_MODEL=<modelo>`. Trocar o modelo de **embeddings** exige reindexar.

## Convenções de código

- Python 3.11, PEP 8, type hints nas assinaturas públicas, docstrings e mensagens ao usuário em português do Brasil.
- Regra de negócio fica em `rag_engine.py`; `app.py` só apresenta.
- Parâmetros de chunking (700 / 120) e `k = 5` são citados no artigo e na documentação: mudar exige atualizar ambos.
- TDD: escreva o teste, veja falhar, implemente. Testes de integração levam `@pytest.mark.ollama`.
- Commits semânticos (`feat:`, `fix:`, `docs:`, `test:`, `chore:`).
- Arquivos de depuração vão em `scratch/` (ignorada pelo Git); o que for útil de verdade migra para `scripts/`.

## Limitações conhecidas (não as esconda)

- O corpus são 3 documentos fictícios da "Empresa Exemplo S.A."; a avaliação é funcional, não um benchmark estatístico.
- `llama3.2:3b` erra a pergunta composta de férias (não associa "vender 10 dias" ao abono da cláusula 5.1). O modelo padrão e o dos testes de geração é `llama3.1:8b`.
- A guarda pós-geração é heurística (expressões regulares): cobre recusas e negativas sem citação, não detecta um valor inventado que venha acompanhado de uma citação.
- O manual de TI **não** menciona "bloqueio remoto" do notebook; a resposta correta é B.O. em 24 h, avisar a TI (`security@empresa.com.br`) e abrir chamado no ServiceDesk.
