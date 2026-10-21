# Implementação do Artigo Científico e Alinhamento do MVP DocAI

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redigir o artigo científico completo com rigor acadêmico (SBC/Unisul), preencher o arquivo oficial `Artigo - Chatbot.docx` preservando estilos e formatação, e alinhar a documentação do MVP DocAI para a apresentação ao Prof. Jorge Werner.

**Architecture:** O conteúdo do artigo será redigido primeiramente em Markdown (`Artigo/ARTIGO_DOCAI.md`) servindo como documento mestre auditável. Em seguida, um script automatizado injetará cada seção, tabela e referências diretamente na árvore XML do template `Artigo - Chatbot.docx`, preservando as fontes, estilos e espaçamentos do formato SBC/Unisul. Por fim, o `README.md` será atualizado com o roteiro de apresentação sincronizado com os dados do artigo.

**Tech Stack:** Python 3.11, Docker, OpenXML / .docx, Streamlit, LangChain, ChromaDB, Ollama.

**Spec:** [docs/superpowers/specs/2026-10-20-artigo-mvp-docai-design.md](file:///d:/Projetos/Chatbot/docs/superpowers/specs/2026-10-20-artigo-mvp-docai-design.md)

## Global Constraints
- Autoria: Jorge Werner (Unisul) e Tiago A. Ventura (Análise e Desenvolvimento de Sistemas, Unisul).
- Template: Preenchimento do arquivo `Artigo - Chatbot.docx` respeitando estritamente o layout de conferência SBC/Unisul sem desconfigurar margens ou fontes.
- Seção de Trabalhos Futuros: Conter obrigatoriamente os 4 pilares estratégicos (Validação Quantitativa RAG, Expansão Multi-departamental, Governança/LGPD, e Ciclo de Vida/Usabilidade).
- MVP: O sistema já está funcional via Docker na porta 8501, e os dados experimentais citados no artigo devem corresponder ao comportamento real do protótipo.

## Review Focus
- Ausência de placeholders não substituídos (`[...]`) no documento final `.docx`.
- Fidelidade matemática e técnica na descrição do RAG (Similaridade de Cossenos, formulação vetorial, embeddings).
- Consistência de métricas de latência e consumo de hardware entre o texto do artigo e os testes do MVP.
- Formatação correta das referências bibliográficas segundo as normas da SBC/ABNT.
- Preservação da integridade estrutural do arquivo `.docx` ao abrir no Microsoft Word / LibreOffice.

---

### Task 1: Redação do Artigo Científico Mestre em Markdown

**Files:**
- Create: `Artigo/ARTIGO_DOCAI.md`
- Reference: `Artigo/conteudo_artigo.txt`
- Reference: `docs/superpowers/specs/2026-10-20-artigo-mvp-docai-design.md`

**Interfaces:**
- Produces: Texto completo e revisado de todas as seções acadêmicas necessárias para a injeção no `.docx`.

- [ ] **Step 1: Redigir o cabeçalho, Resumo, Abstract, Introdução e Fundamentação Teórica**
  - Autores: Jorge Werner e Tiago A. Ventura.
  - Título em Português e Inglês.
  - Resumo e Abstract (até 250 palavras) com palavras-chave / keywords.
  - Seção 1 (Introdução): contextualização corporativa, perda de produtividade na busca de informações, Shadow AI e risco de vazamento de dados, conformidade LGPD.
  - Seção 2 (Fundamentação Teórica): 2.1 Modelos de Linguagem e Quantização GGUF; 2.2 Vetorização e Espaço Semântico; 2.3 RAG vs Fine-tuning; 2.4 Mecanismos de Grounding e Trava Anti-Alucinação.

- [ ] **Step 2: Redigir Metodologia, Resultados Empíricos, Discussão, Conclusões e Trabalhos Futuros**
  - Seção 3 (Materiais e Métodos): Corpus de políticas internas, pipeline de chunking recursivo com overlap, formulação matemática da Similaridade de Cossenos ($S_C(q, d) = \frac{\mathbf{q} \cdot \mathbf{d}}{\|\mathbf{q}\| \|\mathbf{d}\|}$), arquitetura do DocAI (Ollama, ChromaDB, LangChain, Streamlit) e engenharia de prompt restrito.
  - Seção 4 (Resultados): Tabela comparativa de latência (Llama 3.2 3B vs Llama 3.1 8B), análise dos testes práticos de RH/TI e evidência empírica da recusa categórica em perguntas fora do corpus (teste de estresse anti-alucinação).
  - Seção 5 (Discussão): Viabilidade de implementação on-premises em PMEs e trade-offs de hardware.
  - Seção 6 (Conclusões e Trabalhos Futuros): Síntese das contribuições e detalhamento dos 4 pilares de evolução.
  - Seções Finais: Agradecimentos, Declaração de Uso de IA e Referências Bibliográficas (Lewis et al., Vaswani et al., etc.).

- [ ] **Step 3: Validar a completude do texto**
  - Verificar se todas as seções mapeiam 100% para os placeholders do arquivo `conteudo_artigo.txt`.

---

### Task 2: Implementação do Script de Injeção e Geração do Documento Word (.docx)

**Files:**
- Create: `scratch/fill_docx.py`
- Modify: `Artigo/Artigo - Chatbot.docx` (com backup criado previamente em `Artigo/Artigo - Chatbot_original.docx`)

**Interfaces:**
- Consumes: `Artigo/ARTIGO_DOCAI.md` e `Artigo/Artigo - Chatbot.docx`
- Produces: `Artigo/Artigo - Chatbot.docx` preenchido e formatado.

- [ ] **Step 1: Criar backup do documento original**
  - Copiar `Artigo/Artigo - Chatbot.docx` para `Artigo/Artigo - Chatbot_original.docx`.

- [ ] **Step 2: Desenvolver o script de injeção em Python**
  - Utilizar manipulação do pacote OpenXML (`word/document.xml`) ou biblioteca `python-docx` para substituir cada nó de placeholder preservando a formatação (fontes, tamanhos, itálicos, estilos de parágrafo e estrutura de colunas).
  - Inserir a tabela formatada de resultados comparativos (Tabela 1).
  - Inserir a formulação matemática da Equação (1) adaptada para similaridade de cossenos.

- [ ] **Step 3: Executar a injeção no container Docker**
  - Executar o script no container `docai-chatbot` para garantir que as dependências necessárias estejam disponíveis.

---

### Task 3: Verificação e Auditoria do Artigo Gerado

**Files:**
- Test: `scratch/verify_docx.ps1`
- Target: `Artigo/Artigo - Chatbot.docx`

**Interfaces:**
- Consumes: `Artigo/Artigo - Chatbot.docx`
- Produces: Relatório de integridade do documento.

- [ ] **Step 1: Desenvolver script de verificação de placeholders remanescentes**
  - Extrair o texto do novo `.docx` e buscar por termos não preenchidos (`[Title`, `[Nome`, `[Texto`, `[Keyword`, etc.).
  - Confirmar presença de nomes dos autores, título, seções 1 a 6 e os 4 pilares de trabalhos futuros.

- [ ] **Step 2: Executar a verificação**
  - Validar que a contagem de placeholders residuais é zero e que a estrutura está intacta.

---

### Task 4: Atualização da Documentação e Roteiro de Demonstração (README.md)

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: Dados finais do artigo e status do MVP
- Produces: `README.md` atualizado com o resumo acadêmico e guia da demonstração para a banca/professor.

- [ ] **Step 1: Atualizar o README.md**
  - Adicionar dados do Artigo (Título, Autoria, Resumo Acadêmico).
  - Detalhar a correspondência entre os botões de demonstração da UI e os experimentos descritos na Seção 4 do artigo.
  - Documentar os 4 pilares de Trabalhos Futuros.

- [ ] **Step 2: Validar o MVP em execução**
  - Testar novamente a interface no navegador e garantir que as perguntas do roteiro respondem com exatidão e citam as fontes esperadas.
