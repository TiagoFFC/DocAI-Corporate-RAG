# Especificação de Design: Artigo Científico e MVP DocAI

- **Projeto**: DocAI - Assistente Corporativo Inteligente com RAG Local
- **Autores**: 
  - Tiago A. Ventura (Análise e Desenvolvimento de Sistemas, Unisul)
  - Jorge Werner (Docente / Orientador, Unisul)
- **Data**: 20/10/2026
- **Status**: Proposta para aprovação

---

## 1. Contexto e Motivação (O Porquê)
Em organizações modernas, o volume de documentos normativos (manuais de compliance, políticas de RH, procedimentos de TI e segurança da informação) gera dois grandes gargalos operacionais:
1. **Ineficiência produtiva**: Colaboradores despendem tempo excessivo buscando cláusulas específicas em arquivos extensos.
2. **Risco de vazamento de dados corporativos (Shadow AI)**: A utilização não supervisionada de LLMs comerciais em nuvem expõe segredos industriais, dados pessoais de funcionários e regras internas a servidores terceiros.

A solução proposta neste trabalho é o **DocAI**, um assistente baseado na arquitetura **RAG (Retrieval-Augmented Generation)** com operação **100% on-premises/local**, unindo busca semântica em banco vetorial à síntese contextual por modelos de linguagem abertos, garantindo **privacidade absoluta, custo zero recorrente de API e combate efetivo a alucinações**.

---

## 2. Objetivos e Critérios de Sucesso

### 2.1 Objetivos do Artigo Acadêmico
- Preencher integralmente o template oficial (`Artigo - Chatbot.docx`) no padrão de conferência da SBC / Unisul.
- Formalizar a arquitetura técnica do RAG local de ponta a ponta.
- Apresentar fundamentação teórica rigorosa sobre Embeddings, Similaridade de Cossenos, Chunking e Janela de Contexto.
- Demonstrar resultados empíricos reais medidos no MVP (latência de inferência, número de fontes citadas e taxa de contenção anti-alucinação).
- Estruturar a evolução científica nos 4 pilares de Trabalhos Futuros definidos.

### 2.2 Objetivos do MVP (Aplicação Funcional)
- Manter o ecossistema Docker orquestrado (`docai-chatbot` Streamlit + `ollama` local).
- Prover interface limpa, moderna e reativa com:
  - Painel de controle na barra lateral (conexão com Ollama, escolha de modelo, temperatura, upload de novos documentos e reindexação).
  - Chat conversacional com histórico.
  - Métricas em tempo real para cada resposta (tempo de resposta em segundos e contagem de fontes citadas).
  - Expansor auditável contendo os trechos exatos e scores de similaridade recuperados da base vetorial.
  - Perguntas prontas de demonstração cobrindo casos reais de RH, TI e teste de estresse da trava anti-alucinação.

### 2.3 Não-Objetivos (Fora do Escopo desta Entrega)
- Implementação de autenticação complexa multi-usuário com banco relacional externo (será abordada em Trabalhos Futuros).
- Conexão com bases SQL corporativas legadas ou APIs de terceiros.
- Migração para nuvem paga (o foco do artigo é justamente soberania e privacidade local).

---

## 3. Arquitetura e Decisões Técnicas

| Componente | Tecnologia Adotada | Justificativa |
| :--- | :--- | :--- |
| **Interface de Usuário** | Streamlit 1.38+ | Desenvolvimento ágil em Python, layout corporativo e reatividade nativa |
| **Orquestração RAG** | LangChain 0.2+ | Pipelines padronizados de loaders, splitters, chains e prompt templates |
| **Banco Vetorial** | ChromaDB 0.5+ | Persistência local (`./chroma_db`), baixo consumo de memória, sem necessidade de cluster |
| **Modelo de Embeddings** | `nomic-embed-text` | Especializado em textos longos (contexto de 2048 tokens), alta precisão semântica |
| **Modelo de Linguagem (LLM)** | `llama3.1:8b` / `llama3.2:3b` | Alta capacidade de raciocínio em português, executável em hardware local (GPU/CPU) |
| **Servidor de Inferência** | Ollama | Gestão simplificada de pesos GGUF, aceleração por GPU NVIDIA e API REST local |
| **Isolamento e Entrega** | Docker + Docker Compose | Reprodutibilidade total em qualquer ambiente sem necessidade de dependências do host |

---

## 4. Estrutura do Artigo Científico

O artigo será estruturado rigorosamente conforme o template em `Artigo - Chatbot.docx`:

1. **Identificação e Título**:
   - Título em Português: *DocAI: Assistente Corporativo Inteligente Baseado em Arquitetura RAG Local para Recuperação de Documentos e Mitigação de Vazamento de Dados*
   - Title in English: *DocAI: An Intelligent Corporate Assistant Based on Local RAG Architecture for Document Retrieval and Data Leakage Mitigation*
   - Autores: Jorge Werner (Unisul) e Tiago A. Ventura (Unisul).
2. **Resumo e Abstract (até 250 palavras)**:
   - Apresentação do problema corporativo, metodologia RAG local, resultados de latência/precisão e conclusões.
   - Palavras-chave: *RAG, Modelos de Linguagem Locais, ChromaDB, Segurança da Informação, Assistentes Corporativos.*
3. **1. Introdução**:
   - Cenário atual da IA generativa nas empresas;
   - O dilema entre produtividade vs. privacidade (Shadow AI e LGPD);
   - Proposta e justificativa do DocAI.
4. **2. Fundamentação Teórica**:
   - 2.1 Modelos de Linguagem Locais e Quantização (GGUF);
   - 2.2 Representação Vetorial de Textos (Embeddings e Espaço Vetorial);
   - 2.3 Recuperação Aumentada por Geração (RAG vs. Fine-tuning);
   - 2.4 Alucinações em LLMs e Mecanismos de Grounding.
5. **3. Materiais e Métodos**:
   - 3.1 Corpus Documental (políticas de férias, benefícios e segurança da informação);
   - 3.2 Pipeline de Ingestão e Chunking (divisão recursiva com overlap);
   - 3.3 Formulação Matemática da Busca Vetorial (Similaridade de Cossenos entre query e chunks);
   - 3.4 Engenharia de Prompt e Restrição de Contexto (System Prompt estrito);
   - 3.5 Ambiente Experimental e Infraestrutura de Execução.
6. **4. Resultados e Avaliação**:
   - Tabela comparativa de latência e consumo entre `llama3.2:3b` e `llama3.1:8b`;
   - Análise de casos de uso práticos (perguntas de RH e TI);
   - Validação da contenção de alucinações (teste com perguntas fora do corpus).
7. **5. Discussão**:
   - Análise de viabilidade para pequenas e médias empresas;
   - Trade-offs de acurácia vs. latência em hardware local;
   - Implicações na governança e privacidade de dados.
8. **6. Conclusões e Trabalhos Futuros**:
   - Síntese das contribuições do protótipo;
   - As 4 direções de evolução: (1) Validação quantitativa (RAG Triad), (2) Expansão multi-departamental com RBAC, (3) Governança e auditoria LGPD, e (4) Reindexação incremental e estudos de usabilidade com usuários.
9. **Agradecimentos e Declaração de Uso de IA**.
10. **Referências Bibliográficas**:
    - Normas da SBC / ABNT com artigos canônicos (Lewis et al. sobre RAG, Vaswani et al. sobre Transformers, etc.).

---

## 5. Plano de Entrega e Verificação
1. **Geração do Artigo Formatado**:
   - Criação de script automatizado em Python para preencher os nós do `Artigo - Chatbot.docx` mantendo os estilos tipográficos, margens e formatação oficial da Unisul/SBC.
   - Geração de versão em Markdown para leitura e revisão rápida no repositório.
2. **Validação do MVP**:
   - Verificação do container Docker e teste de inferência em tempo real com as 4 perguntas do roteiro de apresentação.
   - Atualização do README com as instruções completas de apresentação acadêmica para o professor.
