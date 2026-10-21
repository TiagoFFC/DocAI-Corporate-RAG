# DocAI: Assistente Virtual Corporativo com IA Local (RAG)

> **Artigo Científico & Prova de Conceito (MVP)**  
> **Autores**: Tiago A. Ventura e Jorge Werner  
> **Instituição**: Unisul — Universidade do Sul de Santa Catarina (Curso de Análise e Desenvolvimento de Sistemas)  
> 📄 **Artigo Completo em Word (Template Unisul/SBC)**: [Artigo/Artigo - Chatbot.docx](file:///d:/Projetos/Chatbot/Artigo/Artigo%20-%20Chatbot.docx)  
> 📝 **Versão em Markdown para Leitura Rápida**: [Artigo/ARTIGO_DOCAI.md](file:///d:/Projetos/Chatbot/Artigo/ARTIGO_DOCAI.md)  

---

## 📌 Visão Geral do Projeto

O **DocAI** é um assistente virtual corporativo inteligente baseado na arquitetura **RAG (*Retrieval-Augmented Generation*)** com operação **100% local (on-premises)**. O sistema resolve dois desafios fundamentais enfrentados pelas organizações modernas:

1. **Eficiência e Produtividade**: Elimina o tempo gasto pelos colaboradores navegando manualmente em dezenas de manuais de RH, TI e políticas internas extensas.
2. **Privacidade Absoluta e Soberania de Dados (Zero Vazamento)**: Ao executar modelos de linguagem abertos localmente (`llama3.2:3b` e `llama3.1:8b`) integrados ao banco vetorial `ChromaDB`, nenhum dado confidencial, segredo de negócio ou informação de colaborador trafega para servidores de terceiros ou nuvens públicas, garantindo conformidade com a **LGPD (Lei nº 13.709/2018)** e governança de TI.
3. **Mitigação Rigorosa de Alucinações**: O modelo possui diretrizes estritas no prompt de sistema (*grounding*) e se recusa categoricamente a inventar respostas sobre informações que não estejam documentadas na base.

---

## 🚀 Arquitetura Técnica

```
  [ Colaborador / Interface Web Streamlit ] (Porta 8501)
                     │
                     ▼
        [ Orquestrador LangChain ]
         ├── RecursiveCharacterTextSplitter (chunk: 700, overlap: 120)
         ├── Similaridade de Cossenos S_C(q, d)
         └── System Prompt com Trava Anti-Alucinação
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
 [ ChromaDB Local ]      [ Ollama Container ] (Porta 11434)
 (./chroma_db)           ├── nomic-embed-text (Embeddings)
                         └── llama3.2:3b / llama3.1:8b (LLM)
```

- **Interface**: `Streamlit` (Dashboard corporativo com chat reativo, métricas de latência e citação auditável de fontes).
- **Orquestração RAG**: `LangChain` com divisores semânticos e cadeias de prompt restrito.
- **Banco Vetorial**: `ChromaDB` persistido localmente em `./chroma_db`.
- **Embeddings**: `nomic-embed-text` (Ollama).
- **LLM Local**: `llama3.2:3b` (ultrarrápido, ~1,4s) e `llama3.1:8b` (alta densidade sintática, ~3,6s).
- **Containerização**: `Docker` e `Docker Compose`.

---

## 🛠️ Como Executar a Aplicação

Como o projeto está totalmente orquestrado em **Docker Compose**, não é necessário configurar o Python ou o Ollama no sistema operacional hospedeiro.

### 1. Iniciar os Containers (Ollama + DocAI)
No terminal PowerShell:
```powershell
docker compose up -d --build
```

### 2. Acessar a Interface
Abra no navegador:
👉 **[http://localhost:8501](http://localhost:8501)**

---

## 🎬 Roteiro de Demonstração Prática (Para a Banca e Apresentação)

O DocAI foi projetado para permitir uma validação empírica clara e imediata perante o professor e avaliadores:

### Passo 1: Apresentação da Barra Lateral
- Mostre o status de conexão ativa com o Ollama (`🔌 Testar`).
- Destaque os 3 documentos regulatórios disponíveis em [documentos/](file:///d:/Projetos/Chatbot/documentos):
  - `politica_de_ferias.txt` (regras CLT, antecedência e abono);
  - `guia_de_beneficios.txt` (coparticipação e planos de saúde);
  - `manual_ti_e_seguranca.txt` (procedimentos em caso de furto e senhas).
- Demonstre o botão **`⚡ Indexar Documentos no ChromaDB`**, explicando a vetorização semântica em chunks de 700 caracteres.

### Passo 2: Execução das Consultas Empíricas (Botões Rápidos)
1. **🏖️ 1. Regra de Férias e Abono**:
   - Pergunta: *"Qual a antecedência mínima para solicitar férias e posso vender 10 dias?"*
   - Observe a resposta instantânea citando a **cláusula 3.1** (antecedência de 30 dias) e **cláusula 5.1** (abono pecuniário de 1/3).
   - Abra o expander *"🔍 Ver trechos e fontes recuperadas da base"* para auditar o score de similaridade e o texto exato.
2. **🏥 2. Coparticipação Saúde**:
   - Pergunta: *"Como funciona a coparticipação em consultas e exames no plano de saúde?"*
   - O assistente recupera os valores exatos de consultas eletivas (R$ 35,00) e exames de alta complexidade (R$ 60,00).
3. **💻 3. Furto de Notebook / TI**:
   - Pergunta: *"O que devo fazer caso meu notebook corporativo seja roubado?"*
   - O bot orienta o registro de B.O. em 24h e acionamento imediato da TI para bloqueio remoto.
4. **⚠️ 4. Pegadinha (Trava de Alucinação - O Grande Destaque)**:
   - Pergunta: *"Qual é o valor do bônus de Natal pago para estagiários?"*
   - **Resultado do Bot**: *"Essa informação não consta nas políticas e manuais internos disponíveis na minha base. Recomendo abrir um chamado junto ao setor responsável (Gente & Gestão ou TI)."*
   - **Ponto Alto da Demonstração**: Destaque ao professor que, diferentemente de IAs comuns que inventam valores plausíveis (alucinação), o DocAI respeita guardrails rígidos de governança.

---

## 🔮 Trabalhos Futuros (Roteiro de Evolução Científica)

Como evolução e expansão do projeto DocAI, propõem-se **quatro direções estratégicas**:

1. **Validação Quantitativa do Sistema**: Implementação de um benchmark padronizado por meio de um conjunto de perguntas de referência (*ground truth*) e métricas objetivas baseadas na Tríade RAG (*precision* da recuperação, *faithfulness* da resposta e taxa de recusa em perguntas fora da base), com o objetivo de mensurar formalmente a mitigação de respostas não fundamentadas.
2. **Expansão para Múltiplos Setores Organizacionais**: Escalonamento da solução para atender áreas corporativas distintas (RH, TI, Jurídico e Desenvolvimento de Software), com coleções segregadas no banco vetorial, metadados de controle de acesso e regras de resposta específicas por área, investigando o comportamento da arquitetura em corpora com blocos de código e sintaxes técnicas.
3. **Fortalecimento dos Mecanismos de Segurança e Governança**: Incorporação de controle de acesso baseado em papéis (*Role-Based Access Control* - RBAC), trilha de auditoria criptografada e verificação formal de conformidade contínua com a LGPD.
4. **Manutenção Contínua da Base e Estudos de Produtividade**: Análise de rotinas de versionamento e reindexação incremental de documentos (evitando reprocessamento total da base), complementada por estudos empíricos com colaboradores para quantificar o ganho de produtividade em horas economizadas e comparar diferentes modelos locais quanto a latência e consumo de hardware.

---

## 📂 Estrutura de Arquivos do Repositório

```text
d:/Projetos/Chatbot/
├── Artigo/
│   ├── Artigo - Chatbot.docx        # Artigo acadêmico preenchido no template oficial (Unisul/SBC)
│   ├── ARTIGO_DOCAI.md              # Artigo acadêmico completo em formato Markdown
│   └── Artigo - Chatbot_original.docx # Backup do template original
├── app.py                           # Interface Web Streamlit com chat, métricas e auditoria
├── rag_engine.py                    # Motor RAG com LangChain, ChromaDB e Ollama
├── Dockerfile                       # Configuração da imagem do assistente
├── docker-compose.yml               # Orquestração conjunta dos containers (Streamlit + Ollama)
├── requirements.txt                 # Dependências Python do projeto
├── README.md                        # Documentação executiva e roteiro de apresentação
├── chroma_db/                       # Banco de vetores local persistido
└── documentos/                      # Repositório de políticas e manuais internos
    ├── politica_de_ferias.txt
    ├── guia_de_beneficios.txt
    └── manual_ti_e_seguranca.txt
```
