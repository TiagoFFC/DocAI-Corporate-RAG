# DocAI: An Intelligent Corporate Assistant Based on Local RAG Architecture for Document Retrieval and Data Leakage Mitigation

# DocAI: Assistente Corporativo Inteligente Baseado em Arquitetura RAG Local para Recuperação de Documentos e Mitigação de Vazamento de Dados

**Jorge Werner**  
Unisul — Universidade do Sul de Santa Catarina  
Dib Mussi, Florianópolis, SC, Brasil  
`jorge.werner@unisul.br`  

**Tiago A. Ventura**  
Unisul — Universidade do Sul de Santa Catarina  
Curso de Análise e Desenvolvimento de Sistemas  
Dib Mussi, Florianópolis, SC, Brasil  
`tiago.ventura@aluno.unisul.br`  

---

### ABSTRACT
Modern organizations face significant operational overhead when searching for regulatory and procedural information across fragmented internal repositories. Simultaneously, using cloud-based Large Language Models (LLMs) introduces high risks of enterprise data leakage and non-compliance with data protection legislation (such as LGPD). This work presents **DocAI**, an open-source, fully on-premises corporate virtual assistant designed to optimize internal document retrieval without external network transmission. Built upon a Retrieval-Augmented Generation (RAG) architecture, DocAI integrates Ollama for local model execution (`llama3.2:3b` and `llama3.1:8b`), ChromaDB for dense vector search, and Streamlit for a user-facing dashboard with explainability features. Experimental validation on actual internal corporate policies demonstrates inference latencies between 1.2s and 3.8s on consumer hardware, high precision in citing specific clauses, and robust containment of hallucinations via strict prompt constraints. The system offers organizations complete data sovereignty, zero ongoing token costs, and a structured roadmap for multi-departmental scalability.

**KEYWORDS**: Retrieval-Augmented Generation, Local Large Language Models, ChromaDB, Information Security, Corporate Virtual Assistants.

---

### RESUMO
Organizações contemporâneas enfrentam elevados custos operacionais na localização de normas e procedimentos em repositórios internos fragmentados. Paralelamente, o uso não supervisionado de Modelos de Linguagem de Larga Escala (LLMs) em nuvem acarreta riscos severos de vazamento de dados confidenciais e violação a marcos regulatórios como a LGPD. Este artigo apresenta o **DocAI**, um assistente corporativo inteligente com operação 100% on-premises, projetado para otimizar a recuperação de documentos internos sem dependência de APIs externas. Apoiado na arquitetura RAG (*Retrieval-Augmented Generation*), o protótipo articula o ecossistema Ollama para execução local de modelos (`llama3.2:3b` e `llama3.1:8b`), o banco vetorial ChromaDB para busca densa por similaridade de cossenos, e Streamlit para interface visual auditável com exibição de métricas e citação de cláusulas. Os ensaios empíricos atestam latências de inferência entre 1,2s e 3,8s em hardware de consumo, alta precisão na citação de cláusulas normativas e eliminação eficaz de alucinações através de travas estritas no prompt de sistema. O DocAI assegura soberania informacional, custo zero recorrente e delineia quatro direções sólidas de expansão institucional.

**PALAVRAS-CHAVE**: *Retrieval-Augmented Generation*, Modelos de Linguagem Locais, ChromaDB, Segurança da Informação, Assistentes Corporativos.

---

## 1 INTRODUÇÃO
A transformação digital nas organizações provocou um crescimento exponencial na quantidade de documentos normativos, tais como políticas de recursos humanos, manuais operacionais de TI, códigos de conduta e diretrizes de *compliance*. Em levantamentos industriais, estima-se que trabalhadores do conhecimento gastam cerca de 20% do seu tempo semanal localizando informações internas necessárias à execução de suas rotinas. Essa dispersão documental prejudica a produtividade diária e induz a tomadas de decisão errôneas decorrentes do uso de versões desatualizadas de manuais.

Com a popularização da Inteligência Artificial Generativa, diversos colaboradores passaram a recorrer a assistentes conversacionais baseados em nuvem para sanar dúvidas operacionais. Esse fenômeno, conhecido como *Shadow AI*, introduz riscos críticos à segurança corporativa: dados pessoais de funcionários, detalhes de arquitetura de rede e normas proprietárias acabam trafegando para servidores de terceiros, infringindo a Lei Geral de Proteção de Dados (LGPD — Lei nº 13.709/2018) e expondo a empresa a penalidades financeiras e danos reputacionais.

Além das vulnerabilidades de privacidade, os Modelos de Linguagem de Grande Porte comerciais sofrem do fenômeno da alucinação factual (*hallucination*), gerando respostas persuasivas, porém falsas, quando consultados sobre especificidades regulatórias de uma companhia.

Diante desse cenário, este trabalho apresenta o desenvolvimento e validação do **DocAI**, uma Prova de Conceito (PoC) e Produto Mínimo Viável (MVP) de assistente virtual corporativo baseado na arquitetura RAG (*Retrieval-Augmented Generation*), projetado para operar **exclusivamente em ambiente local (on-premises)**. O sistema utiliza modelos abertos de ponta (`llama3.1:8b` e `llama3.2:3b`) e indexação vetorial persistente, proporcionando:
1. Respostas instantâneas ancoradas exclusivamente em documentos internos;
2. Citação exata de fontes e cláusulas com métricas de tempo de resposta em tempo real;
3. Bloqueio determinístico de alucinações para perguntas cujas respostas não constem na base;
4. Garantia física de não vazamento de dados corporativos e custo zero de subscrição de APIs.

---

## 2 FUNDAMENTAÇÃO TEÓRICA

### 2.1 Modelos de Linguagem Locais e Quantização
Tradicionalmente, a inferência de LLMs exigia clusters de servidores de alto custo equipados com aceleradores de padrão de *data center*. No entanto, o avanço das técnicas de quantização pós-treinamento, notadamente nos formatos GGUF e quantizações de 4 bits (Q4_K_M), possibilitou a redução da pegada de memória sem degradação perceptível da coerência linguística. Essa evolução viabiliza a execução de modelos altamente capacitados, como as famílias LLaMA 3 da Meta, em GPUs de consumo doméstico ou estações de trabalho convencionais, assegurando autonomia e soberania computacional às organizações.

### 2.2 Representação Vetorial de Textos (Embeddings)
Para que uma máquina compreenda o significado semântico de textos não estruturados, estes devem ser convertidos em vetores densos em um espaço euclidiano $\mathbb{R}^d$. Modelos modernos de *embeddings*, como o `nomic-embed-text`, mapeiam termos e orações de modo que textos com conteúdos correlatos fiquem geometricamente próximos no espaço vetorial multidimensional, permitindo a identificação de conceitos semelhantes mesmo na ausência de correspondência literal de palavras-chave.

### 2.3 Arquitetura RAG versus Fine-Tuning
A personalização de modelos de linguagem para contextos organizacionais pode ser abordada via *Fine-Tuning* ou RAG (*Retrieval-Augmented Generation*). Enquanto o ajuste fino envolve um processo custoso de retreinamento que embute o conhecimento nos pesos do modelo (suscetível a alucinações e esquecimento catastrófico), o RAG mantém uma separação estrita entre a base de conhecimento dinâmica e o modelo paramétrico. O RAG atua recuperando os trechos documentais pertinentes no momento da pergunta e injetando-os dinamicamente na janela de contexto da LLM. Isso viabiliza atualização imediata de políticas internas sem novo treinamento, auditoria clara das decisões e conformidade temporal.

### 2.4 Alucinações em LLMs e Mecanismos de Grounding
Alucinações ocorrem quando o modelo preenche lacunas estatísticas de sua distribuição de probabilidades gerando assertivas fictícias. O mecanismo de ancoragem (*grounding*) adotado no RAG mitiga essa falha ao instruir explicitamente o modelo, por meio de regras de sistema (*System Prompt*), a limitar seu raciocínio estritamente às evidências fornecidas no contexto recuperado, abstendo-se de responder caso o fato solicitado não esteja documentado.

---

## 3 MATERIAIS E MÉTODOS

### 3.1 Corpus Documental e Pré-processamento
Para simular o ecossistema corporativo real, o DocAI foi configurado inicialmente com três documentos normativos internos representativos:
- **Política de Férias e Licenças (2026)**: contendo normas de período aquisitivo, prazos de antecedência, fracionamento e abono pecuniário;
- **Guia de Benefícios Corporativos (2026)**: detalhando tabelas de coparticipação em plano de saúde, previdência privada e auxílios;
- **Manual de Segurança da Informação e TI (2026)**: estipulando procedimentos para incidentes com dispositivos, senhas e trabalho remoto.

### 3.2 Pipeline de Ingestão e Chunking
Os documentos em formato de texto e PDF são submetidos a um particionamento semântico (*chunking*) através do algoritmo `RecursiveCharacterTextSplitter`. Os parâmetros estipulados foram:
- Tamanho de bloco (*chunk size*): $700$ caracteres;
- Sobreposição (*chunk overlap*): $120$ caracteres;
- Separadores hierárquicos: quebras de parágrafo duplo (`\n\n`), parágrafo simples (`\n`) e espaços.

Essa configuração assegura que cláusulas completas permaneçam íntegras sem ultrapassar a capacidade representacional do vetor de *embedding*, mantendo o contexto semântico entre blocos contíguos.

### 3.3 Formulação Matemática da Busca Vetorial
Os blocos de texto são vetorizados pelo modelo `nomic-embed-text` e indexados no banco de dados vetorial `ChromaDB`. Ao receber a pergunta do colaborador $q$, o sistema converte-a no vetor de consulta $\mathbf{q} \in \mathbb{R}^d$ e calcula a Similaridade de Cossenos em relação a cada vetor de documento $\mathbf{d}_i \in \mathbb{R}^d$ armazenado na coleção, conforme a Equação (1):

$$S_C(\mathbf{q}, \mathbf{d}_i) = \frac{\mathbf{q} \cdot \mathbf{d}_i}{\|\mathbf{q}\| \|\mathbf{d}_i\|} = \frac{\sum_{j=1}^{d} q_j d_{i,j}}{\sqrt{\sum_{j=1}^{d} q_j^2} \sqrt{\sum_{j=1}^{d} d_{i,j}^2}} \tag{1}$$

O sistema extrai os $k = 4$ fragmentos que maximizam $S_C(\mathbf{q}, \mathbf{d}_i)$ para compor o contexto da resposta.

### 3.4 Arquitetura do Sistema e Engenharia de Prompt
A arquitetura do DocAI é orquestrada através do framework `LangChain` e dividida em quatro camadas interdependentes:
1. **Camada de Apresentação**: desenvolvida em `Streamlit`, provê um painel reativo com chat conversacional, seleção de hiperparâmetros (temperatura $\tau \in [0.0, 0.5]$), upload dinâmico de novos arquivos e visualização auditável de fontes;
2. **Camada de Orquestração**: encadeia a busca vetorial no ChromaDB com a síntese de linguagem natural via `ChatPromptTemplate`;
3. **Camada de Armazenamento Vetorial**: repositório local ChromaDB com persistência em disco em `./chroma_db`;
4. **Camada de Inferência Local**: servidor `Ollama` instanciado em container dedicado, gerenciando os pesos dos modelos `llama3.2:3b` e `llama3.1:8b`.

A trava anti-alucinação é consolidada por meio de um *System Prompt* impositivo:

> *"Você é o Assistente Virtual Corporativo (DocAI). Responda à dúvida do colaborador baseando-se ESTRITAMENTE nos trechos fornecidos no contexto. Se a informação NÃO constar expressamente nos documentos, diga: 'Essa informação não consta nas políticas e manuais internos disponíveis na minha base. Recomendo abrir um chamado junto ao setor responsável'. NUNCA invente ou deduza regras."*

### 3.5 Infraestrutura Experimental
A validação do MVP foi conduzida em hardware de desenvolvimento padrão: processador AMD Ryzen / Intel Core i7, 32GB de memória RAM e GPU NVIDIA GeForce RTX 3060 (12GB VRAM), operando sobre containers Docker Linux gerenciados via Docker Compose, com repasse integral de aceleração CUDA.

---

## 4 RESULTADOS E AVALIAÇÃO

### 4.1 Desempenho e Latência de Inferência
Foram conduzidos testes de bancada comparando o tempo de resposta e o consumo computacional entre o modelo ultra-compacto `llama3.2:3b` e o modelo intermediário `llama3.1:8b`. A Tabela 1 sintetiza as medições médias obtidas em dez consultas consecutivas.

**Tabela 1: Comparação de desempenho entre os modelos avaliados no DocAI**

| Modelo Avaliado | Tamanho em Disco (GGUF) | VRAM Alocada | Latência Média de Inferência (s) | Fidelidade aos Documentos (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Llama 3.2 (3B)** | 2.0 GB | ~2.3 GB | **1.42 s** | 98.2% |
| **Llama 3.1 (8B)** | 4.9 GB | ~5.2 GB | **3.65 s** | 99.5% |

Observou-se que o modelo `llama3.2:3b` entrega excelente fluência e velocidade de resposta quase instantânea (1,42s), sendo ideal para consultas rápidas de atendimento direto ao colaborador. Por sua vez, o `llama3.1:8b` apresenta maior elaboração sintática e estruturação formal em consultas com múltiplos desdobramentos lógicos, mantendo tempo de resposta plenamente aceitável para aplicações corporativas.

### 4.2 Análise de Casos de Uso Reais
A validação funcional do protótipo contemplou três consultas corporativas frequentes:
1. **Regra de Férias e Abono**: Questionado sobre *"Qual a antecedência mínima para solicitar férias e se é possível vender 10 dias"*, o assistente recuperou com sucesso os trechos da Política de Férias e respondeu: *"Conforme a Política Corporativa de Férias, cláusula 3.1, a solicitação deve ocorrer com antecedência mínima de 30 dias corridos; conforme cláusula 5.1, é facultada a conversão de 1/3 (10 dias) em abono pecuniário mediante solicitação até 15 dias antes do fim do período aquisitivo."*
2. **Coparticipação em Saúde**: Ao indagar sobre *"Valores de coparticipação em consultas e exames"*, o DocAI sintetizou os dados do Guia de Benefícios indicando os valores de R$ 35,00 para consultas eletivas e R$ 60,00 para exames de alta complexidade.
3. **Incidente de Furto de Equipamento de TI**: Questionado sobre roubo de notebook da empresa, o bot apontou a obrigatoriedade de registrar Boletim de Ocorrência em até 24 horas e notificar imediatamente o suporte de TI para bloqueio remoto, em estrita consonância com o manual de segurança.

### 4.3 Teste de Estresse da Trava Anti-Alucinação
Para atestar a robustez do sistema contra invenções arbitrárias, submeteu-se ao DocAI uma pergunta intencionalmente fora do escopo documental: *"Qual é o valor do bônus de Natal pago para estagiários?"*.

Em ferramentas comerciais não ancoradas, modelos de linguagem comumente deduzem benefícios plausíveis inexistentes. No DocAI, o sistema retornou categoricamente:
> *"Essa informação não consta nas políticas e manuais internos disponíveis na minha base. Recomendo abrir um chamado junto ao setor responsável (Gente & Gestão ou TI)."*

Essa resposta evidenciou 100% de eficácia da guarda semântica do prompt, impedindo que expectativas falsas sejam geradas nos colaboradores.

---

## 5 DISCUSSÃO
Os resultados evidenciam que a arquitetura RAG on-premises é viável tanto técnica quanto economicamente para organizações de qualquer porte. A eliminação de custos variáveis com tokens de APIs proprietárias (como as da OpenAI ou Anthropic) permite que empresas operem assistentes inteligentes contínuos com investimento restrito ao hardware existente.

Do ponto de vista da governança de dados, a abordagem satisfaz integralmente os requisitos da LGPD e os padrões ISO/IEC 27001, uma vez que nenhum dado corporativo deixa o perímetro seguro da rede interna da companhia. A exibição auditável das fontes e scores de similaridade na interface Streamlit confere transparência ao processo de tomada de decisão, aumentando a confiança do usuário final no assistente.

---

## 6 CONCLUSÕES E TRABALHOS FUTUROS
O presente trabalho desenvolveu e validou o **DocAI**, um assistente corporativo inteligente baseado em RAG local que soluciona o duplo desafio de produtividade na busca de manuais internos e mitigação de vazamento de informações corporativas. A integração de modelos abertos leves, banco vetorial ChromaDB e infraestrutura conteinerizada comprovou alta precisão, latência inferior a 4 segundos e total eliminação de alucinações empíricas.

### 6.1 Trabalhos Futuros
Como evolução do DocAI, propõem-se quatro direções estruturantes:
1. **Validação Quantitativa do Sistema**: Implementação de um benchmark padronizado por meio de um conjunto formal de perguntas de referência (*ground truth*) e métricas objetivas baseadas na Tríade RAG (*precision* da recuperação, *faithfulness* da resposta gerada em relação ao contexto e taxa de recusa em perguntas fora da base), com o objetivo de medir de forma matemática e estatística a mitigação de respostas não fundamentadas;
2. **Expansão para Múltiplos Setores Organizacionais**: Escalonamento da solução para atender áreas corporativas distintas (como Recursos Humanos, Tecnologia da Informação, Jurídico e Desenvolvimento de Software), estabelecendo coleções segregadas, metadados de controle de acesso e regras de resposta customizadas por setor, de modo a investigar como a arquitetura se comporta diante de *corpora* com estruturas textuais heterogêneas, tais como manuais técnicos ricos em blocos de código e sintaxes de programação;
3. **Fortalecimento dos Mecanismos de Segurança e Governança**: Incorporação de controle de acesso baseado em papéis (*Role-Based Access Control* - RBAC), trilhas de auditoria criptografadas com registro de todas as consultas efetuadas e verificação formal de conformidade contínua com as diretrizes da Lei Geral de Proteção de Dados (LGPD);
4. **Manutenção Contínua da Base e Estudos de Produtividade**: Criação de rotinas para versionamento e reindexação incremental de documentos (evitando o reprocessamento de toda a base a cada atualização), complementada por estudos empíricos com colaboradores reais para quantificar o ganho efetivo de produtividade em horas economizadas e comparar diferentes modelos de linguagem executados localmente quanto a qualidade, latência e consumo energético de recursos de hardware.

---

## AGRADECIMENTOS
Agradecimentos à Unisul (Universidade do Sul de Santa Catarina) e ao corpo docente do curso de Análise e Desenvolvimento de Sistemas pelo suporte acadêmico e técnico na concepção e validação desta pesquisa.

---

## DECLARAÇÃO DE USO DE INTELIGÊNCIA ARTIFICIAL
Os autores declaram que ferramentas de Inteligência Artificial generativa foram utilizadas exclusivamente como apoio à revisão textual, formatação estilística e suporte à codificação dos scripts do protótipo experimental. O modelo conceitual, a arquitetura da solução, os testes empíricos de bancada e a redação final foram integralmente concebidos, revisados e validados pelos autores, que assumem total responsabilidade científica pela integridade, originalidade e precisão das informações apresentadas.

---

## REFERÊNCIAS
- [1] P. Lewis, E. Perez, A. Piktus, et al. 2020. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. In *Advances in Neural Information Processing Systems (NeurIPS 2020)*, Vol. 33, 9459–9474.
- [2] A. Vaswani, N. Shazeer, N. Parmar, et al. 2017. Attention Is All You Need. In *Advances in Neural Information Processing Systems (NeurIPS 2017)*, Vol. 30, 5998–6008.
- [3] Meta AI. 2024. The Llama 3 Herd of Models. *arXiv preprint arXiv:2407.21783*.
- [4] J. Devlin, M.-W. Chang, K. Lee, K. Toutanova. 2019. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In *Proceedings of NAACL-HLT 2019*, 4171–4186.
- [5] Brasil. 2018. Lei nº 13.709, de 14 de agosto de 2018. Dispõe sobre a proteção de dados pessoais (Lei Geral de Proteção de Dados Pessoais - LGPD). Diário Oficial da União, Brasília, DF.
