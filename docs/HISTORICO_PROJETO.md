# DocAI — Histórico Executivo do Projeto

**Projeto**: DocAI: Assistente Corporativo Inteligente com RAG Local
**Autores**: Tiago A. Ventura e Jorge Werner (orientador)
**Instituição**: Unisul — Universidade do Sul de Santa Catarina, curso de Análise e Desenvolvimento de Sistemas
**Versão**: 1.0.0 (entrega acadêmica e MVP)

---

## 1. Problema de negócio

| Dor | Consequência |
| --- | --- |
| **Tempo perdido em manuais legados** | Políticas de RH, benefícios e TI ficam espalhadas em documentos longos. O colaborador gasta tempo procurando uma cláusula e, muitas vezes, consulta uma versão desatualizada. |
| **Shadow AI** | Para ganhar tempo, colaboradores colam trechos de documentos internos em assistentes de IA em nuvem, sem aprovação da empresa. |
| **Vazamento de dados e LGPD** | Dados pessoais de funcionários e normas proprietárias trafegam para servidores de terceiros, expondo a empresa à Lei nº 13.709/2018 e a dano reputacional. |
| **Alucinação** | Um modelo genérico, perguntado sobre uma regra interna que desconhece, inventa uma resposta plausível. Em RH isso cria expectativa falsa (por exemplo, um bônus que não existe). |

O DocAI responde às dúvidas usando apenas os documentos internos, cita a fonte e a cláusula, recusa o que não está documentado e roda inteiramente na infraestrutura da empresa.

---

## 2. Decisões de arquitetura

### 2.1 Por que RAG local, e não fine-tuning ou API em nuvem

| Critério | RAG local (escolhido) | Fine-tuning | API em nuvem |
| --- | --- | --- | --- |
| Dados saem da rede? | Não | Não (se treinado localmente) | Sim |
| Atualizar uma política | Trocar o arquivo e reindexar (segundos) | Retreinar o modelo | Reindexar, mas os trechos seguem para o provedor |
| Citação auditável da fonte | Sim, trecho e score visíveis | Não (conhecimento embutido nos pesos) | Sim |
| Risco de alucinação | Mitigado por contexto + trava | Alto (sem evidência para conferir) | Mitigado por contexto |
| Custo recorrente | Zero por consulta | Custo de GPU a cada retreino | Cobrança por token |
| Qualidade bruta do modelo | Menor (modelos de 3 a 8 bilhões de parâmetros) | Depende da base | Maior |
| Hardware | GPU com ~12 GB de VRAM | GPU de treino | Nenhum |

O levantamento inicial ([Apresentação/levantamento_projeto_chatbot.md](../Apresentação/levantamento_projeto_chatbot.md)) comparou as três rotas. A nuvem venceria em qualidade de resposta e simplicidade; o RAG local foi escolhido porque privacidade e conformidade eram o requisito central do trabalho, e porque o corpus é pequeno e muda com frequência.

### 2.2 Stack

| Camada | Tecnologia | Motivo |
| --- | --- | --- |
| Interface | Streamlit | Chat, métricas de latência e painel de fontes com pouco código |
| Orquestração | LangChain | Loaders, splitter e cadeia de prompt prontos |
| Base vetorial | ChromaDB (persistência em `./chroma_db`) | Embutido, sem servidor extra |
| Embeddings | `nomic-embed-text` via Ollama | Local, bom desempenho em textos longos |
| LLM | `llama3.1:8b` (padrão) e `llama3.2:3b` via Ollama | Cabem em GPU de consumo |
| Empacotamento | Docker Compose (serviços `ollama` e `docai-chatbot`) | Sobe com um comando |

### 2.3 Parâmetros do pipeline

- Chunking: `RecursiveCharacterTextSplitter`, 700 caracteres, sobreposição de 120. O corpus atual gera 21 trechos.
- Recuperação: busca 10 candidatos, remove trechos redundantes e mantém os 5 mais similares.
- Geração: temperatura padrão 0,1.

---

## 3. Modelos e benchmarks

### 3.1 Números publicados no artigo (Tabela 1)

Medidos pelos autores em estação com NVIDIA RTX 3060 (12 GB), média de dez execuções por modelo.

| Modelo | VRAM alocada | Latência média |
| --- | --- | --- |
| Llama 3.2 (3B) | ~2,3 GB | 1,42 s |
| Llama 3.1 (8B) | ~5,2 GB | 3,65 s |

### 3.2 Medição de conferência da versão final

Executada com `scripts/benchmark_latency.py 3` na mesma classe de hardware (RTX 3060 12 GB, Docker), com as 4 perguntas do roteiro, 3 rodadas cada, após aquecimento. O tempo é ponta a ponta: embedding da pergunta, busca e geração.

| Modelo | Memória carregada (`ollama ps`) | Mediana | Mínimo | Máximo |
| --- | --- | --- | --- | --- |
| `llama3.2:3b` | 2,6 GB | 2,20 s | 0,61 s | 3,23 s |
| `llama3.1:8b` | 5,3 GB | 4,35 s | 1,13 s | 6,51 s |

Leitura dos dois conjuntos:

- A relação entre os modelos se mantém: o 3B é cerca de duas vezes mais rápido que o 8B.
- Os valores absolutos desta conferência ficaram acima dos da Tabela 1 (2,20 s contra 1,42 s; 4,35 s contra 3,65 s). As metodologias diferem (mediana de 12 chamadas ponta a ponta aqui, média de dez execuções no artigo) e a latência depende do tamanho da resposta: a recusa sai em cerca de 1 s, a resposta de coparticipação leva mais de 5 s no 8B. **Os números do artigo não foram alterados**; cabe aos autores decidir se querem atualizá-los.
- A primeira chamada após subir o container é muito mais lenta, pois carrega os pesos na GPU (foram observados 250 s no 8B a frio). Faça uma pergunta de aquecimento antes de uma demonstração.

### 3.3 Qualidade das respostas (temperatura 0)

| Pergunta | `llama3.1:8b` | `llama3.2:3b` |
| --- | --- | --- |
| Férias e abono | Correta, cita 3.1 e 5.1 | **Incorreta**: acerta os 30 dias, mas não associa "vender 10 dias" ao abono da cláusula 5.1 |
| Coparticipação | Correta (R$ 35,00 e R$ 60,00) | Correta |
| Furto de notebook | Correta (B.O. em 24 h, avisar a TI) | Correta |
| Bônus de Natal (fora da base) | Recusa padrão | Recusa padrão (com a guarda pós-geração) |

Por isso `llama3.1:8b` é o modelo padrão do app e o usado nos testes de geração.

---

## 4. Mecanismos de segurança

1. **Isolamento de rede**: LLM, embeddings e base vetorial rodam em containers locais. Nenhuma chamada a API externa; não há chave de API no projeto.
2. **Prompt grounding**: o `SYSTEM_PROMPT` obriga o modelo a responder apenas com os trechos recuperados, citar documento e cláusula, e usar a recusa padrão para o que não constar.
3. **Similaridade de cossenos**: a coleção usa métrica de cosseno e o score mostrado ao usuário é S_C(q, d), a Equação (1) do artigo. Um teste recalcula o cosseno a partir dos embeddings e compara com o score exibido.
4. **Guarda determinística pós-geração** (`apply_refusal_guard`): se a resposta sinaliza que a informação não consta e não cita nenhum documento nem cláusula, o texto é substituído pela recusa padrão exata. Isso elimina preâmbulos e variações de redação entre modelos.
5. **Auditabilidade**: a interface exibe os trechos recuperados, o arquivo de origem e o score de cada um.
6. **Temperatura baixa** (0,1 por padrão; limite de 0,5 na interface).

Limite conhecido: a guarda é uma heurística por expressões regulares. Ela garante a forma da recusa, mas não detecta um valor inventado que venha acompanhado de uma citação. Medir isso formalmente é a primeira direção de trabalhos futuros do artigo.

---

## 5. Registro de mudanças (changelog)

### v1.0.0 — versão final (novembro de 2026)

- **Suíte de testes** `tests/test_rag_pipeline.py`: 30 testes em 5 grupos (conectividade, ingestão, recuperação, geração, trava anti-alucinação); 10 rodam sem Ollama.
- **Métrica corrigida para cosseno**: a base usava distância L2 (padrão do ChromaDB) e exibia a distância como "score de similaridade". Agora a coleção é criada com cosseno e o score é a similaridade real. Bases antigas são detectadas e exigem reindexação.
- **Trava anti-alucinação reforçada**: constante `REFUSAL_MESSAGE`, instrução explícita de responder somente com a recusa, e guarda determinística pós-geração.
- **Motor refatorado**: `load_and_split()` e `retrieve()` separados de `index_directory()` e `ask()`; parâmetro `seed` para respostas reprodutíveis.
- **Artigo**: `ARTIGO_DOCAI.md` passou a ser gerado do Word por `scripts/docx_to_md.py`. A versão anterior do Markdown era um texto mais longo que divergia do documento oficial (seções extras, tabela com colunas diferentes); ela permanece no histórico do Git (commit `4acb4f0`). `scripts/verify_docx.py` agora verifica a paridade e retorna código de saída 1 em caso de pendência.
- **Artigo, correções de template**: o cabeçalho ainda trazia o placeholder "[Sobrenome do Primeiro Autor] et al." (agora "Werner e Ventura") e a Figura 1 era a imagem de exemplo do template; foi substituída por uma captura real da interface (`scripts/capture_interface.py`). A auditoria passou a verificar cabeçalhos, rodapés e a figura. PDF exportado em `Artigo/Artigo - Chatbot.pdf`.
- **Governança**: `.gitignore`, `.env.example`, `.dockerignore`, `requirements-dev.txt`, CI no GitHub Actions; `chroma_db/` saiu do versionamento por ser artefato gerado.
- **Documentação**: README reescrito, `AGENTS.md` do projeto, este histórico.

### Outubro de 2026 — protótipo e artigo

| Commit | Mudança |
| --- | --- |
| `c1e441e` | Commit inicial: app Streamlit, motor RAG, Docker Compose, três documentos de exemplo |
| `4acb4f0` | Primeira versão completa do artigo em Markdown |
| `a1717b9` | Preenchimento do template Word oficial (SBC/Unisul) |
| `3998f2f` | Script de auditoria automatizada do Word |
| `1b0b10d` | README com metadados do artigo e roteiro de apresentação |
| `5bd1dfe` | `llama3.1:8b` vira o modelo padrão (maior precisão) |
| `1d86c3b` | Deduplicação de trechos recuperados e ajuste do prompt para perguntas compostas |

### Fase de ideação

Levantamento de abordagens (local, nuvem, low-code) e roteiro de pitch, em [Apresentação/](../Apresentação/).

---

## 6. Próximos passos

As quatro direções de trabalhos futuros do artigo: validação quantitativa com perguntas de referência; expansão para múltiplos setores com coleções segregadas; controle de acesso por perfil e trilha de auditoria; reindexação incremental e estudo de produtividade com usuários.
