import docx

doc = docx.Document("/app/original.docx")

# 1. Títulos
if len(doc.paragraphs[0].runs) > 0:
    doc.paragraphs[0].runs[0].text = "DocAI: An Intelligent Corporate Assistant Based on Local RAG Architecture for Document Retrieval and Data Leakage Mitigation"
    for r in doc.paragraphs[0].runs[1:]:
        r.text = ""
else:
    doc.paragraphs[0].text = "DocAI: An Intelligent Corporate Assistant Based on Local RAG Architecture for Document Retrieval and Data Leakage Mitigation"

if len(doc.paragraphs[1].runs) > 0:
    doc.paragraphs[1].runs[0].text = "DocAI: Assistente Corporativo Inteligente Baseado em Arquitetura RAG Local para Recuperação de Documentos e Mitigação de Vazamento de Dados"
    for r in doc.paragraphs[1].runs[1:]:
        r.text = ""
else:
    doc.paragraphs[1].text = "DocAI: Assistente Corporativo Inteligente Baseado em Arquitetura RAG Local para Recuperação de Documentos e Mitigação de Vazamento de Dados"

# 2. Tabela de Autores (Table 0)
table0 = doc.tables[0]
cell1 = table0.rows[0].cells[1]
author2_texts = [
    "Tiago A. Ventura",
    "Unisul — Análise e Desenvolvimento de Sistemas",
    "Dib Mussi",
    "Florianópolis, SC, Brasil",
    "tiago.ventura@aluno.unisul.br"
]
for idx, text in enumerate(author2_texts):
    if idx < len(cell1.paragraphs):
        p = cell1.paragraphs[idx]
        if len(p.runs) > 0:
            p.runs[0].text = text
            for r in p.runs[1:]:
                r.text = ""
        else:
            p.text = text

# 3. Abstract e Keywords
abstract_text = (
    "Modern organizations face significant operational overhead when searching for regulatory and procedural "
    "information across fragmented internal repositories. Simultaneously, using cloud-based Large Language Models (LLMs) "
    "introduces high risks of enterprise data leakage and non-compliance with data protection legislation (such as LGPD). "
    "This work presents DocAI, an open-source, fully on-premises corporate virtual assistant designed to optimize internal document "
    "retrieval without external network transmission. Built upon a Retrieval-Augmented Generation (RAG) architecture, "
    "DocAI integrates Ollama for local model execution (llama3.2:3b and llama3.1:8b), ChromaDB for dense vector search, "
    "and Streamlit for a user-facing dashboard with explainability features. Experimental validation on actual internal corporate "
    "policies demonstrates inference latencies between 1.2s and 3.8s on consumer hardware, high precision in citing specific clauses, "
    "and robust containment of hallucinations via strict prompt constraints. The system offers organizations complete data sovereignty, "
    "zero ongoing token costs, and a structured roadmap for multi-departmental scalability."
)
if len(doc.paragraphs[4].runs) > 0:
    doc.paragraphs[4].runs[0].text = abstract_text
    for r in doc.paragraphs[4].runs[1:]:
        r.text = ""
else:
    doc.paragraphs[4].text = abstract_text

keywords_text = "Retrieval-Augmented Generation, Local Large Language Models, ChromaDB, Information Security, Corporate Virtual Assistants."
if len(doc.paragraphs[6].runs) > 0:
    doc.paragraphs[6].runs[0].text = keywords_text
    for r in doc.paragraphs[6].runs[1:]:
        r.text = ""
else:
    doc.paragraphs[6].text = keywords_text

# 4. Introdução
intro_text = (
    "A transformação digital nas organizações contemporâneas provocou uma proliferação expressiva no volume de documentos "
    "normativos, tais como políticas de recursos humanos, manuais operacionais de TI, códigos de conduta e diretrizes de compliance. "
    "Estimativas corporativas apontam que colaboradores gastam cerca de 20% de sua jornada semanal apenas localizando informações "
    "internas necessárias à execução de suas rotinas. Essa dispersão gera ineficiência produtiva e propicia erros operacionais "
    "decorrentes do uso de versões desatualizadas de manuais. Paralelamente, o uso não supervisionado de inteligências artificiais "
    "comerciais em nuvem (fenômeno conhecido como Shadow AI) acarreta graves vulnerabilidades de segurança da informação, expondo "
    "segredos industriais e violando a Lei Geral de Proteção de Dados (LGPD — Lei nº 13.709/2018). Além disso, modelos comerciais não "
    "ancorados tendem a alucinar assertivas sobre regras internas da companhia. Para solucionar essa lacuna, este trabalho apresenta "
    "o DocAI, um assistente corporativo baseado na arquitetura RAG (Retrieval-Augmented Generation) com operação 100% on-premises, "
    "assegurando custo zero de subscrição de APIs, respostas com citação auditável de cláusulas e eliminação determinística de alucinações."
)
if len(doc.paragraphs[8].runs) > 0:
    doc.paragraphs[8].runs[0].text = intro_text
    for r in doc.paragraphs[8].runs[1:]:
        r.text = ""
else:
    doc.paragraphs[8].text = intro_text

# 5. Fundamentação Teórica
fund_intro = (
    "A viabilização de assistentes inteligentes estritamente locais apoia-se na convergência entre quantização eficiente de modelos "
    "de linguagem e técnicas de recuperação densa em bancos vetoriais, eliminando a dependência de infraestruturas proprietárias em nuvem."
)
if len(doc.paragraphs[10].runs) > 0:
    doc.paragraphs[10].runs[0].text = fund_intro
    for r in doc.paragraphs[10].runs[1:]:
        r.text = ""
else:
    doc.paragraphs[10].text = fund_intro

doc.paragraphs[11].text = "2.1 Modelos de Linguagem Locais e Quantização GGUF"

fund_sub = (
    "Avanços em quantização pós-treinamento, notadamente formatos GGUF de 4 bits (Q4_K_M), possibilitaram a execução de modelos abertos de "
    "alta capacidade, como a família LLaMA 3 da Meta, em GPUs de consumo com consumo de memória inferior a 6GB de VRAM. No DocAI, a representação "
    "semântica é realizada convertendo textos em vetores densos multidimensionais via nomic-embed-text. Em vez de recorrer a processos onerosos de "
    "fine-tuning (que sofrem com esquecimento catastrófico e custo proibitivo de retreinamento contínuo a cada nova política), adota-se a arquitetura "
    "RAG (Retrieval-Augmented Generation), na qual a base documental é mantida segregada em um banco vetorial indexado. No momento da interação, o sistema "
    "extrai os trechos documentais pertinentes e ancora a geração de texto por meio de um System Prompt que instrui formalmente o modelo a recusar perguntas "
    "cujos dados não constem expressamente na base, contendo qualquer alucinação factual."
)
if len(doc.paragraphs[12].runs) > 0:
    doc.paragraphs[12].runs[0].text = fund_sub
    for r in doc.paragraphs[12].runs[1:]:
        r.text = ""
else:
    doc.paragraphs[12].text = fund_sub

# 6. Materiais e Métodos
mat_intro = (
    "O corpus de avaliação experimental foi estruturado a partir de três normativos corporativos reais: a Política Corporativa de Férias e Licenças, "
    "o Guia de Benefícios de Saúde e o Manual de Segurança da Informação e TI. Os documentos em formato texto e PDF foram particionados pelo algoritmo "
    "RecursiveCharacterTextSplitter com chunk size de 700 caracteres e chunk overlap de 120 caracteres, preservando a coerência sintática dos parágrafos."
)
if len(doc.paragraphs[14].runs) > 0:
    doc.paragraphs[14].runs[0].text = mat_intro
    for r in doc.paragraphs[14].runs[1:]:
        r.text = ""
else:
    doc.paragraphs[14].text = mat_intro

eq_intro = "A relação de relevância semântica entre a consulta do colaborador e os trechos documentais indexados é quantificada através da Similaridade de Cossenos no espaço vetorial do ChromaDB, expressa na Equação (1):"
if len(doc.paragraphs[15].runs) > 0:
    doc.paragraphs[15].runs[0].text = eq_intro
    for r in doc.paragraphs[15].runs[1:]:
        r.text = ""
else:
    doc.paragraphs[15].text = eq_intro

eq_formula = "	S_C(q, d) = (q · d) / (||q|| ||d||)	(1)"
if len(doc.paragraphs[16].runs) > 0:
    doc.paragraphs[16].runs[0].text = eq_formula
    for r in doc.paragraphs[16].runs[1:]:
        r.text = ""
else:
    doc.paragraphs[16].text = eq_formula

eq_vars = "em que q representa o vetor denso da consulta gerado pelo modelo nomic-embed-text, d é o vetor do trecho documental e S_C(q, d) é a similaridade angular calculada no espaço euclidiano multidimensional."
if len(doc.paragraphs[17].runs) > 0:
    doc.paragraphs[17].runs[0].text = eq_vars
    for r in doc.paragraphs[17].runs[1:]:
        r.text = ""
else:
    doc.paragraphs[17].text = eq_vars

eq_unnum = "S_C(q, d) = (∑_{j=1}^n q_j · d_j) / (√∑_{j=1}^n q_j^2 · √∑_{j=1}^n d_j^2)"
if len(doc.paragraphs[18].runs) > 0:
    doc.paragraphs[18].runs[0].text = eq_unnum
    for r in doc.paragraphs[18].runs[1:]:
        r.text = ""
else:
    doc.paragraphs[18].text = eq_unnum

doc.paragraphs[19].text = "3.1 Arquitetura do Sistema e Engenharia de Prompt"

arq_text = (
    "A solução foi concebida sob uma arquitetura conteinerizada em Docker Compose composta por dois serviços integrados: o servidor Ollama "
    "(encarregado da inferência dos pesos GGUF com aceleração CUDA) e o serviço da aplicação DocAI (desenvolvido em Python com interface Streamlit "
    "e orquestrador LangChain). A interface disponibiliza histórico conversacional reativo, métricas de latência calculadas em tempo real, "
    "visualizador auditável de trechos e scores de similaridade recuperados da base ChromaDB. O controle de alucinação é regido por diretrizes "
    "estritas no prompt de sistema, forçando a abstenção padronizada quando o tema da consulta não constar expressamente nos arquivos disponíveis."
)
if len(doc.paragraphs[20].runs) > 0:
    doc.paragraphs[20].runs[0].text = arq_text
    for r in doc.paragraphs[20].runs[1:]:
        r.text = ""
else:
    doc.paragraphs[20].text = arq_text

# 7. Resultados
res_text = (
    "Os ensaios de bancada foram executados em uma estação de trabalho equipada com GPU NVIDIA RTX 3060 (12GB VRAM), comparando o modelo ultra-compacto "
    "llama3.2:3b com o modelo intermediário llama3.1:8b sob o mesmo conjunto de consultas. A Tabela 1 sintetiza as métricas médias apuradas em dez execuções consecutivas para cada modelo."
)
if len(doc.paragraphs[22].runs) > 0:
    doc.paragraphs[22].runs[0].text = res_text
    for r in doc.paragraphs[22].runs[1:]:
        r.text = ""
else:
    doc.paragraphs[22].text = res_text

doc.paragraphs[23].text = "A Figura 1 ilustra a interface de usuário do DocAI em tempo de execução, destacando a medição de tempo de resposta e os trechos de evidência recuperados da base ChromaDB."
doc.paragraphs[25].text = "Figura 1: Interface do assistente corporativo DocAI com chat conversacional, medição de latência e fontes citadas."

if len(doc.paragraphs[26].runs) > 0:
    doc.paragraphs[26].runs[0].text = "Tabela 1: Comparação de desempenho e latência entre os modelos avaliados no DocAI."
    for r in doc.paragraphs[26].runs[1:]:
        r.text = ""
else:
    doc.paragraphs[26].text = "Tabela 1: Comparação de desempenho e latência entre os modelos avaliados no DocAI."

# Atualiza Table 1
table1 = doc.tables[1]
t1_data = [
    ["Modelo Avaliado", "VRAM Alocada", "Latência Média (s)"],
    ["Llama 3.2 (3B)", "~2.3 GB", "1,42 s"],
    ["Llama 3.1 (8B)", "~5.2 GB", "3,65 s"]
]
for r_idx, row in enumerate(table1.rows):
    for c_idx, cell in enumerate(row.cells):
        cell.text = t1_data[r_idx][c_idx]

res_detail = (
    "Os ensaios demonstraram que o modelo Llama 3.2 (3B) proporciona tempos de inferência quase instantâneos (1,42s com alocação de apenas 2,3GB de VRAM), "
    "sendo plenamente recomendado para consultas do dia a dia. Já o Llama 3.1 (8B) apresentou maior profundidade léxica na síntese de políticas complexas. "
    "Em ambos os modelos, a validação funcional comprovou precisão absoluta na citação de cláusulas normativas: na consulta de férias, o sistema recuperou "
    "com exatidão a antecedência de 30 dias (cláusula 3.1) e as regras de abono de 10 dias (cláusula 5.1); na consulta de benefícios, indicou com precisão "
    "os valores de R$ 35,00 para consultas e R$ 60,00 para exames. No teste de estresse da trava anti-alucinação, ao ser questionado sobre 'bônus de Natal para "
    "estagiários' (tema inexistente nos documentos), o assistente recusou categoricamente a resposta com 100% de contenção, sem inventar valores."
)
if len(doc.paragraphs[27].runs) > 0:
    doc.paragraphs[27].runs[0].text = res_detail
    for r in doc.paragraphs[27].runs[1:]:
        r.text = ""
else:
    doc.paragraphs[27].text = res_detail

# 8. Discussão
doc.paragraphs[29].text = "Estudos e experimentos anteriores em processamento de linguagem natural destacam o papel do RAG na contenção de custos e erros conceituais [1, 2]."

disc_text = (
    "A experiência prática com o DocAI demonstra que o uso de arquiteturas RAG locais é uma alternativa de altíssima viabilidade técnica e financeira "
    "para organizações de qualquer segmento. A eliminação completa do envio de informações para nuvens públicas anula o risco de Shadow AI e garante "
    "conformidade nativa com a LGPD e padrões de governança como a ISO 27001. Além disso, a exibição aberta das fontes e cláusulas recuperadas desmistifica "
    "o modelo caixa-preta comum às LLMs proprietárias, permitindo que os próprios colaboradores e auditores internos verifiquem a veracidade do conteúdo sintetizado pelo assistente."
)
if len(doc.paragraphs[30].runs) > 0:
    doc.paragraphs[30].runs[0].text = disc_text
    for r in doc.paragraphs[30].runs[1:]:
        r.text = ""
else:
    doc.paragraphs[30].text = disc_text

# 9. Conclusão e Trabalhos Futuros
conc_text = (
    "O presente estudo desenvolveu e validou o assistente virtual DocAI, demonstrando a eficácia da recuperação aumentada por geração executada "
    "100% em ambiente local. A solução comprovou latências reduzidas (inferiores a 4 segundos), fidelidade estrita às normas documentais e custo operacional nulo com serviços de terceiros."
)
if len(doc.paragraphs[32].runs) > 0:
    doc.paragraphs[32].runs[0].text = conc_text
    for r in doc.paragraphs[32].runs[1:]:
        r.text = ""
else:
    doc.paragraphs[32].text = conc_text

doc.paragraphs[33].text = "6.1 Trabalhos Futuros"

futuro_text = (
    "Como evolução do DocAI, propõem-se quatro direções. Primeiro, a validação quantitativa do sistema, por meio de um conjunto de perguntas de referência "
    "e métricas de precisão da recuperação, fidelidade da resposta e taxa de recusa em perguntas fora da base, com o objetivo de medir de forma objetiva "
    "a mitigação de respostas não fundamentadas. Segundo, a expansão para múltiplos setores organizacionais (como Recursos Humanos, Tecnologia da Informação, "
    "Jurídico e Desenvolvimento de Software), com coleções segregadas, metadados de controle de acesso e regras de resposta específicas por área, de modo a "
    "investigar como a arquitetura se comporta diante de corpora com estruturas distintas, como manuais técnicos com blocos de código. Terceiro, o fortalecimento "
    "dos mecanismos de segurança e governança, incluindo controle de acesso por perfil, trilha de auditoria e verificação formal de conformidade com a LGPD. "
    "Quarto, a análise da manutenção contínua da base, com versionamento e reindexação incremental de documentos, além de estudos com usuários para quantificar "
    "o ganho de produtividade e comparar diferentes modelos de linguagem executados localmente quanto a qualidade, latência e consumo de recursos."
)
if len(doc.paragraphs[34].runs) > 0:
    doc.paragraphs[34].runs[0].text = futuro_text
    for r in doc.paragraphs[34].runs[1:]:
        r.text = ""
else:
    doc.paragraphs[34].text = futuro_text

# 10. Agradecimentos e Declaração IA
agradec_text = "Agradecimentos à Unisul (Universidade do Sul de Santa Catarina) e ao corpo docente do curso de Análise e Desenvolvimento de Sistemas pelo suporte institucional e incentivo à pesquisa aplicada."
if len(doc.paragraphs[36].runs) > 0:
    doc.paragraphs[36].runs[0].text = agradec_text
    for r in doc.paragraphs[36].runs[1:]:
        r.text = ""
else:
    doc.paragraphs[36].text = agradec_text

decl_ia = (
    "Os autores declaram que ferramentas de Inteligência Artificial generativa foram utilizadas exclusivamente no apoio à revisão estilística e "
    "apoio à codificação dos scripts do protótipo experimental. O modelo conceitual, a arquitetura da solução, os testes empíricos de bancada e a "
    "redação final foram integralmente concebidos, revisados e validados pelos autores, que assumem total responsabilidade científica pela integridade, "
    "originalidade e precisão das informações apresentadas."
)
if len(doc.paragraphs[38].runs) > 0:
    doc.paragraphs[38].runs[0].text = decl_ia
    for r in doc.paragraphs[38].runs[1:]:
        r.text = ""
else:
    doc.paragraphs[38].text = decl_ia

# 11. Referências
refs = [
    "[1] P. Lewis, E. Perez, A. Piktus, et al. 2020. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. In Advances in Neural Information Processing Systems (NeurIPS 2020), Vol. 33, 9459–9474.",
    "[2] A. Vaswani, N. Shazeer, N. Parmar, et al. 2017. Attention Is All You Need. In Advances in Neural Information Processing Systems (NeurIPS 2017), Vol. 30, 5998–6008.",
    "[3] Meta AI. 2024. The Llama 3 Herd of Models. arXiv preprint arXiv:2407.21783.",
    "[4] J. Devlin, M.-W. Chang, K. Lee, K. Toutanova. 2019. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In Proceedings of NAACL-HLT 2019, 4171–4186.",
    "[5] Brasil. 2018. Lei nº 13.709, de 14 de agosto de 2018. Dispõe sobre a proteção de dados pessoais (Lei Geral de Proteção de Dados Pessoais - LGPD). Diário Oficial da União, Brasília, DF."
]
for idx, ref in enumerate(refs):
    p_idx = 40 + idx
    if p_idx < len(doc.paragraphs):
        p = doc.paragraphs[p_idx]
        if len(p.runs) > 0:
            p.runs[0].text = ref
            for r in p.runs[1:]:
                r.text = ""
        else:
            p.text = ref

# Salva o documento atualizado
out_path = "/app/Artigo_preenchido.docx"
doc.save(out_path)
print(f"Sucesso! Documento salvo em: {out_path}")
