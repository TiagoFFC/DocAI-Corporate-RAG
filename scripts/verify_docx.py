"""Auditoria automatizada do artigo científico do DocAI.

Uso:
    python scripts/verify_docx.py "Artigo/Artigo - Chatbot.docx" [Artigo/ARTIGO_DOCAI.md]

Verifica (1) ausência de placeholders do template SBC/Unisul, (2) presença dos
elementos obrigatórios e (3) paridade entre o Word e o Markdown.
Código de saída: 0 = aprovado, 1 = há pendências.
"""
import re
import sys
from pathlib import Path

import docx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from docx_to_md import render  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

doc_path = Path(sys.argv[1] if len(sys.argv) > 1 else "Artigo/Artigo - Chatbot.docx")
md_path = Path(sys.argv[2]) if len(sys.argv) > 2 else doc_path.parent / "ARTIGO_DOCAI.md"
doc = docx.Document(str(doc_path))

all_text = [p.text for p in doc.paragraphs]
table_rows = []
for t in doc.tables:
    for row in t.rows:
        table_rows.append(" | ".join(cell.text for cell in row.cells))
        for cell in row.cells:
            all_text.append(cell.text)

# Cabeçalhos e rodapés também carregam placeholders do template
for section in doc.sections:
    for part in (section.header, section.first_page_header, section.even_page_header,
                 section.footer, section.first_page_footer, section.even_page_footer):
        all_text.extend(p.text for p in part.paragraphs)

full_content = "\n".join(all_text)


def image_blobs(document):
    return [rel.target_part.blob for rel in document.part.rels.values() if "image" in rel.reltype]


template_path = doc_path.parent / "Artigo - Chatbot_original.docx"
template_images = image_blobs(docx.Document(str(template_path))) if template_path.exists() else []
figure_is_real = bool(image_blobs(doc)) and not any(b in template_images for b in image_blobs(doc))

print(f"Documento auditado: {doc_path}")
print(f"Total lines extracted: {len(all_text)}")
print(f"Total characters: {len(full_content)}")

# Procura por colchetes ou placeholders remanescentes do template
placeholders = re.findall(r'\[[A-Za-zÀ-ÿ0-9\s\.\,\:\;\-\_]{3,}\]', full_content)
# Ignora referências legítimas como [1], [2], [1, 2]
bad_placeholders = [p for p in placeholders if not re.match(r'^\[\d+(,\s*\d+)*\]$', p)]
# Textos de instrução típicos de templates
template_leftovers = [
    marker for marker in ("Lorem ipsum", "Nome do Autor", "Título do Artigo", "Insira aqui", "XXXX")
    if marker.lower() in full_content.lower()
]

print("\n--- Auditoria de Placeholders ---")
if bad_placeholders or template_leftovers:
    print(f"ALERTA: {len(bad_placeholders) + len(template_leftovers)} placeholders encontrados:")
    for bp in bad_placeholders + template_leftovers:
        print(f"  - {bp}")
else:
    print("Sucesso! Nenhum placeholder do template restou no documento.")


def table_row_has(*terms):
    return any(all(term in row for term in terms) for row in table_rows)


# Corpo do artigo = parágrafos antes do título "REFERÊNCIAS"
body_paragraphs = []
reference_numbers = []
in_references = False
for p in doc.paragraphs:
    if p.style.name.startswith("Heading") and p.text.strip().upper().startswith("REFERÊNCIAS"):
        in_references = True
        continue
    if in_references:
        m = re.match(r"^\[(\d+)\]", p.text.strip())
        if m:
            reference_numbers.append(int(m.group(1)))
    else:
        body_paragraphs.append(p.text)
body_text = "\n".join(body_paragraphs)

cited_numbers = set()
for group in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", body_text):
    cited_numbers.update(int(n) for n in re.split(r"\s*,\s*", group))
orphan_references = [n for n in reference_numbers if n not in cited_numbers]


def paragraph_with(*terms):
    """True se algum parágrafo do corpo contém todos os termos."""
    return any(all(t in p for t in terms) for p in body_paragraphs)


def section_text(start_heading, stop_headings):
    """Texto dos parágrafos entre o título start_heading e o próximo título listado."""
    out, inside = [], False
    for p in doc.paragraphs:
        is_heading = p.style.name.startswith("Heading")
        title = p.text.strip().upper()
        if is_heading and title == start_heading:
            inside = True
            continue
        if inside and is_heading and any(title.startswith(s) for s in stop_headings):
            break
        if inside:
            out.append(p.text)
    return "\n".join(out)


headings = [p.text.strip().upper() for p in doc.paragraphs if p.style.name.startswith("Heading")]
resumo_text = section_text("RESUMO", ["PALAVRAS-CHAVE"])
palavras_chave_text = section_text("PALAVRAS-CHAVE", ["1 "])
portuguese_body = section_text("1 INTRODUÇÃO", ["REFERÊNCIAS"])

# Termos em inglês que, no texto em português, só podem aparecer uma vez e entre
# parênteses (glosa do original na 1ª ocorrência). Nomes próprios ficam de fora.
ENGLISH_TERMS = [
    "compliance", "Shadow AI", "on-premises", "fine-tuning", "Retrieval-Augmented Generation",
    "System Prompt", "chunk size", "chunk overlap", "scores", "corpora", "tokens",
]


def bare_english_terms(text):
    """Termos em inglês usados fora de glosa ou repetidos após a 1ª ocorrência."""
    problems = []
    for term in ENGLISH_TERMS:
        pattern = re.escape(term)
        total = len(re.findall(rf"\b{pattern}\b", text, re.IGNORECASE))
        glossed = len(re.findall(rf"\([^()]*\b{pattern}\b[^()]*\)", text, re.IGNORECASE))
        if total > 1 or total != glossed:
            problems.append(f"{term} ({total}x, {glossed} entre parênteses)")
    return problems


bare_in_body = bare_english_terms(portuguese_body)
bare_in_resumo = bare_english_terms(resumo_text + "\n" + palavras_chave_text)


print("\n--- Verificação de Elementos Obrigatórios ---")
checks = {
    "Autor Tiago A. Ventura": "Tiago A. Ventura" in full_content,
    "Autor Jorge Werner": "Jorge Werner" in full_content,
    "Instituição Unisul": "Unisul" in full_content and "Universidade do Sul de Santa Catarina" in full_content,
    "Curso de Análise e Desenvolvimento de Sistemas": "Análise e Desenvolvimento de Sistemas" in full_content,
    "Título em Inglês": "DocAI: An Intelligent Corporate Assistant" in full_content,
    "Título em Português": "DocAI: Assistente Corporativo Inteligente" in full_content,
    "Similaridade de Cossenos (Eq. 1)": "S_C(q, d) = (q · d) / (||q|| ||d||)" in full_content and "(1)" in full_content,
    "Cabeçalho com sobrenomes dos autores": "Werner e Ventura" in full_content,
    "Figura 1 é captura real (não a imagem de exemplo do template)": figure_is_real,
    "Tabela 1 - legenda": "Tabela 1:" in full_content,
    "Tabela 1 - Llama 3.2 (3B) com latência 1,42 s": table_row_has("Llama 3.2 (3B)", "1,42 s"),
    "Tabela 1 - Llama 3.1 (8B) com latência 3,65 s": table_row_has("Llama 3.1 (8B)", "3,65 s"),
    "Resultados - acurácia nas cláusulas e trava anti-alucinação": "cláusula 3.1" in full_content and "100% de contenção" in full_content,
    "Trabalhos Futuros (4 direções)": "propõem-se quatro direções" in full_content,
    "Validação quantitativa (Pilar 1)": "validação quantitativa do sistema" in full_content,
    "Expansão multi-departamental (Pilar 2)": "expansão para múltiplos setores organizacionais" in full_content,
    "Segurança e Governança (Pilar 3)": "fortalecimento dos mecanismos de segurança e governança" in full_content,
    "Ciclo de vida e usabilidade (Pilar 4)": "manutenção contínua da base" in full_content,
    "Agradecimentos Unisul": "Agradecimentos à Unisul" in full_content,
    "Declaração de Uso de IA": "ferramentas de Inteligência Artificial generativa foram utilizadas" in full_content,
    "Referências Bibliográficas": "P. Lewis, E. Perez" in full_content and "Lei nº 13.709" in full_content,
    f"Nenhuma referência órfã (sem citação no corpo: {orphan_references or 'nenhuma'})": bool(reference_numbers) and not orphan_references,
    "Citação [3] junto à família LLaMA 3": paragraph_with("LLaMA 3 da Meta [3]"),
    "Citação [4] junto aos vetores densos": paragraph_with("vetores densos multidimensionais [4]"),
    "Citação [5] junto à LGPD na Introdução": paragraph_with("Lei nº 13.709/2018) [5]"),
    "Citação [5] junto à LGPD na Discussão": paragraph_with("conformidade nativa com a LGPD [5]"),
    "Sem fórmula duplicada em notação crua": "∑_{" not in body_text and "^n" not in body_text,
    "Modelos com grafia editorial no texto (sem tags llama3.x:yb)": not re.search(r"llama3\.\d:\d+b", body_text),
    "Hardware de teste intacto (RTX 3060, 12GB)": "RTX 3060 (12GB VRAM)" in body_text,
    "ABSTRACT e KEYWORDS em inglês mantidos (exigência do template)": "ABSTRACT" in headings and "KEYWORDS" in headings,
    "RESUMO em português presente e não vazio": len(resumo_text) > 500,
    "PALAVRAS-CHAVE em português presentes": len(palavras_chave_text.split(",")) >= 4,
    "RESUMO vem antes da Introdução": "RESUMO" in headings and headings.index("RESUMO") < headings.index("1 INTRODUÇÃO"),
    f"Corpo em português sem termos em inglês soltos ({'; '.join(bare_in_body) or 'nenhum'})": bool(portuguese_body) and not bare_in_body,
    f"Resumo sem termos em inglês soltos ({'; '.join(bare_in_resumo) or 'nenhum'})": bool(resumo_text) and not bare_in_resumo,
}

print("\n--- Paridade Word x Markdown ---")
if md_path.exists():
    expected_md = render(str(doc_path))
    actual_md = md_path.read_text(encoding="utf-8").replace("\r\n", "\n")
    checks[f"Paridade total com {md_path.name}"] = actual_md == expected_md
    if actual_md != expected_md:
        exp_lines, act_lines = expected_md.splitlines(), actual_md.splitlines()
        for i, (e, a) in enumerate(zip(exp_lines, act_lines), 1):
            if e != a:
                print(f"Primeira divergência na linha {i} do Markdown:")
                print(f"  Word    : {e[:110]}")
                print(f"  Markdown: {a[:110]}")
                break
        else:
            print(f"Quantidade de linhas divergente: Word={len(exp_lines)} Markdown={len(act_lines)}")
        print('Corrija com: python scripts/docx_to_md.py')
    else:
        print(f"{md_path.name} é idêntico à transcrição do Word ({len(actual_md)} caracteres).")
else:
    checks[f"Paridade total com {md_path.name}"] = False
    print(f"Arquivo Markdown não encontrado: {md_path}")

print()
all_passed = True
for check_name, passed in checks.items():
    status = "OK" if passed else "FALHOU"
    print(f"[{status}] {check_name}")
    if not passed:
        all_passed = False

errors = sum(not ok for ok in checks.values()) + len(bad_placeholders) + len(template_leftovers)
if all_passed and errors == 0:
    print(f"\n>>> AUDITORIA CONCLUÍDA COM 100% DE SUCESSO! {len(checks)} verificações, 0 erros.")
    sys.exit(0)

print(f"\n>>> AUDITORIA INDICOU {errors} PENDÊNCIA(S). VERIFIQUE OS ITENS ACIMA.")
sys.exit(1)
