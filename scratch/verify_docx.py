import re
import docx

import sys
doc_path = sys.argv[1] if len(sys.argv) > 1 else "/workspace/Artigo/Artigo - Chatbot.docx"
doc = docx.Document(doc_path)

all_text = []

for p in doc.paragraphs:
    all_text.append(p.text)

for t in doc.tables:
    for row in t.rows:
        for cell in row.cells:
            all_text.append(cell.text)

full_content = "\n".join(all_text)

print(f"Total lines extracted: {len(all_text)}")
print(f"Total characters: {len(full_content)}")

# Procura por colchetes ou placeholders remanescentes do template
placeholders = re.findall(r'\[[A-Za-zÀ-ÿ0-9\s\.\,\:\;\-\_]{3,}\]', full_content)
# Ignora referências legítimas como [1], [2], [1, 2]
bad_placeholders = [p for p in placeholders if not re.match(r'^\[\d+(,\s*\d+)*\]$', p)]

print("\n--- Auditoria de Placeholders ---")
if bad_placeholders:
    print(f"ALERTA: {len(bad_placeholders)} placeholders encontrados:")
    for bp in bad_placeholders:
        print(f"  - {bp}")
else:
    print("Sucesso! Nenhum placeholder do template restou no documento.")

print("\n--- Verificação de Elementos Obrigatórios ---")
checks = {
    "Autor Tiago A. Ventura": "Tiago A. Ventura" in full_content,
    "Autor Jorge Werner": "Jorge Werner" in full_content,
    "Título em Inglês": "DocAI: An Intelligent Corporate Assistant" in full_content,
    "Título em Português": "DocAI: Assistente Corporativo Inteligente" in full_content,
    "Similaridade de Cossenos (Eq. 1)": "S_C(q, d)" in full_content,
    "Tabela 1 de Resultados": "Llama 3.2 (3B)" in full_content and "1,42 s" in full_content,
    "Trabalhos Futuros (4 direções)": "propõem-se quatro direções" in full_content,
    "Validação quantitativa (Pilar 1)": "validação quantitativa do sistema" in full_content,
    "Expansão multi-departamental (Pilar 2)": "expansão para múltiplos setores organizacionais" in full_content,
    "Segurança e Governança (Pilar 3)": "fortalecimento dos mecanismos de segurança e governança" in full_content,
    "Ciclo de vida e usabilidade (Pilar 4)": "manutenção contínua da base" in full_content,
    "Agradecimentos Unisul": "Agradecimentos à Unisul" in full_content,
    "Declaração de Uso de IA": "ferramentas de Inteligência Artificial generativa foram utilizadas" in full_content,
    "Referências Bibliográficas": "P. Lewis, E. Perez" in full_content and "Lei nº 13.709" in full_content
}

all_passed = True
for check_name, passed in checks.items():
    status = "OK" if passed else "FALHOU"
    print(f"[{status}] {check_name}")
    if not passed:
        all_passed = False

if all_passed and not bad_placeholders:
    print("\n>>> AUDITORIA CONCLUÍDA COM 100% DE SUCESSO! O DOCUMENTO ESTÁ PERFEITO.")
else:
    print("\n>>> AUDITORIA INDICOU PENDÊNCIAS. VERIFIQUE OS ITENS ACIMA.")
