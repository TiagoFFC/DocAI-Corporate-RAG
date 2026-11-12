"""Gera Artigo/ARTIGO_DOCAI.md a partir do Word oficial (fonte única da verdade).

Uso:
    python scripts/docx_to_md.py ["Artigo/Artigo - Chatbot.docx"] [Artigo/ARTIGO_DOCAI.md]

O Markdown é uma transcrição fiel do .docx: nunca edite o .md à mão — altere o
Word e rode este script. A paridade é conferida por scripts/verify_docx.py.
"""
import re
import sys
from pathlib import Path

import docx
from docx.table import Table
from docx.text.paragraph import Paragraph

DEFAULT_DOCX = "Artigo/Artigo - Chatbot.docx"
DEFAULT_MD = "Artigo/ARTIGO_DOCAI.md"
FIGURE_NAME = "figura1_interface_docai.png"

HEADER = (
    "<!-- Gerado automaticamente por scripts/docx_to_md.py a partir de "
    "'Artigo - Chatbot.docx'. Não edite manualmente. -->"
)


def iter_blocks(document):
    """Percorre parágrafos e tabelas na ordem em que aparecem no documento."""
    for child in document.element.body.iterchildren():
        if child.tag.endswith("}p"):
            yield Paragraph(child, document)
        elif child.tag.endswith("}tbl"):
            yield Table(child, document)


def _render_table(table, first_table):
    rows = [[cell.text.strip() for cell in row.cells] for row in table.rows]
    if first_table:
        # Tabela de autoria do template: uma célula por autor, uma linha por dado.
        blocks = []
        for cell in rows[0]:
            lines = [line.strip() for line in cell.splitlines() if line.strip()]
            blocks.append("  \n".join([f"**{lines[0]}**"] + lines[1:]))
        return "\n\n".join(blocks)
    header, body = rows[0], rows[1:]
    out = ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
    out += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(out)


def _render_paragraph(par, index):
    text = par.text.strip()
    if "<w:drawing" in par._p.xml or "<w:pict" in par._p.xml:
        return f"![Figura 1 — Interface do DocAI]({FIGURE_NAME})"
    if not text:
        return None
    style = par.style.name
    if index < 2:
        return f"# {text}"
    if style == "Heading 1":
        return f"## {text}"
    if style == "Heading 2":
        return f"### {text}"
    if text.startswith("S_C(q, d) ="):
        return "```text\n" + re.sub(r"\s*\t\s*", "    ", text) + "\n```"
    if re.match(r"^(Figura|Tabela) \d+:", text):
        return f"**{text}**"
    if re.match(r"^\[\d+\] ", text):
        return f"- {text}"
    return text


def render(docx_path):
    """Devolve o Markdown completo correspondente ao documento Word."""
    document = docx.Document(docx_path)
    parts = [HEADER]
    par_index = 0
    table_index = 0
    for block in iter_blocks(document):
        if isinstance(block, Table):
            parts.append(_render_table(block, first_table=table_index == 0))
            table_index += 1
        else:
            rendered = _render_paragraph(block, par_index)
            par_index += 1
            if rendered:
                parts.append(rendered)

    # Itens de lista consecutivos (referências) ficam em linhas adjacentes.
    md = "\n\n".join(parts)
    md = re.sub(r"(?m)^(- \[\d+\] .*)\n\n(?=- \[\d+\] )", r"\1\n", md)
    return md + "\n"


def extract_figure(docx_path, out_dir):
    """Extrai a primeira imagem embutida no Word para acompanhar o Markdown."""
    document = docx.Document(docx_path)
    for rel in document.part.rels.values():
        if "image" in rel.reltype:
            target = Path(out_dir) / FIGURE_NAME
            target.write_bytes(rel.target_part.blob)
            return target
    return None


def main():
    docx_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DOCX
    md_path = Path(sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MD)
    md_path.write_text(render(docx_path), encoding="utf-8", newline="\n")
    figure = extract_figure(docx_path, md_path.parent)
    print(f"Markdown gerado: {md_path}")
    if figure:
        print(f"Figura extraída: {figure}")


if __name__ == "__main__":
    main()
