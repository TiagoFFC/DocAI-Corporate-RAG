"""Funções puras de apresentação, testáveis sem iniciar o Streamlit."""
import html


def source_box_html(content: str) -> str:
    """Monta a caixa HTML de um trecho recuperado, escapando o conteúdo.

    O texto vem dos documentos da base (incluindo uploads), portanto nunca deve
    ser injetado como HTML bruto: ``<script>``, ``<img onerror=...>`` etc. são
    exibidos como texto. As quebras de linha são preservadas com ``<br>``.
    """
    seguro = html.escape(content or "", quote=True).replace("\n", "<br>")
    return f'<div class="source-box">{seguro}</div>'
