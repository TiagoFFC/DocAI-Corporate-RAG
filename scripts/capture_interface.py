"""Captura a tela real do DocAI (Figura 1 do artigo) com o Playwright.

Uso (com os containers no ar e a base indexada):
    python scripts/capture_interface.py [url] [saida.png]

Faz a pergunta de férias pelo botão da barra lateral, espera a resposta e salva
uma captura 4:3 (1200x900), a mesma proporção da figura no template do artigo.
"""
import sys

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8501"
OUT = sys.argv[2] if len(sys.argv) > 2 else "Artigo/figura1_interface_docai.png"

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1200, "height": 900}, device_scale_factor=2)
    page.goto(URL, wait_until="networkidle")
    page.get_by_role("button", name="1. Regra de Férias e Abono").click()
    page.get_by_text("fontes citadas").first.wait_for(timeout=300_000)
    # O Streamlit rola para o fim da conversa; volta ao topo do chat e da barra lateral.
    page.get_by_text("Painel de Controle").first.scroll_into_view_if_needed()
    page.get_by_text("DocAI: Assistente Corporativo Inteligente").first.scroll_into_view_if_needed()
    page.wait_for_timeout(1500)
    page.screenshot(path=OUT)
    browser.close()
print(f"Captura salva em {OUT}")
