"""Mede a latência ponta a ponta (busca + geração) do DocAI por modelo.

Uso:
    python scripts/benchmark_latency.py [rodadas]

Usa uma base vetorial temporária (não toca no ./chroma_db) e as 4 perguntas do
roteiro de demonstração. A primeira chamada de cada modelo é descartada
(aquecimento / carga dos pesos na GPU).
"""
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag_engine import RAGEngine  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

QUESTIONS = [
    "Qual a antecedência mínima para solicitar férias e posso vender 10 dias?",
    "Como funciona a coparticipação em consultas e exames no plano de saúde?",
    "O que devo fazer caso meu notebook corporativo seja roubado?",
    "Qual é o valor do bônus de Natal pago para estagiários?",
]
MODELS = ["llama3.2:3b", "llama3.1:8b"]


def main():
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        engine = RAGEngine(ollama_url=url, persist_dir=tmp)
        result = engine.index_directory(str(ROOT / "documentos"))
        if not result["success"]:
            sys.exit(result["message"])

        print(f"Rodadas por pergunta: {rounds} | Ollama: {url}\n")
        print("| Modelo | Pergunta | Mediana (s) | Mín (s) | Máx (s) |")
        print("| --- | --- | --- | --- | --- |")
        for model in MODELS:
            engine.llm_model = model
            engine.ask(QUESTIONS[0], temperature=0.0)  # aquecimento
            todas = []
            for idx, question in enumerate(QUESTIONS, 1):
                tempos = []
                for _ in range(rounds):
                    start = time.perf_counter()
                    engine.ask(question, temperature=0.1)
                    tempos.append(time.perf_counter() - start)
                todas.extend(tempos)
                print(
                    f"| {model} | {idx} | {statistics.median(tempos):.2f} "
                    f"| {min(tempos):.2f} | {max(tempos):.2f} |"
                )
            print(
                f"| **{model}** | **todas** | **{statistics.median(todas):.2f}** "
                f"| {min(todas):.2f} | {max(todas):.2f} |"
            )
        engine.vectorstore = None


if __name__ == "__main__":
    main()
