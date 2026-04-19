from __future__ import annotations

import argparse

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.graph import ask_graph
from app.agent.simple_agent import ask_agent
from app.console import StepTracer
from app.models import get_chat_model
from app.rag.ask_rag import answer_with_rag
from app.rag.indexer import index_documents
from app.rag.retriever import retrieve


def ask_llm(question: str, *, verbose: bool = False) -> str:
    tracer = StepTracer(verbose)
    tracer.title("LLM directo")
    tracer.concept("Esta etapa prueba solo Gemini: todavia no hay embeddings, RAG, tools ni agente.")
    tracer.step("Creo chat model Gemini.")
    model = get_chat_model(temperature=0.0)
    tracer.step("Envio system prompt + pregunta del usuario.")
    response = model.invoke(
        [
            SystemMessage(content="Sos un asistente didactico. Responde claro y breve."),
            HumanMessage(content=question),
        ]
    )
    tracer.result("Respuesta recibida desde Gemini.")
    return str(response.content)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Laboratorio local de agente IA + RAG con Gemini, LangChain y LangGraph."
    )
    parser.add_argument(
        "mode",
        choices=["llm", "index", "rag-search", "rag-answer", "agent", "graph"],
        help="Que queres ejecutar.",
    )
    parser.add_argument("question", nargs="*", help="Pregunta. Si se omite, se solicita por consola.")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Oculta la salida educativa paso a paso y muestra solo resultados.",
    )
    args = parser.parse_args()
    verbose = not args.quiet

    if args.mode == "index":
        result = index_documents(verbose=verbose)
        print("Resultado de indexacion incremental:")
        for key, value in result.items():
            print(f"- {key}: {value}")
        return

    question = " ".join(args.question).strip() or input("Pregunta: ").strip()
    if not question:
        raise SystemExit("No ingresaste una pregunta.")

    if args.mode == "llm":
        print(ask_llm(question, verbose=verbose))
    elif args.mode == "rag-search":
        context = retrieve(question, verbose=verbose)
        print(context.to_prompt_context())
        print("\nFuentes:")
        for source in context.sources:
            print(f"- {source}")
    elif args.mode == "rag-answer":
        print(answer_with_rag(question, verbose=verbose))
    elif args.mode == "agent":
        print(ask_agent(question, verbose=verbose))
    elif args.mode == "graph":
        print(ask_graph(question, verbose=verbose))


if __name__ == "__main__":
    main()
