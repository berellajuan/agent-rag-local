from __future__ import annotations

from typing import Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from app.agent.prompts import GRAPH_ANSWER_PROMPT, GRAPH_DECIDER_PROMPT
from app.console import StepTracer
from app.models import get_chat_model
from app.rag.retriever import retrieve


class RagGraphState(TypedDict):
    question: str
    route: Literal["retrieve", "direct"]
    retrieved_context: str
    sources: list[str]
    answer: str
    verbose: bool


def _decide(state: RagGraphState) -> RagGraphState:
    tracer = StepTracer(state.get("verbose", False))
    tracer.step("Nodo decide: el modelo decide si necesita consultar documentos locales.")
    model = get_chat_model(temperature=0.0)
    response = model.invoke(
        [
            SystemMessage(content=GRAPH_DECIDER_PROMPT),
            HumanMessage(content=state["question"]),
        ]
    )
    decision = str(response.content).strip().upper()
    route: Literal["retrieve", "direct"] = "retrieve" if "RETRIEVE" in decision else "direct"
    tracer.detail("decision_modelo", decision)
    tracer.detail("route", route)
    return {**state, "route": route}


def _retrieve(state: RagGraphState) -> RagGraphState:
    tracer = StepTracer(state.get("verbose", False))
    tracer.step("Nodo retrieve: recupero contexto desde el vector store.")
    context = retrieve(state["question"], verbose=state.get("verbose", False))
    tracer.detail("sources", context.sources)
    return {
        **state,
        "retrieved_context": context.to_prompt_context(),
        "sources": context.sources,
    }


def _generate_answer(state: RagGraphState) -> RagGraphState:
    tracer = StepTracer(state.get("verbose", False))
    tracer.step("Nodo generate: genero respuesta final con Gemini.")
    model = get_chat_model(temperature=0.0)
    if state.get("route") == "retrieve":
        content = (
            f"Pregunta: {state['question']}\n\n"
            f"<context>\n{state.get('retrieved_context', '')}\n</context>\n\n"
            f"Fuentes disponibles: {state.get('sources', [])}\n"
        )
    else:
        content = state["question"]

    response = model.invoke(
        [
            SystemMessage(content=GRAPH_ANSWER_PROMPT),
            HumanMessage(content=content),
        ]
    )
    answer = str(response.content)
    if state.get("sources"):
        answer = f"{answer}\n\nFuentes:\n" + "\n".join(f"- {s}" for s in state["sources"])
    tracer.result("Respuesta final del grafo generada.")
    return {**state, "answer": answer}


def _route_after_decide(state: RagGraphState) -> Literal["retrieve", "generate"]:
    return "retrieve" if state.get("route") == "retrieve" else "generate"


def build_graph():
    graph = StateGraph(RagGraphState)
    graph.add_node("decide", _decide)
    graph.add_node("retrieve", _retrieve)
    graph.add_node("generate", _generate_answer)

    graph.add_edge(START, "decide")
    graph.add_conditional_edges(
        "decide",
        _route_after_decide,
        {"retrieve": "retrieve", "generate": "generate"},
    )
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


def ask_graph(question: str, *, verbose: bool = False) -> str:
    tracer = StepTracer(verbose)
    tracer.title("LangGraph explicito")
    tracer.concept("LangGraph ejecuta nodos que leen y actualizan un estado compartido.")
    tracer.step("Compilo grafo: START -> decide -> retrieve/generate -> END.")
    graph = build_graph()
    tracer.step("Invoco grafo con estado inicial.")
    result = graph.invoke(
        {
            "question": question,
            "route": "direct",
            "retrieved_context": "",
            "sources": [],
            "answer": "",
            "verbose": verbose,
        }
    )
    tracer.result(f"Ruta tomada: {result.get('route')}")
    return result["answer"]


def main() -> None:
    question = input("Pregunta al grafo LangGraph: ").strip()
    if not question:
        raise SystemExit("No ingresaste una pregunta.")
    print(ask_graph(question, verbose=True))


if __name__ == "__main__":
    main()
