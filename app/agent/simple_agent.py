from __future__ import annotations

from typing import Any

from langchain.agents import create_agent

from app.agent.prompts import AGENT_SYSTEM_PROMPT
from app.agent.tools import retrieve_context
from app.console import StepTracer
from app.models import get_chat_model


def create_rag_agent():
    """Agente simple recomendado: LangChain create_agent sobre runtime LangGraph."""

    model = get_chat_model(temperature=0.0)
    return create_agent(
        model=model,
        tools=[retrieve_context],
        system_prompt=AGENT_SYSTEM_PROMPT,
    )


def _describe_message(message: Any) -> str:
    message_type = message.__class__.__name__
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        names = [call.get("name", "unknown") for call in tool_calls]
        return f"{message_type} tool_calls={names}"
    name = getattr(message, "name", None)
    if name:
        return f"{message_type} name={name}"
    return message_type


def ask_agent(question: str, *, verbose: bool = False) -> str:
    tracer = StepTracer(verbose)
    tracer.title("Agente LangChain")
    tracer.concept("Un agente es modelo + instrucciones + herramientas + loop de decision.")
    tracer.step("Creo agente con Gemini y tool retrieve_context.")
    agent = create_rag_agent()
    tracer.step("Invoco el agente con limite de recursion 8.")
    result: dict[str, Any] = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config={"recursion_limit": 8},
    )
    messages = result.get("messages", [])
    if not messages:
        return "El agente no devolvio mensajes."
    tracer.result(f"Mensajes producidos por el agente: {len(messages)}")
    for index, message in enumerate(messages, start=1):
        tracer.detail(f"mensaje_{index}", _describe_message(message))
    tracer.concept(
        "Si aparece un AIMessage con tool_calls y luego un ToolMessage, el agente decidio usar una herramienta."
    )
    return str(messages[-1].content)


def main() -> None:
    question = input("Pregunta al agente: ").strip()
    if not question:
        raise SystemExit("No ingresaste una pregunta.")
    print(ask_agent(question, verbose=True))


if __name__ == "__main__":
    main()
