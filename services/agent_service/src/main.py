"""
Точка входа Agent Service: Gradio UI, смонтированный в FastAPI.

Lifespan управляет краткосрочной памятью агента: при старте открывает
SQLite checkpointer, при остановке закрывает соединение.
"""
import os
import uuid
from contextlib import asynccontextmanager

import gradio as gr
import uvicorn
from fastapi import FastAPI
from langchain_core.messages import HumanMessage

from shared.utils.logger import get_logger
from shared.utils.text import sanitize_llm_text

from core.agent import get_root_agent, init_root_agent, shutdown_root_agent
from core.callbacks import TokenUsageCallback
from core.rag import close_qdrant
from embedding_service.src.client import embedding_client

logger = get_logger(name="agent_service.main")


async def respond(message_text, history, session_state):
    # Каждая сессия браузера = отдельный thread_id: диалоги пользователей изолированы
    if "thread_id" not in session_state:
        session_state["thread_id"] = str(uuid.uuid4())
        session_state["user_id"] = os.getenv("DEFAULT_USER_ID", "manager_default")

    root_agent = get_root_agent()
    if root_agent is None:
        return "Агент ещё инициализируется, повторите запрос через пару секунд."

    result = await root_agent.ainvoke(
        {"messages": [HumanMessage(message_text)]},
        config={
            "configurable": {
                "thread_id": session_state["thread_id"],
                "user_id": session_state["user_id"],
            },
            "metadata": {"thread_id": session_state["thread_id"]},
            "callbacks": [TokenUsageCallback()],
        },
    )
    return sanitize_llm_text(result["messages"][-1].content)


def build_ui() -> gr.Blocks:
    return gr.ChatInterface(
        respond,
        title="Churn Copilot Agent",
        # gr.State(value={}) — собственный экземпляр состояния на каждую сессию браузера
        additional_inputs=[gr.State(value={})],
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_root_agent()
    logger.info("Agent Service готов к работе")
    yield
    await shutdown_root_agent()
    # Закрываем фоновые HTTP-соединения RAG-контура (созданы при импорте модулей)
    await embedding_client.close()
    await close_qdrant()
    logger.info("Соединения embedding-сервиса и Qdrant закрыты")


app = FastAPI(lifespan=lifespan)
app = gr.mount_gradio_app(app, build_ui(), path="/")

if __name__ == "__main__":
    logger.info("Запуск Agent Service UI (Gradio) на http://0.0.0.0:7860")
    uvicorn.run(app, host="0.0.0.0", port=7860)
