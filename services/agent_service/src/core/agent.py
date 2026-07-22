"""
Описание и параметры агентов.
"""
import os
import sys
import asyncio

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage

from .llm_client import llm
from .prompts import root_agent_prompt, churn_analisys_and_retention_agent_prompt
from .tools import get_churn_info
# from .middleware import TrimMessagesMiddleware

from shared.utils.logger import app_logger, get_logger


# =========  Суб-агент для получения экспертизы по корпоративным политикам и маркетинговым предложениям по удержанию.  =========

churn_analisys_and_retention_agent = create_agent(
    llm,
    tools=[get_churn_info],
    system_prompt=churn_analisys_and_retention_agent_prompt
)
# Тул, в рамках которого будет работать суб-агент policies_and_retention_rules_agent
@tool(
    "get_churn_analisys_and_retention",
    description="Используй для получения экспертизы по корпоративным политикам и маркетинговым предложениям по удержанию. \
        Не используй для обычного диалога или приветствий."
)
async def get_churn_analisys_and_retention(query: str):
    app_logger.info(f"Задача передана агенту churn_analisys_and_retention_agent")
    result = await churn_analisys_and_retention_agent.ainvoke(
        {            
            "messages": [{"role": "user", "content": query}],
            "config": {"max_tool_calls": 5}
        }
    )
    return f"Результат по твоему запросу {query}: \n{result['messages'][-1].content.split('</think>')[-1].strip()}"


# ==========   Основной агент, взаимодействует с пользователем и раздает задачи суб-агентам   ==========

root_agent = create_agent(
    llm,
    tools=[get_churn_analisys_and_retention],
    system_prompt=root_agent_prompt,
    # middleware=[TrimMessagesMiddleware(max_messages=15)],
    checkpointer=InMemorySaver(),
)
