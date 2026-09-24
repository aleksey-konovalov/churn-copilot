"""
Кастомные middleware агента.

Базовая оптимизация контекста построена на встроенных middleware LangChain
(SummarizationMiddleware, ToolRetryMiddleware, ToolCallLimitMiddleware,
ModelFallbackMiddleware — подключены в core/agent.py).

Этот модуль зарезервирован под MemoryInjectionMiddleware (этап 5 плана
оптимизации контекста): инъекция долговременной памяти через LangGraph Store —
семантический поиск фактов о пользователе (store.asearch по namespace
("memories", user_id)) и добавление их в system prompt через
request.override(system_message=...).
"""
