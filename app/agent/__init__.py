# -*- coding: utf-8 -*-
"""
app.agent - Agent 编排层

将 RAG 从"固定管道"升级为"LLM 自主决策调用工具"的 Agent 架构。

模块组成：
- tools.py   工具注册表，把现有检索能力包装为 LLM 可调用的工具
- memory.py  会话记忆读写，复用现有 SQLite conversations 表
- loop.py    ReAct 推理循环（Planner + Executor），含步数上限/成本预算/循环检测

设计原则：
- 不修改 app.core / app.index / app.api 中已有的正确逻辑
- 工具函数复用 app.core.query 中的检索能力，仅做"包装"
- LLM 调用统一走 cost_tracker，自动纳入成本统计
"""
from app.agent.loop import agent_chat, agent_chat_sse
from app.agent.tools import get_tool_schemas, execute_tool
from app.agent.memory import get_recent_messages, save_message
from app.agent.planner import plan_retrieval, format_plan_summary
from app.agent.validator import validate_answer, format_validation_summary

__all__ = [
    "agent_chat",
    "agent_chat_sse",
    "get_tool_schemas",
    "execute_tool",
    "get_recent_messages",
    "save_message",
    "plan_retrieval",
    "format_plan_summary",
    "validate_answer",
    "format_validation_summary",
]
