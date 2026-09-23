# -*- coding: utf-8 -*-
"""API 路由层：仅做协议适配，不写业务逻辑"""
from app.api.auth import auth_bp
from app.api.ask import ask_bp
from app.api.knowledge import knowledge_bp
from app.api.stats import stats_bp
from app.api.conversation import conversation_bp
from app.api.agent_chat import agent_bp

__all__ = ["auth_bp", "ask_bp", "knowledge_bp", "stats_bp", "conversation_bp", "agent_bp"]
