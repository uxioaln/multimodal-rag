# -*- coding: utf-8 -*-
"""
app.api.stats - 全局统计路由：返回大模型调用的累计使用情况
"""
from flask import Blueprint, jsonify

from app.api.auth import require_login
from app.core.cost_tracker import get_global_stats

stats_bp = Blueprint("stats", __name__)


@stats_bp.route("/api/stats", methods=["GET"])
def get_stats():
    """返回所有模块的大模型调用累计统计"""
    guard = require_login()
    if guard:
        return guard
    summary = get_global_stats().get_summary()
    return jsonify(summary)


@stats_bp.route("/api/stats/reset", methods=["POST"])
def reset_stats():
    """清空全局统计（管理用途）"""
    guard = require_login()
    if guard:
        return guard
    get_global_stats().reset()
    return jsonify({"status": "ok", "message": "统计已重置"})
