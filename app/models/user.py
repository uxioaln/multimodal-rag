# -*- coding: utf-8 -*-
"""
app.models.user - 用户表的数据访问封装
"""
import sqlite3

from app.config import DB_FILE


def get_conn():
    """获取 SQLite 连接（统一入口）"""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn
