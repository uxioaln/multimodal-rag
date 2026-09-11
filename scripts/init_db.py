# -*- coding: utf-8 -*-
"""
scripts/init_db.py - 数据库初始化

使用：
    export DASHSCOPE_API_KEY=xxx
    export AGICTO_API_KEY=xxx
    python scripts/init_db.py
"""
import hashlib
import os
import secrets
import sqlite3
import sys

# 把工程根加入 path，使 app.* 包可被导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import DB_DIR, DB_FILE


def hash_password(password: str, salt: str) -> str:
    """对密码进行加盐哈希，返回十六进制字符串"""
    return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()


def init_database():
    """初始化数据库：建表并写入默认管理员"""
    os.makedirs(DB_DIR, exist_ok=True)
    db_path = os.path.abspath(DB_FILE)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # 创建用户表
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            password_salt TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'user',
            created_at TEXT NOT NULL
        )
    """)

    # 创建对话表
    cur.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            meta_json TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
        )
    """)

    # 创建会话生命周期表：用于标记多轮对话的起止时间
    # ended_at IS NOT NULL 视为"已结束"，knowledge_distill 只会从这类会话中提取
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chat_sessions (
            session_id TEXT PRIMARY KEY,
            visitor_id TEXT,
            started_at TEXT NOT NULL,
            last_active_at TEXT NOT NULL,
            ended_at TEXT
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_chat_sessions_ended_at ON chat_sessions(ended_at)")

    # 数据迁移：把 conversations 表中已有但 chat_sessions 中没有的 session 标记为"已结束"
    # 结束时间取该 session 最后一条消息的 created_at，避免老数据被默认当作进行中
    cur.execute("""
        INSERT OR IGNORE INTO chat_sessions (session_id, visitor_id, started_at, last_active_at, ended_at)
        SELECT
            c.session_id,
            '' AS visitor_id,
            MIN(c.created_at) AS started_at,
            MAX(c.created_at) AS last_active_at,
            MAX(c.created_at) AS ended_at
        FROM conversations c
        LEFT JOIN chat_sessions s ON s.session_id = c.session_id
        WHERE s.session_id IS NULL
        GROUP BY c.session_id
    """)

    # 创建默认管理员
    cur.execute("SELECT COUNT(*) FROM users WHERE username = ?", ("admin",))
    if cur.fetchone()[0] == 0:
        salt = secrets.token_hex(8)
        pwd_hash = hash_password("admin123", salt)
        cur.execute(
            "INSERT INTO users (username, password_hash, password_salt, role, created_at) VALUES (?, ?, ?, ?, datetime('now', 'localtime'))",
            ("admin", pwd_hash, salt, "admin"),
        )
        print("已创建默认管理员：admin / admin123")

    conn.commit()
    conn.close()
    print(f"数据库初始化完成 -> {db_path}")


if __name__ == "__main__":
    init_database()
