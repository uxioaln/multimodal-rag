# -*- coding: utf-8 -*-
"""
app.api.auth - 用户鉴权相关路由（Flask Blueprint）

包含：
- /api/login, /api/logout, /api/login_status 三个路由
- require_login 守卫（紧耦合 Flask，不要做框架无关抽象）
"""
import hashlib

from flask import Blueprint, jsonify, request, session

from app.models.user import get_conn

auth_bp = Blueprint("auth", __name__)


# ========== DB 相关 ==========
def get_db_conn():
    """获取 SQLite 连接（兼容旧调用点，推荐直接使用 app.models.user.get_conn）"""
    return get_conn()


def verify_user(username: str, password: str):
    """
    校验用户账号密码
    返回 True 表示数据库中存在该用户且密码匹配
    """
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT password_hash, salt FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    conn.close()
    if row is None:
        return False
    # 加盐哈希比对
    expect_hash = hashlib.sha256((row["salt"] + password).encode("utf-8")).hexdigest()
    return expect_hash == row["password_hash"]


# ========== 鉴权守卫 ==========
def require_login():
    """未登录则返回 401 响应（紧耦合 Flask request/response，不做框架无关抽象）"""
    if not session.get("user"):
        return jsonify({"code": 1, "message": "请先登录管理员账号"}), 401
    return None


# ========== 路由 ==========
@auth_bp.route("/api/login", methods=["POST"])
def api_login():
    """登录接口：校验数据库用户"""
    data = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""
    if not username or not password:
        return jsonify({"code": 1, "message": "账号或密码不能为空"}), 400

    if verify_user(username, password):
        session["user"] = username
        return jsonify({"code": 0, "data": {"username": username}, "message": "登录成功"})
    return jsonify({"code": 1, "message": "账号或密码错误，数据库中不存在该用户"}), 401


@auth_bp.route("/api/logout", methods=["POST"])
def api_logout():
    """退出登录"""
    session.pop("user", None)
    return jsonify({"code": 0, "message": "已退出"})


@auth_bp.route("/api/login_status", methods=["GET"])
def api_login_status():
    """查询当前登录状态"""
    user = session.get("user")
    return jsonify({"code": 0, "data": {"username": user, "logged_in": bool(user)}})
