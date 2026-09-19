from functools import wraps
from flask import Blueprint, flash, redirect, render_template, request, session, url_for

import auth_utils

auth_bp = Blueprint("auth", __name__)


def login_required(f):
    """Decorator bảo vệ các route yêu cầu đăng nhập Admin."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get("admin_logged_in"):
            flash("Vui lòng đăng nhập để truy cập trang quản trị.", "warning")
            return redirect(url_for("auth.login", next=request.url))
        return f(*args, **kwargs)
    return decorated_function


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    """Trang đăng nhập Web Admin."""
    if session.get("admin_logged_in"):
        return redirect(url_for("dashboard.index"))

    next_url = request.args.get("next") or url_for("dashboard.index")

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        success, msg = auth_utils.verify_admin_login(username, password)
        if success:
            session["admin_logged_in"] = True
            session["admin_username"] = username
            flash("Đăng nhập hệ thống quản trị thành công!", "success")
            return redirect(next_url)
        else:
            flash(msg or "Tên đăng nhập hoặc mật khẩu không chính xác.", "danger")

    return render_template("login.html")


@auth_bp.route("/logout")
def logout():
    """Đăng xuất khỏi Web Admin."""
    session.clear()
    flash("Bạn đã đăng xuất khỏi hệ thống thành công.", "info")
    return redirect(url_for("auth.login"))
