import os
import sys
from datetime import datetime

# 1. Đảm bảo thư mục gốc dự án nằm trong sys.path để import database, config, faiss_utils...
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from flask import Flask, render_template, session

import config
from web_admin.routes.attendance import attendance_bp
from web_admin.routes.auth import auth_bp
from web_admin.routes.dashboard import dashboard_bp
from web_admin.routes.employees import employees_bp
from web_admin.routes.reports import reports_bp
from web_admin.routes.system import system_bp


def create_app() -> Flask:
    """Khởi tạo và cấu hình ứng dụng Flask Web Admin."""
    app = Flask(
        __name__,
        template_folder=os.path.join(CURRENT_DIR, "templates"),
        static_folder=os.path.join(CURRENT_DIR, "static"),
    )

    # Cấu hình Secret Key cho Session
    app.secret_key = config.FLASK_SECRET_KEY or "face_attendance_flask_secret_2026_key"

    # Đăng ký các Blueprints
    app.register_blueprint(auth_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(employees_bp)
    app.register_blueprint(attendance_bp)
    app.register_blueprint(reports_bp)
    app.register_blueprint(system_bp)

    # Inject context toàn cục cho template Jinja2
    @app.context_processor
    def inject_global_vars():
        return {
            "current_year": datetime.now().year,
            "admin_user": session.get("admin_username", "Admin"),
            "is_logged_in": bool(session.get("admin_logged_in")),
            "app_title": "Face Attendance Admin",
        }

    # Trang lỗi 404
    @app.errorhandler(404)
    def page_not_found(e):
        return render_template("base.html", not_found=True), 404

    # Trang lỗi 500
    @app.errorhandler(500)
    def server_error(e):
        return render_template("base.html", server_error=True), 500

    return app


app = create_app()

if __name__ == "__main__":
    host = config.FLASK_HOST or "0.0.0.0"
    port = config.FLASK_PORT or 5000
    print("=" * 65)
    print("🚀 FLASK WEB ADMIN ĐANG KHỞI ĐỘNG...")
    print(f"🌐 Truy cập trên máy tính: http://localhost:{port} hoặc http://127.0.0.1:{port}")
    print(f"📱 Truy cập từ điện thoại/máy cùng Wi-Fi: http://<IP_MAY_TINH>:{port}")
    print("🔑 Tài khoản đăng nhập mặc định: admin / admin123")
    print("=" * 65)
    app.run(host=host, port=port, debug=False)
