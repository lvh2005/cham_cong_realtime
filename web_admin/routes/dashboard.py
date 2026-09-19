import json
from flask import Blueprint, render_template

import database
import faiss_utils
from web_admin.routes.auth import login_required

dashboard_bp = Blueprint("dashboard", __name__)


@dashboard_bp.route("/")
@dashboard_bp.route("/dashboard")
@login_required
def index():
    """Trang tổng quan Dashboard."""
    # 1. Thống kê số lượng tổng quan
    stats = database.get_dashboard_stats()

    # 2. Số lượng vectors trong FAISS
    engine = faiss_utils.get_faiss_engine()
    if engine.index is None:
        try:
            engine.init_engine()
        except Exception:
            pass
    faiss_total = engine.index.ntotal if engine.index is not None else 0
    stats["faiss_total"] = faiss_total

    # 3. Lịch sử chấm công mới nhất (kèm ảnh Cloudinary)
    recent_attendance = database.get_recent_attendance(limit=10)

    # 4. Dữ liệu biểu đồ chấm công theo giờ hôm nay
    hourly_data = database.get_hourly_attendance_today()

    # 5. Dữ liệu biểu đồ nhân viên theo phòng ban
    dept_data = database.get_department_employee_counts()

    return render_template(
        "dashboard.html",
        stats=stats,
        recent_attendance=recent_attendance,
        hourly_data_json=json.dumps(hourly_data),
        dept_data_json=json.dumps(dept_data),
    )
