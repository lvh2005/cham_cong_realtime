from datetime import datetime
from flask import Blueprint, render_template, request

import database
from web_admin.routes.auth import login_required

attendance_bp = Blueprint("attendance", __name__)


@attendance_bp.route("/attendance")
@login_required
def index():
    """Trang xem và lọc lịch sử chấm công phân trang."""
    page = request.args.get("page", 1, type=int)
    start_date = request.args.get("start_date", "").strip()
    end_date = request.args.get("end_date", "").strip()
    search = request.args.get("search", "").strip()
    department = request.args.get("department", "").strip()

    # Mặc định lọc ngày hôm nay nếu chưa chọn ngày
    if not start_date and not end_date and not search and not department:
        start_date = datetime.now().strftime("%Y-%m-%d")
        end_date = start_date

    per_page = 20
    logs, total_count, total_pages = database.get_attendance_paginated(
        page=page,
        per_page=per_page,
        start_date=start_date,
        end_date=end_date,
        search=search,
        department=department,
    )
    departments = database.get_departments()

    return render_template(
        "attendance.html",
        logs=logs,
        page=page,
        total_pages=total_pages,
        total_count=total_count,
        start_date=start_date,
        end_date=end_date,
        search=search,
        department=department,
        departments=departments,
    )
