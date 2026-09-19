from flask import Blueprint, flash, redirect, render_template, request, url_for

import cloudinary_utils
import database
import faiss_utils
from web_admin.routes.auth import login_required

employees_bp = Blueprint("employees", __name__)


@employees_bp.route("/employees")
@login_required
def index():
    """Danh sách nhân viên (có phân trang, tìm kiếm, lọc trạng thái và phòng ban)."""
    page = request.args.get("page", 1, type=int)
    search = request.args.get("search", "").strip()
    status = request.args.get("status", "").strip()
    department = request.args.get("department", "").strip()

    per_page = 15
    employees, total_count, total_pages = database.get_employees_paginated(
        page=page,
        per_page=per_page,
        search=search,
        status=status,
        department=department,
    )
    departments = database.get_departments()

    return render_template(
        "employees.html",
        employees=employees,
        page=page,
        total_pages=total_pages,
        total_count=total_count,
        search=search,
        status=status,
        department=department,
        departments=departments,
    )


@employees_bp.route("/employees/create", methods=["GET", "POST"])
@login_required
def create():
    """Tạo mới hồ sơ nhân viên từ Web Admin (Trạng thái ban đầu: NOT REGISTERED khuôn mặt)."""
    departments = database.get_departments()

    if request.method == "POST":
        employee_code = request.form.get("employee_code", "").strip().upper()
        full_name = request.form.get("full_name", "").strip()
        department = request.form.get("department", "").strip()
        position = request.form.get("position", "").strip()

        if not employee_code or not full_name:
            flash("Vui lòng điền đầy đủ Mã nhân viên và Họ tên!", "danger")
            return render_template(
                "employee_create.html",
                departments=departments,
                employee_code=employee_code,
                full_name=full_name,
                department=department,
                position=position,
            )

        success, msg, emp_id = database.create_employee_web(
            employee_code=employee_code,
            full_name=full_name,
            department=department,
            position=position,
        )

        if success and emp_id:
            flash(
                f"Đã tạo hồ sơ nhân viên '{full_name}' ({employee_code}) thành công! "
                "Nhân viên có thể đến máy chấm công Tkinter để đăng ký khuôn mặt.",
                "success",
            )
            return redirect(url_for("employees.detail", employee_id=emp_id))
        else:
            flash(msg or "Không thể tạo nhân viên.", "danger")
            return render_template(
                "employee_create.html",
                departments=departments,
                employee_code=employee_code,
                full_name=full_name,
                department=department,
                position=position,
            )

    return render_template("employee_create.html", departments=departments)


@employees_bp.route("/employees/<int:employee_id>")
@login_required
def detail(employee_id: int):
    """Trang chi tiết hồ sơ nhân viên."""
    data = database.get_employee_detail(employee_id)
    if not data:
        flash("Không tìm thấy nhân viên trong hệ thống.", "danger")
        return redirect(url_for("employees.index"))

    return render_template(
        "employee_detail.html",
        employee=data["employee"],
        embeddings=data["embeddings"],
        embedding_count=data["embedding_count"],
        recent_attendance=data["recent_attendance"],
    )


@employees_bp.route("/employees/<int:employee_id>/edit", methods=["GET", "POST"])
@login_required
def edit(employee_id: int):
    """Sửa thông tin nhân viên."""
    emp = database.get_employee_by_id(employee_id)
    if not emp:
        flash("Không tìm thấy nhân viên trong hệ thống.", "danger")
        return redirect(url_for("employees.index"))

    departments = database.get_departments()

    if request.method == "POST":
        employee_code = request.form.get("employee_code", "").strip().upper()
        full_name = request.form.get("full_name", "").strip()
        department = request.form.get("department", "").strip()
        position = request.form.get("position", "").strip()
        status = request.form.get("status", "ACTIVE").strip().upper()

        success, msg = database.update_employee(
            employee_id=employee_id,
            employee_code=employee_code,
            full_name=full_name,
            department=department,
            position=position,
            status=status,
        )

        if success:
            # Đồng bộ lại FAISS nếu có thay đổi trạng thái
            try:
                engine = faiss_utils.get_faiss_engine()
                engine.rebuild_from_database()
            except Exception:
                pass

            flash("Cập nhật thông tin nhân viên thành công!", "success")
            return redirect(url_for("employees.detail", employee_id=employee_id))
        else:
            flash(msg or "Lỗi khi cập nhật nhân viên.", "danger")

    return render_template(
        "employee_edit.html",
        employee=emp,
        departments=departments,
    )


@employees_bp.route("/employees/<int:employee_id>/deactivate", methods=["POST"])
@login_required
def deactivate(employee_id: int):
    """Vô hiệu hóa nhân viên (Soft Delete)."""
    emp = database.get_employee_by_id(employee_id)
    if not emp:
        flash("Không tìm thấy nhân viên.", "danger")
        return redirect(url_for("employees.index"))

    success = database.soft_delete_employee(emp["employee_code"])
    if success:
        # Cập nhật FAISS Index
        try:
            engine = faiss_utils.get_faiss_engine()
            engine.remove_employee(employee_id)
        except Exception:
            pass
        flash(f"Đã vô hiệu hóa nhân viên '{emp['full_name']}' thành công.", "warning")
    else:
        flash("Không thể vô hiệu hóa nhân viên.", "danger")

    return redirect(url_for("employees.detail", employee_id=employee_id))


@employees_bp.route("/employees/<int:employee_id>/activate", methods=["POST"])
@login_required
def activate(employee_id: int):
    """Kích hoạt lại nhân viên."""
    emp = database.get_employee_by_id(employee_id)
    if not emp:
        flash("Không tìm thấy nhân viên.", "danger")
        return redirect(url_for("employees.index"))

    success = database.activate_employee(emp["employee_code"])
    if success:
        # Rebuild lại FAISS Index để đưa các vector của nhân viên trở lại
        try:
            engine = faiss_utils.get_faiss_engine()
            engine.rebuild_from_database()
        except Exception:
            pass
        flash(f"Đã kích hoạt lại nhân viên '{emp['full_name']}' thành công.", "success")
    else:
        flash("Không thể kích hoạt lại nhân viên.", "danger")

    return redirect(url_for("employees.detail", employee_id=employee_id))


@employees_bp.route("/employees/<int:employee_id>/delete", methods=["POST"])
@login_required
def delete(employee_id: int):
    """Xóa vĩnh viễn nhân viên (Hard Delete)."""
    emp = database.get_employee_by_id(employee_id)
    if not emp:
        flash("Không tìm thấy nhân viên.", "danger")
        return redirect(url_for("employees.index"))

    success, cloudinary_pid = database.hard_delete_employee(emp["employee_code"])
    if success:
        # Xóa avatar trên Cloudinary nếu có
        if cloudinary_pid:
            cloudinary_utils.delete_employee_avatar(cloudinary_pid)

        # Rebuild lại FAISS Index
        try:
            engine = faiss_utils.get_faiss_engine()
            engine.rebuild_from_database()
        except Exception:
            pass

        flash(f"Đã xóa vĩnh viễn nhân viên '{emp['full_name']}' khỏi hệ thống!", "success")
        return redirect(url_for("employees.index"))
    else:
        flash("Không thể xóa nhân viên.", "danger")
        return redirect(url_for("employees.detail", employee_id=employee_id))
