import io
from datetime import datetime
from flask import Blueprint, flash, redirect, render_template, request, send_file, url_for

import attendance
import database
from web_admin.routes.auth import login_required

reports_bp = Blueprint("reports", __name__)


@reports_bp.route("/reports")
@login_required
def index():
    """Trang báo cáo chấm công và xuất Excel."""
    start_date = request.args.get("start_date", "").strip()
    end_date = request.args.get("end_date", "").strip()
    employee_code = request.args.get("employee_code", "").strip()

    # Mặc định chọn từ đầu tháng đến hiện tại
    if not start_date and not end_date:
        now = datetime.now()
        start_date = now.strftime("%Y-%m-01")
        end_date = now.strftime("%Y-%m-%d")

    departments = database.get_departments()
    employees = database.get_employees(status=None)

    # Lấy dữ liệu báo cáo xem trước
    df = attendance.get_attendance_report(
        start_date=start_date if start_date else None,
        end_date=end_date if end_date else None,
        employee_code=employee_code if employee_code else None,
    )

    records = df.to_dict(orient="records") if not df.empty else []

    return render_template(
        "reports.html",
        start_date=start_date,
        end_date=end_date,
        employee_code=employee_code,
        departments=departments,
        employees=employees,
        records=records,
        total_records=len(records),
    )


@reports_bp.route("/reports/export")
@login_required
def export_excel():
    """Xuất báo cáo chấm công ra file Excel (.xlsx)."""
    start_date = request.args.get("start_date", "").strip()
    end_date = request.args.get("end_date", "").strip()
    employee_code = request.args.get("employee_code", "").strip()

    try:
        df = attendance.get_attendance_report(
            start_date=start_date if start_date else None,
            end_date=end_date if end_date else None,
            employee_code=employee_code if employee_code else None,
        )

        output = io.BytesIO()
        with pd_excel_writer(output) as writer:
            df.to_excel(writer, index=False, sheet_name="Báo Cáo Chấm Công")

        output.seek(0)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"bao_cao_cham_cong_{timestamp}.xlsx"

        return send_file(
            output,
            as_attachment=True,
            download_name=filename,
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    except Exception as ex:
        flash(f"Lỗi khi xuất file Excel: {str(ex)}", "danger")
        return redirect(url_for("reports.index"))


def pd_excel_writer(buffer):
    """Helper tạo ExcelWriter với engine openpyxl."""
    import pandas as pd
    return pd.ExcelWriter(buffer, engine="openpyxl")
