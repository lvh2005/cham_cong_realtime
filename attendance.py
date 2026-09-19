import threading
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd

import config
from database import (
    create_check_in,
    create_check_out,
    get_connection,
    get_employee,
    get_last_attendance,
    get_open_attendance,
)

_ATTENDANCE_LOCK = threading.RLock()
_IN_MEMORY_COOLDOWN: Dict[str, float] = {}  # {employee_code: timestamp}
_COOLDOWN_LOCK = threading.Lock()


def is_in_cooldown(employee_code: str) -> bool:
    """Kiểm tra nhanh trong RAM xem nhân viên có đang trong thời gian cooldown không (0ms)."""
    with _COOLDOWN_LOCK:
        if employee_code in _IN_MEMORY_COOLDOWN:
            elapsed = datetime.now().timestamp() - _IN_MEMORY_COOLDOWN[employee_code]
            return elapsed < config.ATTENDANCE_COOLDOWN_SECONDS
    return False


def set_in_memory_cooldown(employee_code: str, timestamp: Optional[float] = None) -> None:
    """Ghi nhận timestamp cooldown vào RAM."""
    ts = timestamp if timestamp is not None else datetime.now().timestamp()
    with _COOLDOWN_LOCK:
        _IN_MEMORY_COOLDOWN[employee_code] = ts


def register_attendance(employee_code: str) -> Dict[str, Any]:
    """
    Tự động ghi nhận chấm công (Check-in / Check-out) trên Microsoft SQL Server:
    - Nếu chưa có phiên mở -> Check-in.
    - Nếu đã có phiên mở -> Check-out.
    - Áp dụng Cooldown RAM + Database từ config.ATTENDANCE_COOLDOWN_SECONDS chống chấm công liên tục.
    """
    with _ATTENDANCE_LOCK:
        now = datetime.now()
        now_ts = now.timestamp()
        now_str = now.strftime("%Y-%m-%d %H:%M:%S")
        cooldown_sec = config.ATTENDANCE_COOLDOWN_SECONDS

        # 1. Kiểm tra cache RAM trước (0ms, không tốn tài nguyên DB)
        with _COOLDOWN_LOCK:
            if employee_code in _IN_MEMORY_COOLDOWN:
                elapsed = now_ts - _IN_MEMORY_COOLDOWN[employee_code]
                if elapsed < cooldown_sec:
                    return {
                        "success": False,
                        "cooldown": True,
                        "message": f"Đang trong thời gian chống chấm công trùng (còn {int(cooldown_sec - elapsed)}s).",
                    }

        employee = get_employee(employee_code)
        if employee is None:
            return {
                "success": False,
                "message": "Không tìm thấy thông tin nhân viên trong hệ thống.",
            }

        if employee.get("status") == "INACTIVE":
            return {
                "success": False,
                "message": "Nhân viên này đã ngưng hoạt động (INACTIVE).",
            }

        employee_name = employee["full_name"]
        last = get_last_attendance(employee_code)

        # 2. Kiểm tra Cooldown từ CSDL SQL Server
        if last:
            last_time = datetime.strptime(last["check_in"], "%Y-%m-%d %H:%M:%S")
            if last["check_out"]:
                last_time = datetime.strptime(last["check_out"], "%Y-%m-%d %H:%M:%S")

            if (now - last_time).total_seconds() < cooldown_sec:
                _IN_MEMORY_COOLDOWN[employee_code] = last_time.timestamp()
                return {
                    "success": False,
                    "cooldown": True,
                    "message": f"Đang trong thời gian chống chấm công trùng ({cooldown_sec}s).",
                }

        open_record = get_open_attendance(employee_code)

        if open_record is None:
            attendance_id = create_check_in(employee_code, now_str)
            _IN_MEMORY_COOLDOWN[employee_code] = now_ts
            return {
                "success": True,
                "type": "check-in",
                "attendance_id": attendance_id,
                "employee_code": employee_code,
                "employee_name": employee_name,
                "time": now_str,
            }

        create_check_out(open_record["id"], now_str)
        _IN_MEMORY_COOLDOWN[employee_code] = now_ts
        return {
            "success": True,
            "type": "check-out",
            "attendance_id": open_record["id"],
            "employee_code": employee_code,
            "employee_name": employee_name,
            "time": now_str,
        }


def get_attendance_report(
    start_date=None,
    end_date=None,
    employee_code: Optional[str] = None,
) -> pd.DataFrame:
    """
    Truy vấn lịch sử chấm công từ Microsoft SQL Server và trả về DataFrame:
    Mã NV | Họ tên | Ngày | Giờ vào | Giờ ra | Tổng giờ làm
    """
    conn = get_connection()

    query = """
        SELECT
            a.employee_code AS employee_code,
            e.full_name AS full_name,
            a.check_in AS check_in,
            a.check_out AS check_out
        FROM attendance a
        JOIN employees e
          ON e.employee_code = a.employee_code
        WHERE 1 = 1
    """
    params = []

    if start_date is not None:
        query += " AND a.check_in >= ?"
        params.append(f"{start_date} 00:00:00")

    if end_date is not None:
        query += " AND a.check_in <= ?"
        params.append(f"{end_date} 23:59:59")

    if employee_code:
        query += " AND a.employee_code = ?"
        params.append(employee_code)

    query += " ORDER BY a.check_in DESC"

    cur = conn.cursor()
    cur.execute(query, params)
    rows = cur.fetchall()

    if cur.description:
        cols = [col[0] for col in cur.description]
    else:
        cols = ["employee_code", "full_name", "check_in", "check_out"]

    data = [[val for val in row] for row in rows]
    df = pd.DataFrame(data, columns=cols)
    conn.close()

    if df.empty:
        return pd.DataFrame(
            columns=[
                "Mã NV",
                "Họ tên",
                "Ngày",
                "Giờ vào",
                "Giờ ra",
                "Tổng giờ làm",
            ]
        )

    df["check_in_dt"] = pd.to_datetime(df["check_in"], errors="coerce")
    df["check_out_dt"] = pd.to_datetime(df["check_out"], errors="coerce")

    df["work_hours"] = (
        (df["check_out_dt"] - df["check_in_dt"]).dt.total_seconds() / 3600
    )

    df["Ngày"] = df["check_in_dt"].dt.strftime("%d/%m/%Y")
    df["Giờ vào"] = df["check_in_dt"].dt.strftime("%H:%M:%S")
    df["Giờ ra"] = df["check_out_dt"].dt.strftime("%H:%M:%S")

    # Với phiên chưa checkout, hiển thị trống
    df.loc[df["check_out_dt"].isna(), "Giờ ra"] = ""

    df["Tổng giờ làm"] = df["work_hours"].apply(
        lambda x: f"{x:.2f}" if pd.notna(x) else ""
    )

    result = df[
        [
            "employee_code",
            "full_name",
            "Ngày",
            "Giờ vào",
            "Giờ ra",
            "Tổng giờ làm",
        ]
    ].copy()

    result.columns = [
        "Mã NV",
        "Họ tên",
        "Ngày",
        "Giờ vào",
        "Giờ ra",
        "Tổng giờ làm",
    ]

    return result
