import json
import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Thử import pyodbc nếu đã cài đặt
try:
    import pyodbc
except ImportError:
    pyodbc = None

CONFIG_FILE = "db_config.json"
DEFAULT_SQLITE_PATH = "attendance.db"

DEFAULT_CONFIG = {
    "db_type": "sqlserver",  # "sqlserver" hoặc "sqlite"
    "sqlite_path": DEFAULT_SQLITE_PATH,

    "sqlserver": {
        "server": "localhost",
        "database": "FaceAttendanceDB",
        "driver": "ODBC Driver 17 for SQL Server",
        "trusted_connection": True,
        "username": "sa",
        "password": "",
    },
}


def load_db_config() -> Dict[str, Any]:
    """Đọc cấu hình database từ file JSON."""
    if not os.path.exists(CONFIG_FILE):
        save_db_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG.copy()
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            # Merge with default in case keys are missing
            merged = DEFAULT_CONFIG.copy()
            merged.update(cfg)
            if "sqlserver" in cfg:
                merged["sqlserver"] = DEFAULT_CONFIG["sqlserver"].copy()
                merged["sqlserver"].update(cfg["sqlserver"])
            return merged
    except Exception:
        return DEFAULT_CONFIG.copy()


def save_db_config(cfg: Dict[str, Any]) -> None:
    """Lưu cấu hình database vào file JSON."""
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=4, ensure_ascii=False)


def get_available_sqlserver_drivers() -> List[str]:
    """Lấy danh sách các ODBC Driver của SQL Server có trên máy Windows."""
    if pyodbc is None:
        return []
    drivers = pyodbc.drivers()
    sql_drivers = [
        d for d in drivers
        if "sql server" in d.lower() or "odbc driver" in d.lower()
    ]
    return sql_drivers if sql_drivers else ["ODBC Driver 17 for SQL Server", "SQL Server"]


def build_sqlserver_connection_string(cfg_sql: Dict[str, Any]) -> str:
    """Tạo chuỗi kết nối pyodbc cho SQL Server."""
    server = cfg_sql.get("server", "localhost")
    database = cfg_sql.get("database", "FaceAttendanceDB")
    driver = cfg_sql.get("driver", "ODBC Driver 17 for SQL Server")
    trusted = cfg_sql.get("trusted_connection", True)
    username = cfg_sql.get("username", "sa")
    password = cfg_sql.get("password", "")

    if trusted:
        conn_str = (
            f"DRIVER={{{driver}}};"
            f"SERVER={server};"
            f"DATABASE={database};"
            "Trusted_Connection=yes;"
            "TrustServerCertificate=yes;"
            "Connection Timeout=5;"
        )
    else:
        conn_str = (
            f"DRIVER={{{driver}}};"
            f"SERVER={server};"
            f"DATABASE={database};"
            f"UID={username};"
            f"PWD={password};"
            "TrustServerCertificate=yes;"
            "Connection Timeout=5;"
        )
    return conn_str


def get_connection(custom_config: Optional[Dict[str, Any]] = None):
    """
    Tạo đối tượng kết nối tùy thuộc vào cấu hình (SQLite hoặc SQL Server).
    """
    config = custom_config or load_db_config()
    db_type = config.get("db_type", "sqlite").lower()

    if db_type == "sqlserver":
        if pyodbc is None:
            raise ImportError(
                "Chưa cài đặt thư viện 'pyodbc'. Vui lòng chạy 'pip install pyodbc' để dùng SQL Server."
            )
        conn_str = build_sqlserver_connection_string(config.get("sqlserver", {}))
        conn = pyodbc.connect(conn_str, timeout=5)
        return conn

    else:
        sqlite_path = config.get("sqlite_path", DEFAULT_SQLITE_PATH)
        conn = sqlite3.connect(sqlite_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn


def test_db_connection(custom_config: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
    """Kiểm tra thử kết nối tới database."""
    config = custom_config or load_db_config()
    db_type = config.get("db_type", "sqlite").lower()

    try:
        conn = get_connection(config)
        cur = conn.cursor()
        cur.execute("SELECT 1")
        cur.fetchone()
        conn.close()
        return True, f"Kết nối thành công tới {db_type.upper()}!"
    except Exception as exc:
        return False, f"Lỗi kết nối {db_type.upper()}: {str(exc)}"


def init_db(custom_config: Optional[Dict[str, Any]] = None):
    """Khởi tạo bảng cho SQLite hoặc SQL Server nếu chưa tồn tại."""
    config = custom_config or load_db_config()
    db_type = config.get("db_type", "sqlite").lower()

    conn = get_connection(config)
    cur = conn.cursor()

    if db_type == "sqlserver":
        cur.execute(
            """
            IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='employees' AND xtype='U')
            BEGIN
                CREATE TABLE employees (
                    id INT IDENTITY(1,1) PRIMARY KEY,
                    employee_code NVARCHAR(100) NOT NULL UNIQUE,
                    full_name NVARCHAR(255) NOT NULL,
                    face_encoding VARBINARY(MAX) NOT NULL,
                    created_at NVARCHAR(50) NOT NULL
                )
            END
            """
        )
        cur.execute(
            """
            IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='attendance' AND xtype='U')
            BEGIN
                CREATE TABLE attendance (
                    id INT IDENTITY(1,1) PRIMARY KEY,
                    employee_code NVARCHAR(100) NOT NULL,
                    check_in NVARCHAR(50) NOT NULL,
                    check_out NVARCHAR(50) NULL,
                    FOREIGN KEY(employee_code) REFERENCES employees(employee_code)
                )
            END
            """
        )
        cur.execute(
            """
            IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name='idx_attendance_employee_date')
            BEGIN
                CREATE INDEX idx_attendance_employee_date
                ON attendance(employee_code, check_in)
            END
            """
        )
    else:
        # SQLite
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS employees (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_code TEXT NOT NULL UNIQUE,
                full_name TEXT NOT NULL,
                face_encoding BLOB NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_code TEXT NOT NULL,
                check_in TEXT NOT NULL,
                check_out TEXT,
                FOREIGN KEY(employee_code) REFERENCES employees(employee_code)
            )
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_attendance_employee_date
            ON attendance(employee_code, check_in)
            """
        )

    conn.commit()
    conn.close()


def add_employee(employee_code: str, full_name: str, encoding: np.ndarray):
    """Lưu nhân viên và face encoding dạng BLOB/VARBINARY."""
    encoding = np.asarray(encoding, dtype=np.float64)
    blob = encoding.tobytes()

    config = load_db_config()
    db_type = config.get("db_type", "sqlite").lower()
    binary_data = pyodbc.Binary(blob) if (db_type == "sqlserver" and pyodbc is not None) else blob

    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO employees (employee_code, full_name, face_encoding, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (
            employee_code,
            full_name,
            binary_data,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        ),
    )
    conn.commit()
    conn.close()



def get_employees() -> List[Dict[str, Any]]:
    """Lấy danh sách tất cả nhân viên."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT employee_code, full_name, created_at
        FROM employees
        ORDER BY employee_code
        """
    )
    rows = cur.fetchall()

    results = []
    for row in rows:
        results.append({
            "employee_code": str(row[0]),
            "full_name": str(row[1]),
            "created_at": str(row[2]) if row[2] else "",
        })
    conn.close()
    return results


def get_employee(employee_code: str) -> Optional[Dict[str, Any]]:
    """Lấy thông tin 1 nhân viên theo mã."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT employee_code, full_name, face_encoding, created_at
        FROM employees
        WHERE employee_code = ?
        """,
        (employee_code,),
    )
    row = cur.fetchone()
    conn.close()

    if row is None:
        return None

    encoding_bytes = bytes(row[2])
    return {
        "employee_code": str(row[0]),
        "full_name": str(row[1]),
        "face_encoding": np.frombuffer(encoding_bytes, dtype=np.float64),
        "created_at": str(row[3]) if row[3] else "",
    }


def load_all_face_encodings() -> Tuple[List[np.ndarray], List[str], List[str]]:
    """Load toàn bộ vector encoding vào RAM khi hệ thống khởi động."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT employee_code, full_name, face_encoding
        FROM employees
        ORDER BY employee_code
        """
    )
    rows = cur.fetchall()
    conn.close()

    encodings = []
    codes = []
    names = []

    for row in rows:
        encoding_bytes = bytes(row[2])
        encoding = np.frombuffer(encoding_bytes, dtype=np.float64)

        # Encoding hợp lệ của face_recognition có 128 chiều
        if encoding.shape == (128,):
            encodings.append(encoding)
            codes.append(str(row[0]))
            names.append(str(row[1]))

    return encodings, codes, names


def create_check_in(employee_code: str, check_in: str) -> int:
    """Tạo bản ghi check-in mới."""
    config = load_db_config()
    db_type = config.get("db_type", "sqlite").lower()

    conn = get_connection()
    cur = conn.cursor()

    if db_type == "sqlserver":
        cur.execute(
            """
            INSERT INTO attendance (employee_code, check_in)
            OUTPUT INSERTED.id
            VALUES (?, ?)
            """,
            (employee_code, check_in),
        )
        row = cur.fetchone()
        attendance_id = int(row[0]) if row and row[0] is not None else 0

    else:
        cur.execute(
            """
            INSERT INTO attendance (employee_code, check_in)
            VALUES (?, ?)
            """,
            (employee_code, check_in),
        )
        attendance_id = cur.lastrowid

    conn.commit()
    conn.close()
    return attendance_id


def create_check_out(attendance_id: int, check_out: str):
    """Cập nhật giờ check-out cho bản ghi chấm công."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE attendance
        SET check_out = ?
        WHERE id = ?
        """,
        (check_out, attendance_id),
    )
    conn.commit()
    conn.close()


def get_open_attendance(employee_code: str) -> Optional[Dict[str, Any]]:
    """Lấy phiên chấm công chưa có checkout gần nhất."""
    config = load_db_config()
    db_type = config.get("db_type", "sqlite").lower()

    conn = get_connection()
    cur = conn.cursor()

    if db_type == "sqlserver":
        cur.execute(
            """
            SELECT TOP 1 id, employee_code, check_in, check_out
            FROM attendance
            WHERE employee_code = ?
              AND check_out IS NULL
            ORDER BY id DESC
            """,
            (employee_code,),
        )
    else:
        cur.execute(
            """
            SELECT id, employee_code, check_in, check_out
            FROM attendance
            WHERE employee_code = ?
              AND check_out IS NULL
            ORDER BY id DESC
            LIMIT 1
            """,
            (employee_code,),
        )

    row = cur.fetchone()
    conn.close()

    if not row:
        return None

    return {
        "id": row[0],
        "employee_code": str(row[1]),
        "check_in": str(row[2]),
        "check_out": str(row[3]) if row[3] else None,
    }


def get_last_attendance(employee_code: str) -> Optional[Dict[str, Any]]:
    """Lấy bản ghi chấm công gần nhất của nhân viên."""
    config = load_db_config()
    db_type = config.get("db_type", "sqlite").lower()

    conn = get_connection()
    cur = conn.cursor()

    if db_type == "sqlserver":
        cur.execute(
            """
            SELECT TOP 1 id, employee_code, check_in, check_out
            FROM attendance
            WHERE employee_code = ?
            ORDER BY id DESC
            """,
            (employee_code,),
        )
    else:
        cur.execute(
            """
            SELECT id, employee_code, check_in, check_out
            FROM attendance
            WHERE employee_code = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (employee_code,),
        )

    row = cur.fetchone()
    conn.close()

    if not row:
        return None

    return {
        "id": row[0],
        "employee_code": str(row[1]),
        "check_in": str(row[2]),
        "check_out": str(row[3]) if row[3] else None,
    }


def delete_employee(employee_code: str):
    """Xóa nhân viên và toàn bộ lịch sử chấm công tương ứng."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM attendance WHERE employee_code = ?", (employee_code,))
    cur.execute("DELETE FROM employees WHERE employee_code = ?", (employee_code,))
    conn.commit()
    conn.close()

