import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Import pyodbc (Bắt buộc cho Microsoft SQL Server)
try:
    import pyodbc
except ImportError:
    pyodbc = None

import config


def serialize_embedding(embedding: np.ndarray) -> bytes:
    """
    Chuyển đổi vector khuôn mặt sang mảng byte float32 (128 chiều = 512 bytes).
    Đảm bảo vector được chuẩn hóa L2 norm trước khi lưu.
    """
    vec = np.asarray(embedding, dtype=np.float32).flatten()
    if vec.shape != (128,):
        raise ValueError(f"Vector khuôn mặt phải có đúng 128 chiều, hiện tại: {vec.shape}")

    # Chuẩn hóa L2 norm
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm

    return vec.tobytes()


def deserialize_embedding(blob: Any) -> np.ndarray:
    """
    Chuyển đổi dữ liệu nhị phân từ SQL Server sang mảng NumPy float32 (128 chiều).
    Tự động tương thích ngược với vector float64 (1024 bytes) từ phiên bản cũ.
    """
    if blob is None:
        return np.zeros((128,), dtype=np.float32)

    raw_bytes = bytes(blob)
    if len(raw_bytes) == 1024:
        # Legacy float64 (128 * 8 bytes)
        arr = np.frombuffer(raw_bytes, dtype=np.float64).astype(np.float32)
    elif len(raw_bytes) == 512:
        # Standard float32 (128 * 4 bytes)
        arr = np.frombuffer(raw_bytes, dtype=np.float32)
    else:
        # Thử ép kiểu float32
        arr = np.frombuffer(raw_bytes, dtype=np.float32)

    if arr.shape != (128,):
        arr = arr[:128] if len(arr) >= 128 else np.pad(arr, (0, 128 - len(arr)))

    # Chuẩn hóa L2 norm
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr = arr / norm

    return np.ascontiguousarray(arr, dtype=np.float32)


def load_db_config() -> Dict[str, Any]:
    """Đọc cấu hình database từ file JSON / config.py."""
    if not os.path.exists(config.DB_CONFIG_FILE):
        default_cfg = {
            "db_type": "sqlserver",
            "sqlserver": config.get_db_dict(),
        }
        save_db_config(default_cfg)
        return default_cfg
    try:
        with open(config.DB_CONFIG_FILE, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            return cfg
    except Exception:
        return {
            "db_type": "sqlserver",
            "sqlserver": config.get_db_dict(),
        }


def save_db_config(cfg: Dict[str, Any]) -> None:
    """Lưu cấu hình database vào file JSON và đồng bộ biến môi trường."""
    with open(config.DB_CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=4, ensure_ascii=False)


def get_available_sqlserver_drivers() -> List[str]:
    """Lấy danh sách các ODBC Driver của SQL Server có trên máy Windows."""
    if pyodbc is None:
        return ["ODBC Driver 17 for SQL Server", "SQL Server"]
    try:
        drivers = pyodbc.drivers()
        sql_drivers = [
            d for d in drivers
            if "sql server" in d.lower() or "odbc driver" in d.lower()
        ]
        return sql_drivers if sql_drivers else ["ODBC Driver 17 for SQL Server", "SQL Server"]
    except Exception:
        return ["ODBC Driver 17 for SQL Server", "SQL Server"]


def build_sqlserver_connection_string(cfg_sql: Optional[Dict[str, Any]] = None) -> str:
    """Tạo chuỗi kết nối pyodbc cho Microsoft SQL Server."""
    if cfg_sql is None:
        cfg_sql = config.get_db_dict()

    server = cfg_sql.get("server") or config.DB_SERVER or "localhost"
    database = cfg_sql.get("database") or config.DB_DATABASE or "FaceAttendanceDB"
    driver = cfg_sql.get("driver") or config.DB_DRIVER or "ODBC Driver 17 for SQL Server"
    trusted = cfg_sql.get("trusted_connection", True)
    username = cfg_sql.get("username") or config.DB_USER or "sa"
    password = cfg_sql.get("password") or config.DB_PASSWORD or ""

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
    Tạo đối tượng kết nối pyodbc tới Microsoft SQL Server.
    """
    if pyodbc is None:
        raise ImportError(
            "Chưa cài đặt thư viện 'pyodbc'. Vui lòng chạy 'pip install pyodbc' để kết nối SQL Server."
        )

    if custom_config:
        sql_cfg = custom_config.get("sqlserver", custom_config)
    else:
        cfg = load_db_config()
        sql_cfg = cfg.get("sqlserver", config.get_db_dict())

    conn_str = build_sqlserver_connection_string(sql_cfg)
    conn = pyodbc.connect(conn_str, timeout=5)
    return conn


def test_db_connection(custom_config: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
    """Kiểm tra thử kết nối tới Microsoft SQL Server."""
    try:
        conn = get_connection(custom_config)
        cur = conn.cursor()
        cur.execute("SELECT 1")
        cur.fetchone()
        conn.close()
        return True, "Kết nối thành công tới Microsoft SQL Server!"
    except Exception as exc:
        return False, f"Lỗi kết nối SQL Server: {str(exc)}"


def init_db(custom_config: Optional[Dict[str, Any]] = None):
    """
    Khởi tạo và Migration Database trên Microsoft SQL Server.
    Tạo các bảng employees, face_embeddings, attendance và tự động chuyển đổi dữ liệu cũ.
    """
    conn = get_connection(custom_config)
    cur = conn.cursor()

    # 1. Bảng employees
    cur.execute(
        """
        IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='employees' AND xtype='U')
        BEGIN
            CREATE TABLE employees (
                id INT IDENTITY(1,1) PRIMARY KEY,
                employee_code NVARCHAR(100) NOT NULL UNIQUE,
                full_name NVARCHAR(255) NOT NULL,
                department NVARCHAR(100) NULL,
                position NVARCHAR(100) NULL,
                image_url NVARCHAR(500) NULL,
                cloudinary_public_id NVARCHAR(200) NULL,
                status NVARCHAR(50) NOT NULL DEFAULT 'ACTIVE',
                face_encoding VARBINARY(MAX) NULL,
                created_at NVARCHAR(50) NOT NULL,
                updated_at NVARCHAR(50) NULL
            )
        END
        """
    )
    conn.commit()

    # Migration các cột mới cho employees nếu đã tồn tại bảng cũ
    migration_columns = [
        ("department", "NVARCHAR(100) NULL"),
        ("position", "NVARCHAR(100) NULL"),
        ("image_url", "NVARCHAR(500) NULL"),
        ("cloudinary_public_id", "NVARCHAR(200) NULL"),
        ("status", "NVARCHAR(50) NOT NULL DEFAULT 'ACTIVE'"),
        ("updated_at", "NVARCHAR(50) NULL"),
    ]
    for col_name, col_type in migration_columns:
        cur.execute(
            f"""
            IF NOT EXISTS (SELECT * FROM syscolumns WHERE id = OBJECT_ID('employees') AND name = '{col_name}')
            BEGIN
                ALTER TABLE employees ADD {col_name} {col_type}
            END
            """
        )
    conn.commit()

    # Cho phép face_encoding NULL
    cur.execute(
        """
        IF EXISTS (SELECT * FROM syscolumns WHERE id = OBJECT_ID('employees') AND name = 'face_encoding')
        BEGIN
            ALTER TABLE employees ALTER COLUMN face_encoding VARBINARY(MAX) NULL
        END
        """
    )
    conn.commit()

    # 2. Bảng face_embeddings (Hỗ trợ 3-5 vector/nhân viên, tỉ lệ 30.000 - 50.000 vectors)
    cur.execute(
        """
        IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='face_embeddings' AND xtype='U')
        BEGIN
            CREATE TABLE face_embeddings (
                face_id BIGINT IDENTITY(1,1) PRIMARY KEY,
                employee_id INT NOT NULL,
                embedding VARBINARY(MAX) NOT NULL,
                is_active BIT NOT NULL DEFAULT 1,
                created_at DATETIME2 NOT NULL DEFAULT SYSDATETIME(),
                CONSTRAINT FK_face_embeddings_employees FOREIGN KEY (employee_id)
                    REFERENCES employees(id)
                    ON DELETE CASCADE
            )
        END
        """
    )
    conn.commit()

    # Tạo index cho face_embeddings
    cur.execute(
        """
        IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name='idx_face_embeddings_emp_active')
        BEGIN
            CREATE INDEX idx_face_embeddings_emp_active
            ON face_embeddings(employee_id, is_active)
        END
        """
    )
    conn.commit()

    # 3. Migration dữ liệu face_encoding cũ từ employees sang face_embeddings
    cur.execute(
        """
        IF EXISTS (SELECT 1 FROM employees WHERE face_encoding IS NOT NULL)
        BEGIN
            INSERT INTO face_embeddings (employee_id, embedding, is_active, created_at)
            SELECT e.id, e.face_encoding, 1, SYSDATETIME()
            FROM employees e
            WHERE e.face_encoding IS NOT NULL
              AND NOT EXISTS (
                  SELECT 1 FROM face_embeddings fe WHERE fe.employee_id = e.id
              )
        END
        """
    )
    conn.commit()

    # 4. Bảng attendance
    cur.execute(
        """
        IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='attendance' AND xtype='U')
        BEGIN
            CREATE TABLE attendance (
                id INT IDENTITY(1,1) PRIMARY KEY,
                employee_code NVARCHAR(100) NOT NULL,
                check_in NVARCHAR(50) NOT NULL,
                check_out NVARCHAR(50) NULL,
                CONSTRAINT FK_attendance_employees FOREIGN KEY (employee_code)
                    REFERENCES employees(employee_code)
                    ON DELETE NO ACTION
                    ON UPDATE CASCADE
            )
        END
        """
    )
    conn.commit()

    # Tạo index cho attendance
    cur.execute(
        """
        IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name='idx_attendance_employee_date')
        BEGIN
            CREATE INDEX idx_attendance_employee_date
            ON attendance(employee_code, check_in)
        END
        """
    )
    conn.commit()
    conn.close()


def add_employee(
    employee_code: str,
    full_name: str,
    encoding: Optional[np.ndarray] = None,
    department: str = "",
    position: str = "",
    image_url: str = "",
    cloudinary_public_id: str = "",
) -> int:
    """
    Thêm thông tin nhân viên mới vào bảng employees và trả về employee_id (Primary Key).
    """
    conn = get_connection()
    cur = conn.cursor()

    legacy_binary = pyodbc.Binary(serialize_embedding(encoding)) if encoding is not None else None
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cur.execute(
        """
        INSERT INTO employees (
            employee_code, full_name, department, position,
            image_url, cloudinary_public_id, status, face_encoding, created_at
        )
        OUTPUT INSERTED.id
        VALUES (?, ?, ?, ?, ?, ?, 'ACTIVE', ?, ?)
        """,
        (
            employee_code.strip().upper(),
            full_name.strip(),
            department.strip(),
            position.strip(),
            image_url.strip(),
            cloudinary_public_id.strip(),
            legacy_binary,
            now_str,
        ),
    )
    row = cur.fetchone()
    emp_id = int(row[0]) if row and row[0] is not None else 0

    # Nếu có truyền kèm encoding đơn lẻ -> tự động lưu vào face_embeddings
    if emp_id > 0 and encoding is not None:
        add_face_embeddings(emp_id, [encoding], conn_to_reuse=conn)

    conn.commit()
    conn.close()
    return emp_id


def add_face_embeddings(
    employee_id: int,
    embeddings: List[np.ndarray],
    conn_to_reuse=None,
) -> List[int]:
    """
    Lưu danh sách 3-5 vector embeddings vào bảng face_embeddings.
    Trả về danh sách các face_id mới được sinh ra.
    """
    if not embeddings:
        return []

    should_close = False
    if conn_to_reuse is None:
        conn = get_connection()
        should_close = True
    else:
        conn = conn_to_reuse

    cur = conn.cursor()
    created_face_ids = []

    for emb in embeddings:
        binary_data = pyodbc.Binary(serialize_embedding(emb))
        cur.execute(
            """
            INSERT INTO face_embeddings (employee_id, embedding, is_active)
            OUTPUT INSERTED.face_id
            VALUES (?, ?, 1)
            """,
            (employee_id, binary_data),
        )
        row = cur.fetchone()
        if row and row[0] is not None:
            created_face_ids.append(int(row[0]))

    if should_close:
        conn.commit()
        conn.close()

    return created_face_ids


def get_employees(status: Optional[str] = "ACTIVE") -> List[Dict[str, Any]]:
    """Lấy danh sách nhân viên theo trạng thái ('ACTIVE', 'INACTIVE' hoặc None để lấy tất cả)."""
    conn = get_connection()
    cur = conn.cursor()

    if status:
        cur.execute(
            """
            SELECT id, employee_code, full_name, department, position, image_url, cloudinary_public_id, status, created_at
            FROM employees
            WHERE status = ?
            ORDER BY employee_code
            """,
            (status,),
        )
    else:
        cur.execute(
            """
            SELECT id, employee_code, full_name, department, position, image_url, cloudinary_public_id, status, created_at
            FROM employees
            ORDER BY employee_code
            """
        )
    rows = cur.fetchall()
    conn.close()

    results = []
    for row in rows:
        results.append({
            "id": int(row[0]),
            "employee_code": str(row[1]),
            "full_name": str(row[2]),
            "department": str(row[3]) if row[3] else "",
            "position": str(row[4]) if row[4] else "",
            "image_url": str(row[5]) if row[5] else "",
            "cloudinary_public_id": str(row[6]) if row[6] else "",
            "status": str(row[7]) if row[7] else "ACTIVE",
            "created_at": str(row[8]) if row[8] else "",
        })
    return results


def get_employees_with_stats(status: Optional[str] = "ACTIVE") -> List[Dict[str, Any]]:
    """
    Lấy danh sách nhân viên kèm số lượng vector khuôn mặt (embedding_count).
    Phục vụ chọn nhân viên đăng ký/cập nhật khuôn mặt trên Tkinter.
    """
    conn = get_connection()
    cur = conn.cursor()

    query = """
        SELECT
            e.id,
            e.employee_code,
            e.full_name,
            e.department,
            e.position,
            e.image_url,
            e.cloudinary_public_id,
            e.status,
            e.created_at,
            (
                SELECT COUNT(*)
                FROM face_embeddings fe
                WHERE fe.employee_id = e.id AND fe.is_active = 1
            ) AS embedding_count
        FROM employees e
    """
    params = []
    if status:
        query += " WHERE e.status = ?"
        params.append(status)

    query += " ORDER BY e.employee_code ASC"

    cur.execute(query, params)
    rows = cur.fetchall()
    conn.close()

    results = []
    for row in rows:
        results.append({
            "id": int(row[0]),
            "employee_code": str(row[1]),
            "full_name": str(row[2]),
            "department": str(row[3]) if row[3] else "",
            "position": str(row[4]) if row[4] else "",
            "image_url": str(row[5]) if row[5] else "",
            "cloudinary_public_id": str(row[6]) if row[6] else "",
            "status": str(row[7]) if row[7] else "ACTIVE",
            "created_at": str(row[8]) if row[8] else "",
            "embedding_count": int(row[9] or 0),
        })
    return results


def update_employee_face(
    employee_id: int,
    embeddings: List[np.ndarray],
    image_url: str = "",
    cloudinary_public_id: str = "",
) -> List[int]:
    """
    Cập nhật / Nạp khuôn mặt cho nhân viên đã tồn tại trong CSDL:
    1. Vô hiệu hóa (is_active = 0) toàn bộ vector cũ của nhân viên.
    2. Thêm danh sách các vector mới vào bảng face_embeddings (is_active = 1).
    3. Cập nhật image_url & cloudinary_public_id trong bảng employees (nếu có).
    Trả về danh sách face_id mới được tạo.
    """
    if not embeddings:
        return []

    conn = get_connection()
    cur = conn.cursor()

    # 1. Đặt các vector cũ thành is_active = 0
    cur.execute("UPDATE face_embeddings SET is_active = 0 WHERE employee_id = ?", (employee_id,))

    # 2. Cập nhật thông tin ảnh đại diện và thời gian cập nhật
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if image_url or cloudinary_public_id:
        cur.execute(
            """
            UPDATE employees
            SET image_url = ?, cloudinary_public_id = ?, updated_at = ?
            WHERE id = ?
            """,
            (image_url.strip(), cloudinary_public_id.strip(), now_str, employee_id),
        )
    else:
        cur.execute("UPDATE employees SET updated_at = ? WHERE id = ?", (now_str, employee_id))

    # 3. Thêm các face embeddings mới
    created_face_ids = []
    for emb in embeddings:
        binary_data = pyodbc.Binary(serialize_embedding(emb))
        cur.execute(
            """
            INSERT INTO face_embeddings (employee_id, embedding, is_active)
            OUTPUT INSERTED.face_id
            VALUES (?, ?, 1)
            """,
            (employee_id, binary_data),
        )
        row = cur.fetchone()
        if row and row[0] is not None:
            created_face_ids.append(int(row[0]))

    conn.commit()
    conn.close()
    return created_face_ids


def get_employee(employee_code: str) -> Optional[Dict[str, Any]]:
    """Lấy thông tin 1 nhân viên theo mã nhân viên."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, employee_code, full_name, department, position, image_url, cloudinary_public_id, status, created_at
        FROM employees
        WHERE employee_code = ?
        """,
        (employee_code.strip().upper(),),
    )
    row = cur.fetchone()
    conn.close()

    if row is None:
        return None

    return {
        "id": int(row[0]),
        "employee_code": str(row[1]),
        "full_name": str(row[2]),
        "department": str(row[3]) if row[3] else "",
        "position": str(row[4]) if row[4] else "",
        "image_url": str(row[5]) if row[5] else "",
        "cloudinary_public_id": str(row[6]) if row[6] else "",
        "status": str(row[7]) if row[7] else "ACTIVE",
        "created_at": str(row[8]) if row[8] else "",
    }


def get_employee_by_id(employee_id: int) -> Optional[Dict[str, Any]]:
    """Lấy thông tin 1 nhân viên theo id."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, employee_code, full_name, department, position, image_url, cloudinary_public_id, status, created_at
        FROM employees
        WHERE id = ?
        """,
        (employee_id,),
    )
    row = cur.fetchone()
    conn.close()

    if row is None:
        return None

    return {
        "id": int(row[0]),
        "employee_code": str(row[1]),
        "full_name": str(row[2]),
        "department": str(row[3]) if row[3] else "",
        "position": str(row[4]) if row[4] else "",
        "image_url": str(row[5]) if row[5] else "",
        "cloudinary_public_id": str(row[6]) if row[6] else "",
        "status": str(row[7]) if row[7] else "ACTIVE",
        "created_at": str(row[8]) if row[8] else "",
    }


def load_all_active_face_embeddings() -> List[Dict[str, Any]]:
    """
    Nạp toàn bộ vector khuôn mặt active của nhân viên ACTIVE để phục vụ nạp và Rebuild FAISS.
    Trả về danh sách: [{'face_id': int, 'employee_id': int, 'employee_code': str, 'full_name': str, 'embedding': np.ndarray}]
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT fe.face_id, fe.employee_id, e.employee_code, e.full_name, fe.embedding
        FROM face_embeddings fe
        JOIN employees e ON e.id = fe.employee_id
        WHERE fe.is_active = 1
          AND e.status = 'ACTIVE'
        ORDER BY fe.face_id ASC
        """
    )
    rows = cur.fetchall()
    conn.close()

    embeddings_data = []
    for row in rows:
        face_id = int(row[0])
        emp_id = int(row[1])
        code = str(row[2])
        name = str(row[3])
        emb_blob = row[4]

        vec = deserialize_embedding(emb_blob)
        embeddings_data.append({
            "face_id": face_id,
            "employee_id": emp_id,
            "employee_code": code,
            "full_name": name,
            "embedding": vec,
        })

    return embeddings_data


def load_all_face_encodings() -> Tuple[List[np.ndarray], List[str], List[str]]:
    """Hàm tương thích ngược: nạp danh sách vector, mã và tên nhân viên."""
    active_faces = load_all_active_face_embeddings()
    encodings = [item["embedding"] for item in active_faces]
    codes = [item["employee_code"] for item in active_faces]
    names = [item["full_name"] for item in active_faces]
    return encodings, codes, names


def soft_delete_employee(employee_code: str) -> bool:
    """
    Xóa mềm nhân viên (Soft Delete):
    - Đặt employees.status = 'INACTIVE'
    - Đặt face_embeddings.is_active = 0
    - Giữ nguyên toàn bộ lịch sử chấm công trong bảng attendance
    """
    conn = get_connection()
    cur = conn.cursor()

    # Lấy id nhân viên
    cur.execute("SELECT id FROM employees WHERE employee_code = ?", (employee_code,))
    row = cur.fetchone()
    if not row:
        conn.close()
        return False

    emp_id = int(row[0])
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cur.execute(
        """
        UPDATE employees
        SET status = 'INACTIVE', updated_at = ?
        WHERE id = ?
        """,
        (now_str, emp_id),
    )

    cur.execute(
        """
        UPDATE face_embeddings
        SET is_active = 0
        WHERE employee_id = ?
        """,
        (emp_id,),
    )

    conn.commit()
    conn.close()
    return True


def activate_employee(employee_code: str) -> bool:
    """
    Kích hoạt lại nhân viên (Re-activate):
    - Đặt employees.status = 'ACTIVE'
    - Đặt face_embeddings.is_active = 1
    """
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT id FROM employees WHERE employee_code = ?", (employee_code,))
    row = cur.fetchone()
    if not row:
        conn.close()
        return False

    emp_id = int(row[0])
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cur.execute(
        """
        UPDATE employees
        SET status = 'ACTIVE', updated_at = ?
        WHERE id = ?
        """,
        (now_str, emp_id),
    )

    cur.execute(
        """
        UPDATE face_embeddings
        SET is_active = 1
        WHERE employee_id = ?
        """,
        (emp_id,),
    )

    conn.commit()
    conn.close()
    return True


def hard_delete_employee(employee_code: str) -> Tuple[bool, Optional[str]]:
    """
    Xóa vĩnh viễn nhân viên khỏi CSDL (Hard Delete):
    - Xóa toàn bộ lượt chấm công trong attendance
    - Xóa toàn bộ face embeddings trong face_embeddings
    - Xóa bản ghi trong employees
    - Trả về (True, cloudinary_public_id) nếu thành công
    """
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT id, cloudinary_public_id FROM employees WHERE employee_code = ?", (employee_code,))
    row = cur.fetchone()
    if not row:
        conn.close()
        return False, None

    emp_id = int(row[0])
    cloudinary_pid = str(row[1]) if row[1] else None

    # 1. Xóa lịch sử chấm công
    cur.execute("DELETE FROM attendance WHERE employee_code = ?", (employee_code,))
    # 2. Xóa các vector khuôn mặt
    cur.execute("DELETE FROM face_embeddings WHERE employee_id = ?", (emp_id,))
    # 3. Xóa nhân viên
    cur.execute("DELETE FROM employees WHERE id = ?", (emp_id,))

    conn.commit()
    conn.close()
    return True, cloudinary_pid


def delete_employee(employee_code: str):
    """Alias cho soft_delete_employee để tương thích với các lệnh gọi cũ."""
    return soft_delete_employee(employee_code)


def deactivate_employee_embeddings(employee_id: int):
    """Vô hiệu hóa toàn bộ face embeddings cũ của 1 nhân viên khi cập nhật khuôn mặt mới."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE face_embeddings
        SET is_active = 0
        WHERE employee_id = ?
        """,
        (employee_id,),
    )
    conn.commit()
    conn.close()


def create_check_in(employee_code: str, check_in: str) -> int:
    """Tạo bản ghi check-in mới vào bảng attendance."""
    conn = get_connection()
    cur = conn.cursor()
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
    """Lấy phiên chấm công gần nhất chưa có check-out của nhân viên."""
    conn = get_connection()
    cur = conn.cursor()
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
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT TOP 1 id, employee_code, check_in, check_out
        FROM attendance
        WHERE employee_code = ?
        ORDER BY id DESC
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


# =========================================================================
# WEB ADMIN HELPER FUNCTIONS (DÙNG CHUNG CSDL SQL SERVER)
# =========================================================================

def get_dashboard_stats() -> Dict[str, Any]:
    """
    Truy vấn số liệu thống kê tổng quan cho trang Dashboard:
    - Tổng nhân viên ACTIVE / INACTIVE
    - Số nhân viên có mặt / chưa chấm công hôm nay
    - Số lượt check-in / check-out hôm nay
    - Tổng số face embeddings trong CSDL
    """
    conn = get_connection()
    cur = conn.cursor()
    today_str = datetime.now().strftime("%Y-%m-%d")

    try:
        # 1. Số nhân viên ACTIVE & INACTIVE
        cur.execute("SELECT COUNT(*) FROM employees WHERE status = 'ACTIVE'")
        active_employees = int(cur.fetchone()[0] or 0)

        cur.execute("SELECT COUNT(*) FROM employees WHERE status = 'INACTIVE'")
        inactive_employees = int(cur.fetchone()[0] or 0)

        # 2. Tổng Face Embeddings active
        cur.execute("SELECT COUNT(*) FROM face_embeddings WHERE is_active = 1")
        total_embeddings = int(cur.fetchone()[0] or 0)

        # 3. Lượt check-in & check-out hôm nay
        cur.execute("SELECT COUNT(*) FROM attendance WHERE check_in LIKE ?", (f"{today_str}%",))
        checkin_today = int(cur.fetchone()[0] or 0)

        cur.execute("SELECT COUNT(*) FROM attendance WHERE check_out LIKE ?", (f"{today_str}%",))
        checkout_today = int(cur.fetchone()[0] or 0)

        # 4. Số nhân viên đã có mặt hôm nay (distinct employee_code)
        cur.execute("SELECT COUNT(DISTINCT employee_code) FROM attendance WHERE check_in LIKE ?", (f"{today_str}%",))
        present_today = int(cur.fetchone()[0] or 0)

        absent_today = max(0, active_employees - present_today)

        return {
            "active_employees": active_employees,
            "inactive_employees": inactive_employees,
            "total_embeddings": total_embeddings,
            "checkin_today": checkin_today,
            "checkout_today": checkout_today,
            "present_today": present_today,
            "absent_today": absent_today,
        }
    finally:
        conn.close()


def get_recent_attendance(limit: int = 10) -> List[Dict[str, Any]]:
    """Lấy danh sách các lượt chấm công gần nhất kèm avatar Cloudinary."""
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT TOP (?)
                a.id,
                a.employee_code,
                COALESCE(e.full_name, a.employee_code) AS full_name,
                COALESCE(e.department, N'Chưa phân bổ') AS department,
                COALESCE(e.position, '') AS position,
                e.image_url,
                a.check_in,
                a.check_out
            FROM attendance a
            LEFT JOIN employees e ON a.employee_code = e.employee_code
            ORDER BY a.id DESC
            """,
            (limit,),
        )
        rows = cur.fetchall()
        results = []
        for r in rows:
            results.append({
                "id": int(r[0]),
                "employee_code": str(r[1]),
                "full_name": str(r[2]),
                "department": str(r[3]),
                "position": str(r[4]),
                "image_url": str(r[5]) if r[5] else "",
                "check_in": str(r[6]),
                "check_out": str(r[7]) if r[7] else None,
            })
        return results
    finally:
        conn.close()


def get_hourly_attendance_today() -> Dict[str, List[Any]]:
    """Thống kê số lượt Check-in / Check-out theo từng khung giờ trong ngày hôm nay (0h - 23h)."""
    conn = get_connection()
    cur = conn.cursor()
    today_str = datetime.now().strftime("%Y-%m-%d")

    hours_labels = [f"{h:02d}:00" for h in range(24)]
    checkin_counts = [0] * 24
    checkout_counts = [0] * 24

    try:
        cur.execute(
            """
            SELECT check_in, check_out
            FROM attendance
            WHERE check_in LIKE ? OR check_out LIKE ?
            """,
            (f"{today_str}%", f"{today_str}%"),
        )
        rows = cur.fetchall()
        for r in rows:
            cin = str(r[0]) if r[0] else ""
            cout = str(r[1]) if r[1] else ""
            if cin and cin.startswith(today_str):
                try:
                    dt = datetime.strptime(cin, "%Y-%m-%d %H:%M:%S")
                    checkin_counts[dt.hour] += 1
                except Exception:
                    pass
            if cout and cout.startswith(today_str):
                try:
                    dt = datetime.strptime(cout, "%Y-%m-%d %H:%M:%S")
                    checkout_counts[dt.hour] += 1
                except Exception:
                    pass

        return {
            "labels": hours_labels,
            "checkins": checkin_counts,
            "checkouts": checkout_counts,
        }
    finally:
        conn.close()


def get_department_employee_counts() -> List[Dict[str, Any]]:
    """Lấy số lượng nhân viên ACTIVE theo từng phòng ban phục vụ vẽ biểu đồ."""
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT
                CASE
                    WHEN department IS NULL OR LTRIM(RTRIM(department)) = '' THEN N'Chưa phân bổ'
                    ELSE department
                END AS dept,
                COUNT(*) AS cnt
            FROM employees
            WHERE status = 'ACTIVE'
            GROUP BY
                CASE
                    WHEN department IS NULL OR LTRIM(RTRIM(department)) = '' THEN N'Chưa phân bổ'
                    ELSE department
                END
            ORDER BY cnt DESC
            """
        )
        rows = cur.fetchall()
        return [{"department": str(r[0]), "count": int(r[1])} for r in rows]
    finally:
        conn.close()


def get_departments() -> List[str]:
    """Lấy danh sách các phòng ban phân biệt có trong hệ thống."""
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT DISTINCT department
            FROM employees
            WHERE department IS NOT NULL AND LTRIM(RTRIM(department)) <> ''
            ORDER BY department ASC
            """
        )
        rows = cur.fetchall()
        return [str(r[0]) for r in rows]
    finally:
        conn.close()


def get_employees_paginated(
    page: int = 1,
    per_page: int = 20,
    search: str = "",
    status: str = "",
    department: str = "",
) -> Tuple[List[Dict[str, Any]], int, int]:
    """
    Lấy danh sách nhân viên có phân trang (Pagination), tìm kiếm và bộ lọc.
    Trả về: (danh_sách_nhân_viên, tổng_bản_ghi, tổng_số_trang)
    """
    page = max(1, page)
    per_page = max(1, min(per_page, 200))
    offset = (page - 1) * per_page

    conn = get_connection()
    cur = conn.cursor()

    conditions = []
    params = []

    if search.strip():
        s = f"%{search.strip()}%"
        conditions.append("(e.employee_code LIKE ? OR e.full_name LIKE ? OR e.position LIKE ?)")
        params.extend([s, s, s])

    if status.strip():
        conditions.append("e.status = ?")
        params.append(status.strip().upper())

    if department.strip():
        conditions.append("e.department = ?")
        params.append(department.strip())

    where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""

    try:
        # 1. Đếm tổng số bản ghi
        count_query = f"SELECT COUNT(*) FROM employees e {where_clause}"
        cur.execute(count_query, params)
        total_count = int(cur.fetchone()[0] or 0)
        total_pages = max(1, (total_count + per_page - 1) // per_page)

        # 2. Truy vấn dữ liệu trang
        data_query = f"""
            SELECT
                e.id,
                e.employee_code,
                e.full_name,
                e.department,
                e.position,
                e.image_url,
                e.cloudinary_public_id,
                e.status,
                e.created_at,
                (
                    SELECT COUNT(*)
                    FROM face_embeddings fe
                    WHERE fe.employee_id = e.id AND fe.is_active = 1
                ) AS embedding_count
            FROM employees e
            {where_clause}
            ORDER BY e.id DESC
            OFFSET ? ROWS FETCH NEXT ? ROWS ONLY
        """
        data_params = list(params) + [offset, per_page]
        cur.execute(data_query, data_params)
        rows = cur.fetchall()

        items = []
        for r in rows:
            items.append({
                "id": int(r[0]),
                "employee_code": str(r[1]),
                "full_name": str(r[2]),
                "department": str(r[3]) if r[3] else "",
                "position": str(r[4]) if r[4] else "",
                "image_url": str(r[5]) if r[5] else "",
                "cloudinary_public_id": str(r[6]) if r[6] else "",
                "status": str(r[7]) if r[7] else "ACTIVE",
                "created_at": str(r[8]) if r[8] else "",
                "embedding_count": int(r[9] or 0),
            })

        return items, total_count, total_pages
    finally:
        conn.close()


def get_employee_detail(employee_id: int) -> Optional[Dict[str, Any]]:
    """Lấy chi tiết hồ sơ nhân viên, số vector khuôn mặt và 15 lượt chấm công gần nhất."""
    conn = get_connection()
    cur = conn.cursor()
    try:
        # 1. Thông tin nhân viên
        cur.execute(
            """
            SELECT id, employee_code, full_name, department, position, image_url, cloudinary_public_id, status, created_at, updated_at
            FROM employees
            WHERE id = ?
            """,
            (employee_id,),
        )
        row = cur.fetchone()
        if not row:
            return None

        emp = {
            "id": int(row[0]),
            "employee_code": str(row[1]),
            "full_name": str(row[2]),
            "department": str(row[3]) if row[3] else "",
            "position": str(row[4]) if row[4] else "",
            "image_url": str(row[5]) if row[5] else "",
            "cloudinary_public_id": str(row[6]) if row[6] else "",
            "status": str(row[7]) if row[7] else "ACTIVE",
            "created_at": str(row[8]) if row[8] else "",
            "updated_at": str(row[9]) if row[9] else "",
        }

        # 2. Danh sách embeddings
        cur.execute(
            """
            SELECT face_id, is_active, created_at
            FROM face_embeddings
            WHERE employee_id = ?
            ORDER BY face_id ASC
            """,
            (employee_id,),
        )
        emb_rows = cur.fetchall()
        embeddings = [
            {
                "face_id": int(er[0]),
                "is_active": bool(er[1]),
                "created_at": str(er[2]),
            }
            for er in emb_rows
        ]
        active_embedding_count = sum(1 for e in embeddings if e["is_active"])

        # 3. Lịch sử chấm công gần nhất
        cur.execute(
            """
            SELECT TOP 15 id, check_in, check_out
            FROM attendance
            WHERE employee_code = ?
            ORDER BY id DESC
            """,
            (emp["employee_code"],),
        )
        att_rows = cur.fetchall()
        recent_attendance = [
            {
                "id": int(ar[0]),
                "check_in": str(ar[1]),
                "check_out": str(ar[2]) if ar[2] else None,
            }
            for ar in att_rows
        ]

        return {
            "employee": emp,
            "embeddings": embeddings,
            "embedding_count": active_embedding_count,
            "recent_attendance": recent_attendance,
        }
    finally:
        conn.close()


def create_employee_web(
    employee_code: str,
    full_name: str,
    department: str = "",
    position: str = "",
) -> Tuple[bool, str, Optional[int]]:
    """
    Tạo hồ sơ nhân viên mới từ Web Admin (Trạng thái ban đầu: NOT REGISTERED khuôn mặt).
    Nhân viên sau đó có thể đến app Tkinter để chụp và lưu khuôn mặt AI.
    """
    code = employee_code.strip().upper()
    name = full_name.strip()
    dept = department.strip()
    pos = position.strip()

    if not code or not name:
        return False, "Mã nhân viên và Họ tên không được để trống!", None

    conn = get_connection()
    cur = conn.cursor()
    try:
        # Kiểm tra trùng mã nhân viên
        cur.execute("SELECT id FROM employees WHERE UPPER(employee_code) = ?", (code,))
        if cur.fetchone():
            return False, f"Mã nhân viên '{code}' đã tồn tại trong hệ thống!", None

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cur.execute(
            """
            INSERT INTO employees (
                employee_code, full_name, department, position,
                image_url, cloudinary_public_id, status, face_encoding, created_at
            )
            OUTPUT INSERTED.id
            VALUES (?, ?, ?, ?, '', '', 'ACTIVE', NULL, ?)
            """,
            (code, name, dept, pos, now_str),
        )
        row = cur.fetchone()
        emp_id = int(row[0]) if row and row[0] is not None else 0
        conn.commit()
        return True, "Tạo hồ sơ nhân viên thành công!", emp_id
    except Exception as ex:
        return False, f"Lỗi tạo nhân viên: {str(ex)}", None
    finally:
        conn.close()


def update_employee(
    employee_id: int,
    employee_code: str,
    full_name: str,
    department: str = "",
    position: str = "",
    status: str = "ACTIVE",
) -> Tuple[bool, str]:
    """
    Cập nhật thông tin nhân viên từ Web Admin.
    Đồng bộ trạng thái is_active của face_embeddings tương ứng.
    """
    code = employee_code.strip().upper()
    name = full_name.strip()
    dept = department.strip()
    pos = position.strip()
    st = status.strip().upper() if status else "ACTIVE"

    if not code or not name:
        return False, "Mã nhân viên và Họ tên không được để trống!"

    conn = get_connection()
    cur = conn.cursor()
    try:
        # Kiểm tra trùng mã nhân viên với người khác
        cur.execute(
            "SELECT id FROM employees WHERE UPPER(employee_code) = ? AND id <> ?",
            (code, employee_id),
        )
        if cur.fetchone():
            return False, f"Mã nhân viên '{code}' đã được sử dụng bởi nhân viên khác!"

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Cập nhật thông tin bảng employees
        cur.execute(
            """
            UPDATE employees
            SET employee_code = ?, full_name = ?, department = ?, position = ?, status = ?, updated_at = ?
            WHERE id = ?
            """,
            (code, name, dept, pos, st, now_str, employee_id),
        )

        # Đồng bộ is_active cho face_embeddings
        is_active_val = 1 if st == "ACTIVE" else 0
        cur.execute(
            "UPDATE face_embeddings SET is_active = ? WHERE employee_id = ?",
            (is_active_val, employee_id),
        )

        conn.commit()
        return True, "Cập nhật thông tin nhân viên thành công!"
    except Exception as ex:
        return False, f"Lỗi cập nhật nhân viên: {str(ex)}"
    finally:
        conn.close()


def get_attendance_paginated(
    page: int = 1,
    per_page: int = 20,
    start_date: str = "",
    end_date: str = "",
    search: str = "",
    department: str = "",
) -> Tuple[List[Dict[str, Any]], int, int]:
    """
    Lấy danh sách lịch sử chấm công có phân trang và bộ lọc nâng cao.
    Trả về: (danh_sách_chấm_công, tổng_bản_ghi, tổng_số_trang)
    """
    page = max(1, page)
    per_page = max(1, min(per_page, 200))
    offset = (page - 1) * per_page

    conn = get_connection()
    cur = conn.cursor()

    conditions = []
    params = []

    if start_date.strip():
        conditions.append("a.check_in >= ?")
        params.append(f"{start_date.strip()} 00:00:00")

    if end_date.strip():
        conditions.append("a.check_in <= ?")
        params.append(f"{end_date.strip()} 23:59:59")

    if search.strip():
        s = f"%{search.strip()}%"
        conditions.append("(a.employee_code LIKE ? OR e.full_name LIKE ?)")
        params.extend([s, s])

    if department.strip():
        conditions.append("e.department = ?")
        params.append(department.strip())

    where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""

    try:
        # 1. Đếm tổng số bản ghi
        count_query = f"""
            SELECT COUNT(*)
            FROM attendance a
            LEFT JOIN employees e ON a.employee_code = e.employee_code
            {where_clause}
        """
        cur.execute(count_query, params)
        total_count = int(cur.fetchone()[0] or 0)
        total_pages = max(1, (total_count + per_page - 1) // per_page)

        # 2. Truy vấn dữ liệu trang
        data_query = f"""
            SELECT
                a.id,
                a.employee_code,
                COALESCE(e.full_name, a.employee_code) AS full_name,
                COALESCE(e.department, N'Chưa phân bổ') AS department,
                COALESCE(e.position, '') AS position,
                e.image_url,
                a.check_in,
                a.check_out
            FROM attendance a
            LEFT JOIN employees e ON a.employee_code = e.employee_code
            {where_clause}
            ORDER BY a.id DESC
            OFFSET ? ROWS FETCH NEXT ? ROWS ONLY
        """
        data_params = list(params) + [offset, per_page]
        cur.execute(data_query, data_params)
        rows = cur.fetchall()

        items = []
        for r in rows:
            cin = str(r[6])
            cout = str(r[7]) if r[7] else None

            # Tính thời gian làm việc nếu đã check-out
            work_duration = ""
            if cin and cout:
                try:
                    dt_in = datetime.strptime(cin, "%Y-%m-%d %H:%M:%S")
                    dt_out = datetime.strptime(cout, "%Y-%m-%d %H:%M:%S")
                    diff_sec = (dt_out - dt_in).total_seconds()
                    if diff_sec >= 0:
                        hrs = int(diff_sec // 3600)
                        mins = int((diff_sec % 3600) // 60)
                        work_duration = f"{hrs}h {mins}m"
                except Exception:
                    pass

            items.append({
                "id": int(r[0]),
                "employee_code": str(r[1]),
                "full_name": str(r[2]),
                "department": str(r[3]),
                "position": str(r[4]),
                "image_url": str(r[5]) if r[5] else "",
                "check_in": cin,
                "check_out": cout,
                "work_duration": work_duration,
            })

        return items, total_count, total_pages
    finally:
        conn.close()

