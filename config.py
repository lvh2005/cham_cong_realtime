import os
from typing import Any, Dict
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
FAISS_DIR = os.path.join(DATA_DIR, "faiss")
TEMP_DIR = os.path.join(BASE_DIR, "temp")

os.makedirs(FAISS_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

ENV_FILE = os.path.join(BASE_DIR, ".env")
DB_CONFIG_FILE = os.path.join(BASE_DIR, "db_config.json")

# Nạp biến môi trường từ .env
load_dotenv(ENV_FILE, override=True)


def _get_str_env(key: str, default: str) -> str:
    val = os.getenv(key)
    return val.strip() if val is not None else default


def _get_bool_env(key: str, default: bool) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return val.strip().lower() in ("true", "1", "yes", "y", "t")


def _get_float_env(key: str, default: float) -> float:
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return float(val.strip())
    except ValueError:
        return default


def _get_int_env(key: str, default: int) -> int:
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return int(val.strip())
    except ValueError:
        return default


# 1. SQL Server Config
DB_SERVER = _get_str_env("DB_SERVER", "localhost")
DB_DATABASE = _get_str_env("DB_DATABASE", "FaceAttendanceDB")
DB_DRIVER = _get_str_env("DB_DRIVER", "ODBC Driver 17 for SQL Server")
DB_TRUSTED_CONNECTION = _get_bool_env("DB_TRUSTED_CONNECTION", True)
DB_USER = _get_str_env("DB_USER", "sa")
DB_PASSWORD = _get_str_env("DB_PASSWORD", "")

# 2. Cloudinary Config
CLOUDINARY_CLOUD_NAME = _get_str_env("CLOUDINARY_CLOUD_NAME", "")
CLOUDINARY_API_KEY = _get_str_env("CLOUDINARY_API_KEY", "")
CLOUDINARY_API_SECRET = _get_str_env("CLOUDINARY_API_SECRET", "")
CLOUDINARY_FOLDER = _get_str_env("CLOUDINARY_FOLDER", "face_attendance/employees")

# 3. Recognition & Algorithm Parameters
FACE_MATCH_THRESHOLD = _get_float_env("FACE_MATCH_THRESHOLD", 0.38)
FACE_DUPLICATE_THRESHOLD = _get_float_env("FACE_DUPLICATE_THRESHOLD", 0.35)
FACE_AMBIGUOUS_MARGIN = _get_float_env("FACE_AMBIGUOUS_MARGIN", 0.05)
ATTENDANCE_COOLDOWN_SECONDS = _get_int_env("ATTENDANCE_COOLDOWN_SECONDS", 60)
RECOGNITION_FRAME_SKIP = _get_int_env("RECOGNITION_FRAME_SKIP", 3)

FAISS_INDEX_PATH = os.path.join(FAISS_DIR, "face.index")


# 4. Flask Web Admin Config
FLASK_HOST = _get_str_env("FLASK_HOST", "0.0.0.0")
FLASK_PORT = _get_int_env("FLASK_PORT", 5000)
FLASK_SECRET_KEY = _get_str_env("FLASK_SECRET_KEY", "face_attendance_flask_secret_2026_key")


def reload_config():
    """Tải lại các biến cấu hình từ file .env."""
    global DB_SERVER, DB_DATABASE, DB_DRIVER, DB_TRUSTED_CONNECTION, DB_USER, DB_PASSWORD
    global CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, CLOUDINARY_API_SECRET, CLOUDINARY_FOLDER
    global FACE_MATCH_THRESHOLD, FACE_DUPLICATE_THRESHOLD, FACE_AMBIGUOUS_MARGIN
    global ATTENDANCE_COOLDOWN_SECONDS, RECOGNITION_FRAME_SKIP
    global FLASK_HOST, FLASK_PORT, FLASK_SECRET_KEY

    load_dotenv(ENV_FILE, override=True)

    DB_SERVER = _get_str_env("DB_SERVER", "localhost")
    DB_DATABASE = _get_str_env("DB_DATABASE", "FaceAttendanceDB")
    DB_DRIVER = _get_str_env("DB_DRIVER", "ODBC Driver 17 for SQL Server")
    DB_TRUSTED_CONNECTION = _get_bool_env("DB_TRUSTED_CONNECTION", True)
    DB_USER = _get_str_env("DB_USER", "sa")
    DB_PASSWORD = _get_str_env("DB_PASSWORD", "")

    CLOUDINARY_CLOUD_NAME = _get_str_env("CLOUDINARY_CLOUD_NAME", "")
    CLOUDINARY_API_KEY = _get_str_env("CLOUDINARY_API_KEY", "")
    CLOUDINARY_API_SECRET = _get_str_env("CLOUDINARY_API_SECRET", "")
    CLOUDINARY_FOLDER = _get_str_env("CLOUDINARY_FOLDER", "face_attendance/employees")

    FACE_MATCH_THRESHOLD = _get_float_env("FACE_MATCH_THRESHOLD", 0.38)
    FACE_DUPLICATE_THRESHOLD = _get_float_env("FACE_DUPLICATE_THRESHOLD", 0.35)
    FACE_AMBIGUOUS_MARGIN = _get_float_env("FACE_AMBIGUOUS_MARGIN", 0.05)
    ATTENDANCE_COOLDOWN_SECONDS = _get_int_env("ATTENDANCE_COOLDOWN_SECONDS", 60)
    RECOGNITION_FRAME_SKIP = _get_int_env("RECOGNITION_FRAME_SKIP", 3)

    FLASK_HOST = _get_str_env("FLASK_HOST", "0.0.0.0")
    FLASK_PORT = _get_int_env("FLASK_PORT", 5000)
    FLASK_SECRET_KEY = _get_str_env("FLASK_SECRET_KEY", "face_attendance_flask_secret_2026_key")


def get_db_dict() -> Dict[str, Any]:
    """Trả về dict cấu hình kết nối SQL Server."""
    return {
        "server": DB_SERVER,
        "database": DB_DATABASE,
        "driver": DB_DRIVER,
        "trusted_connection": DB_TRUSTED_CONNECTION,
        "username": DB_USER,
        "password": DB_PASSWORD,
    }

