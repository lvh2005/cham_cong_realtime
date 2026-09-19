import hashlib
import json
import os
from typing import Any, Dict, Tuple

import config

AUTH_CONFIG_FILE = os.path.join(config.BASE_DIR, "admin_config.json")

DEFAULT_ADMIN = {
    "username": "admin",
    # Mật khẩu mặc định: admin123
    # SHA-256("admin123") = 240be518fabd2724ddb6f04eeb1da5967448d7e831c08c8fa822809f74c720a9
    "password_hash": "240be518fabd2724ddb6f04eeb1da5967448d7e831c08c8fa822809f74c720a9",
    "role": "Super Admin",
    "updated_at": "2026-09-14 00:00:00",
}


def _hash_password(password: str) -> str:
    """Băm mật khẩu bằng SHA-256."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def load_admin_config() -> Dict[str, Any]:
    """Tải cấu hình tài khoản admin từ file JSON."""
    if not os.path.exists(AUTH_CONFIG_FILE):
        save_admin_config(DEFAULT_ADMIN)
        return DEFAULT_ADMIN.copy()
    try:
        with open(AUTH_CONFIG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data
    except Exception:
        return DEFAULT_ADMIN.copy()


def save_admin_config(cfg: Dict[str, Any]) -> None:
    """Lưu cấu hình tài khoản admin vào file JSON."""
    with open(AUTH_CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=4, ensure_ascii=False)


def verify_admin_login(username: str, password: str) -> Tuple[bool, str]:
    """
    Xác thực thông tin đăng nhập của Admin.
    Trả về (True, "") nếu thành công, (False, "Lý do lỗi") nếu thất bại.
    """
    u = username.strip()
    p = password.strip()

    if not u or not p:
        return False, "Vui lòng nhập đầy đủ tên đăng nhập và mật khẩu."

    cfg = load_admin_config()
    target_user = cfg.get("username", "admin")
    target_hash = cfg.get("password_hash", "")

    if u.lower() != target_user.lower():
        return False, "Tên đăng nhập không chính xác."

    input_hash = _hash_password(p)
    if input_hash != target_hash:
        return False, "Mật khẩu không đúng. Vui lòng thử lại!"

    return True, "Đăng nhập thành công!"


def change_admin_password(old_pass: str, new_pass: str, confirm_pass: str) -> Tuple[bool, str]:
    """
    Đổi mật khẩu tài khoản Admin.
    """
    if not old_pass or not new_pass or not confirm_pass:
        return False, "Vui lòng nhập đầy đủ các trường thông tin."

    if new_pass != confirm_pass:
        return False, "Mật khẩu mới và mật khẩu xác nhận không trùng khớp."

    if len(new_pass) < 4:
        return False, "Mật khẩu mới phải có tối thiểu 4 ký tự."

    cfg = load_admin_config()
    target_hash = cfg.get("password_hash", "")

    if _hash_password(old_pass) != target_hash:
        return False, "Mật khẩu hiện tại không chính xác."

    cfg["password_hash"] = _hash_password(new_pass)
    save_admin_config(cfg)
    return True, "Đổi mật khẩu Admin thành công!"
