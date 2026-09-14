"""
Face Attendance Desktop Application (Tkinter GUI)
Hệ thống chấm công bằng nhận diện khuôn mặt
Hỗ trợ SQLite & Microsoft SQL Server
"""

import json
import os
import shutil
import threading
import time
from datetime import date, datetime, timedelta
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import List, Optional

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageTk

from attendance import COOLDOWN_SECONDS, get_attendance_report, register_attendance
from database import (
    add_employee,
    delete_employee,
    get_available_sqlserver_drivers,
    get_employee,
    get_employees,
    init_db,
    load_all_face_encodings,
    load_db_config,
    save_db_config,
    test_db_connection,
)
from auth_utils import change_admin_password, verify_admin_login
from face_utils import (
    build_employee_encoding,
    ensure_rgb_uint8,
    load_known_faces,
    recognize_faces,
)
from voice_utils import speak_async

DATASET_DIR = "dataset"
os.makedirs(DATASET_DIR, exist_ok=True)


class CameraGrabber:
    """
    Luồng độc lập chạy ngầm đọc frame từ webcam (Non-blocking).
    Tách biệt hoàn toàn việc đọc phần cứng camera ra khỏi Main Thread của Tkinter,
    giúp giao diện đạt 30+ FPS siêu mượt, không bao giờ bị Not Responding hay giật lag.
    """
    def __init__(self, src: int = 0):
        self.src = src
        self.cap = None
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        self.latest_frame = None

    def start(self) -> bool:
        if self.running:
            return True
        try:
            # Ưu tiên cv2.CAP_DSHOW trên Windows để bật camera tức thì và không bị buffer trễ
            self.cap = cv2.VideoCapture(self.src, cv2.CAP_DSHOW)
            if not self.cap or not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.src)

            if not self.cap or not self.cap.isOpened():
                return False

            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.running = True
            self.thread = threading.Thread(target=self._worker, daemon=True)
            self.thread.start()
            return True
        except Exception as e:
            print(f"Lỗi khởi tạo camera grabber: {e}")
            return False

    def _worker(self):
        while self.running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.lock:
                    self.latest_frame = frame
            else:
                time.sleep(0.01)

    def get_frame(self) -> Optional[np.ndarray]:
        with self.lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
            return None

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=0.3)
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        with self.lock:
            self.latest_frame = None


class FaceAttendanceApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Face Attendance System • Hệ Thống Chấm Công Khuôn Mặt")

        # Căn giữa màn hình
        win_w, win_h = 1260, 800
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        pos_x = max(0, (screen_w - win_w) // 2)
        pos_y = max(0, (screen_h - win_h) // 2 - 20)
        self.geometry(f"{win_w}x{win_h}+{pos_x}+{pos_y}")
        self.minsize(1080, 680)

        # Trạng thái toàn cục (khởi tạo nhanh rỗng, tải dữ liệu sau)
        self.known_faces = {"encodings": [], "codes": [], "names": []}
        self.active_tab = "realtime"

        # Phân quyền người dùng (Mặc định: Nhân viên / Kiosk Chấm công)
        self.is_admin = False
        self.current_admin_user = None
        self.ADMIN_TABS = {"dashboard", "register", "history", "export", "employees", "database"}

        # Camera Grabbers chuyên dụng non-blocking
        self.reg_grabber = CameraGrabber(0)
        self.rec_grabber = CameraGrabber(0)

        # Biến cho chức năng Đăng ký
        self.register_photos: List[np.ndarray] = []
        self.reg_cam_running = False

        # Biến cho chức năng Chấm công Realtime
        self.rec_cam_running = False
        self.rec_tolerance = 0.50
        self.last_detected_faces = []
        self.recent_attendance_attempts = {}
        self.recent_logs = []

        self._setup_theme()
        self._build_layout()
        self.show_tab("realtime")

        # Bắt sự kiện đóng cửa sổ để giải phóng camera
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Khởi tạo CSDL và nạp dữ liệu nền (Non-blocking để không bao giờ bị Not Responding)
        self.after(100, self._async_init_system)

    def _async_init_system(self):
        def worker():
            try:
                init_db()
            except Exception as e:
                print(f"Lỗi khởi tạo CSDL nền: {e}")

            try:
                faces = load_known_faces()
                self.known_faces = faces
            except Exception as e:
                print(f"Lỗi tải face encodings: {e}")

            self.after(0, self.refresh_dashboard)
            self.after(0, self.update_db_badge)

        threading.Thread(target=worker, daemon=True).start()



    def _setup_theme(self):
        self.style = ttk.Style(self)
        self.style.theme_use("clam")

        # Bảng màu
        self.BG_DARK = "#0f172a"      # Slate 900
        self.SIDEBAR_BG = "#1e293b"  # Slate 800
        self.SIDEBAR_HOVER = "#334155"
        self.SIDEBAR_ACTIVE = "#2563eb"
        self.CONTENT_BG = "#f8fafc"  # Slate 50
        self.CARD_BG = "#ffffff"
        self.BORDER_COLOR = "#e2e8f0"
        self.PRIMARY_COLOR = "#2563eb"
        self.SUCCESS_COLOR = "#16a34a"
        self.DANGER_COLOR = "#dc2626"
        self.TEXT_MAIN = "#1e293b"
        self.TEXT_MUTED = "#64748b"

        self.configure(bg=self.CONTENT_BG)

        # Style cho Treeview
        self.style.configure(
            "Custom.Treeview",
            background="#ffffff",
            foreground=self.TEXT_MAIN,
            rowheight=32,
            fieldbackground="#ffffff",
            font=("Segoe UI", 10),
            bordercolor=self.BORDER_COLOR,
            borderwidth=1,
        )
        self.style.configure(
            "Custom.Treeview.Heading",
            background="#f1f5f9",
            foreground="#0f172a",
            font=("Segoe UI", 10, "bold"),
            relief="flat",
            padding=6,
        )
        self.style.map(
            "Custom.Treeview",
            background=[("selected", "#dbeafe")],
            foreground=[("selected", "#1e3a8a")],
        )

        # Style cho Button
        self.style.configure(
            "Primary.TButton",
            background=self.PRIMARY_COLOR,
            foreground="#ffffff",
            font=("Segoe UI", 10, "bold"),
            padding=(12, 6),
            relief="flat",
        )
        self.style.map(
            "Primary.TButton",
            background=[("active", "#1d4ed8")],
        )

        self.style.configure(
            "Success.TButton",
            background=self.SUCCESS_COLOR,
            foreground="#ffffff",
            font=("Segoe UI", 10, "bold"),
            padding=(12, 6),
            relief="flat",
        )
        self.style.map(
            "Success.TButton",
            background=[("active", "#15803d")],
        )

        self.style.configure(
            "Danger.TButton",
            background=self.DANGER_COLOR,
            foreground="#ffffff",
            font=("Segoe UI", 10, "bold"),
            padding=(12, 6),
            relief="flat",
        )
        self.style.map(
            "Danger.TButton",
            background=[("active", "#b91c1c")],
        )

    def _build_layout(self):
        # Frame chính
        self.main_container = tk.Frame(self, bg=self.CONTENT_BG)
        self.main_container.pack(fill="both", expand=True)

        # 1. Sidebar (bên trái)
        self.sidebar = tk.Frame(self.main_container, bg=self.SIDEBAR_BG, width=250)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        # Logo / Tiêu đề Sidebar
        title_frame = tk.Frame(self.sidebar, bg=self.SIDEBAR_BG, pady=18)
        title_frame.pack(fill="x")
        lbl_logo = tk.Label(
            title_frame,
            text="👤 Face Attendance",
            font=("Segoe UI", 14, "bold"),
            fg="#ffffff",
            bg=self.SIDEBAR_BG,
        )
        lbl_logo.pack(anchor="w", padx=20)
        lbl_sub = tk.Label(
            title_frame,
            text="Hệ thống chấm công AI",
            font=("Segoe UI", 9),
            fg="#94a3b8",
            bg=self.SIDEBAR_BG,
        )
        lbl_sub.pack(anchor="w", padx=20, pady=(2, 0))

        # Role Badge
        self.lbl_role_badge = tk.Label(
            title_frame,
            text="🟢 Nhân viên (Chấm công)",
            font=("Segoe UI", 8, "bold"),
            fg="#6ee7b7",
            bg="#064e3b",
            padx=8,
            pady=3,
        )
        self.lbl_role_badge.pack(anchor="w", padx=20, pady=(6, 0))

        # Divider
        tk.Frame(self.sidebar, bg="#334155", height=1).pack(fill="x", padx=15, pady=5)

        # Menu buttons
        self.nav_buttons = {}
        self.menu_items_config = [
            ("realtime", "📷  Chấm công Realtime", False),
            ("dashboard", "📊  Tổng quan", True),
            ("register", "👤  Đăng ký nhân viên", True),
            ("employees", "⚙️  Quản lý nhân viên", True),
            ("history", "📋  Lịch sử chấm công", True),
            ("export", "📈  Xuất báo cáo Excel", True),
            ("database", "🗄️  Cấu hình Database", True),
        ]

        menu_frame = tk.Frame(self.sidebar, bg=self.SIDEBAR_BG)
        menu_frame.pack(fill="x", pady=6)

        for key, text, is_admin_tab in self.menu_items_config:
            btn = tk.Button(
                menu_frame,
                text=text,
                font=("Segoe UI", 10),
                fg="#e2e8f0",
                bg=self.SIDEBAR_BG,
                activebackground=self.SIDEBAR_ACTIVE,
                activeforeground="#ffffff",
                relief="flat",
                anchor="w",
                padx=20,
                pady=8,
                cursor="hand2",
                command=lambda k=key: self.show_tab(k),
            )
            btn.pack(fill="x", pady=1)
            self.nav_buttons[key] = btn

        # Divider
        tk.Frame(self.sidebar, bg="#334155", height=1).pack(fill="x", padx=15, pady=5)

        # Khung nút chức năng Admin (Đăng nhập / Đổi mật khẩu / Đăng xuất)
        self.auth_frame = tk.Frame(self.sidebar, bg=self.SIDEBAR_BG, pady=6, padx=15)
        self.auth_frame.pack(fill="x")

        self.btn_admin_login = tk.Button(
            self.auth_frame,
            text="🔐 Đăng nhập Quản trị (Admin)",
            font=("Segoe UI", 9, "bold"),
            fg="#ffffff",
            bg="#2563eb",
            activebackground="#1d4ed8",
            activeforeground="#ffffff",
            relief="flat",
            pady=8,
            cursor="hand2",
            command=self.open_admin_login_dialog,
        )
        self.btn_admin_login.pack(fill="x")

        self.admin_logged_frame = tk.Frame(self.auth_frame, bg=self.SIDEBAR_BG)

        self.btn_change_pwd = tk.Button(
            self.admin_logged_frame,
            text="🔑 Đổi mật khẩu Admin",
            font=("Segoe UI", 8),
            fg="#cbd5e1",
            bg="#334155",
            activebackground="#475569",
            activeforeground="#ffffff",
            relief="flat",
            pady=4,
            cursor="hand2",
            command=self.open_change_password_dialog,
        )
        self.btn_change_pwd.pack(fill="x", pady=(0, 4))

        self.btn_admin_logout = tk.Button(
            self.admin_logged_frame,
            text="🔒 Đăng xuất Quản trị",
            font=("Segoe UI", 8, "bold"),
            fg="#fca5a5",
            bg="#7f1d1d",
            activebackground="#991b1b",
            activeforeground="#ffffff",
            relief="flat",
            pady=4,
            cursor="hand2",
            command=self.admin_logout,
        )
        self.btn_admin_logout.pack(fill="x")

        # Bottom Info Badge in Sidebar
        bottom_frame = tk.Frame(self.sidebar, bg="#0f172a", pady=10, padx=15)
        bottom_frame.pack(side="bottom", fill="x")

        self.lbl_db_status = tk.Label(
            bottom_frame,
            text="🟢 DB: SQLite",
            font=("Segoe UI", 9, "bold"),
            fg="#4ade80",
            bg="#0f172a",
        )
        self.lbl_db_status.pack(anchor="w")

        lbl_ver = tk.Label(
            bottom_frame,
            text="Face Recognition v1.3 • OpenCV",
            font=("Segoe UI", 8),
            fg="#64748b",
            bg="#0f172a",
        )
        lbl_ver.pack(anchor="w", pady=(1, 6))

        # Nút Thoát ứng dụng
        btn_exit = tk.Button(
            bottom_frame,
            text="🚪 Thoát ứng dụng",
            font=("Segoe UI", 9, "bold"),
            fg="#ffffff",
            bg="#dc2626",
            activebackground="#b91c1c",
            activeforeground="#ffffff",
            relief="flat",
            pady=6,
            cursor="hand2",
            command=self.confirm_exit_app,
        )
        btn_exit.pack(fill="x")

        # 2. Vùng Content (bên phải)
        self.content_area = tk.Frame(self.main_container, bg=self.CONTENT_BG)
        self.content_area.pack(side="right", fill="both", expand=True, padx=25, pady=20)

        # Dictionary lưu các frame tab
        self.tabs = {}
        self._init_realtime_tab()
        self._init_dashboard_tab()
        self._init_register_tab()
        self._init_employees_tab()
        self._init_history_tab()
        self._init_export_tab()
        self._init_database_tab()

        self.update_sidebar_auth_state()

    def update_sidebar_auth_state(self):
        """Cập nhật trạng thái hiển thị của Sidebar theo quyền Admin / Nhân viên."""
        if self.is_admin:
            u = self.current_admin_user or "admin"
            self.lbl_role_badge.config(
                text=f"⭐ Quản trị viên ({u})",
                bg="#1e3a8a",
                fg="#93c5fd",
            )
            self.btn_admin_login.pack_forget()
            self.admin_logged_frame.pack(fill="x")

            for key, base_text, is_admin_tab in self.menu_items_config:
                if key in self.nav_buttons:
                    self.nav_buttons[key].config(text=base_text)
        else:
            self.lbl_role_badge.config(
                text="🟢 Nhân viên (Chấm công)",
                bg="#064e3b",
                fg="#6ee7b7",
            )
            self.admin_logged_frame.pack_forget()
            self.btn_admin_login.pack(fill="x")

            for key, base_text, is_admin_tab in self.menu_items_config:
                if key in self.nav_buttons:
                    text = base_text if not is_admin_tab else f"{base_text} 🔒"
                    self.nav_buttons[key].config(text=text)

    def open_admin_login_dialog(self, target_tab: Optional[str] = None):
        """Hộp thoại đăng nhập dành cho Quản trị viên."""
        dialog = tk.Toplevel(self)
        dialog.title("🔐 Đăng nhập Quản trị viên (Admin)")
        dialog.geometry("420x330")
        dialog.resizable(False, False)
        dialog.transient(self)
        dialog.grab_set()

        # Căn giữa cửa sổ con
        x = self.winfo_x() + max(0, (self.winfo_width() - 420) // 2)
        y = self.winfo_y() + max(0, (self.winfo_height() - 330) // 2)
        dialog.geometry(f"+{x}+{y}")
        dialog.configure(bg="#ffffff")

        # Header
        hdr = tk.Frame(dialog, bg="#1e293b", pady=15, padx=20)
        hdr.pack(fill="x")
        tk.Label(hdr, text="🔐 Đăng nhập Quản trị viên", font=("Segoe UI", 13, "bold"), fg="#ffffff", bg="#1e293b").pack(anchor="w")
        tk.Label(hdr, text="Vui lòng xác thực để truy cập quản lý nhân viên, lịch sử, báo cáo & CSDL", font=("Segoe UI", 8), fg="#94a3b8", bg="#1e293b").pack(anchor="w", pady=(2, 0))

        body = tk.Frame(dialog, bg="#ffffff", padx=25, pady=15)
        body.pack(fill="both", expand=True)

        tk.Label(body, text="Tên đăng nhập:", font=("Segoe UI", 9, "bold"), bg="#ffffff", fg="#1e293b").grid(row=0, column=0, sticky="w", pady=(0, 2))
        ent_user = ttk.Entry(body, font=("Segoe UI", 10), width=32)
        ent_user.grid(row=1, column=0, sticky="w", pady=(0, 8))
        ent_user.insert(0, "admin")

        tk.Label(body, text="Mật khẩu:", font=("Segoe UI", 9, "bold"), bg="#ffffff", fg="#1e293b").grid(row=2, column=0, sticky="w", pady=(0, 2))
        ent_pass = ttk.Entry(body, font=("Segoe UI", 10), width=32, show="*")
        ent_pass.grid(row=3, column=0, sticky="w", pady=(0, 4))

        var_show_pwd = tk.BooleanVar(value=False)
        def toggle_show():
            ent_pass.config(show="" if var_show_pwd.get() else "*")

        chk_show = tk.Checkbutton(body, text="Hiện mật khẩu", variable=var_show_pwd, bg="#ffffff", font=("Segoe UI", 8), command=toggle_show)
        chk_show.grid(row=4, column=0, sticky="w", pady=(0, 6))

        lbl_hint = tk.Label(body, text="💡 Tài khoản mặc định: admin / admin123", font=("Segoe UI", 8), fg="#64748b", bg="#ffffff")
        lbl_hint.grid(row=5, column=0, sticky="w")

        btn_f = tk.Frame(dialog, bg="#f8fafc", pady=12, padx=20)
        btn_f.pack(fill="x", side="bottom")

        def do_login():
            u = ent_user.get().strip()
            p = ent_pass.get().strip()
            ok, msg = verify_admin_login(u, p)
            if ok:
                self.is_admin = True
                self.current_admin_user = u
                self.update_sidebar_auth_state()
                dialog.destroy()
                messagebox.showinfo("Đăng nhập thành công", f"Chào mừng {u}! Bạn đã mở khóa toàn bộ quyền Quản trị viên.")
                self.show_tab(target_tab or "dashboard")
            else:
                messagebox.showerror("Đăng nhập thất bại", msg, parent=dialog)

        btn_submit = tk.Button(
            btn_f,
            text="Đăng nhập",
            font=("Segoe UI", 9, "bold"),
            bg=self.PRIMARY_COLOR,
            fg="#ffffff",
            activebackground="#1d4ed8",
            activeforeground="#ffffff",
            relief="flat",
            padx=16,
            pady=5,
            cursor="hand2",
            command=do_login,
        )
        btn_submit.pack(side="right", padx=(5, 0))

        btn_cancel = tk.Button(
            btn_f,
            text="Hủy",
            font=("Segoe UI", 9),
            bg="#e2e8f0",
            fg="#1e293b",
            relief="flat",
            padx=15,
            pady=5,
            cursor="hand2",
            command=dialog.destroy,
        )
        btn_cancel.pack(side="right")

        dialog.bind("<Return>", lambda e: do_login())
        dialog.bind("<Escape>", lambda e: dialog.destroy())
        ent_pass.focus_set()

    def open_change_password_dialog(self):
        """Hộp thoại đổi mật khẩu tài khoản Quản trị viên."""
        dialog = tk.Toplevel(self)
        dialog.title("🔑 Đổi mật khẩu Quản trị viên")
        dialog.geometry("400x340")
        dialog.resizable(False, False)
        dialog.transient(self)
        dialog.grab_set()

        x = self.winfo_x() + max(0, (self.winfo_width() - 400) // 2)
        y = self.winfo_y() + max(0, (self.winfo_height() - 340) // 2)
        dialog.geometry(f"+{x}+{y}")
        dialog.configure(bg="#ffffff")

        hdr = tk.Frame(dialog, bg="#1e293b", pady=15, padx=20)
        hdr.pack(fill="x")
        tk.Label(hdr, text="🔑 Đổi mật khẩu Quản trị", font=("Segoe UI", 13, "bold"), fg="#ffffff", bg="#1e293b").pack(anchor="w")

        body = tk.Frame(dialog, bg="#ffffff", padx=25, pady=15)
        body.pack(fill="both", expand=True)

        tk.Label(body, text="Mật khẩu hiện tại:", font=("Segoe UI", 9, "bold"), bg="#ffffff").grid(row=0, column=0, sticky="w", pady=(0, 2))
        ent_old = ttk.Entry(body, font=("Segoe UI", 10), width=32, show="*")
        ent_old.grid(row=1, column=0, sticky="w", pady=(0, 8))

        tk.Label(body, text="Mật khẩu mới:", font=("Segoe UI", 9, "bold"), bg="#ffffff").grid(row=2, column=0, sticky="w", pady=(0, 2))
        ent_new = ttk.Entry(body, font=("Segoe UI", 10), width=32, show="*")
        ent_new.grid(row=3, column=0, sticky="w", pady=(0, 8))

        tk.Label(body, text="Xác nhận mật khẩu mới:", font=("Segoe UI", 9, "bold"), bg="#ffffff").grid(row=4, column=0, sticky="w", pady=(0, 2))
        ent_confirm = ttk.Entry(body, font=("Segoe UI", 10), width=32, show="*")
        ent_confirm.grid(row=5, column=0, sticky="w", pady=(0, 5))

        btn_f = tk.Frame(dialog, bg="#f8fafc", pady=12, padx=20)
        btn_f.pack(fill="x", side="bottom")

        def do_change():
            old_p = ent_old.get().strip()
            new_p = ent_new.get().strip()
            cfm_p = ent_confirm.get().strip()
            ok, msg = change_admin_password(old_p, new_p, cfm_p)
            if ok:
                dialog.destroy()
                messagebox.showinfo("Thành công", msg)
            else:
                messagebox.showerror("Lỗi", msg, parent=dialog)

        btn_save = tk.Button(
            btn_f,
            text="Lưu mật khẩu",
            font=("Segoe UI", 9, "bold"),
            bg=self.SUCCESS_COLOR,
            fg="#ffffff",
            relief="flat",
            padx=15,
            pady=5,
            cursor="hand2",
            command=do_change,
        )
        btn_save.pack(side="right", padx=(5, 0))

        btn_cancel = tk.Button(
            btn_f,
            text="Hủy",
            font=("Segoe UI", 9),
            bg="#e2e8f0",
            fg="#1e293b",
            relief="flat",
            padx=15,
            pady=5,
            cursor="hand2",
            command=dialog.destroy,
        )
        btn_cancel.pack(side="right")
        ent_old.focus_set()

    def admin_logout(self):
        """Đăng xuất quyền Admin và quay về chế độ chấm công của nhân viên."""
        confirm = messagebox.askyesno("Đăng xuất Quản trị", "Bạn có chắc chắn muốn đăng xuất khỏi quyền Quản trị viên?")
        if confirm:
            self.is_admin = False
            self.current_admin_user = None
            self.update_sidebar_auth_state()
            self.show_tab("realtime")
            messagebox.showinfo("Thông báo", "Đã đăng xuất! Ứng dụng đã chuyển về chế độ Chấm công nhân viên.")

    def confirm_exit_app(self):
        """Hỏi xác nhận và đóng ứng dụng an toàn."""
        confirm = messagebox.askyesno("Thoát ứng dụng", "Bạn có chắc chắn muốn thoát khỏi ứng dụng Chấm công không?")
        if confirm:
            self.on_closing()

    def update_db_badge(self):
        cfg = load_db_config()
        db_type = cfg.get("db_type", "sqlite").upper()

        def worker():
            ok, _ = test_db_connection()
            def apply():
                try:
                    if ok:
                        self.lbl_db_status.config(text=f"🟢 DB: {db_type}", fg="#4ade80")
                    else:
                        self.lbl_db_status.config(text=f"🔴 DB: {db_type} (Lỗi)", fg="#f87171")
                except Exception:
                    pass
            self.after(0, apply)

        threading.Thread(target=worker, daemon=True).start()

    def reload_faces(self):
        """Tải lại danh sách encoding vào bộ nhớ RAM (chạy ngầm)."""
        def worker():
            try:
                faces = load_known_faces()
                self.known_faces = faces
            except Exception as e:
                print(f"Lỗi tải lại faces: {e}")
        threading.Thread(target=worker, daemon=True).start()

    def show_tab(self, tab_key: str):
        # Kiểm tra phân quyền: Nếu tab yêu cầu quyền Admin mà chưa đăng nhập
        if tab_key in self.ADMIN_TABS and not self.is_admin:
            self.open_admin_login_dialog(target_tab=tab_key)
            return

        # Dừng camera các tab khác nếu đang chuyển
        if self.active_tab == "register" and tab_key != "register":
            self.stop_register_camera()
        if self.active_tab == "realtime" and tab_key != "realtime":
            self.stop_realtime_camera()

        self.active_tab = tab_key

        # Highlight nút menu
        for key, btn in self.nav_buttons.items():
            if key == tab_key:
                btn.config(bg=self.SIDEBAR_ACTIVE, fg="#ffffff", font=("Segoe UI", 10, "bold"))
            else:
                btn.config(bg=self.SIDEBAR_BG, fg="#e2e8f0", font=("Segoe UI", 10))

        # Hiển thị frame tương ứng
        for key, frame in self.tabs.items():
            if key == tab_key:
                frame.pack(fill="both", expand=True)
            else:
                frame.pack_forget()

        # Gọi hàm refresh dữ liệu nếu cần
        if tab_key == "dashboard":
            self.refresh_dashboard()
        elif tab_key == "history":
            self.filter_history()
        elif tab_key == "employees":
            self.refresh_employee_list()
        elif tab_key == "database":
            self.refresh_db_settings_tab()

        self.update_db_badge()

    # ==========================================
    # 1. TAB: DASHBOARD
    # ==========================================
    def _init_dashboard_tab(self):
        tab = tk.Frame(self.content_area, bg=self.CONTENT_BG)
        self.tabs["dashboard"] = tab

        # Tiêu đề
        lbl_h = tk.Label(
            tab,
            text="📊 Tổng quan hệ thống",
            font=("Segoe UI", 18, "bold"),
            fg=self.TEXT_MAIN,
            bg=self.CONTENT_BG,
        )
        lbl_h.pack(anchor="w", pady=(0, 15))

        # Khối thẻ thống kê (Metric Cards)
        cards_frame = tk.Frame(tab, bg=self.CONTENT_BG)
        cards_frame.pack(fill="x", pady=(0, 20))

        self.card_total_emp = self._create_metric_card(cards_frame, "Tổng số nhân viên", "0", "#3b82f6")
        self.card_total_emp.pack(side="left", fill="both", expand=True, padx=(0, 10))

        self.card_today_checkin = self._create_metric_card(cards_frame, "Có chấm công hôm nay", "0", "#10b981")
        self.card_today_checkin.pack(side="left", fill="both", expand=True, padx=5)

        self.card_today_completed = self._create_metric_card(cards_frame, "Đã có giờ ra (Check-out)", "0", "#8b5cf6")
        self.card_today_completed.pack(side="left", fill="both", expand=True, padx=(10, 0))

        # Bảng dữ liệu hôm nay
        table_container = tk.Frame(tab, bg=self.CARD_BG, highlightbackground=self.BORDER_COLOR, highlightthickness=1)
        table_container.pack(fill="both", expand=True)

        header_frame = tk.Frame(table_container, bg=self.CARD_BG, pady=12, padx=15)
        header_frame.pack(fill="x")

        tk.Label(
            header_frame,
            text="Danh sách chấm công hôm nay",
            font=("Segoe UI", 12, "bold"),
            fg=self.TEXT_MAIN,
            bg=self.CARD_BG,
        ).pack(side="left")

        btn_refresh = tk.Button(
            header_frame,
            text="🔄 Làm mới",
            font=("Segoe UI", 9),
            bg="#f1f5f9",
            fg=self.TEXT_MAIN,
            relief="flat",
            padx=10,
            pady=4,
            command=self.refresh_dashboard,
        )
        btn_refresh.pack(side="right")

        # Treeview
        columns = ("emp_code", "full_name", "date", "in_time", "out_time", "hours")
        self.dash_tree = ttk.Treeview(
            table_container,
            columns=columns,
            show="headings",
            style="Custom.Treeview",
        )
        self.dash_tree.heading("emp_code", text="Mã NV")
        self.dash_tree.heading("full_name", text="Họ và Tên")
        self.dash_tree.heading("date", text="Ngày")
        self.dash_tree.heading("in_time", text="Giờ vào")
        self.dash_tree.heading("out_time", text="Giờ ra")
        self.dash_tree.heading("hours", text="Tổng giờ làm")

        self.dash_tree.column("emp_code", width=100, anchor="center")
        self.dash_tree.column("full_name", width=220, anchor="w")
        self.dash_tree.column("date", width=110, anchor="center")
        self.dash_tree.column("in_time", width=110, anchor="center")
        self.dash_tree.column("out_time", width=110, anchor="center")
        self.dash_tree.column("hours", width=120, anchor="center")

        scroll_y = ttk.Scrollbar(table_container, orient="vertical", command=self.dash_tree.yview)
        self.dash_tree.configure(yscrollcommand=scroll_y.set)
        scroll_y.pack(side="right", fill="y")
        self.dash_tree.pack(fill="both", expand=True, padx=15, pady=(0, 15))

    def _create_metric_card(self, parent, title: str, initial_val: str, color_accent: str):
        card = tk.Frame(parent, bg=self.CARD_BG, highlightbackground=self.BORDER_COLOR, highlightthickness=1, pady=18, padx=20)
        lbl_t = tk.Label(card, text=title, font=("Segoe UI", 10), fg=self.TEXT_MUTED, bg=self.CARD_BG)
        lbl_t.pack(anchor="w")
        lbl_v = tk.Label(card, text=initial_val, font=("Segoe UI", 22, "bold"), fg=color_accent, bg=self.CARD_BG)
        lbl_v.pack(anchor="w", pady=(5, 0))
        card.val_label = lbl_v
        return card

    def refresh_dashboard(self):
        def worker():
            try:
                employees = get_employees()
                total_emp = len(employees)

                today = date.today()
                df = get_attendance_report(start_date=today, end_date=today)

                checked_in = int((df["Giờ vào"].notna() & (df["Giờ vào"] != "")).sum()) if not df.empty else 0
                completed = int((df["Giờ ra"].notna() & (df["Giờ ra"] != "")).sum()) if not df.empty else 0

                rows_data = []
                if not df.empty:
                    for _, row in df.iterrows():
                        rows_data.append((
                            row["Mã NV"],
                            row["Họ tên"],
                            row["Ngày"],
                            row["Giờ vào"],
                            row["Giờ ra"],
                            row["Tổng giờ làm"],
                        ))

                def apply():
                    try:
                        self.card_total_emp.val_label.config(text=str(total_emp))
                        self.card_today_checkin.val_label.config(text=str(checked_in))
                        self.card_today_completed.val_label.config(text=str(completed))

                        for item in self.dash_tree.get_children():
                            self.dash_tree.delete(item)

                        for val in rows_data:
                            self.dash_tree.insert("", "end", values=val)
                    except Exception:
                        pass

                self.after(0, apply)
            except Exception as e:
                print(f"Lỗi refresh dashboard: {e}")

        threading.Thread(target=worker, daemon=True).start()


    # ==========================================
    # 2. TAB: ĐĂNG KÝ NHÂN VIÊN
    # ==========================================
    def _init_register_tab(self):
        tab = tk.Frame(self.content_area, bg=self.CONTENT_BG)
        self.tabs["register"] = tab

        lbl_h = tk.Label(
            tab,
            text="👤 Đăng ký nhân viên mới",
            font=("Segoe UI", 18, "bold"),
            fg=self.TEXT_MAIN,
            bg=self.CONTENT_BG,
        )
        lbl_h.pack(anchor="w", pady=(0, 15))

        content_grid = tk.Frame(tab, bg=self.CONTENT_BG)
        content_grid.pack(fill="both", expand=True)

        # Cột trái: Form nhập & Video Preview
        left_col = tk.Frame(content_grid, bg=self.CARD_BG, highlightbackground=self.BORDER_COLOR, highlightthickness=1, padx=20, pady=20)
        left_col.pack(side="left", fill="both", expand=True, padx=(0, 10))

        # Form fields
        form_frame = tk.Frame(left_col, bg=self.CARD_BG)
        form_frame.pack(fill="x", pady=(0, 10))

        tk.Label(form_frame, text="Mã nhân viên:", font=("Segoe UI", 10, "bold"), bg=self.CARD_BG).grid(row=0, column=0, sticky="w", pady=5)
        self.ent_reg_code = ttk.Entry(form_frame, font=("Segoe UI", 10), width=20)
        self.ent_reg_code.grid(row=0, column=1, sticky="w", padx=10, pady=5)

        tk.Label(form_frame, text="Họ và tên:", font=("Segoe UI", 10, "bold"), bg=self.CARD_BG).grid(row=1, column=0, sticky="w", pady=5)
        self.ent_reg_name = ttk.Entry(form_frame, font=("Segoe UI", 10), width=30)
        self.ent_reg_name.grid(row=1, column=1, sticky="w", padx=10, pady=5)

        # Video Frame
        self.lbl_reg_video = tk.Label(
            left_col,
            text="Camera đang tắt\nBấm 'Bật Camera' để chụp ảnh",
            font=("Segoe UI", 11),
            bg="#1e293b",
            fg="#94a3b8",
            width=50,
            height=16,
        )
        self.lbl_reg_video.pack(fill="both", expand=True, pady=10)

        # Button Controls
        btn_box = tk.Frame(left_col, bg=self.CARD_BG)
        btn_box.pack(fill="x", pady=5)

        self.btn_reg_cam_toggle = ttk.Button(
            btn_box,
            text="▶ Bật Camera",
            style="Primary.TButton",
            command=self.toggle_register_camera,
        )
        self.btn_reg_cam_toggle.pack(side="left", padx=5)

        self.btn_reg_capture = ttk.Button(
            btn_box,
            text="📸 Chụp ảnh",
            style="Primary.TButton",
            command=self.capture_register_photo,
        )
        self.btn_reg_capture.pack(side="left", padx=5)

        btn_upload = ttk.Button(
            btn_box,
            text="📁 Chọn file ảnh",
            command=self.upload_register_photos,
        )
        btn_upload.pack(side="left", padx=5)

        btn_clear = ttk.Button(
            btn_box,
            text="🗑️ Xóa hết ảnh",
            command=self.clear_register_photos,
        )
        btn_clear.pack(side="left", padx=5)

        # Cột phải: Danh sách ảnh đã chụp & Lưu hồ sơ
        right_col = tk.Frame(content_grid, bg=self.CARD_BG, highlightbackground=self.BORDER_COLOR, highlightthickness=1, padx=20, pady=20, width=380)
        right_col.pack(side="right", fill="both", padx=(10, 0))
        right_col.pack_propagate(False)

        self.lbl_photo_count = tk.Label(
            right_col,
            text="Ảnh đã chụp: 0/8–10 ảnh",
            font=("Segoe UI", 12, "bold"),
            fg=self.PRIMARY_COLOR,
            bg=self.CARD_BG,
        )
        self.lbl_photo_count.pack(anchor="w")

        tk.Label(
            right_col,
            text="Cần tối thiểu 5 ảnh có 1 mặt rõ nét, các góc chụp/biểu cảm hơi khác nhau.",
            font=("Segoe UI", 9),
            fg=self.TEXT_MUTED,
            bg=self.CARD_BG,
            wraplength=340,
            justify="left",
        ).pack(anchor="w", pady=(2, 10))

        # Gallery Canvas hiển thị thumbnails
        thumb_container = tk.Frame(right_col, bg="#f1f5f9")
        thumb_container.pack(fill="both", expand=True, pady=5)

        self.thumb_canvas = tk.Canvas(thumb_container, bg="#f1f5f9", highlightthickness=0)
        self.thumb_scrollbar = ttk.Scrollbar(thumb_container, orient="vertical", command=self.thumb_canvas.yview)
        self.thumb_inner = tk.Frame(self.thumb_canvas, bg="#f1f5f9")

        self.thumb_inner.bind(
            "<Configure>",
            lambda e: self.thumb_canvas.configure(scrollregion=self.thumb_canvas.bbox("all")),
        )
        self.thumb_canvas.create_window((0, 0), window=self.thumb_inner, anchor="nw")
        self.thumb_canvas.configure(yscrollcommand=self.thumb_scrollbar.set)

        self.thumb_canvas.pack(side="left", fill="both", expand=True)
        self.thumb_scrollbar.pack(side="right", fill="y")

        # Tiến trình & Nút Lưu
        self.reg_progress = ttk.Progressbar(right_col, mode="indeterminate")

        self.btn_save_reg = ttk.Button(
            right_col,
            text="💾 Encode & Lưu Nhân Viên",
            style="Success.TButton",
            command=self.save_employee_registration,
        )
        self.btn_save_reg.pack(fill="x", pady=10)

    def toggle_register_camera(self):
        if self.reg_cam_running:
            self.stop_register_camera()
        else:
            self.start_register_camera()

    def start_register_camera(self):
        if self.reg_cam_running:
            return
        if not self.reg_grabber.start():
            messagebox.showerror("Lỗi Camera", "Không thể mở webcam (Camera index 0).")
            return
        self.reg_cam_running = True
        self.btn_reg_cam_toggle.config(text="⏹ Tắt Camera")
        self._update_register_camera_feed()

    def stop_register_camera(self):
        self.reg_cam_running = False
        self.reg_grabber.stop()
        self.btn_reg_cam_toggle.config(text="▶ Bật Camera")
        self.lbl_reg_video.config(image="", text="Camera đang tắt\nBấm 'Bật Camera' để chụp ảnh")

    def _update_register_camera_feed(self):
        if not self.reg_cam_running:
            return
        frame = self.reg_grabber.get_frame()
        if frame is not None:
            # Resize frame để vừa khung giao diện
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame_rgb.shape[:2]
            target_w = 480
            target_h = int(h * (target_w / w))
            resized = cv2.resize(frame_rgb, (target_w, target_h))
            img_pil = Image.fromarray(resized)
            img_tk = ImageTk.PhotoImage(image=img_pil)
            self.lbl_reg_video.img_tk = img_tk
            self.lbl_reg_video.config(image=img_tk, text="")

        self.after(30, self._update_register_camera_feed)

    def capture_register_photo(self):
        if not self.reg_cam_running:
            messagebox.showwarning("Cảnh báo", "Vui lòng bật Camera trước khi chụp ảnh.")
            return
        frame = self.reg_grabber.get_frame()
        if frame is not None:
            self.register_photos.append(frame)
            self._update_thumbnails()

    def upload_register_photos(self):
        files = filedialog.askopenfilenames(
            title="Chọn các file ảnh nhân viên",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp")],
        )
        if files:
            for p in files:
                img = cv2.imread(p)
                if img is not None:
                    self.register_photos.append(img)
            self._update_thumbnails()

    def clear_register_photos(self):
        self.register_photos.clear()
        self._update_thumbnails()

    def _update_thumbnails(self):
        count = len(self.register_photos)
        self.lbl_photo_count.config(text=f"Ảnh đã chụp: {count}/8–10 ảnh")

        # Xóa widget cũ trong thumbnail frame
        for child in self.thumb_inner.winfo_children():
            child.destroy()

        # Hiển thị thumbnails dạng grid (3 cột)
        cols = 3
        for i, photo_bgr in enumerate(self.register_photos):
            photo_rgb = cv2.cvtColor(photo_bgr, cv2.COLOR_BGR2RGB)
            thumb_pil = Image.fromarray(photo_rgb).resize((90, 70))
            thumb_tk = ImageTk.PhotoImage(thumb_pil)

            card = tk.Frame(self.thumb_inner, bg="#ffffff", bd=1, relief="solid")
            row = i // cols
            col = i % cols
            card.grid(row=row, column=col, padx=4, pady=4)

            lbl_img = tk.Label(card, image=thumb_tk, bg="#ffffff")
            lbl_img.image = thumb_tk
            lbl_img.pack()

            lbl_num = tk.Label(card, text=f"Ảnh {i+1}", font=("Segoe UI", 8), bg="#ffffff", fg=self.TEXT_MUTED)
            lbl_num.pack()

    def save_employee_registration(self):
        code = self.ent_reg_code.get().strip().upper()
        name = self.ent_reg_name.get().strip()

        if not code or not name:
            messagebox.showerror("Thiếu thông tin", "Vui lòng nhập đầy đủ Mã nhân viên và Họ tên.")
            return

        if len(self.register_photos) < 5:
            messagebox.showwarning(
                "Chưa đủ ảnh",
                f"Bạn mới chụp {len(self.register_photos)} ảnh. Vui lòng chụp tối thiểu 5 ảnh (khuyến nghị 8-10 ảnh).",
            )
            return

        if get_employee(code):
            messagebox.showerror("Trùng mã", f"Mã nhân viên '{code}' đã tồn tại trong CSDL!")
            return

        self.btn_save_reg.config(state="disabled")
        self.reg_progress.pack(fill="x", pady=5)
        self.reg_progress.start()

        # Chạy encode trong background thread để không lag giao diện
        def worker():
            try:
                encoding = build_employee_encoding(self.register_photos, min_valid_photos=3)
                if encoding is None:
                    self.after(0, lambda: messagebox.showerror(
                        "Lỗi Encode",
                        "Không thể nhận diện khuôn mặt rõ nét từ các ảnh đã chụp.\n"
                        "Vui lòng đảm bảo mỗi ảnh có đúng 1 khuôn mặt, đủ sáng và không bị che khuất."
                    ))
                    return

                # Lưu ảnh vào thư mục dataset/MãNV
                emp_dir = os.path.join(DATASET_DIR, code)
                os.makedirs(emp_dir, exist_ok=True)
                for idx, photo_bgr in enumerate(self.register_photos, start=1):
                    file_path = os.path.join(emp_dir, f"{idx:02d}.jpg")
                    cv2.imwrite(file_path, photo_bgr)

                # Lưu vào database
                add_employee(code, name, encoding)
                self.reload_faces()

                def on_success():
                    messagebox.showinfo(
                        "Thành công",
                        f"Đã đăng ký thành công nhân viên:\n[{code}] {name}\nĐã lưu {len(self.register_photos)} ảnh và vector khuôn mặt vào CSDL.",
                    )
                    self.ent_reg_code.delete(0, tk.END)
                    self.ent_reg_name.delete(0, tk.END)
                    self.clear_register_photos()
                    self.stop_register_camera()

                self.after(0, on_success)
            except Exception as ex:
                err_msg = str(ex)
                def show_err(msg=err_msg):
                    messagebox.showerror("Lỗi khi lưu dữ liệu", f"Chi tiết lỗi: {msg}")
                self.after(0, show_err)
            finally:
                def reset_ui():
                    self.reg_progress.stop()
                    self.reg_progress.pack_forget()
                    self.btn_save_reg.config(state="normal")
                self.after(0, reset_ui)


        threading.Thread(target=worker, daemon=True).start()

    # ==========================================
    # 3. TAB: CHẤM CÔNG REALTIME
    # ==========================================
    def _init_realtime_tab(self):
        tab = tk.Frame(self.content_area, bg=self.CONTENT_BG)
        self.tabs["realtime"] = tab

        lbl_h = tk.Label(
            tab,
            text="📷 Chấm công nhận diện khuôn mặt Realtime",
            font=("Segoe UI", 18, "bold"),
            fg=self.TEXT_MAIN,
            bg=self.CONTENT_BG,
        )
        lbl_h.pack(anchor="w", pady=(0, 10))

        content_grid = tk.Frame(tab, bg=self.CONTENT_BG)
        content_grid.pack(fill="both", expand=True)

        # Cột trái: Camera Stream
        left_col = tk.Frame(content_grid, bg=self.CARD_BG, highlightbackground=self.BORDER_COLOR, highlightthickness=1, padx=15, pady=15)
        left_col.pack(side="left", fill="both", expand=True, padx=(0, 10))

        # Controls bar
        top_ctrl = tk.Frame(left_col, bg=self.CARD_BG)
        top_ctrl.pack(fill="x", pady=(0, 10))

        self.btn_rec_toggle = ttk.Button(
            top_ctrl,
            text="▶ Bắt đầu điểm danh",
            style="Success.TButton",
            command=self.toggle_realtime_camera,
        )
        self.btn_rec_toggle.pack(side="left", padx=(0, 10))

        tk.Label(top_ctrl, text="Độ nhạy:", font=("Segoe UI", 9), bg=self.CARD_BG).pack(side="left", padx=(0, 2))
        self.scale_tol = ttk.Scale(top_ctrl, from_=0.35, to=0.65, value=0.50, orient="horizontal", length=90)
        self.scale_tol.pack(side="left", padx=2)

        self.lbl_tol_val = tk.Label(top_ctrl, text="0.50", font=("Segoe UI", 9, "bold"), bg=self.CARD_BG, fg=self.PRIMARY_COLOR)
        self.lbl_tol_val.pack(side="left", padx=(2, 10))
        self.scale_tol.configure(command=lambda v: self.lbl_tol_val.config(text=f"{float(v):.2f}"))

        self.var_voice_enabled = tk.BooleanVar(value=True)
        chk_voice = tk.Checkbutton(
            top_ctrl,
            text="🔊 Giọng nói",
            variable=self.var_voice_enabled,
            bg=self.CARD_BG,
            font=("Segoe UI", 9),
            activebackground=self.CARD_BG,
        )
        chk_voice.pack(side="left", padx=5)

        self.var_auto_stop = tk.BooleanVar(value=True)
        chk_autostop = tk.Checkbutton(
            top_ctrl,
            text="⏹ Tự động kết thúc khi xong",
            variable=self.var_auto_stop,
            bg=self.CARD_BG,
            font=("Segoe UI", 9),
            activebackground=self.CARD_BG,
        )
        chk_autostop.pack(side="left", padx=5)

        # Video canvas
        self.lbl_rec_video = tk.Label(
            left_col,
            text="Camera đang tắt\nBấm 'Bắt đầu điểm danh' để nhận diện",
            font=("Segoe UI", 12),
            bg="#0f172a",
            fg="#94a3b8",
            width=65,
            height=20,
        )
        self.lbl_rec_video.pack(fill="both", expand=True, pady=5)

        # Status Banner dưới camera
        self.rec_status_banner = tk.Label(
            left_col,
            text="Sẵn sàng điểm danh",
            font=("Segoe UI", 12, "bold"),
            bg="#f1f5f9",
            fg=self.TEXT_MUTED,
            pady=10,
        )
        self.rec_status_banner.pack(fill="x", pady=(5, 0))

        # Cột phải: Log sự kiện tức thì
        right_col = tk.Frame(content_grid, bg=self.CARD_BG, highlightbackground=self.BORDER_COLOR, highlightthickness=1, padx=15, pady=15, width=360)
        right_col.pack(side="right", fill="both", padx=(10, 0))
        right_col.pack_propagate(False)

        tk.Label(
            right_col,
            text="Lịch sử nhận diện tức thì",
            font=("Segoe UI", 12, "bold"),
            fg=self.TEXT_MAIN,
            bg=self.CARD_BG,
        ).pack(anchor="w", pady=(0, 10))

        # Log Treeview
        log_cols = ("time", "code", "name", "type")
        self.log_tree = ttk.Treeview(right_col, columns=log_cols, show="headings", style="Custom.Treeview")
        self.log_tree.heading("time", text="Thời gian")
        self.log_tree.heading("code", text="Mã NV")
        self.log_tree.heading("name", text="Họ tên")
        self.log_tree.heading("type", text="Trạng thái")

        self.log_tree.column("time", width=75, anchor="center")
        self.log_tree.column("code", width=65, anchor="center")
        self.log_tree.column("name", width=110, anchor="w")
        self.log_tree.column("type", width=80, anchor="center")

        scroll_log = ttk.Scrollbar(right_col, orient="vertical", command=self.log_tree.yview)
        self.log_tree.configure(yscrollcommand=scroll_log.set)
        scroll_log.pack(side="right", fill="y")
        self.log_tree.pack(fill="both", expand=True)

    def toggle_realtime_camera(self):
        if self.rec_cam_running:
            self.stop_realtime_camera()
        else:
            self.start_realtime_camera()

    def start_realtime_camera(self):
        if self.rec_cam_running:
            return

        # Luôn tải lại danh sách khuôn mặt mới nhất từ CSDL
        self.known_faces = load_known_faces()
        if not self.known_faces["encodings"]:
            messagebox.showwarning(
                "Chưa có dữ liệu",
                "Chưa có nhân viên nào có face encoding trong CSDL.\nVui lòng vào tab 'Đăng ký nhân viên' trước!",
            )
            return

        if not self.rec_grabber.start():
            messagebox.showerror("Lỗi Camera", "Không thể mở webcam (Camera index 0).")
            return

        self.rec_cam_running = True
        self.last_detected_faces = []
        self.recent_attendance_attempts = {}
        self.btn_rec_toggle.config(text="⏹ Dừng Camera", style="Danger.TButton")
        self.rec_status_banner.config(text="Camera đang hoạt động • Vui lòng nhìn thẳng vào ống kính", bg="#dbeafe", fg="#1e40af")

        # 1. Khởi động AI Background Worker (xử lý nhận diện ngầm không lag UI)
        threading.Thread(target=self._realtime_ai_worker, daemon=True).start()

        # 2. Khởi động vòng lặp render video mượt mà trên UI (30 FPS)
        self._update_realtime_camera_feed()

    def stop_realtime_camera(self):
        self.rec_cam_running = False
        self.rec_grabber.stop()
        self.last_detected_faces = []
        self.btn_rec_toggle.config(text="▶ Bắt đầu điểm danh", style="Success.TButton")
        self.lbl_rec_video.config(image="", text="Camera đang tắt\nBấm 'Bắt đầu điểm danh' để nhận diện")
        self.rec_status_banner.config(text="Đã dừng nhận diện", bg="#f1f5f9", fg=self.TEXT_MUTED)

    def _realtime_ai_worker(self):
        """Worker thread nhận diện khuôn mặt ngầm tách biệt hoàn toàn."""
        while self.rec_cam_running:
            frame = self.rec_grabber.get_frame()
            if frame is None or not self.known_faces["encodings"]:
                time.sleep(0.04)
                continue

            try:
                try:
                    tolerance = float(self.scale_tol.get())
                except Exception:
                    tolerance = 0.50

                small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                locations, names, codes, distances = recognize_faces(
                    small,
                    self.known_faces["encodings"],
                    self.known_faces["names"],
                    known_codes=self.known_faces["codes"],
                    tolerance=tolerance,
                )

                new_faces = []
                now_ts = time.time()
                for (top, right, bottom, left), name, code, distance in zip(
                    locations, names, codes, distances
                ):
                    top *= 2
                    right *= 2
                    bottom *= 2
                    left *= 2

                    if name != "Unknown" and code != "Unknown":
                        color = (0, 255, 0)
                        label = f"{code} - {name} ({distance:.2f})"
                        # Debounce: chỉ gọi register_attendance nếu chưa gọi trong 6 giây gần nhất
                        if now_ts - self.recent_attendance_attempts.get(code, 0) > 6.0:
                            self.recent_attendance_attempts[code] = now_ts
                            res = register_attendance(code)
                            if res and res.get("success"):
                                self.after(0, lambda r=res: self._handle_attendance_result(r))
                    else:
                        color = (0, 0, 255)
                        label = "Unknown"

                    new_faces.append((top, right, bottom, left, label, color))

                self.last_detected_faces = new_faces
            except Exception as ex:
                pass

            # Nghỉ 50ms giữa các lần nhận diện (~15 FPS nhận diện ngầm, giữ CPU mát mẻ)
            time.sleep(0.05)

    def _update_realtime_camera_feed(self):
        """Render frame camera và vẽ bounding box trên luồng UI (30 FPS siêu mượt)."""
        if not self.rec_cam_running:
            return

        frame = self.rec_grabber.get_frame()
        if frame is not None:
            # Vẽ các bounding box mới nhất lên frame
            for (top, right, bottom, left, label, color) in list(self.last_detected_faces):
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.rectangle(frame, (left, bottom - 30), (right, bottom), color, cv2.FILLED)
                cv2.putText(
                    frame,
                    label,
                    (left + 6, bottom - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0) if color == (0, 255, 0) else (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            # Chuyển đổi hiển thị lên Canvas
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame_rgb.shape[:2]
            target_w = 640
            target_h = int(h * (target_w / w))
            resized = cv2.resize(frame_rgb, (target_w, target_h))
            img_pil = Image.fromarray(resized)
            img_tk = ImageTk.PhotoImage(image=img_pil)
            self.lbl_rec_video.img_tk = img_tk
            self.lbl_rec_video.config(image=img_tk, text="")

        self.after(30, self._update_realtime_camera_feed)



    def _handle_attendance_result(self, res: dict):
        if not res:
            return
        if res.get("success"):
            att_type = "CHECK-IN" if res["type"] == "check-in" else "CHECK-OUT"
            code = res["employee_code"]
            name = res["employee_name"]
            now_time = datetime.now().strftime("%H:%M:%S")

            banner_text = f"✅ {att_type} THÀNH CÔNG: [{code}] {name} lúc {now_time}"
            bg_color = "#dcfce7" if att_type == "CHECK-IN" else "#fef3c7"
            fg_color = "#166534" if att_type == "CHECK-IN" else "#92400e"

            self.rec_status_banner.config(text=banner_text, bg=bg_color, fg=fg_color)

            # Thêm vào bảng log
            self.log_tree.insert("", 0, values=(now_time, code, name, att_type))

            # 1. Phát giọng nói: "Xin cảm ơn <Tên nhân viên>!"
            if getattr(self, "var_voice_enabled", None) and self.var_voice_enabled.get():
                speak_async(employee_name=name)

            # 2. Tự động kết thúc điểm danh nếu đang bật chế độ Tự động kết thúc
            if getattr(self, "var_auto_stop", None) and self.var_auto_stop.get():
                def auto_finish():
                    if self.rec_cam_running:
                        self.stop_realtime_camera()
                        self.rec_status_banner.config(
                            text=f"🎉 Đã hoàn tất điểm danh cho [{code}] {name} ({att_type}). Đã kết thúc phiên!",
                            bg=bg_color,
                            fg=fg_color,
                        )
                        self.refresh_dashboard()

                self.after(1600, auto_finish)

    # ==========================================
    # 4. TAB: LỊCH SỬ CHẤM CÔNG
    # ==========================================
    def _init_history_tab(self):
        tab = tk.Frame(self.content_area, bg=self.CONTENT_BG)
        self.tabs["history"] = tab

        lbl_h = tk.Label(
            tab,
            text="📋 Lịch sử chấm công",
            font=("Segoe UI", 18, "bold"),
            fg=self.TEXT_MAIN,
            bg=self.CONTENT_BG,
        )
        lbl_h.pack(anchor="w", pady=(0, 15))

        # Thanh lọc (Filter bar)
        filter_card = tk.Frame(tab, bg=self.CARD_BG, highlightbackground=self.BORDER_COLOR, highlightthickness=1, padx=15, pady=12)
        filter_card.pack(fill="x", pady=(0, 15))

        today_str = date.today().strftime("%Y-%m-%d")

        tk.Label(filter_card, text="Từ ngày (YYYY-MM-DD):", font=("Segoe UI", 9, "bold"), bg=self.CARD_BG).pack(side="left", padx=(0, 5))
        self.ent_hist_start = ttk.Entry(filter_card, width=12, font=("Segoe UI", 9))
        self.ent_hist_start.insert(0, today_str)
        self.ent_hist_start.pack(side="left", padx=(0, 15))

        tk.Label(filter_card, text="Đến ngày (YYYY-MM-DD):", font=("Segoe UI", 9, "bold"), bg=self.CARD_BG).pack(side="left", padx=(0, 5))
        self.ent_hist_end = ttk.Entry(filter_card, width=12, font=("Segoe UI", 9))
        self.ent_hist_end.insert(0, today_str)
        self.ent_hist_end.pack(side="left", padx=(0, 15))

        tk.Label(filter_card, text="Nhân viên:", font=("Segoe UI", 9, "bold"), bg=self.CARD_BG).pack(side="left", padx=(0, 5))
        self.cb_hist_emp = ttk.Combobox(filter_card, state="readonly", width=25, font=("Segoe UI", 9))
        self.cb_hist_emp.pack(side="left", padx=(0, 15))

        btn_filter = ttk.Button(filter_card, text="🔍 Tìm kiếm", style="Primary.TButton", command=self.filter_history)
        btn_filter.pack(side="left", padx=5)

        # Bảng hiển thị kết quả
        table_container = tk.Frame(tab, bg=self.CARD_BG, highlightbackground=self.BORDER_COLOR, highlightthickness=1)
        table_container.pack(fill="both", expand=True)

        cols = ("emp_code", "full_name", "date", "in_time", "out_time", "hours")
        self.hist_tree = ttk.Treeview(table_container, columns=cols, show="headings", style="Custom.Treeview")
        self.hist_tree.heading("emp_code", text="Mã NV")
        self.hist_tree.heading("full_name", text="Họ và Tên")
        self.hist_tree.heading("date", text="Ngày")
        self.hist_tree.heading("in_time", text="Giờ vào")
        self.hist_tree.heading("out_time", text="Giờ ra")
        self.hist_tree.heading("hours", text="Tổng giờ làm")

        self.hist_tree.column("emp_code", width=110, anchor="center")
        self.hist_tree.column("full_name", width=240, anchor="w")
        self.hist_tree.column("date", width=120, anchor="center")
        self.hist_tree.column("in_time", width=120, anchor="center")
        self.hist_tree.column("out_time", width=120, anchor="center")
        self.hist_tree.column("hours", width=120, anchor="center")

        scroll_y = ttk.Scrollbar(table_container, orient="vertical", command=self.hist_tree.yview)
        self.hist_tree.configure(yscrollcommand=scroll_y.set)
        scroll_y.pack(side="right", fill="y")
        self.hist_tree.pack(fill="both", expand=True, padx=10, pady=10)

    def filter_history(self):
        start_s = self.ent_hist_start.get().strip()
        end_s = self.ent_hist_end.get().strip()
        sel_emp = self.cb_hist_emp.get()
        code_filter = None if (not sel_emp or sel_emp == "Tất cả") else sel_emp.split(" — ")[0]

        def worker():
            try:
                employees = get_employees()
                emp_options = ["Tất cả"] + [f"{e['employee_code']} — {e['full_name']}" for e in employees]

                df = get_attendance_report(
                    start_date=start_s if start_s else None,
                    end_date=end_s if end_s else None,
                    employee_code=code_filter,
                )

                rows_data = []
                if not df.empty:
                    for _, row in df.iterrows():
                        rows_data.append((
                            row["Mã NV"],
                            row["Họ tên"],
                            row["Ngày"],
                            row["Giờ vào"],
                            row["Giờ ra"],
                            row["Tổng giờ làm"],
                        ))

                def apply():
                    try:
                        curr_sel = self.cb_hist_emp.get()
                        self.cb_hist_emp["values"] = emp_options
                        if not curr_sel or curr_sel not in emp_options:
                            self.cb_hist_emp.current(0)

                        for item in self.hist_tree.get_children():
                            self.hist_tree.delete(item)

                        for val in rows_data:
                            self.hist_tree.insert("", "end", values=val)
                    except Exception:
                        pass

                self.after(0, apply)
            except Exception as e:
                err_text = str(e)
                def show_err(msg=err_text):
                    messagebox.showerror("Lỗi truy vấn", f"Không thể lấy dữ liệu lịch sử: {msg}")
                self.after(0, show_err)

        threading.Thread(target=worker, daemon=True).start()



    # ==========================================
    # 5. TAB: XUẤT BÁO CÁO EXCEL
    # ==========================================
    def _init_export_tab(self):
        tab = tk.Frame(self.content_area, bg=self.CONTENT_BG)
        self.tabs["export"] = tab

        lbl_h = tk.Label(
            tab,
            text="📈 Xuất báo cáo chấm công Excel",
            font=("Segoe UI", 18, "bold"),
            fg=self.TEXT_MAIN,
            bg=self.CONTENT_BG,
        )
        lbl_h.pack(anchor="w", pady=(0, 15))

        card = tk.Frame(tab, bg=self.CARD_BG, highlightbackground=self.BORDER_COLOR, highlightthickness=1, padx=25, pady=25)
        card.pack(fill="x", pady=(0, 20))

        today_str = date.today().strftime("%Y-%m-%d")
        month_start_str = date.today().replace(day=1).strftime("%Y-%m-%d")

        form_f = tk.Frame(card, bg=self.CARD_BG)
        form_f.pack(fill="x", pady=(0, 15))

        tk.Label(form_f, text="Từ ngày:", font=("Segoe UI", 10, "bold"), bg=self.CARD_BG).grid(row=0, column=0, sticky="w", pady=5)
        self.ent_exp_start = ttk.Entry(form_f, width=15, font=("Segoe UI", 10))
        self.ent_exp_start.insert(0, month_start_str)
        self.ent_exp_start.grid(row=0, column=1, sticky="w", padx=10, pady=5)

        tk.Label(form_f, text="Đến ngày:", font=("Segoe UI", 10, "bold"), bg=self.CARD_BG).grid(row=1, column=0, sticky="w", pady=5)
        self.ent_exp_end = ttk.Entry(form_f, width=15, font=("Segoe UI", 10))
        self.ent_exp_end.insert(0, today_str)
        self.ent_exp_end.grid(row=1, column=1, sticky="w", padx=10, pady=5)

        btn_export = ttk.Button(
            card,
            text="📊 Xuất File Excel (.xlsx)",
            style="Success.TButton",
            command=self.export_to_excel,
        )
        btn_export.pack(anchor="w", pady=10)

    def export_to_excel(self):
        start_s = self.ent_exp_start.get().strip()
        end_s = self.ent_exp_end.get().strip()

        df = get_attendance_report(start_date=start_s, end_date=end_s)
        if df.empty:
            messagebox.showinfo("Thông báo", "Không có dữ liệu trong khoảng thời gian đã chọn để xuất file.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel Workbook", "*.xlsx")],
            initialfile=f"Bao_Cao_Cham_Cong_{start_s}_{end_s}.xlsx",
            title="Chọn nơi lưu file Excel",
        )

        if not file_path:
            return

        try:
            with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="ChamCong")
                ws = writer.book["ChamCong"]
                ws.freeze_panes = "A2"
                ws.auto_filter.ref = ws.dimensions

                # Tự động điều chỉnh độ rộng cột
                for col in ws.columns:
                    max_len = max(len(str(cell.value or "")) for cell in col)
                    col_letter = col[0].column_letter
                    ws.column_dimensions[col_letter].width = max(max_len + 4, 14)

            messagebox.showinfo("Xuất thành công", f"Đã xuất file Excel thành công tại:\n{file_path}")
        except Exception as ex:
            messagebox.showerror("Lỗi khi xuất file", f"Chi tiết lỗi: {str(ex)}")

    # ==========================================
    # 6. TAB: QUẢN LÝ NHÂN VIÊN
    # ==========================================
    def _init_employees_tab(self):
        tab = tk.Frame(self.content_area, bg=self.CONTENT_BG)
        self.tabs["employees"] = tab

        lbl_h = tk.Label(
            tab,
            text="⚙️ Quản lý danh sách nhân viên",
            font=("Segoe UI", 18, "bold"),
            fg=self.TEXT_MAIN,
            bg=self.CONTENT_BG,
        )
        lbl_h.pack(anchor="w", pady=(0, 15))

        # Action Bar
        act_card = tk.Frame(tab, bg=self.CARD_BG, highlightbackground=self.BORDER_COLOR, highlightthickness=1, padx=15, pady=10)
        act_card.pack(fill="x", pady=(0, 15))

        btn_reload = ttk.Button(act_card, text="🔄 Tải lại danh sách", command=self.refresh_employee_list)
        btn_reload.pack(side="left", padx=5)

        btn_del = ttk.Button(act_card, text="🗑️ Xóa nhân viên đã chọn", style="Danger.TButton", command=self.delete_selected_employee)
        btn_del.pack(side="left", padx=5)

        # Table
        table_container = tk.Frame(tab, bg=self.CARD_BG, highlightbackground=self.BORDER_COLOR, highlightthickness=1)
        table_container.pack(fill="both", expand=True)

        cols = ("emp_code", "full_name", "created_at", "photo_count")
        self.emp_tree = ttk.Treeview(table_container, columns=cols, show="headings", style="Custom.Treeview")
        self.emp_tree.heading("emp_code", text="Mã NV")
        self.emp_tree.heading("full_name", text="Họ và Tên")
        self.emp_tree.heading("created_at", text="Ngày đăng ký")
        self.emp_tree.heading("photo_count", text="Số ảnh Dataset")

        self.emp_tree.column("emp_code", width=120, anchor="center")
        self.emp_tree.column("full_name", width=260, anchor="w")
        self.emp_tree.column("created_at", width=180, anchor="center")
        self.emp_tree.column("photo_count", width=140, anchor="center")

        scroll_y = ttk.Scrollbar(table_container, orient="vertical", command=self.emp_tree.yview)
        self.emp_tree.configure(yscrollcommand=scroll_y.set)
        scroll_y.pack(side="right", fill="y")
        self.emp_tree.pack(fill="both", expand=True, padx=10, pady=10)

    def refresh_employee_list(self):
        def worker():
            try:
                employees = get_employees()
                rows_data = []
                for emp in employees:
                    code = emp["employee_code"]
                    emp_dir = os.path.join(DATASET_DIR, code)
                    photo_count = len(os.listdir(emp_dir)) if os.path.isdir(emp_dir) else 0
                    rows_data.append((
                        code,
                        emp["full_name"],
                        emp["created_at"],
                        f"{photo_count} ảnh",
                    ))

                def apply():
                    try:
                        for item in self.emp_tree.get_children():
                            self.emp_tree.delete(item)
                        for val in rows_data:
                            self.emp_tree.insert("", "end", values=val)
                    except Exception:
                        pass

                self.after(0, apply)
            except Exception as e:
                err_text = str(e)
                def show_err(msg=err_text):
                    messagebox.showerror("Lỗi", f"Không thể lấy danh sách nhân viên: {msg}")
                self.after(0, show_err)

        threading.Thread(target=worker, daemon=True).start()



    def delete_selected_employee(self):
        sel = self.emp_tree.selection()
        if not sel:
            messagebox.showwarning("Chưa chọn", "Vui lòng chọn 1 nhân viên trong bảng để xóa.")
            return

        item = self.emp_tree.item(sel[0])
        code = item["values"][0]
        name = item["values"][1]

        confirm = messagebox.askyesno(
            "Xác nhận xóa",
            f"Bạn có chắc chắn muốn xóa nhân viên:\n[{code}] {name}\nToàn bộ lịch sử chấm công và ảnh khuôn mặt sẽ bị xóa?",
        )

        if confirm:
            try:
                delete_employee(code)
                self.reload_faces()

                emp_dir = os.path.join(DATASET_DIR, code)
                if os.path.isdir(emp_dir):
                    shutil.rmtree(emp_dir, ignore_errors=True)

                messagebox.showinfo("Thành công", f"Đã xóa nhân viên {code} thành công.")
                self.refresh_employee_list()
            except Exception as ex:
                messagebox.showerror("Lỗi khi xóa", f"Chi tiết lỗi: {str(ex)}")

    # ==========================================
    # 7. TAB: CẤU HÌNH DATABASE
    # ==========================================
    def _init_database_tab(self):
        tab = tk.Frame(self.content_area, bg=self.CONTENT_BG)
        self.tabs["database"] = tab

        lbl_h = tk.Label(
            tab,
            text="🗄️ Cấu hình Cơ sở dữ liệu (Database Settings)",
            font=("Segoe UI", 18, "bold"),
            fg=self.TEXT_MAIN,
            bg=self.CONTENT_BG,
        )
        lbl_h.pack(anchor="w", pady=(0, 15))

        card = tk.Frame(tab, bg=self.CARD_BG, highlightbackground=self.BORDER_COLOR, highlightthickness=1, padx=25, pady=25)
        card.pack(fill="x")

        # Lựa chọn loại CSDL
        tk.Label(card, text="Chọn loại Cơ sở dữ liệu:", font=("Segoe UI", 11, "bold"), bg=self.CARD_BG).pack(anchor="w", pady=(0, 10))

        self.var_db_type = tk.StringVar(value="sqlite")
        rb_frame = tk.Frame(card, bg=self.CARD_BG)
        rb_frame.pack(anchor="w", pady=(0, 15))

        rb_sqlite = tk.Radiobutton(
            rb_frame,
            text="SQLite (Cục bộ, nhẹ & không cần cài server)",
            variable=self.var_db_type,
            value="sqlite",
            font=("Segoe UI", 10),
            bg=self.CARD_BG,
            command=self._on_db_type_changed,
        )
        rb_sqlite.pack(side="left", padx=(0, 20))

        rb_sqlserver = tk.Radiobutton(
            rb_frame,
            text="Microsoft SQL Server (Doanh nghiệp, máy chủ)",
            variable=self.var_db_type,
            value="sqlserver",
            font=("Segoe UI", 10),
            bg=self.CARD_BG,
            command=self._on_db_type_changed,
        )
        rb_sqlserver.pack(side="left")

        # 1. Khung cấu hình SQLite
        self.frame_sqlite_cfg = tk.LabelFrame(card, text="Cấu hình SQLite", font=("Segoe UI", 10, "bold"), bg=self.CARD_BG, padx=15, pady=15)
        self.frame_sqlite_cfg.pack(fill="x", pady=10)

        tk.Label(self.frame_sqlite_cfg, text="Đường dẫn file DB:", font=("Segoe UI", 9), bg=self.CARD_BG).grid(row=0, column=0, sticky="w", pady=5)
        self.ent_sqlite_path = ttk.Entry(self.frame_sqlite_cfg, font=("Segoe UI", 9), width=40)
        self.ent_sqlite_path.grid(row=0, column=1, sticky="w", padx=10, pady=5)

        # 2. Khung cấu hình SQL Server
        self.frame_sqlserver_cfg = tk.LabelFrame(card, text="Cấu hình Microsoft SQL Server", font=("Segoe UI", 10, "bold"), bg=self.CARD_BG, padx=15, pady=15)
        self.frame_sqlserver_cfg.pack(fill="x", pady=10)

        tk.Label(self.frame_sqlserver_cfg, text="Server:", font=("Segoe UI", 9), bg=self.CARD_BG).grid(row=0, column=0, sticky="w", pady=5)
        self.ent_sql_server = ttk.Entry(self.frame_sqlserver_cfg, font=("Segoe UI", 9), width=35)
        self.ent_sql_server.grid(row=0, column=1, sticky="w", padx=10, pady=5)

        tk.Label(self.frame_sqlserver_cfg, text="Database Name:", font=("Segoe UI", 9), bg=self.CARD_BG).grid(row=1, column=0, sticky="w", pady=5)
        self.ent_sql_db = ttk.Entry(self.frame_sqlserver_cfg, font=("Segoe UI", 9), width=35)
        self.ent_sql_db.grid(row=1, column=1, sticky="w", padx=10, pady=5)

        tk.Label(self.frame_sqlserver_cfg, text="ODBC Driver:", font=("Segoe UI", 9), bg=self.CARD_BG).grid(row=2, column=0, sticky="w", pady=5)
        self.cb_sql_driver = ttk.Combobox(self.frame_sqlserver_cfg, font=("Segoe UI", 9), width=33, state="readonly")
        drivers = get_available_sqlserver_drivers()
        self.cb_sql_driver["values"] = drivers
        if drivers:
            self.cb_sql_driver.current(0)
        self.cb_sql_driver.grid(row=2, column=1, sticky="w", padx=10, pady=5)

        self.var_sql_trusted = tk.BooleanVar(value=True)
        self.chk_sql_trusted = tk.Checkbutton(
            self.frame_sqlserver_cfg,
            text="Sử dụng Windows Authentication (Trusted Connection)",
            variable=self.var_sql_trusted,
            font=("Segoe UI", 9),
            bg=self.CARD_BG,
            command=self._on_trusted_auth_changed,
        )
        self.chk_sql_trusted.grid(row=3, column=0, columnspan=2, sticky="w", pady=5)

        tk.Label(self.frame_sqlserver_cfg, text="Username:", font=("Segoe UI", 9), bg=self.CARD_BG).grid(row=4, column=0, sticky="w", pady=5)
        self.ent_sql_user = ttk.Entry(self.frame_sqlserver_cfg, font=("Segoe UI", 9), width=25)
        self.ent_sql_user.grid(row=4, column=1, sticky="w", padx=10, pady=5)

        tk.Label(self.frame_sqlserver_cfg, text="Password:", font=("Segoe UI", 9), bg=self.CARD_BG).grid(row=5, column=0, sticky="w", pady=5)
        self.ent_sql_pwd = ttk.Entry(self.frame_sqlserver_cfg, font=("Segoe UI", 9), width=25, show="*")
        self.ent_sql_pwd.grid(row=5, column=1, sticky="w", padx=10, pady=5)

        # Buttons
        btn_box = tk.Frame(card, bg=self.CARD_BG)
        btn_box.pack(anchor="w", pady=15)

        btn_test = ttk.Button(btn_box, text="🔍 Kiểm tra kết nối", command=self.test_database_settings)
        btn_test.pack(side="left", padx=(0, 10))

        btn_save = ttk.Button(btn_box, text="💾 Lưu cấu hình & Khởi tạo CSDL", style="Success.TButton", command=self.save_database_settings)
        btn_save.pack(side="left")

        self.refresh_db_settings_tab()

    def _on_db_type_changed(self):
        db_t = self.var_db_type.get()
        if db_t == "sqlite":
            self.frame_sqlite_cfg.pack(fill="x", pady=10)
            self.frame_sqlserver_cfg.pack_forget()
        else:
            self.frame_sqlserver_cfg.pack(fill="x", pady=10)
            self.frame_sqlite_cfg.pack_forget()

    def _on_trusted_auth_changed(self):
        is_trusted = self.var_sql_trusted.get()
        if is_trusted:
            self.ent_sql_user.config(state="disabled")
            self.ent_sql_pwd.config(state="disabled")
        else:
            self.ent_sql_user.config(state="normal")
            self.ent_sql_pwd.config(state="normal")

    def refresh_db_settings_tab(self):
        cfg = load_db_config()
        db_type = cfg.get("db_type", "sqlite")
        self.var_db_type.set(db_type)

        self.ent_sqlite_path.delete(0, tk.END)
        self.ent_sqlite_path.insert(0, cfg.get("sqlite_path", "attendance.db"))

        sql_cfg = cfg.get("sqlserver", {})
        self.ent_sql_server.delete(0, tk.END)
        self.ent_sql_server.insert(0, sql_cfg.get("server", "localhost"))

        self.ent_sql_db.delete(0, tk.END)
        self.ent_sql_db.insert(0, sql_cfg.get("database", "FaceAttendanceDB"))

        driver = sql_cfg.get("driver", "ODBC Driver 17 for SQL Server")
        if driver in self.cb_sql_driver["values"]:
            self.cb_sql_driver.set(driver)

        self.var_sql_trusted.set(sql_cfg.get("trusted_connection", True))
        self.ent_sql_user.delete(0, tk.END)
        self.ent_sql_user.insert(0, sql_cfg.get("username", "sa"))

        self.ent_sql_pwd.delete(0, tk.END)
        self.ent_sql_pwd.insert(0, sql_cfg.get("password", ""))

        self._on_db_type_changed()
        self._on_trusted_auth_changed()

    def _get_current_ui_db_config(self):
        return {
            "db_type": self.var_db_type.get(),
            "sqlite_path": self.ent_sqlite_path.get().strip() or "attendance.db",
            "sqlserver": {
                "server": self.ent_sql_server.get().strip() or "localhost",
                "database": self.ent_sql_db.get().strip() or "FaceAttendanceDB",
                "driver": self.cb_sql_driver.get().strip() or "ODBC Driver 17 for SQL Server",
                "trusted_connection": self.var_sql_trusted.get(),
                "username": self.ent_sql_user.get().strip(),
                "password": self.ent_sql_pwd.get(),
            },
        }

    def test_database_settings(self):
        cfg = self._get_current_ui_db_config()
        ok, msg = test_db_connection(cfg)
        if ok:
            messagebox.showinfo("Thành công", f"✅ {msg}")
        else:
            messagebox.showerror("Thất bại", f"❌ {msg}")

    def save_database_settings(self):
        cfg = self._get_current_ui_db_config()
        ok, msg = test_db_connection(cfg)
        if not ok:
            confirm = messagebox.askyesno(
                "Cảnh báo kết nối",
                f"Kiểm tra kết nối thất bại:\n{msg}\n\nBạn có chắc chắn vẫn muốn lưu cấu hình này?",
            )
            if not confirm:
                return

        save_db_config(cfg)
        try:
            init_db(cfg)
            self.reload_faces()
            self.update_db_badge()
            messagebox.showinfo("Thành công", "Đã lưu cấu hình và khởi tạo CSDL thành công!")
        except Exception as ex:
            messagebox.showerror("Lỗi khởi tạo", f"Không thể khởi tạo CSDL: {str(ex)}")

    def on_closing(self):
        self.stop_register_camera()
        self.stop_realtime_camera()
        self.destroy()


def main():
    app = FaceAttendanceApp()
    app.mainloop()


if __name__ == "__main__":
    main()
