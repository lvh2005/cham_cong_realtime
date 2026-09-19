"""
Face Attendance Desktop Application (Tkinter GUI)
Hệ thống chấm công bằng nhận diện khuôn mặt Realtime
Kiến trúc: Tkinter + OpenCV + face_recognition + FAISS + SQL Server + Cloudinary
Quy mô: 10.000+ nhân viên (30.000 - 50.000 face vectors)
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import List, Optional

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageTk

import config
from attendance import get_attendance_report, register_attendance
from auth_utils import change_admin_password, verify_admin_login
from cloudinary_utils import delete_employee_avatar, upload_employee_avatar
from database import (
    activate_employee,
    add_employee,
    add_face_embeddings,
    get_available_sqlserver_drivers,
    get_employee,
    get_employees,
    get_employees_with_stats,
    hard_delete_employee,
    init_db,
    load_db_config,
    save_db_config,
    soft_delete_employee,
    test_db_connection,
    update_employee_face,
)
from faiss_utils import get_faiss_engine
from face_utils import (
    build_employee_embeddings,
    recognize_faces,
    recognize_faces_detailed,
)
from voice_utils import speak_async


class CameraGrabber:
    """
    Luồng độc lập chạy ngầm đọc frame từ webcam (Non-blocking).
    Tách biệt hoàn toàn việc đọc phần cứng camera ra khỏi Main Thread của Tkinter,
    sử dụng MJPG 640x480 @ 30 FPS để đạt 0ms trễ phần cứng, giao diện siêu mượt 30-60 FPS.
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
            self.cap = cv2.VideoCapture(self.src, cv2.CAP_DSHOW)
            if not self.cap or not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.src)

            if not self.cap or not self.cap.isOpened():
                return False

            # Cấu hình MJPG + 640x480 + 30 FPS + Buffer 1 để loại bỏ hoàn toàn độ trễ phần cứng
            try:
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            except Exception:
                pass
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
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
                time.sleep(0.005)

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

        self.title("Face Attendance System • Hệ Thống Chấm Công AI (FAISS + SQL Server)")

        # Căn giữa màn hình
        win_w, win_h = 1260, 800
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        pos_x = max(0, (screen_w - win_w) // 2)
        pos_y = max(0, (screen_h - win_h) // 2 - 20)
        self.geometry(f"{win_w}x{win_h}+{pos_x}+{pos_y}")
        self.minsize(1080, 680)

        # Trạng thái toàn cục
        self.active_tab = "realtime"
        self.is_admin = False
        self.current_admin_user = None
        self.ADMIN_TABS = {"dashboard", "register", "history", "export", "employees", "database"}

        # Camera Grabbers chuyên dụng non-blocking
        self.reg_grabber = CameraGrabber(0)
        self.rec_grabber = CameraGrabber(0)

        # ThreadPoolExecutor cho các tác vụ I/O chạy ngầm (Attendance, DB, Voice)
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="AttendanceIO")

        # Biến cho chức năng Đăng ký
        self.register_photos: List[np.ndarray] = []
        self.reg_cam_running = False

        # Biến cho chức năng Chấm công Realtime
        self.rec_cam_running = False
        self._detected_faces_lock = threading.Lock()
        self.last_detected_faces = []
        self.recent_attendance_attempts = {}  # {employee_code: timestamp}
        self.cached_employees_stats = []

        self._setup_theme()
        self._build_layout()
        self.show_tab("realtime")

        # Bắt sự kiện đóng cửa sổ để giải phóng camera
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Khởi tạo CSDL và nạp dữ liệu FAISS nền
        self.after(100, self._async_init_system)

    def _async_init_system(self):
        def worker():
            try:
                init_db()
            except Exception as e:
                print(f"Lỗi khởi tạo/migration CSDL SQL Server: {e}")

            try:
                engine = get_faiss_engine()
                engine.init_engine()
            except Exception as e:
                print(f"Lỗi khởi động FAISS Engine: {e}")

            self.after(0, self.refresh_dashboard)
            self.after(0, self.update_db_badge)

            def auto_start_if_realtime():
                if self.active_tab == "realtime" and getattr(self, "var_auto_cam", None) and self.var_auto_cam.get():
                    if not self.rec_cam_running:
                        self.start_realtime_camera()

            self.after(400, auto_start_if_realtime)

        threading.Thread(target=worker, daemon=True).start()

    def _setup_theme(self):
        self.style = ttk.Style(self)
        self.style.theme_use("clam")

        self.BG_DARK = "#0f172a"      # Slate 900
        self.SIDEBAR_BG = "#1e293b"  # Slate 800
        self.SIDEBAR_HOVER = "#334155"
        self.SIDEBAR_ACTIVE = "#2563eb"
        self.CONTENT_BG = "#f8fafc"  # Slate 50
        self.CARD_BG = "#ffffff"
        self.BORDER_COLOR = "#e2e8f0"
        self.PRIMARY_COLOR = "#2563eb"
        self.SUCCESS_COLOR = "#16a34a"
        self.WARNING_COLOR = "#d97706"
        self.DANGER_COLOR = "#dc2626"
        self.TEXT_MAIN = "#1e293b"
        self.TEXT_MUTED = "#64748b"

        self.configure(bg=self.CONTENT_BG)

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

        self.style.configure(
            "Primary.TButton",
            background=self.PRIMARY_COLOR,
            foreground="#ffffff",
            font=("Segoe UI", 10, "bold"),
            padding=(12, 6),
            relief="flat",
        )
        self.style.map("Primary.TButton", background=[("active", "#1d4ed8")])

        self.style.configure(
            "Success.TButton",
            background=self.SUCCESS_COLOR,
            foreground="#ffffff",
            font=("Segoe UI", 10, "bold"),
            padding=(12, 6),
            relief="flat",
        )
        self.style.map("Success.TButton", background=[("active", "#15803d")])

        self.style.configure(
            "Danger.TButton",
            background=self.DANGER_COLOR,
            foreground="#ffffff",
            font=("Segoe UI", 10, "bold"),
            padding=(12, 6),
            relief="flat",
        )
        self.style.map("Danger.TButton", background=[("active", "#b91c1c")])

    def _build_layout(self):
        self.main_container = tk.Frame(self, bg=self.CONTENT_BG)
        self.main_container.pack(fill="both", expand=True)

        # 1. Sidebar
        self.sidebar = tk.Frame(self.main_container, bg=self.SIDEBAR_BG, width=250)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

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
            text="FAISS • SQL Server • Cloudinary",
            font=("Segoe UI", 8),
            fg="#94a3b8",
            bg=self.SIDEBAR_BG,
        )
        lbl_sub.pack(anchor="w", padx=20, pady=(2, 0))

        self.lbl_role_badge = tk.Label(
            title_frame,
            text="🟢 Chế độ: Chấm công (Kiosk)",
            font=("Segoe UI", 8, "bold"),
            fg="#6ee7b7",
            bg="#064e3b",
            padx=8,
            pady=3,
        )
        self.lbl_role_badge.pack(anchor="w", padx=20, pady=(6, 0))

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

        tk.Frame(self.sidebar, bg="#334155", height=1).pack(fill="x", padx=15, pady=5)

        # Khung xác thực Admin
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

        self.btn_rebuild_faiss = tk.Button(
            self.admin_logged_frame,
            text="⚡ Rebuild FAISS Index",
            font=("Segoe UI", 8),
            fg="#fef08a",
            bg="#854d0e",
            activebackground="#a16207",
            activeforeground="#ffffff",
            relief="flat",
            pady=4,
            cursor="hand2",
            command=self.rebuild_faiss_manually,
        )
        self.btn_rebuild_faiss.pack(fill="x", pady=(0, 4))

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

        # Bottom Info Badge
        bottom_frame = tk.Frame(self.sidebar, bg="#0f172a", pady=10, padx=15)
        bottom_frame.pack(side="bottom", fill="x")

        self.lbl_db_status = tk.Label(
            bottom_frame,
            text="🟢 DB: SQL Server",
            font=("Segoe UI", 9, "bold"),
            fg="#4ade80",
            bg="#0f172a",
        )
        self.lbl_db_status.pack(anchor="w")

        self.lbl_faiss_status = tk.Label(
            bottom_frame,
            text="⚡ FAISS: 0 vectors",
            font=("Segoe UI", 8),
            fg="#94a3b8",
            bg="#0f172a",
        )
        self.lbl_faiss_status.pack(anchor="w", pady=(1, 6))

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

        # 2. Content Area
        self.content_area = tk.Frame(self.main_container, bg=self.CONTENT_BG)
        self.content_area.pack(side="right", fill="both", expand=True, padx=25, pady=20)

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
                text="🟢 Chế độ: Chấm công (Kiosk)",
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
        dialog = tk.Toplevel(self)
        dialog.title("🔐 Đăng nhập Quản trị viên (Admin)")
        dialog.geometry("420x330")
        dialog.resizable(False, False)
        dialog.transient(self)
        dialog.grab_set()

        x = self.winfo_x() + max(0, (self.winfo_width() - 420) // 2)
        y = self.winfo_y() + max(0, (self.winfo_height() - 330) // 2)
        dialog.geometry(f"+{x}+{y}")
        dialog.configure(bg="#ffffff")

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
        confirm = messagebox.askyesno("Đăng xuất Quản trị", "Bạn có chắc chắn muốn đăng xuất khỏi quyền Quản trị viên?")
        if confirm:
            self.is_admin = False
            self.current_admin_user = None
            self.update_sidebar_auth_state()
            self.show_tab("realtime")
            messagebox.showinfo("Thông báo", "Đã đăng xuất! Ứng dụng đã chuyển về chế độ Chấm công nhân viên.")

    def rebuild_faiss_manually(self):
        def worker():
            try:
                engine = get_faiss_engine()
                engine.rebuild_from_database()
                count = engine.index.ntotal if engine.index else 0
                self.after(0, lambda: messagebox.showinfo("FAISS Rebuild", f"✅ Đã Rebuild thành công FAISS Index!\nTổng số vectors: {count}"))
                self.after(0, self.update_db_badge)
            except Exception as e:
                self.after(0, lambda err=str(e): messagebox.showerror("Lỗi FAISS", f"Không thể rebuild FAISS: {err}"))

        threading.Thread(target=worker, daemon=True).start()

    def confirm_exit_app(self):
        confirm = messagebox.askyesno("Thoát ứng dụng", "Bạn có chắc chắn muốn thoát khỏi ứng dụng Chấm công không?")
        if confirm:
            self.on_closing()

    def update_db_badge(self):
        def worker():
            ok, _ = test_db_connection()
            engine = get_faiss_engine()
            vector_count = engine.index.ntotal if engine.index else 0

            def apply():
                try:
                    if ok:
                        self.lbl_db_status.config(text="🟢 DB: SQL Server", fg="#4ade80")
                    else:
                        self.lbl_db_status.config(text="🔴 DB: SQL Server (Lỗi)", fg="#f87171")
                    self.lbl_faiss_status.config(text=f"⚡ FAISS: {vector_count:,} vectors")
                except Exception:
                    pass

            self.after(0, apply)

        threading.Thread(target=worker, daemon=True).start()

    def show_tab(self, tab_key: str):
        if tab_key in self.ADMIN_TABS and not self.is_admin:
            self.open_admin_login_dialog(target_tab=tab_key)
            return

        if self.active_tab == "register" and tab_key != "register":
            self.stop_register_camera()
        if self.active_tab == "realtime" and tab_key != "realtime":
            self.stop_realtime_camera()

        self.active_tab = tab_key

        for key, btn in self.nav_buttons.items():
            if key == tab_key:
                btn.config(bg=self.SIDEBAR_ACTIVE, fg="#ffffff", font=("Segoe UI", 10, "bold"))
            else:
                btn.config(bg=self.SIDEBAR_BG, fg="#e2e8f0", font=("Segoe UI", 10))

        for key, frame in self.tabs.items():
            if key == tab_key:
                frame.pack(fill="both", expand=True)
            else:
                frame.pack_forget()

        if tab_key == "dashboard":
            self.refresh_dashboard()
        elif tab_key == "register":
            self.refresh_register_employees_dropdown()
        elif tab_key == "realtime":
            if getattr(self, "var_auto_cam", None) and self.var_auto_cam.get():
                if not self.rec_cam_running:
                    self.after(200, self.start_realtime_camera)
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

        lbl_h = tk.Label(
            tab,
            text="📊 Tổng quan hệ thống",
            font=("Segoe UI", 18, "bold"),
            fg=self.TEXT_MAIN,
            bg=self.CONTENT_BG,
        )
        lbl_h.pack(anchor="w", pady=(0, 15))

        cards_frame = tk.Frame(tab, bg=self.CONTENT_BG)
        cards_frame.pack(fill="x", pady=(0, 20))

        self.card_total_emp = self._create_metric_card(cards_frame, "Tổng nhân viên (Active)", "0", "#3b82f6")
        self.card_total_emp.pack(side="left", fill="both", expand=True, padx=(0, 10))

        self.card_today_checkin = self._create_metric_card(cards_frame, "Có chấm công hôm nay", "0", "#10b981")
        self.card_today_checkin.pack(side="left", fill="both", expand=True, padx=5)

        self.card_today_completed = self._create_metric_card(cards_frame, "Đã có giờ ra (Check-out)", "0", "#8b5cf6")
        self.card_today_completed.pack(side="left", fill="both", expand=True, padx=(10, 0))

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
                employees = get_employees(status="ACTIVE")
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
                        self.card_total_emp.val_label.config(text=f"{total_emp:,}")
                        self.card_today_checkin.val_label.config(text=f"{checked_in:,}")
                        self.card_today_completed.val_label.config(text=f"{completed:,}")

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
    # 2. TAB: ĐĂNG KÝ & CẬP NHẬT KHUÔN MẶT NHÂN VIÊN
    # ==========================================
    def _init_register_tab(self):
        tab = tk.Frame(self.content_area, bg=self.CONTENT_BG)
        self.tabs["register"] = tab

        lbl_h = tk.Label(
            tab,
            text="👤 Đăng ký & Cập nhật khuôn mặt nhân viên (Đa Face Embeddings & Cloudinary)",
            font=("Segoe UI", 17, "bold"),
            fg=self.TEXT_MAIN,
            bg=self.CONTENT_BG,
        )
        lbl_h.pack(anchor="w", pady=(0, 12))

        content_grid = tk.Frame(tab, bg=self.CONTENT_BG)
        content_grid.pack(fill="both", expand=True)

        left_col = tk.Frame(content_grid, bg=self.CARD_BG, highlightbackground=self.BORDER_COLOR, highlightthickness=1, padx=18, pady=15)
        left_col.pack(side="left", fill="both", expand=True, padx=(0, 10))

        form_frame = tk.Frame(left_col, bg=self.CARD_BG)
        form_frame.pack(fill="x", pady=(0, 8))

        # Hàng 0: Chọn nhân viên có sẵn trong CSDL
        tk.Label(form_frame, text="Chọn nhân viên:", font=("Segoe UI", 9, "bold"), bg=self.CARD_BG, fg=self.PRIMARY_COLOR).grid(row=0, column=0, sticky="w", pady=4)
        self.cb_reg_select_emp = ttk.Combobox(form_frame, font=("Segoe UI", 9), width=32, state="readonly")
        self.cb_reg_select_emp.grid(row=0, column=1, columnspan=2, sticky="we", padx=(10, 5), pady=4)
        self.cb_reg_select_emp.bind("<<ComboboxSelected>>", self.on_reg_employee_selected)

        btn_reload_emp = ttk.Button(form_frame, text="🔄 Tải lại DS", command=self.refresh_register_employees_dropdown)
        btn_reload_emp.grid(row=0, column=3, sticky="w", padx=5, pady=4)

        # Hàng 1: Mã NV và Họ tên
        tk.Label(form_frame, text="Mã nhân viên (*):", font=("Segoe UI", 9, "bold"), bg=self.CARD_BG).grid(row=1, column=0, sticky="w", pady=4)
        self.ent_reg_code = ttk.Entry(form_frame, font=("Segoe UI", 9), width=16)
        self.ent_reg_code.grid(row=1, column=1, sticky="w", padx=10, pady=4)
        self.ent_reg_code.bind("<KeyRelease>", self.on_reg_code_changed)

        tk.Label(form_frame, text="Họ và tên (*):", font=("Segoe UI", 9, "bold"), bg=self.CARD_BG).grid(row=1, column=2, sticky="w", padx=(10, 0), pady=4)
        self.ent_reg_name = ttk.Entry(form_frame, font=("Segoe UI", 9), width=22)
        self.ent_reg_name.grid(row=1, column=3, sticky="w", padx=10, pady=4)

        # Hàng 2: Phòng ban và Chức vụ
        tk.Label(form_frame, text="Phòng ban:", font=("Segoe UI", 9), bg=self.CARD_BG).grid(row=2, column=0, sticky="w", pady=4)
        self.ent_reg_dept = ttk.Entry(form_frame, font=("Segoe UI", 9), width=16)
        self.ent_reg_dept.grid(row=2, column=1, sticky="w", padx=10, pady=4)

        tk.Label(form_frame, text="Chức vụ:", font=("Segoe UI", 9), bg=self.CARD_BG).grid(row=2, column=2, sticky="w", padx=(10, 0), pady=4)
        self.ent_reg_pos = ttk.Entry(form_frame, font=("Segoe UI", 9), width=22)
        self.ent_reg_pos.grid(row=2, column=3, sticky="w", padx=10, pady=4)

        # Hàng 3: Banner trạng thái chế độ
        self.lbl_reg_mode_status = tk.Label(
            form_frame,
            text="✨ [Tạo mới] Đăng ký nhân viên mới hoặc chọn từ danh sách trên để cập nhật.",
            font=("Segoe UI", 9),
            bg="#f0fdf4",
            fg="#166534",
            padx=10,
            pady=4,
            relief="groove",
            anchor="w",
        )
        self.lbl_reg_mode_status.grid(row=3, column=0, columnspan=4, sticky="we", padx=2, pady=(6, 2))

        # Video Frame
        self.lbl_reg_video = tk.Label(
            left_col,
            text="Camera đang tắt\nBấm 'Bật Camera' để chụp 3-5 góc mặt",
            font=("Segoe UI", 11),
            bg="#1e293b",
            fg="#94a3b8",
            width=50,
            height=15,
        )
        self.lbl_reg_video.pack(fill="both", expand=True, pady=8)

        # Button Controls
        btn_box = tk.Frame(left_col, bg=self.CARD_BG)
        btn_box.pack(fill="x", pady=4)

        self.btn_reg_cam_toggle = ttk.Button(
            btn_box,
            text="▶ Bật Camera",
            style="Primary.TButton",
            command=self.toggle_register_camera,
        )
        self.btn_reg_cam_toggle.pack(side="left", padx=4)

        self.btn_reg_capture = ttk.Button(
            btn_box,
            text="📸 Chụp ảnh",
            style="Primary.TButton",
            command=self.capture_register_photo,
        )
        self.btn_reg_capture.pack(side="left", padx=4)

        btn_upload = ttk.Button(
            btn_box,
            text="📁 Chọn file ảnh",
            command=self.upload_register_photos,
        )
        btn_upload.pack(side="left", padx=4)

        btn_clear = ttk.Button(
            btn_box,
            text="🗑️ Xóa hết ảnh",
            command=self.clear_register_photos,
        )
        btn_clear.pack(side="left", padx=4)

        # Cột phải: Thumbnails & Save
        right_col = tk.Frame(content_grid, bg=self.CARD_BG, highlightbackground=self.BORDER_COLOR, highlightthickness=1, padx=18, pady=18, width=380)
        right_col.pack(side="right", fill="both", padx=(10, 0))
        right_col.pack_propagate(False)

        self.lbl_photo_count = tk.Label(
            right_col,
            text="Ảnh đã chụp: 0/5 ảnh",
            font=("Segoe UI", 12, "bold"),
            fg=self.PRIMARY_COLOR,
            bg=self.CARD_BG,
        )
        self.lbl_photo_count.pack(anchor="w")

        tk.Label(
            right_col,
            text="Hệ thống tạo 3-5 embeddings riêng biệt, tự động kiểm tra trùng khuôn mặt và lưu ảnh đại diện lên Cloudinary.",
            font=("Segoe UI", 8),
            fg=self.TEXT_MUTED,
            bg=self.CARD_BG,
            wraplength=340,
            justify="left",
        ).pack(anchor="w", pady=(2, 8))

        thumb_container = tk.Frame(right_col, bg="#f1f5f9")
        thumb_container.pack(fill="both", expand=True, pady=4)

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

        self.reg_progress = ttk.Progressbar(right_col, mode="indeterminate")

        self.btn_save_reg = ttk.Button(
            right_col,
            text="💾 Tạo Vector, Kiểm Tra Trùng & Lưu",
            style="Success.TButton",
            command=self.save_employee_registration,
        )
        self.btn_save_reg.pack(fill="x", pady=8)

    def refresh_register_employees_dropdown(self):
        def worker():
            try:
                employees = get_employees_with_stats(status=None)
                self.cached_employees_stats = employees
                options = ["➕ [Đăng ký nhân viên mới hoàn toàn]"]
                for emp in employees:
                    code = emp["employee_code"]
                    name = emp["full_name"]
                    cnt = emp["embedding_count"]
                    if cnt == 0:
                        options.append(f"[{code}] {name} — ⚠️ Chưa có khuôn mặt (Web Admin)")
                    else:
                        options.append(f"[{code}] {name} — ✅ {cnt} vectors")

                def apply():
                    try:
                        curr = self.cb_reg_select_emp.get()
                        self.cb_reg_select_emp["values"] = options
                        if not curr or curr not in options:
                            self.cb_reg_select_emp.current(0)
                    except Exception:
                        pass

                self.after(0, apply)
            except Exception as e:
                print(f"Lỗi nạp danh sách nhân viên cho dropdown: {e}")

        threading.Thread(target=worker, daemon=True).start()

    def on_reg_employee_selected(self, event=None):
        sel = self.cb_reg_select_emp.get()
        if not sel or sel.startswith("➕"):
            self.ent_reg_code.delete(0, tk.END)
            self.ent_reg_name.delete(0, tk.END)
            self.ent_reg_dept.delete(0, tk.END)
            self.ent_reg_pos.delete(0, tk.END)
            self.lbl_reg_mode_status.config(
                text="✨ [Tạo mới] Đăng ký nhân viên mới hoàn toàn vào SQL Server & FAISS.",
                bg="#f0fdf4",
                fg="#166534",
            )
            self.btn_save_reg.config(text="💾 Tạo Mới & Lưu Khuôn Mặt")
            return

        if sel.startswith("[") and "]" in sel:
            code = sel[1:sel.index("]")].strip().upper()
            self.load_employee_into_register(code, auto_switch_tab=False)

    def on_reg_code_changed(self, event=None):
        code = self.ent_reg_code.get().strip().upper()
        if not code:
            self.lbl_reg_mode_status.config(
                text="✨ [Tạo mới] Nhập mã để tạo mới hoặc chọn nhân viên từ danh sách.",
                bg="#f0fdf4",
                fg="#166534",
            )
            self.btn_save_reg.config(text="💾 Tạo Mới & Lưu Khuôn Mặt")
            return

        matched_emp = next((e for e in getattr(self, "cached_employees_stats", []) if e["employee_code"].upper() == code), None)
        if not matched_emp:
            try:
                matched_emp = get_employee(code)
            except Exception:
                matched_emp = None

        if matched_emp:
            current_name = self.ent_reg_name.get().strip()
            if not current_name and matched_emp.get("full_name"):
                self.ent_reg_name.delete(0, tk.END)
                self.ent_reg_name.insert(0, matched_emp["full_name"])
                self.ent_reg_dept.delete(0, tk.END)
                self.ent_reg_dept.insert(0, matched_emp.get("department", ""))
                self.ent_reg_pos.delete(0, tk.END)
                self.ent_reg_pos.insert(0, matched_emp.get("position", ""))

            cnt = matched_emp.get("embedding_count", 0)
            if cnt == 0:
                self.lbl_reg_mode_status.config(
                    text=f"⚠️ [Chưa có khuôn mặt] Hồ sơ '{code}' đã có trên CSDL (Web Admin) • Sẵn sàng nạp vector!",
                    bg="#fef3c7",
                    fg="#92400e",
                )
                self.btn_save_reg.config(text="💾 Nạp Khuôn Mặt Cho Nhân Viên")
            else:
                self.lbl_reg_mode_status.config(
                    text=f"🔄 [Đã có {cnt} vector] Nhân viên '{code}' • Chụp ảnh mới sẽ CẬP NHẬT thay thế khuôn mặt cũ.",
                    bg="#eff6ff",
                    fg="#1e40af",
                )
                self.btn_save_reg.config(text="💾 Cập Nhật Khuôn Mặt Mới")
        else:
            self.lbl_reg_mode_status.config(
                text=f"✨ [Mã mới: {code}] Chưa có trong CSDL • Sẽ tạo mới hồ sơ và nạp khuôn mặt.",
                bg="#f0fdf4",
                fg="#166534",
            )
            self.btn_save_reg.config(text="💾 Tạo Mới & Lưu Khuôn Mặt")

    def load_employee_into_register(self, code: str, auto_switch_tab: bool = True):
        if auto_switch_tab:
            self.show_tab("register")

        emp = get_employee(code)
        if not emp:
            return

        self.ent_reg_code.delete(0, tk.END)
        self.ent_reg_code.insert(0, emp["employee_code"])

        self.ent_reg_name.delete(0, tk.END)
        self.ent_reg_name.insert(0, emp["full_name"])

        self.ent_reg_dept.delete(0, tk.END)
        self.ent_reg_dept.insert(0, emp.get("department", ""))

        self.ent_reg_pos.delete(0, tk.END)
        self.ent_reg_pos.insert(0, emp.get("position", ""))

        stats_emp = next((e for e in getattr(self, "cached_employees_stats", []) if e["employee_code"].upper() == code.upper()), None)
        cnt = stats_emp["embedding_count"] if stats_emp else 0

        if cnt == 0:
            self.lbl_reg_mode_status.config(
                text=f"⚠️ [Chưa có khuôn mặt] Hồ sơ [{code}] '{emp['full_name']}' từ Web Admin • Sẵn sàng nạp vector!",
                bg="#fef3c7",
                fg="#92400e",
            )
            self.btn_save_reg.config(text="💾 Nạp Khuôn Mặt Cho Nhân Viên")
        else:
            self.lbl_reg_mode_status.config(
                text=f"🔄 [Đã có {cnt} vector] Nhân viên [{code}] '{emp['full_name']}' • Chụp ảnh mới sẽ CẬP NHẬT thay thế khuôn mặt cũ.",
                bg="#eff6ff",
                fg="#1e40af",
            )
            self.btn_save_reg.config(text="💾 Cập Nhật Khuôn Mặt Mới")

        if hasattr(self, "cb_reg_select_emp") and self.cb_reg_select_emp.get():
            for val in self.cb_reg_select_emp["values"]:
                if f"[{code}]" in val:
                    self.cb_reg_select_emp.set(val)
                    break

        if not self.reg_cam_running:
            self.start_register_camera()

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
        self.lbl_reg_video.config(image="", text="Camera đang tắt\nBấm 'Bật Camera' để chụp 3-5 góc mặt")

    def _update_register_camera_feed(self):
        if not self.reg_cam_running:
            return
        frame = self.reg_grabber.get_frame()
        if frame is not None:
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
        self.lbl_photo_count.config(text=f"Ảnh đã chụp: {count}/5 ảnh (Tối thiểu 3)")

        for child in self.thumb_inner.winfo_children():
            child.destroy()

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
        dept = self.ent_reg_dept.get().strip()
        pos = self.ent_reg_pos.get().strip()

        if not code or not name:
            messagebox.showerror("Thiếu thông tin", "Vui lòng nhập đầy đủ Mã nhân viên và Họ tên.")
            return

        if len(self.register_photos) < 3:
            messagebox.showwarning(
                "Chưa đủ ảnh",
                f"Bạn mới chụp {len(self.register_photos)} ảnh. Vui lòng chụp tối thiểu 3-5 ảnh có khuôn mặt rõ nét.",
            )
            return

        # Kiểm tra xem nhân viên đã tồn tại trong CSDL chưa
        existing_emp = get_employee(code)
        is_update = existing_emp is not None

        self.btn_save_reg.config(state="disabled")
        self.reg_progress.pack(fill="x", pady=5)
        self.reg_progress.start()

        def worker():
            try:
                # 1. Tạo danh sách các vector embeddings riêng lẻ (float32)
                embeddings, best_photo, best_box = build_employee_embeddings(self.register_photos, min_valid_photos=3)

                if not embeddings or len(embeddings) < 3:
                    self.after(0, lambda: messagebox.showerror(
                        "Lỗi Face Embeddings",
                        "Không thể nhận diện đủ tối thiểu 3 ảnh có khuôn mặt rõ nét.\n"
                        "Vui lòng đảm bảo mỗi ảnh có đúng 1 khuôn mặt, đủ sáng và không bị che khuất."
                    ))
                    return

                # 2. Kiểm tra trùng khuôn mặt bằng FAISS (Duplicate Face Detection)
                # Nếu đang cập nhật -> loại trừ chính nhân viên này ra để không báo trùng với bản thân
                engine = get_faiss_engine()
                exclude_emp_id = existing_emp["id"] if is_update else None
                is_dup, dup_info = engine.check_duplicate_face(
                    embeddings,
                    duplicate_threshold=config.FACE_DUPLICATE_THRESHOLD,
                    exclude_employee_id=exclude_emp_id,
                    exclude_employee_code=code,
                )

                if is_dup and dup_info is not None:
                    dup_code = dup_info.get("employee_code", "")
                    dup_name = dup_info.get("full_name", "")
                    dup_dist = dup_info.get("distance", 0.0)

                    self.after(0, lambda: messagebox.showwarning(
                        "Cảnh báo: Khuôn mặt đã tồn tại!",
                        f"⚠️ Khuôn mặt này có khả năng trùng với nhân viên khác trong hệ thống!\n\n"
                        f"• Mã nhân viên trùng: {dup_code}\n"
                        f"• Họ và tên: {dup_name}\n"
                        f"• Khoảng cách sai số (Distance): {dup_dist:.3f}\n\n"
                        f"Hệ thống từ chối lưu để ngăn ngừa trùng lặp dữ liệu.",
                    ))
                    return

                # 3. Upload ảnh đại diện tốt nhất lên Cloudinary (nếu đã cấu hình)
                image_url, cloudinary_pid = "", ""
                if best_photo is not None:
                    image_url, cloudinary_pid = upload_employee_avatar(code, best_photo, best_box)

                if is_update:
                    # CẬP NHẬT KHUÔN MẶT CHO NHÂN VIÊN ĐÃ TỒN TẠI (TẠO TỪ WEB ADMIN HOẶC CSDL)
                    emp_id = existing_emp["id"]
                    face_ids = update_employee_face(
                        employee_id=emp_id,
                        embeddings=embeddings,
                        image_url=image_url,
                        cloudinary_public_id=cloudinary_pid,
                    )

                    # Xóa vector cũ trong FAISS Index và nạp vector mới
                    engine.remove_employee(emp_id)
                    engine.add_embeddings(face_ids, embeddings, emp_id, code, name)

                    def on_success_update():
                        cloud_msg = "Đã cập nhật ảnh đại diện lên Cloudinary.\n" if image_url else ""
                        messagebox.showinfo(
                            "Cập nhật thành công!",
                            f"✅ Đã nạp/cập nhật thành công khuôn mặt cho nhân viên:\n"
                            f"• [{code}] {name}\n"
                            f"• Đã nạp {len(face_ids)} vector khuôn mặt AI vào FAISS & SQL Server.\n"
                            f"• {cloud_msg}"
                            f"Nhân viên đã có thể bắt đầu chấm công nhận diện ngay!",
                        )
                        self.ent_reg_code.delete(0, tk.END)
                        self.ent_reg_name.delete(0, tk.END)
                        self.ent_reg_dept.delete(0, tk.END)
                        self.ent_reg_pos.delete(0, tk.END)
                        self.clear_register_photos()
                        self.stop_register_camera()
                        self.refresh_register_employees_dropdown()
                        self.update_db_badge()

                    self.after(0, on_success_update)
                else:
                    # TẠO MỚI NHÂN VIÊN
                    emp_id = add_employee(
                        employee_code=code,
                        full_name=name,
                        department=dept,
                        position=pos,
                        image_url=image_url,
                        cloudinary_public_id=cloudinary_pid,
                    )

                    if emp_id <= 0:
                        raise RuntimeError("Không thể tạo bản ghi nhân viên trong CSDL SQL Server.")

                    face_ids = add_face_embeddings(emp_id, embeddings)
                    engine.add_embeddings(face_ids, embeddings, emp_id, code, name)

                    def on_success_new():
                        cloud_msg = "Đã lưu ảnh đại diện lên Cloudinary.\n" if image_url else "Chưa cấu hình Cloudinary (ảnh lưu cục bộ).\n"
                        messagebox.showinfo(
                            "Đăng ký thành công!",
                            f"✅ Đã thêm mới thành công nhân viên:\n"
                            f"• [{code}] {name}\n"
                            f"• Đã tạo & index {len(face_ids)} vector khuôn mặt vào FAISS.\n"
                            f"• {cloud_msg}"
                            f"• Đã dọn dẹp ảnh tạm thành công.",
                        )
                        self.ent_reg_code.delete(0, tk.END)
                        self.ent_reg_name.delete(0, tk.END)
                        self.ent_reg_dept.delete(0, tk.END)
                        self.ent_reg_pos.delete(0, tk.END)
                        self.clear_register_photos()
                        self.stop_register_camera()
                        self.refresh_register_employees_dropdown()
                        self.update_db_badge()

                    self.after(0, on_success_new)

            except Exception as ex:
                err_msg = str(ex)
                self.after(0, lambda msg=err_msg: messagebox.showerror("Lỗi khi lưu nhân viên", f"Chi tiết lỗi: {msg}"))
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
            text="📷 Chấm công nhận diện khuôn mặt Realtime (FAISS O(1) Speed)",
            font=("Segoe UI", 18, "bold"),
            fg=self.TEXT_MAIN,
            bg=self.CONTENT_BG,
        )
        lbl_h.pack(anchor="w", pady=(0, 10))

        content_grid = tk.Frame(tab, bg=self.CONTENT_BG)
        content_grid.pack(fill="both", expand=True)

        left_col = tk.Frame(content_grid, bg=self.CARD_BG, highlightbackground=self.BORDER_COLOR, highlightthickness=1, padx=15, pady=15)
        left_col.pack(side="left", fill="both", expand=True, padx=(0, 10))

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
        self.scale_tol = ttk.Scale(top_ctrl, from_=0.35, to=0.65, value=config.FACE_MATCH_THRESHOLD, orient="horizontal", length=90)
        self.scale_tol.pack(side="left", padx=2)

        self.lbl_tol_val = tk.Label(top_ctrl, text=f"{config.FACE_MATCH_THRESHOLD:.2f}", font=("Segoe UI", 9, "bold"), bg=self.CARD_BG, fg=self.PRIMARY_COLOR)
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

        # Mặc định var_auto_stop = False để chấm công liên tục tự động
        self.var_auto_stop = tk.BooleanVar(value=False)
        chk_autostop = tk.Checkbutton(
            top_ctrl,
            text="⏹ Tắt camera sau khi chấm công",
            variable=self.var_auto_stop,
            bg=self.CARD_BG,
            font=("Segoe UI", 9),
            activebackground=self.CARD_BG,
        )
        chk_autostop.pack(side="left", padx=5)

        self.var_auto_cam = tk.BooleanVar(value=True)
        chk_autocam = tk.Checkbutton(
            top_ctrl,
            text="⚡ Tự động bật camera",
            variable=self.var_auto_cam,
            bg=self.CARD_BG,
            font=("Segoe UI", 9),
            activebackground=self.CARD_BG,
        )
        chk_autocam.pack(side="left", padx=5)

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

        self.rec_status_banner = tk.Label(
            left_col,
            text="Sẵn sàng điểm danh",
            font=("Segoe UI", 12, "bold"),
            bg="#f1f5f9",
            fg=self.TEXT_MUTED,
            pady=10,
        )
        self.rec_status_banner.pack(fill="x", pady=(5, 0))

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

        engine = get_faiss_engine()
        if engine.index is None or engine.index.ntotal == 0:
            engine.init_engine()

        if engine.index is None or engine.index.ntotal == 0:
            messagebox.showwarning(
                "Chưa có dữ liệu",
                "Chưa có nhân viên active nào trong FAISS Index.\nVui lòng vào tab 'Đăng ký nhân viên' trước!",
            )
            return

        if not self.rec_grabber.start():
            messagebox.showerror("Lỗi Camera", "Không thể mở webcam (Camera index 0).")
            return

        self.rec_cam_running = True
        with self._detected_faces_lock:
            self.last_detected_faces = []
        self.recent_attendance_attempts.clear()
        self.btn_rec_toggle.config(text="⏹ Dừng Camera", style="Danger.TButton")
        self.rec_status_banner.config(text="🟢 Camera đang hoạt động • Vui lòng nhìn thẳng vào ống kính", bg="#dbeafe", fg="#1e40af")

        # Khởi chạy Thread 2: AI Background Worker
        threading.Thread(target=self._realtime_ai_worker, daemon=True).start()
        # Khởi chạy Thread 3: UI Main Thread render loop
        self._update_realtime_camera_feed()

    def stop_realtime_camera(self):
        self.rec_cam_running = False
        self.rec_grabber.stop()
        with self._detected_faces_lock:
            self.last_detected_faces = []
        self.btn_rec_toggle.config(text="▶ Bắt đầu điểm danh", style="Success.TButton")
        self.lbl_rec_video.config(image="", text="Camera đang tắt\nBấm 'Bắt đầu điểm danh' để nhận diện")
        self.rec_status_banner.config(text="Đã dừng nhận diện", bg="#f1f5f9", fg=self.TEXT_MUTED)

    def _realtime_ai_worker(self):
        """
        Thread 2 - AI Worker:
        Chạy ngầm liên tục, lấy frame mới nhất từ CameraGrabber, trích xuất vector và tìm kiếm FAISS.
        - Tối ưu tải CPU: Xử lý frame ở scale_factor=0.5 (nhanh & cực nhạy).
        - Độc lập 1-to-1: Mỗi khuôn mặt có danh tính riêng, không kế thừa dữ liệu của mặt bên cạnh.
        - Bất đồng bộ hóa: Tác vụ ghi DB / loa được đẩy sang ThreadPoolExecutor, không làm nghẽn AI loop.
        """
        while self.rec_cam_running:
            frame = self.rec_grabber.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            try:
                try:
                    tolerance = float(self.scale_tol.get())
                except Exception:
                    tolerance = config.FACE_MATCH_THRESHOLD

                # Nhận diện đa khuôn mặt chi tiết độc lập
                detailed_faces = recognize_faces_detailed(
                    frame=frame,
                    tolerance=tolerance,
                    scale_factor=0.5,
                    is_bgr=True,
                )

                new_faces = []
                now_ts = time.time()

                for face in detailed_faces:
                    top, right, bottom, left = face["box"]
                    status = face["status"]
                    code = face["code"]
                    name = face["name"]
                    dist = face["distance"]

                    if status == "AMBIGUOUS":
                        color = (0, 165, 255)  # Cam cảnh báo
                        label = f"AMBIGUOUS ({dist:.2f})"
                        self.after(0, lambda: self.rec_status_banner.config(
                            text="⚠️ Không thể xác định chính xác nhân viên (AMBIGUOUS). Vui lòng xác minh thêm.",
                            bg="#fef3c7",
                            fg="#92400e",
                        ))
                    elif status == "MATCH" and code != "Unknown":
                        color = (0, 255, 0)  # Xanh lá
                        label = f"{code} - {name} ({dist:.2f})"

                        # Kiểm tra Cooldown độc lập từng nhân viên trước khi kích hoạt
                        last_attempt = self.recent_attendance_attempts.get(code, 0.0)
                        if (now_ts - last_attempt) >= config.ATTENDANCE_COOLDOWN_SECONDS:
                            self.recent_attendance_attempts[code] = now_ts
                            # Đẩy tác vụ I/O sang ThreadPoolExecutor bất đồng bộ
                            self.executor.submit(self._async_register_attendance_task, code, name)
                    else:
                        color = (0, 0, 255)  # Đỏ (Chưa đăng ký / Unknown)
                        label = "Unknown"

                    new_faces.append((top, right, bottom, left, label, color))

                with self._detected_faces_lock:
                    self.last_detected_faces = new_faces

            except Exception:
                pass

            # Nghỉ nhẹ 25ms để nhường CPU cho luồng hiển thị giao diện đạt FPS tối đa
            time.sleep(0.025)

    def _async_register_attendance_task(self, employee_code: str, employee_name: str):
        """
        Background I/O Worker (ThreadPoolExecutor):
        Thực hiện ghi nhận chấm công vào SQL Server độc lập, không chặn luồng Camera hay UI.
        """
        try:
            res = register_attendance(employee_code)
            if res and res.get("success"):
                self.after(0, lambda r=res: self._handle_attendance_result(r))
        except Exception as ex:
            print(f"Lỗi ghi nhận chấm công bất đồng bộ ({employee_code}): {ex}")

    def _update_realtime_camera_feed(self):
        """
        Thread 3 - UI Loop (Tkinter Main Thread):
        Chỉ nhận frame và danh sách bounding box để vẽ đè lên canvas/label với tần số mượt mà (~40 FPS).
        Tuyệt đối không chờ AI hay I/O mạng.
        """
        if not self.rec_cam_running:
            return

        frame = self.rec_grabber.get_frame()
        if frame is not None:
            # Lấy bản sao an toàn của các bounding box từ AI Worker
            with self._detected_faces_lock:
                faces_to_draw = list(self.last_detected_faces)

            for (top, right, bottom, left, label, color) in faces_to_draw:
                # Vẽ viền bounding box
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                # Vẽ nền nhãn
                cv2.rectangle(frame, (left, max(0, bottom - 26)), (right, bottom), color, cv2.FILLED)
                # Vẽ chữ nhãn
                text_color = (0, 0, 0) if color == (0, 255, 0) else (255, 255, 255)
                cv2.putText(
                    frame,
                    label,
                    (left + 6, max(14, bottom - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    text_color,
                    1,
                    cv2.LINE_AA,
                )

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame_rgb.shape[:2]
            target_w = 640
            target_h = int(h * (target_w / w))
            if target_w != w or target_h != h:
                frame_rgb = cv2.resize(frame_rgb, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

            img_pil = Image.fromarray(frame_rgb)
            img_tk = ImageTk.PhotoImage(image=img_pil)
            self.lbl_rec_video.img_tk = img_tk
            self.lbl_rec_video.config(image=img_tk, text="")

        self.after(25, self._update_realtime_camera_feed)

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
            self.log_tree.insert("", 0, values=(now_time, code, name, att_type))

            # Phát giọng nói tiếng Việt bất đồng bộ (0ms UI lag)
            if getattr(self, "var_voice_enabled", None) and self.var_voice_enabled.get():
                speak_async(employee_name=name)

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
            else:
                # Chế độ tự động liên tục: sau 2.5s đưa banner về trạng thái sẵn sàng đón người tiếp theo
                def reset_status():
                    if self.rec_cam_running:
                        self.rec_status_banner.config(
                            text="🟢 Camera đang hoạt động • Vui lòng nhìn thẳng vào ống kính để điểm danh",
                            bg="#dbeafe",
                            fg="#1e40af",
                        )
                self.after(2500, reset_status)

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
                employees = get_employees(status=None)
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
                self.after(0, lambda msg=err_text: messagebox.showerror("Lỗi truy vấn", f"Không thể lấy dữ liệu lịch sử: {msg}"))

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
            text="⚙️ Quản lý danh sách nhân viên (Soft Delete & Cloudinary)",
            font=("Segoe UI", 18, "bold"),
            fg=self.TEXT_MAIN,
            bg=self.CONTENT_BG,
        )
        lbl_h.pack(anchor="w", pady=(0, 15))

        act_card = tk.Frame(tab, bg=self.CARD_BG, highlightbackground=self.BORDER_COLOR, highlightthickness=1, padx=15, pady=10)
        act_card.pack(fill="x", pady=(0, 15))

        btn_face = ttk.Button(act_card, text="📸 Đăng ký / Cập nhật mặt", style="Primary.TButton", command=self.open_selected_employee_face_update)
        btn_face.pack(side="left", padx=(0, 10))

        btn_del = ttk.Button(act_card, text="⏸️ Ngưng HĐ (INACTIVE)", style="Danger.TButton", command=self.delete_selected_employee)
        btn_del.pack(side="left", padx=5)

        btn_act = ttk.Button(act_card, text="✅ Kích hoạt (ACTIVE)", style="Success.TButton", command=self.activate_selected_employee)
        btn_act.pack(side="left", padx=5)

        btn_hard_del = ttk.Button(act_card, text="🔥 Xóa vĩnh viễn", style="Danger.TButton", command=self.hard_delete_selected_employee)
        btn_hard_del.pack(side="left", padx=5)

        table_container = tk.Frame(tab, bg=self.CARD_BG, highlightbackground=self.BORDER_COLOR, highlightthickness=1)
        table_container.pack(fill="both", expand=True)

        cols = ("emp_code", "full_name", "dept", "pos", "status", "face_ai", "created_at", "avatar")
        self.emp_tree = ttk.Treeview(table_container, columns=cols, show="headings", style="Custom.Treeview")
        self.emp_tree.heading("emp_code", text="Mã NV")
        self.emp_tree.heading("full_name", text="Họ và Tên")
        self.emp_tree.heading("dept", text="Phòng ban")
        self.emp_tree.heading("pos", text="Chức vụ")
        self.emp_tree.heading("status", text="Trạng thái")
        self.emp_tree.heading("face_ai", text="Khuôn mặt AI")
        self.emp_tree.heading("created_at", text="Ngày tạo")
        self.emp_tree.heading("avatar", text="Cloudinary Avatar")

        self.emp_tree.column("emp_code", width=85, anchor="center")
        self.emp_tree.column("full_name", width=180, anchor="w")
        self.emp_tree.column("dept", width=100, anchor="w")
        self.emp_tree.column("pos", width=100, anchor="w")
        self.emp_tree.column("status", width=85, anchor="center")
        self.emp_tree.column("face_ai", width=160, anchor="center")
        self.emp_tree.column("created_at", width=130, anchor="center")
        self.emp_tree.column("avatar", width=120, anchor="center")

        # Double click to update face
        self.emp_tree.bind("<Double-1>", lambda e: self.open_selected_employee_face_update())

        scroll_y = ttk.Scrollbar(table_container, orient="vertical", command=self.emp_tree.yview)
        self.emp_tree.configure(yscrollcommand=scroll_y.set)
        scroll_y.pack(side="right", fill="y")
        self.emp_tree.pack(fill="both", expand=True, padx=10, pady=10)

    def open_selected_employee_face_update(self):
        sel = self.emp_tree.selection()
        if not sel:
            messagebox.showwarning("Chưa chọn", "Vui lòng chọn 1 nhân viên trong bảng để nạp/cập nhật khuôn mặt.")
            return

        item = self.emp_tree.item(sel[0])
        code = str(item["values"][0]).strip()
        self.load_employee_into_register(code, auto_switch_tab=True)

    def refresh_employee_list(self):
        def worker():
            try:
                employees = get_employees_with_stats(status=None)
                self.cached_employees_stats = employees
                rows_data = []
                for emp in employees:
                    avatar_status = "Đã tải lên" if emp.get("image_url") else "Chưa có"
                    cnt = emp.get("embedding_count", 0)
                    if cnt == 0:
                        face_status = "⚠️ Chưa có (Web Admin)"
                    else:
                        face_status = f"✅ {cnt} vectors"

                    rows_data.append((
                        emp["employee_code"],
                        emp["full_name"],
                        emp["department"],
                        emp["position"],
                        emp["status"],
                        face_status,
                        emp["created_at"],
                        avatar_status,
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
                self.after(0, lambda msg=err_text: messagebox.showerror("Lỗi", f"Không thể lấy danh sách nhân viên: {msg}"))

        threading.Thread(target=worker, daemon=True).start()

    def delete_selected_employee(self):
        sel = self.emp_tree.selection()
        if not sel:
            messagebox.showwarning("Chưa chọn", "Vui lòng chọn 1 nhân viên trong bảng để thực hiện Soft Delete.")
            return

        item = self.emp_tree.item(sel[0])
        code = item["values"][0]
        name = item["values"][1]

        confirm = messagebox.askyesno(
            "Xác nhận Soft Delete",
            f"Bạn có chắc chắn muốn ngưng hoạt động (INACTIVE) nhân viên:\n[{code}] {name}?\n\n"
            f"• Nhân viên sẽ bị gỡ khỏi FAISS Index (không còn được nhận diện).\n"
            f"• Toàn bộ lịch sử chấm công vẫn được bảo toàn trong CSDL SQL Server.",
        )

        if confirm:
            try:
                emp = get_employee(code)
                soft_delete_employee(code)

                if emp:
                    get_faiss_engine().remove_employee(emp["id"])

                messagebox.showinfo("Thành công", f"Đã chuyển trạng thái nhân viên {code} sang INACTIVE thành công.")
                self.refresh_employee_list()
                self.update_db_badge()
            except Exception as ex:
                messagebox.showerror("Lỗi khi xóa mềm", f"Chi tiết lỗi: {str(ex)}")

    def activate_selected_employee(self):
        sel = self.emp_tree.selection()
        if not sel:
            messagebox.showwarning("Chưa chọn", "Vui lòng chọn 1 nhân viên trong bảng để kích hoạt lại.")
            return

        item = self.emp_tree.item(sel[0])
        code = item["values"][0]
        name = item["values"][1]

        confirm = messagebox.askyesno(
            "Xác nhận Kích hoạt",
            f"Bạn có chắc chắn muốn kích hoạt lại (ACTIVE) nhân viên:\n[{code}] {name}?\n\n"
            f"• Nhân viên sẽ được đưa trở lại FAISS Index để tiếp tục chấm công.",
        )

        if confirm:
            try:
                activate_employee(code)
                get_faiss_engine().rebuild_from_database()
                messagebox.showinfo("Thành công", f"Đã kích hoạt lại nhân viên {code} (ACTIVE) và cập nhật FAISS Index!")
                self.refresh_employee_list()
                self.update_db_badge()
            except Exception as ex:
                messagebox.showerror("Lỗi khi kích hoạt", f"Chi tiết lỗi: {str(ex)}")

    def hard_delete_selected_employee(self):
        sel = self.emp_tree.selection()
        if not sel:
            messagebox.showwarning("Chưa chọn", "Vui lòng chọn 1 nhân viên trong bảng để xóa vĩnh viễn.")
            return

        item = self.emp_tree.item(sel[0])
        code = item["values"][0]
        name = item["values"][1]

        confirm = messagebox.askyesno(
            "CẢNH BÁO: XÓA VĨNH VIỄN!",
            f"⚠️ Bạn có chắc chắn muốn XÓA HOÀN TOÀN nhân viên:\n[{code}] {name}?\n\n"
            f"• Toàn bộ hồ sơ nhân viên sẽ bị xóa khỏi SQL Server.\n"
            f"• Toàn bộ lịch sử chấm công và các vector khuôn mặt sẽ bị xóa vĩnh viễn.\n"
            f"• Ảnh đại diện trên Cloudinary sẽ bị xóa.\n"
            f"• Hành động này KHÔNG THỂ khôi phục lại!",
            icon="warning",
        )

        if confirm:
            try:
                ok, cloudinary_pid = hard_delete_employee(code)
                if ok:
                    if cloudinary_pid:
                        threading.Thread(target=delete_employee_avatar, args=(cloudinary_pid,), daemon=True).start()

                    get_faiss_engine().rebuild_from_database()
                    messagebox.showinfo("Thành công", f"Đã xóa vĩnh viễn nhân viên {code} khỏi toàn bộ hệ thống!")
                    self.refresh_employee_list()
                    self.update_db_badge()
                else:
                    messagebox.showerror("Lỗi", "Không tìm thấy nhân viên để xóa.")
            except Exception as ex:
                messagebox.showerror("Lỗi khi xóa vĩnh viễn", f"Chi tiết lỗi: {str(ex)}")

    # ==========================================
    # 7. TAB: CẤU HÌNH DATABASE (SQL SERVER ONLY)
    # ==========================================
    def _init_database_tab(self):
        tab = tk.Frame(self.content_area, bg=self.CONTENT_BG)
        self.tabs["database"] = tab

        lbl_h = tk.Label(
            tab,
            text="🗄️ Cấu hình Microsoft SQL Server (Doanh Nghiệp)",
            font=("Segoe UI", 18, "bold"),
            fg=self.TEXT_MAIN,
            bg=self.CONTENT_BG,
        )
        lbl_h.pack(anchor="w", pady=(0, 15))

        card = tk.Frame(tab, bg=self.CARD_BG, highlightbackground=self.BORDER_COLOR, highlightthickness=1, padx=25, pady=25)
        card.pack(fill="x")

        # Khung cấu hình SQL Server
        self.frame_sqlserver_cfg = tk.LabelFrame(card, text="Thông số kết nối Microsoft SQL Server", font=("Segoe UI", 10, "bold"), bg=self.CARD_BG, padx=15, pady=15)
        self.frame_sqlserver_cfg.pack(fill="x", pady=10)

        tk.Label(self.frame_sqlserver_cfg, text="Server Host / Instance:", font=("Segoe UI", 9, "bold"), bg=self.CARD_BG).grid(row=0, column=0, sticky="w", pady=6)
        self.ent_sql_server = ttk.Entry(self.frame_sqlserver_cfg, font=("Segoe UI", 9), width=35)
        self.ent_sql_server.grid(row=0, column=1, sticky="w", padx=10, pady=6)

        tk.Label(self.frame_sqlserver_cfg, text="Database Name:", font=("Segoe UI", 9, "bold"), bg=self.CARD_BG).grid(row=1, column=0, sticky="w", pady=6)
        self.ent_sql_db = ttk.Entry(self.frame_sqlserver_cfg, font=("Segoe UI", 9), width=35)
        self.ent_sql_db.grid(row=1, column=1, sticky="w", padx=10, pady=6)

        tk.Label(self.frame_sqlserver_cfg, text="ODBC Driver:", font=("Segoe UI", 9, "bold"), bg=self.CARD_BG).grid(row=2, column=0, sticky="w", pady=6)
        self.cb_sql_driver = ttk.Combobox(self.frame_sqlserver_cfg, font=("Segoe UI", 9), width=33, state="readonly")
        drivers = get_available_sqlserver_drivers()
        self.cb_sql_driver["values"] = drivers
        if drivers:
            self.cb_sql_driver.current(0)
        self.cb_sql_driver.grid(row=2, column=1, sticky="w", padx=10, pady=6)

        self.var_sql_trusted = tk.BooleanVar(value=True)
        self.chk_sql_trusted = tk.Checkbutton(
            self.frame_sqlserver_cfg,
            text="Sử dụng Windows Authentication (Trusted Connection)",
            variable=self.var_sql_trusted,
            font=("Segoe UI", 9),
            bg=self.CARD_BG,
            command=self._on_trusted_auth_changed,
        )
        self.chk_sql_trusted.grid(row=3, column=0, columnspan=2, sticky="w", pady=6)

        tk.Label(self.frame_sqlserver_cfg, text="SQL Username (nếu dùng SQL Auth):", font=("Segoe UI", 9), bg=self.CARD_BG).grid(row=4, column=0, sticky="w", pady=6)
        self.ent_sql_user = ttk.Entry(self.frame_sqlserver_cfg, font=("Segoe UI", 9), width=25)
        self.ent_sql_user.grid(row=4, column=1, sticky="w", padx=10, pady=6)

        tk.Label(self.frame_sqlserver_cfg, text="SQL Password:", font=("Segoe UI", 9), bg=self.CARD_BG).grid(row=5, column=0, sticky="w", pady=6)
        self.ent_sql_pwd = ttk.Entry(self.frame_sqlserver_cfg, font=("Segoe UI", 9), width=25, show="*")
        self.ent_sql_pwd.grid(row=5, column=1, sticky="w", padx=10, pady=6)

        # Buttons
        btn_box = tk.Frame(card, bg=self.CARD_BG)
        btn_box.pack(anchor="w", pady=15)

        btn_test = ttk.Button(btn_box, text="🔍 Kiểm tra kết nối", command=self.test_database_settings)
        btn_test.pack(side="left", padx=(0, 10))

        btn_save = ttk.Button(btn_box, text="💾 Lưu cấu hình & Khởi tạo CSDL", style="Success.TButton", command=self.save_database_settings)
        btn_save.pack(side="left")

        self.refresh_db_settings_tab()

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
        sql_cfg = cfg.get("sqlserver", config.get_db_dict())

        self.ent_sql_server.delete(0, tk.END)
        self.ent_sql_server.insert(0, sql_cfg.get("server", config.DB_SERVER))

        self.ent_sql_db.delete(0, tk.END)
        self.ent_sql_db.insert(0, sql_cfg.get("database", config.DB_DATABASE))

        driver = sql_cfg.get("driver", config.DB_DRIVER)
        if driver in self.cb_sql_driver["values"]:
            self.cb_sql_driver.set(driver)

        self.var_sql_trusted.set(sql_cfg.get("trusted_connection", config.DB_TRUSTED_CONNECTION))
        self.ent_sql_user.delete(0, tk.END)
        self.ent_sql_user.insert(0, sql_cfg.get("username", config.DB_USER))

        self.ent_sql_pwd.delete(0, tk.END)
        self.ent_sql_pwd.insert(0, sql_cfg.get("password", config.DB_PASSWORD))

        self._on_trusted_auth_changed()

    def _get_current_ui_db_config(self):
        return {
            "db_type": "sqlserver",
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
            get_faiss_engine().rebuild_from_database()
            self.update_db_badge()
            messagebox.showinfo("Thành công", "Đã lưu cấu hình và khởi tạo CSDL SQL Server thành công!")
        except Exception as ex:
            messagebox.showerror("Lỗi khởi tạo", f"Không thể khởi tạo CSDL: {str(ex)}")

    def on_closing(self):
        self.stop_register_camera()
        self.stop_realtime_camera()
        if hasattr(self, "executor"):
            try:
                self.executor.shutdown(wait=False)
            except Exception:
                pass
        self.destroy()


def main():
    app = FaceAttendanceApp()
    app.mainloop()


if __name__ == "__main__":
    main()
