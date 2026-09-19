# Face Attendance - Hệ Thống Chấm Công Nhận Diện Khuôn Mặt Tốc Độ Cao
## Kiến trúc: Tkinter App (Máy Chấm Công) + Flask Web Admin (Quản Trị Local/LAN) + FAISS + SQL Server + Cloudinary
### Khả năng đáp ứng: 10.000+ nhân viên • 30.000 - 50.000 Face Vectors

---

## 🌟 1. Giới Thiệu & Kiến Trúc Hệ Thống

Hệ thống kết hợp mô hình 2 tiến trình độc lập hoạt động song song trên máy tính cục bộ (Local/LAN), dùng chung CSDL Microsoft SQL Server, Cloudinary và thuật toán FAISS:

1. **Tkinter Kiosk App (`python main.py`)**: Đóng vai trò làm máy chấm công camera tại cửa ra vào (nhận diện Realtime, chụp ảnh đăng ký khuôn mặt AI 3-5 góc độ, cảnh báo âm thanh tiếng Việt).
2. **Flask Web Admin Portal (`python web_admin/app.py`)**: Đóng vai trò trang web quản trị Dashboard, quản lý hồ sơ nhân viên, xem lịch sử chấm công, vẽ biểu đồ Chart.js, xuất báo cáo Excel và Rebuild FAISS Index. Có thể truy cập qua trình duyệt máy tính hoặc điện thoại cùng mạng Wi-Fi.

```text
┌────────────────────────────────────────────────────────┐
│                   MÁY TÍNH CỤC BỘ                      │
│                                                        │
│   [Tiến trình 1]                    [Tiến trình 2]     │
│   Tkinter App (GUI)                 Flask Web Admin    │
│   (Camera + Chấm công)              (Port 5000 / LAN)  │
│         │                                  │           │
│         │         ┌────────────────────────┤           │
│         ▼         ▼                        ▼           │
│    Microsoft SQL Server               Cloudinary       │
│    (FaceAttendanceDB)            (Lưu ảnh đại diện)    │
│         ▲                                              │
│         │                                              │
│    FAISS Engine                                        │
│    (IndexIDMap2 - 128D)                                │
└────────────────────────────────────────────────────────┘
```

---

## 💻 2. Yêu Cầu Hệ Thống & Cài Đặt

- **Hệ điều hành**: Windows 10 / Windows 11 (64-bit)
- **Python**: Python 3.10 hoặc 3.11
- **Cơ sở dữ liệu**: Microsoft SQL Server 2016+ (hoặc SQL Express / Developer)
- **ODBC Driver**: `ODBC Driver 17 for SQL Server`
- **Dependencies**: `pip install -r requirements.txt`

---

## ⚙️ 3. Cấu Hình Biến Môi Trường (.env)

File `.env` tại thư mục gốc `face_attendance/`:

```env
# 1. KẾT NỐI MICROSOFT SQL SERVER
DB_SERVER=localhost
DB_DATABASE=FaceAttendanceDB
DB_DRIVER=ODBC Driver 17 for SQL Server
DB_TRUSTED_CONNECTION=true
DB_USER=sa
DB_PASSWORD=

# 2. CẤU HÌNH CLOUDINARY (LƯU ẢNH NHÂN VIÊN)
CLOUDINARY_CLOUD_NAME=your_cloud_name
CLOUDINARY_API_KEY=your_api_key
CLOUDINARY_API_SECRET=your_api_secret
CLOUDINARY_FOLDER=face_attendance/employees

# 3. THAM SỐ NHẬN DIỆN & FAISS
FACE_MATCH_THRESHOLD=0.48
FACE_DUPLICATE_THRESHOLD=0.40
FACE_AMBIGUOUS_MARGIN=0.06
ATTENDANCE_COOLDOWN_SECONDS=60
RECOGNITION_FRAME_SKIP=3

# 4. CẤU HÌNH FLASK WEB ADMIN
FLASK_HOST=0.0.0.0
FLASK_PORT=5000
FLASK_SECRET_KEY=face_attendance_flask_secret_2026_key
```

---

## 🚀 4. Hướng Dẫn Khởi Chạy Hệ Thống

### 🔹 Cách 1: Chạy Máy Chấm Công (Tkinter)
Mở Terminal 1:
```powershell
python main.py
```
*(Dùng để điểm danh nhận diện khuôn mặt realtime và chụp ảnh đăng ký nhân viên mới).*

### 🔹 Cách 2: Chạy Web Admin Quản Trị (Flask)
Mở Terminal 2:
```powershell
python web_admin/app.py
```

- **Truy cập trên máy tính:** [http://localhost:5000](http://localhost:5000) hoặc [http://127.0.0.1:5000](http://127.0.0.1:5000)
- **Truy cập từ điện thoại / Laptop cùng Wi-Fi:** `http://<IP_MAY_TINH>:5000` (Ví dụ: `http://192.168.1.15:5000`)
- **Tài khoản đăng nhập mặc định:**
  - Tên đăng nhập: `admin`
  - Mật khẩu: `admin123`

---

## 🌐 5. Các Chức Năng Chính Trên Web Admin

| Route | Chức Năng & Đặc Điểm |
| :--- | :--- |
| `/login`, `/logout` | Đăng nhập/Đăng xuất Admin bảo vệ session, tái sử dụng `auth_utils.py` & `admin_config.json`. |
| `/` hoặc `/dashboard` | Dashboard tổng quan: 8 metric cards, biểu đồ Chart.js (lưu lượng chấm công theo giờ, nhân viên theo phòng ban), bảng 10 lượt chấm công gần nhất kèm avatar Cloudinary. |
| `/employees` | Quản lý danh sách nhân viên: Tìm kiếm, lọc phòng ban, lọc ACTIVE/INACTIVE, hiển thị badge `REGISTERED` / `NOT REGISTERED`, phân trang (Pagination). |
| `/employees/create` | Tạo mới hồ sơ nhân viên trực tiếp từ Web (trạng thái `NOT REGISTERED`), sau đó nhân viên đến máy Tkinter để chụp mặt AI. |
| `/employees/<id>` | Xem chi tiết nhân viên, thông tin vector khuôn mặt trong SQL Server, 15 lượt chấm công gần nhất, nút Sửa, Vô hiệu hóa, Kích hoạt lại và Xóa vĩnh viễn. |
| `/employees/<id>/edit` | Chỉnh sửa Họ tên, Phòng ban, Chức vụ, Trạng thái (tự động kiểm tra Unique Mã NV). |
| `/attendance` | Xem và lọc nhật ký chấm công toàn diện: Lọc theo khoảng ngày, phòng ban, tìm kiếm mã/tên, phân trang. |
| `/reports`, `/reports/export` | Báo cáo chấm công tổng hợp và nút xuất file Excel `.xlsx` tải trực tiếp về máy. |
| `/system` | Giám sát trạng thái SQL Server, Cloudinary, thông số FAISS Vector Engine và nút **Rebuild FAISS Index** từ database. |

---

## 📁 6. Cấu Trúc Dự Án

```text
face_attendance/
│
├── assets/                    # Âm thanh thông báo
│   └── sounds/
│
├── data/                      # Lưu trữ FAISS Index
│   └── faiss/
│       └── face.index
│
├── temp/                      # Thư mục tạm
├── dataset/                   # Dataset
│
├── .env                       # Cấu hình biến môi trường
├── .env.example               # Mẫu biến môi trường
├── requirements.txt           # Thư viện phụ thuộc
├── main.py                    # Entry point khởi chạy Tkinter Kiosk
├── app_tkinter.py             # Giao diện Desktop Tkinter
├── config.py                  # Module cấu hình tập trung
├── database.py                # SQL Server queries & Web Admin helpers
├── face_utils.py              # Xử lý nhận diện khuôn mặt
├── faiss_utils.py             # Bộ máy tìm kiếm vector FAISS
├── cloudinary_utils.py        # Upload & quản lý avatar Cloudinary
├── attendance.py              # Xử lý chấm công & xuất báo cáo
├── auth_utils.py              # Xác thực Admin
├── voice_utils.py             # Giọng nói thông báo tiếng Việt
├── setup_sqlserver.sql        # Script tạo CSDL SQL Server
│
└── web_admin/                 # 🚀 MODULE FLASK WEB ADMIN MỚI
    ├── app.py                 # Entry point khởi chạy Web Admin
    ├── routes/
    │   ├── auth.py            # Route Login & Logout
    │   ├── dashboard.py       # Route Dashboard & Chart.js data
    │   ├── employees.py       # Route CRUD nhân viên & phân trang
    │   ├── attendance.py      # Route lịch sử chấm công
    │   ├── reports.py         # Route báo cáo & xuất Excel
    │   └── system.py          # Route trạng thái hệ thống & Rebuild FAISS
    ├── templates/
    │   ├── base.html          # Layout chuẩn Bootstrap 5 & Sidebar
    │   ├── login.html         # Giao diện đăng nhập
    │   ├── dashboard.html     # Dashboard thống kê & biểu đồ
    │   ├── employees.html     # Danh sách nhân viên phân trang
    │   ├── employee_create.html # Form thêm nhân viên
    │   ├── employee_detail.html # Chi tiết nhân viên & face vectors
    │   ├── employee_edit.html # Form sửa nhân viên
    │   ├── attendance.html    # Lịch sử chấm công & bộ lọc
    │   ├── reports.html       # Báo cáo & nút xuất Excel
    │   └── system.html        # Trạng thái hệ thống & Rebuild FAISS
    └── static/
        ├── css/style.css      # Custom styling & status badges
        └── js/main.js         # Mobile drawer & live clock
```

