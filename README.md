# Face Attendance - Hệ thống Chấm Công Nhận Diện Khuôn Mặt (Tkinter + SQL Server)

Hệ thống chấm công bằng nhận diện khuôn mặt Realtime viết thuần bằng Python **Tkinter Desktop GUI**, xử lý luồng Camera trực tiếp bằng **OpenCV** + **face_recognition** và kết nối cơ sở dữ liệu **Microsoft SQL Server** (hỗ trợ cả SQLite dự phòng).

---

## 🌟 Tính Năng Chính

- **Nhận diện khuôn mặt chuẩn xác**: Sử dụng `face_recognition` (dlib HOG model) tạo vector đặc trưng 128 chiều chuẩn hóa L2 norm.
- **Giao diện Desktop Tkinter (`main.py` / `app_tkinter.py`)**:
  - 📊 **Tổng quan (Dashboard)**: Thống kê tổng nhân viên, số lượt check-in hôm nay, số ca đã hoàn thành và bảng điểm danh thời gian thực.
  - 👤 **Đăng ký nhân viên**: Luồng webcam trực tiếp, chụp 8–10 ảnh với góc mặt/biểu cảm đa dạng, hỗ trợ tải ảnh từ máy, gallery thumbnail và thanh tiến trình encode.
  - 📷 **Chấm công Realtime**: Camera trực tiếp qua OpenCV không trễ mạng, vẽ khung bounding box, tên, độ khớp sai số (distance), tự động chuyển trạng thái Check-in / Check-out với Cooldown (75s) chống chấm công lặp.
  - 📋 **Lịch sử chấm công**: Lọc theo khoảng ngày (Từ ngày -> Đến ngày), lọc theo nhân viên, xem tổng giờ làm việc.
  - 📈 **Xuất báo cáo Excel**: Xuất file `.xlsx` chuyên nghiệp với độ rộng cột tự động và bộ lọc.
  - ⚙️ **Quản lý nhân viên**: Danh sách nhân viên, xem số lượng ảnh dataset và xóa nhân viên kèm dữ liệu liên quan.
  - 🗄️ **Cấu hình Database**: Kết nối Microsoft SQL Server qua `pyodbc`, kiểm tra kết nối trực tiếp trên giao diện.
- **Hỗ trợ Microsoft SQL Server**: Cung cấp sẵn file script `setup_sqlserver.sql` để chạy trên SQL Server Management Studio (SSMS).

---

## 📁 Cấu Trúc Dự Án

```text
face_attendance/
├── main.py              # File khởi chạy chính của ứng dụng
├── app_tkinter.py       # Giao diện Desktop thuần Tkinter
├── setup_sqlserver.sql  # Script tạo Database & Bảng trên MS SQL Server
├── database.py          # Module kết nối MS SQL Server & SQLite
├── face_utils.py        # Tiền xử lý ảnh an toàn và thuật toán nhận diện
├── attendance.py        # Logic chấm công, cooldown & báo cáo
├── db_config.json       # File cấu hình kết nối CSDL (mặc định SQL Server)
├── requirements.txt     # Danh sách thư viện cần thiết
├── README.md            # Hướng dẫn sử dụng
└── dataset/             # Thư mục lưu ảnh gốc đăng ký của nhân viên
    └── NV001/
        ├── 01.jpg
        ├── 02.jpg
        └── ...
```


---

## ⚙️ Cài Đặt Môi Trường

Khuyến nghị:
- **Hệ điều hành**: Windows 10 / Windows 11
- **Python**: Python 3.10 hoặc 3.11

Mở PowerShell hoặc CMD tại thư mục `face_attendance`:

```powershell
# 1. Kích hoạt môi trường ảo (nếu có)
.\venv\Scripts\activate

# 2. Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

> [!NOTE]
> **Khắc phục lỗi `Unsupported image type, must be 8bit gray or RGB image.`:**
> Lỗi này xảy ra khi dùng NumPy 2.x với dlib cũ hoặc khi truyền ảnh có kênh Alpha (RGBA 4 kênh). Trong dự án này, lỗi đã được khắc phục triệt để bằng cách:
> 1. Khóa phiên bản `numpy>=1.24.0,<2.0.0` (cụ thể `numpy==1.26.4`).
> 2. Hàm `ensure_rgb_uint8()` trong `face_utils.py` tự động chuẩn hóa mọi ảnh về mảng RGB 3 kênh liên tục kiểu `uint8`.

---

## 🚀 Hướng Dẫn Chạy Ứng Dụng

### Chạy Giao Diện Desktop Tkinter (Chính):
```powershell
py app_tkinter.py
# hoặc
python app_tkinter.py
```

### Chạy Giao Diện Web Streamlit (Tùy chọn):
```powershell
streamlit run app.py
```

---

## 🗄️ Hướng Dẫn Cấu Hình Microsoft SQL Server

Hệ thống hỗ trợ 100% Microsoft SQL Server thông qua `pyodbc`. Bạn có thể cấu hình bằng 2 cách:

### Cách 1: Cấu hình trực tiếp trên giao diện Tkinter
1. Mở ứng dụng, vào tab **🗄️ Cấu hình Database**.
2. Chọn **Microsoft SQL Server**.
3. Điền thông tin:
   - **Server**: Tên máy chủ (VD: `localhost`, `127.0.0.1`, `DESKTOP-ABC\SQLEXPRESS`).
   - **Database Name**: Tên database (VD: `FaceAttendanceDB`).
   - **ODBC Driver**: Chọn `ODBC Driver 17 for SQL Server` hoặc `SQL Server`.
   - **Authentication**: Tích chọn *Windows Authentication* (mặc định) hoặc bỏ chọn để nhập Username `sa` và Password.
4. Bấm nút **🔍 Kiểm tra kết nối**.
5. Bấm nút **💾 Lưu cấu hình & Khởi tạo CSDL** để tự động tạo các bảng `employees`, `attendance`.

### Cách 2: Sửa file `db_config.json`
Tạo hoặc chỉnh sửa file `db_config.json` trong thư mục gốc:
```json
{
    "db_type": "sqlserver",
    "sqlite_path": "attendance.db",
    "sqlserver": {
        "server": "localhost",
        "database": "FaceAttendanceDB",
        "driver": "ODBC Driver 17 for SQL Server",
        "trusted_connection": true,
        "username": "sa",
        "password": ""
    }
}
```

---

## 📖 Quy Trình Sử Dụng Điểm Danh

1. **Đăng ký nhân viên**:
   - Vào tab **👤 Đăng ký nhân viên**.
   - Nhập Mã NV (VD: `NV001`) và Họ tên (VD: `Nguyễn Văn A`).
   - Bấm **Bật Camera** rồi bấm **📸 Chụp ảnh** (8–10 ảnh ở các góc mặt, ánh sáng và biểu cảm khác nhau).
   - Bấm **💾 Encode & Lưu Nhân Viên**.
2. **Chấm công Realtime**:
   - Vào tab **📷 Chấm công Realtime**.
   - Bấm **▶ Bắt đầu điểm danh**.
   - Khi đứng trước camera, hệ thống sẽ tự động nhận diện và ghi nhận:
     - Lần 1: **Check-in**
     - Lần 2: **Check-out** (sau khi hết thời gian Cooldown 75 giây).
3. **Xem & Xuất báo cáo**:
   - Vào tab **📋 Lịch sử chấm công** để tra cứu dữ liệu.
   - Vào tab **📈 Xuất báo cáo Excel** để tải file `.xlsx`.

