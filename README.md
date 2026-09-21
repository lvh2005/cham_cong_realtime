# 📸 Face Attendance Realtime

Hệ thống **chấm công realtime bằng nhận diện khuôn mặt** được xây dựng bằng Python.
Ứng dụng hỗ trợ đăng ký nhân viên, nhận diện khuôn mặt qua camera và tự động ghi nhận thời gian chấm công.

## ✨ Tính năng

* 👤 Quản lý nhân viên

  * Thêm nhân viên mới
  * Mã nhân viên
  * Họ và tên
  * Thông tin nhân viên
* 📷 Đăng ký khuôn mặt

  * Sử dụng camera để thu thập hình ảnh khuôn mặt
  * Lưu dữ liệu khuôn mặt phục vụ nhận diện
* 🤖 Nhận diện khuôn mặt realtime

  * Nhận diện trực tiếp qua webcam
  * Tự động xác định nhân viên
  * Hiển thị thông tin người được nhận diện
* ⏰ Chấm công tự động

  * Ghi nhận thời gian vào
  * Hạn chế chấm công trùng
  * Lưu lịch sử chấm công
* 🔊 Phát âm thanh thông báo

  * Thông báo khi nhận diện thành công
  * Hỗ trợ âm thanh riêng cho từng nhân viên
* 🖥️ Giao diện desktop bằng Tkinter
* 💾 Lưu trữ dữ liệu cục bộ
* ⚡ Xử lý realtime thông qua camera

## 🛠️ Công nghệ sử dụng

| Công nghệ          | Mục đích                   |
| ------------------ | -------------------------- |
| Python             | Ngôn ngữ lập trình chính   |
| Tkinter            | Xây dựng giao diện desktop |
| OpenCV             | Xử lý camera và hình ảnh   |
| Face Recognition   | Nhận diện khuôn mặt        |
| NumPy              | Xử lý dữ liệu hình ảnh     |
| SQLite / Database  | Lưu trữ dữ liệu            |
| Pygame / Playsound | Phát âm thanh              |

## 📁 Cấu trúc dự án

```text
face_attendance/
│
├── app_tkinter.py          # Chương trình chính
├── config.py               # Cấu hình hệ thống
├── face_utils.py           # Các hàm xử lý nhận diện khuôn mặt
│
├── assets/
│   └── sounds/             # Âm thanh thông báo
│
├── data/                   # Dữ liệu hệ thống
│
├── faces/                  # Dữ liệu khuôn mặt
│
├── attendance/             # Dữ liệu chấm công
│
├── .env.example            # File cấu hình môi trường mẫu
├── requirements.txt        # Danh sách thư viện
└── README.md               # Tài liệu dự án
```

> Cấu trúc thư mục có thể thay đổi tùy theo phiên bản triển khai của hệ thống.

## ⚙️ Yêu cầu hệ thống

* Windows 10/11
* Python 3.10+
* Webcam
* RAM tối thiểu 4GB
* CPU hỗ trợ chạy xử lý hình ảnh realtime

## 🚀 Cài đặt

### 1. Clone project

```bash
git clone https://github.com/lvh2005/cham_cong_realtime.git
cd cham_cong_realtime
```

### 2. Tạo môi trường ảo

```bash
python -m venv .venv
```

Kích hoạt môi trường ảo trên Windows:

```bash
.venv\Scripts\activate
```

### 3. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

Nếu project chưa có `requirements.txt`, có thể cài các thư viện cần thiết theo cấu hình hiện tại của project.

### 4. Cấu hình môi trường

Tạo file `.env` từ `.env.example`:

```bash
copy .env.example .env
```

Sau đó mở `.env` và thiết lập các thông số cần thiết.

> **Không commit file `.env` lên GitHub** nếu file chứa mật khẩu, API key hoặc thông tin bảo mật.

## ▶️ Chạy chương trình

Sau khi cài đặt xong:

```bash
python app_tkinter.py
```

Ứng dụng sẽ mở giao diện quản lý và chấm công.

## 📷 Quy trình sử dụng

### Bước 1: Thêm nhân viên

Nhập:

* Mã nhân viên
* Họ tên
* Các thông tin cần thiết

Sau đó thực hiện đăng ký khuôn mặt.

### Bước 2: Thu thập khuôn mặt

Sử dụng webcam để chụp dữ liệu khuôn mặt của nhân viên.

Nên đảm bảo:

* Khuôn mặt nhìn rõ camera
* Đủ ánh sáng
* Không che khuất khuôn mặt
* Thay đổi nhẹ góc mặt khi thu thập dữ liệu

### Bước 3: Chấm công

Mở camera realtime.

Hệ thống sẽ:

```text
Camera
   ↓
Phát hiện khuôn mặt
   ↓
So sánh dữ liệu
   ↓
Xác định nhân viên
   ↓
Ghi nhận chấm công
   ↓
Phát thông báo
```

### Bước 4: Kiểm tra lịch sử

Dữ liệu chấm công được lưu lại để phục vụ việc kiểm tra và quản lý.

## 🔐 Lưu ý bảo mật

Không đưa các thông tin nhạy cảm lên GitHub, đặc biệt:

```text
.env
API keys
Database passwords
Secret keys
Private credentials
```

Nên sử dụng `.env.example` để cung cấp cấu trúc cấu hình cho người khác.

## 🐛 Xử lý lỗi thường gặp

### Camera không hoạt động

Kiểm tra webcam và thử đóng các ứng dụng khác đang sử dụng camera.

Nếu project cho phép lựa chọn camera, thử thay đổi camera index:

```python
cv2.VideoCapture(0)
```

Có thể thử:

```python
cv2.VideoCapture(1)
```

### Không nhận diện được khuôn mặt

Kiểm tra:

* Ánh sáng
* Chất lượng camera
* Dữ liệu khuôn mặt đã đăng ký
* Khoảng cách tới camera

### Không cài được thư viện

Nên sử dụng đúng phiên bản Python và cập nhật pip:

```bash
python -m pip install --upgrade pip
```

Sau đó:

```bash
pip install -r requirements.txt
```

## 📌 Mục tiêu dự án

Dự án được xây dựng nhằm nghiên cứu và ứng dụng công nghệ **Computer Vision** và **Face Recognition** vào bài toán quản lý chấm công tự động.

Hệ thống hướng tới việc giảm thao tác chấm công thủ công và hỗ trợ quản lý dữ liệu nhân viên, thời gian làm việc một cách thuận tiện hơn.

## 👨‍💻 Tác giả

**Lê Vũ Hà**

GitHub:

https://github.com/lvh2005

Repository:

https://github.com/lvh2005/cham_cong_realtime

## 📄 License

Dự án được sử dụng cho mục đích **học tập và nghiên cứu**.
