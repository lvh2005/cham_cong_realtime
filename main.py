import sys

if __name__ == "__main__":
    print("=========================================================")
    print("🚀 ĐANG KHỞI ĐỘNG HỆ THỐNG CHẤM CÔNG (FACE ATTENDANCE)...")
    print("🖥️ Cửa sổ giao diện Tkinter đang mở trên màn hình của bạn.")
    print("💡 (Nếu không thấy cửa sổ, vui lòng kiểm tra dưới Taskbar)")
    print("=========================================================")
    sys.stdout.flush()

    from app_tkinter import main
    main()

