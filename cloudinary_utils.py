import os
import time
from typing import Any, Optional, Tuple

import cv2
import numpy as np

import config

# Thử import cloudinary
try:
    import cloudinary
    import cloudinary.uploader
    HAS_CLOUDINARY = True
except ImportError:
    HAS_CLOUDINARY = False


def is_cloudinary_configured() -> bool:
    """Kiểm tra xem thông tin cấu hình Cloudinary đã được nhập đầy đủ hay chưa."""
    config.reload_config()
    return bool(
        config.CLOUDINARY_CLOUD_NAME
        and config.CLOUDINARY_API_KEY
        and config.CLOUDINARY_API_SECRET
    )


def _init_cloudinary():
    """Khởi tạo cấu hình Cloudinary từ file config."""
    if not HAS_CLOUDINARY:
        return
    if is_cloudinary_configured():
        cloudinary.config(
            cloud_name=config.CLOUDINARY_CLOUD_NAME,
            api_key=config.CLOUDINARY_API_KEY,
            api_secret=config.CLOUDINARY_API_SECRET,
            secure=True,
        )


def prepare_avatar_image(
    image_input: Any,
    face_box: Optional[Tuple[int, int, int, int]] = None,
    target_size: Tuple[int, int] = (400, 400),
) -> Optional[bytes]:
    """
    Tiền xử lý ảnh đại diện:
    - Nếu có tọa độ bounding box khuôn mặt (top, right, bottom, left) -> crop mở rộng viền xung quanh 30%.
    - Resize về kích thước chuẩn (mặc định 400x400).
    - Nén JPEG chất lượng 85% để tối ưu dung lượng tải lên.
    """
    if image_input is None:
        return None

    # Chuyển đổi về ndarray BGR nếu cần
    if isinstance(image_input, (bytes, bytearray)):
        nparr = np.frombuffer(image_input, np.uint8)
        bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    elif isinstance(image_input, str):
        if not os.path.exists(image_input):
            return None
        bgr = cv2.imread(image_input, cv2.IMREAD_COLOR)
    elif isinstance(image_input, np.ndarray):
        bgr = image_input.copy()
        if bgr.ndim == 2:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
        elif bgr.shape[2] == 4:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_BGRA2BGR)
    elif hasattr(image_input, "convert"):
        # PIL Image
        rgb = np.array(image_input.convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    else:
        return None

    if bgr is None or bgr.size == 0:
        return None

    h, w = bgr.shape[:2]

    # Crop khuôn mặt có mở rộng padding nếu có face_box
    if face_box:
        top, right, bottom, left = face_box
        box_w = right - left
        box_h = bottom - top

        pad_x = int(box_w * 0.35)
        pad_y = int(box_h * 0.45)

        crop_top = max(0, top - pad_y)
        crop_bottom = min(h, bottom + int(pad_y * 0.8))
        crop_left = max(0, left - pad_x)
        crop_right = min(w, right + pad_x)

        if crop_bottom > crop_top and crop_right > crop_left:
            bgr = bgr[crop_top:crop_bottom, crop_left:crop_right]

    # Resize về target size
    resized = cv2.resize(bgr, target_size, interpolation=cv2.INTER_AREA)

    # Encode sang JPEG bytes
    success, encoded_img = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not success:
        return None

    return encoded_img.tobytes()


def upload_employee_avatar(
    employee_code: str,
    image_input: Any,
    face_box: Optional[Tuple[int, int, int, int]] = None,
) -> Tuple[str, str]:
    """
    Tải ảnh đại diện nhân viên lên Cloudinary.
    Trả về: (secure_url, public_id)
    Nếu Cloudinary chưa được cấu hình hoặc gặp lỗi, trả về ("", "") mà không làm sập ứng dụng.
    """
    _init_cloudinary()

    jpeg_bytes = prepare_avatar_image(image_input, face_box)
    if jpeg_bytes is None:
        return "", ""

    if not HAS_CLOUDINARY or not is_cloudinary_configured():
        # Không có cấu hình Cloudinary: bỏ qua upload online
        return "", ""

    try:
        public_id = f"{employee_code.lower()}_{int(time.time())}"
        folder = config.CLOUDINARY_FOLDER

        response = cloudinary.uploader.upload(
            jpeg_bytes,
            folder=folder,
            public_id=public_id,
            resource_type="image",
            overwrite=True,
            transformation=[
                {"width": 400, "height": 400, "crop": "fill", "gravity": "face"},
                {"quality": "auto", "fetch_format": "auto"}
            ],
        )

        secure_url = response.get("secure_url", "")
        returned_public_id = response.get("public_id", "")
        return secure_url, returned_public_id
    except Exception as ex:
        print(f"[Cloudinary Warning] Không thể upload ảnh lên Cloudinary: {ex}")
        return "", ""


def delete_employee_avatar(public_id: str) -> bool:
    """Xóa ảnh đại diện nhân viên trên Cloudinary bằng public_id."""
    if not public_id or not HAS_CLOUDINARY or not is_cloudinary_configured():
        return False

    _init_cloudinary()
    try:
        res = cloudinary.uploader.destroy(public_id, resource_type="image")
        return res.get("result") == "ok"
    except Exception as ex:
        print(f"[Cloudinary Warning] Lỗi xóa ảnh {public_id}: {ex}")
        return False
