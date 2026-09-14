import io
from typing import Any, List, Optional

import cv2
import face_recognition
import numpy as np
from PIL import Image

from database import load_all_face_encodings


def ensure_rgb_uint8(img_input: Any, is_bgr: bool = True) -> Optional[np.ndarray]:
    """
    Chuẩn hóa mọi đầu vào ảnh (bytes, ndarray BGR/RGB/RGBA/Gray, PIL Image, đường dẫn file)
    thành mảng NumPy 3 kênh RGB liên tục (C-contiguous) với dtype=uint8.
    Ngăn chặn hoàn toàn lỗi 'Unsupported image type, must be 8bit gray or RGB image.' của dlib.
    """
    if img_input is None:
        return None

    # 1. Nếu là bytes
    if isinstance(img_input, (bytes, bytearray)):
        image_array = np.frombuffer(img_input, dtype=np.uint8)
        decoded = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if decoded is None:
            # Thử mở bằng PIL nếu OpenCV decode không thành công
            try:
                pil_img = Image.open(io.BytesIO(img_input))
                return ensure_rgb_uint8(pil_img, is_bgr=False)
            except Exception:
                return None
        # cv2.imdecode trả về BGR
        rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        return np.ascontiguousarray(rgb, dtype=np.uint8)

    # 2. Nếu là PIL Image
    if hasattr(img_input, "convert"):
        rgb_pil = img_input.convert("RGB")
        return np.ascontiguousarray(np.array(rgb_pil), dtype=np.uint8)

    # 3. Nếu là chuỗi đường dẫn file
    if isinstance(img_input, str):
        bgr = cv2.imread(img_input, cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return np.ascontiguousarray(rgb, dtype=np.uint8)

    # 4. Nếu là numpy.ndarray
    if isinstance(img_input, np.ndarray):
        arr = img_input

        # Chuyển đổi dtype sang uint8 nếu đang là float hoặc int khác
        if arr.dtype != np.uint8:
            if np.issubdtype(arr.dtype, np.floating):
                if arr.max() <= 1.0 and arr.min() >= 0.0:
                    arr = (arr * 255.0).astype(np.uint8)
                else:
                    arr = np.clip(arr, 0, 255).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)

        # Xử lý số chiều & kênh màu
        if arr.ndim == 2:  # Grayscale 2D (H, W)
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        elif arr.ndim == 3:
            channels = arr.shape[2]
            if channels == 1:
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
            elif channels == 4:
                # RGBA -> RGB hoặc BGRA -> RGB
                if is_bgr:
                    arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)
                else:
                    arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
            elif channels == 3:
                if is_bgr:
                    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
                else:
                    # Đã là RGB
                    pass
            else:
                return None
        else:
            return None

        return np.ascontiguousarray(arr, dtype=np.uint8)

    return None


def build_employee_encoding(
    photos: List[Any],
    min_valid_photos: int = 3,
    fx: float = 0.75,
    fy: float = 0.75,
) -> Optional[np.ndarray]:
    """
    Từ danh sách ảnh (bytes, file path, PIL Image hoặc ndarray)
    -> Chuẩn hóa RGB uint8
    -> Phát hiện mặt
    -> Tạo encoding
    -> Lấy vector trung bình rồi chuẩn hóa L2 norm.
    """
    encodings = []

    for photo in photos:
        rgb = ensure_rgb_uint8(photo, is_bgr=True)
        if rgb is None or rgb.size == 0:
            continue

        # Resize nhẹ nếu ảnh quá lớn để tăng tốc độ nhận diện
        h, w = rgb.shape[:2]
        if w > 1000 or h > 1000:
            scaled_rgb = cv2.resize(rgb, (0, 0), fx=0.5, fy=0.5)
        elif fx != 1.0 or fy != 1.0:
            scaled_rgb = cv2.resize(rgb, (0, 0), fx=fx, fy=fy)
        else:
            scaled_rgb = rgb

        scaled_rgb = np.ascontiguousarray(scaled_rgb, dtype=np.uint8)

        locations = face_recognition.face_locations(
            scaled_rgb,
            model="hog",
            number_of_times_to_upsample=1,
        )

        # Chỉ nhận ảnh có đúng 1 khuôn mặt
        if len(locations) != 1:
            continue

        face_encs = face_recognition.face_encodings(
            scaled_rgb,
            known_face_locations=locations,
            num_jitters=1,
        )

        if face_encs:
            encodings.append(face_encs[0])

    if len(encodings) < min_valid_photos:
        return None

    mean_encoding = np.mean(np.asarray(encodings, dtype=np.float64), axis=0)

    # Chuẩn hóa vector để khoảng cách Euclidean ổn định
    norm = np.linalg.norm(mean_encoding)
    if norm == 0:
        return None

    return (mean_encoding / norm).astype(np.float64)


def load_known_faces():
    """
    Load toàn bộ encoding vào RAM.
    """
    encodings, codes, names = load_all_face_encodings()

    return {
        "encodings": encodings,
        "codes": codes,
        "names": names,
    }


def recognize_faces(
    frame,
    known_encodings,
    known_names,
    known_codes=None,
    scale: int = 1,
    tolerance: float = 0.50,
    is_bgr: bool = True,
):
    """
    Nhận diện khuôn mặt trên frame ảnh.
    Hỗ trợ trả về tọa độ bounding box, tên nhân viên, mã nhân viên và khoảng cách sai số.
    """
    if not known_encodings:
        if known_codes is not None:
            return [], [], [], []
        return [], [], []

    rgb = ensure_rgb_uint8(frame, is_bgr=is_bgr)
    if rgb is None or rgb.size == 0:
        if known_codes is not None:
            return [], [], [], []
        return [], [], []


    locations = face_recognition.face_locations(
        rgb,
        model="hog",
        number_of_times_to_upsample=1,
    )


    if not locations:
        if known_codes is not None:
            return [], [], [], []
        return [], [], []

    encodings = face_recognition.face_encodings(
        rgb,
        known_face_locations=locations,
        num_jitters=1,
    )

    names = []
    codes = []
    distances = []

    for face_encoding in encodings:
        # Euclidean distance giữa encoding hiện tại và database
        distances_all = face_recognition.face_distance(
            known_encodings,
            face_encoding,
        )

        best_index = int(np.argmin(distances_all))
        best_distance = float(distances_all[best_index])

        if best_distance <= tolerance:
            names.append(known_names[best_index])
            if known_codes is not None:
                codes.append(known_codes[best_index])
        else:
            names.append("Unknown")
            if known_codes is not None:
                codes.append("Unknown")

        distances.append(best_distance)

    if known_codes is not None:
        return locations, names, codes, distances

    return locations, names, distances

