import io
from typing import Any, Dict, List, Optional, Tuple

import cv2
import face_recognition
import numpy as np
from PIL import Image

import config
from faiss_utils import get_faiss_engine


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
            try:
                pil_img = Image.open(io.BytesIO(img_input))
                return ensure_rgb_uint8(pil_img, is_bgr=False)
            except Exception:
                return None
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
        if arr.ndim == 2:  # Grayscale 2D
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        elif arr.ndim == 3:
            channels = arr.shape[2]
            if channels == 1:
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
            elif channels == 4:
                if is_bgr:
                    arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)
                else:
                    arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
            elif channels == 3:
                if is_bgr:
                    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
                else:
                    pass
            else:
                return None
        else:
            return None

        return np.ascontiguousarray(arr, dtype=np.uint8)

    return None


def extract_face_embedding_and_box(
    photo: Any,
    is_bgr: bool = True,
    model: str = "hog",
) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
    """
    Trích xuất vector khuôn mặt (float32, 128 chiều) và bounding box từ 1 ảnh.
    Chỉ chấp nhận ảnh có DUY NHẤT 1 khuôn mặt rõ nét.
    """
    rgb = ensure_rgb_uint8(photo, is_bgr=is_bgr)
    if rgb is None or rgb.size == 0:
        return None, None

    h, w = rgb.shape[:2]
    scaled_rgb = rgb
    fx = fy = 1.0
    if w > 1000 or h > 1000:
        fx = fy = 0.5
        scaled_rgb = cv2.resize(rgb, (0, 0), fx=0.5, fy=0.5)

    scaled_rgb = np.ascontiguousarray(scaled_rgb, dtype=np.uint8)

    locations = face_recognition.face_locations(
        scaled_rgb,
        model=model,
        number_of_times_to_upsample=1,
    )

    if len(locations) != 1:
        return None, None

    top, right, bottom, left = locations[0]
    if fx != 1.0:
        top = int(top / fy)
        right = int(right / fx)
        bottom = int(bottom / fy)
        left = int(left / fx)

    face_box = (top, right, bottom, left)

    face_encs = face_recognition.face_encodings(
        rgb,
        known_face_locations=[face_box],
        num_jitters=1,
    )

    if not face_encs:
        return None, None

    vec = np.asarray(face_encs[0], dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm

    return vec, face_box


def build_employee_embeddings(
    photos: List[Any],
    min_valid_photos: int = 3,
) -> Tuple[List[np.ndarray], Optional[Any], Optional[Tuple[int, int, int, int]]]:
    """
    Tạo danh sách các vector riêng biệt (float32) từ 3-5 (hoặc nhiều hơn) ảnh chụp nhân viên.
    Trả về: (embeddings_list, best_photo_for_avatar, best_face_box)
    """
    embeddings = []
    best_photo = None
    best_box = None
    max_box_area = 0

    for photo in photos:
        vec, box = extract_face_embedding_and_box(photo, is_bgr=True)
        if vec is not None and box is not None:
            embeddings.append(vec)
            # Chọn ảnh có diện tích khuôn mặt lớn và rõ nét nhất làm ảnh đại diện
            top, right, bottom, left = box
            area = (bottom - top) * (right - left)
            if area > max_box_area:
                max_box_area = area
                best_photo = photo
                best_box = box

    if len(embeddings) < min_valid_photos:
        return [], None, None

    return embeddings, best_photo, best_box


def build_employee_encoding(
    photos: List[Any],
    min_valid_photos: int = 3,
    fx: float = 0.75,
    fy: float = 0.75,
) -> Optional[np.ndarray]:
    """
    Hàm tương thích ngược: tính vector trung bình float32 từ danh sách ảnh.
    """
    embs, _, _ = build_employee_embeddings(photos, min_valid_photos=min_valid_photos)
    if not embs:
        return None

    mean_encoding = np.mean(np.asarray(embs, dtype=np.float32), axis=0)
    norm = np.linalg.norm(mean_encoding)
    if norm == 0:
        return None

    return (mean_encoding / norm).astype(np.float32)


def load_known_faces() -> Dict[str, Any]:
    """
    Khởi tạo hoặc nạp lại FAISS Index từ CSDL.
    """
    engine = get_faiss_engine()
    engine.init_engine()
    return {
        "ntotal": engine.index.ntotal if engine.index else 0,
        "metadata_count": len(engine.face_metadata),
    }


def recognize_faces_detailed(
    frame: np.ndarray,
    tolerance: Optional[float] = None,
    scale_factor: float = 0.5,
    is_bgr: bool = True,
    model: str = "hog",
) -> List[Dict[str, Any]]:
    """
    Nhận diện đa khuôn mặt trên frame ảnh bằng FAISS Search Top 5.
    - Xử lý ĐỘC LẬP từng khuôn mặt (1-to-1 mapping), loại bỏ hoàn toàn lỗi gán nhầm/trùng tên.
    - Chống trùng lặp (Deduplication): Trong 1 frame, 1 nhân viên chỉ được gán cho khuôn mặt khớp nhất.
    - Tối ưu HOG upsample=0 & scale_factor=0.5: Tốc độ phát hiện cực nhanh (~10-15ms).
    - Trả về danh sách chi tiết được sắp xếp theo khuôn mặt lớn nhất (gần camera nhất) trước.
    """
    rgb = ensure_rgb_uint8(frame, is_bgr=is_bgr)
    if rgb is None or rgb.size == 0:
        return []

    h, w = rgb.shape[:2]
    match_thresh = tolerance if tolerance is not None else config.FACE_MATCH_THRESHOLD

    # 1. Phát hiện vị trí khuôn mặt với scale_factor (HOG nhạy & nhanh, upsample=0)
    if scale_factor > 0 and scale_factor < 1.0:
        small_rgb = cv2.resize(rgb, (0, 0), fx=scale_factor, fy=scale_factor)
        small_rgb = np.ascontiguousarray(small_rgb, dtype=np.uint8)
        raw_locations = face_recognition.face_locations(
            small_rgb,
            model=model,
            number_of_times_to_upsample=0,
        )
        inv_scale = 1.0 / scale_factor
        scaled_locations = []
        for top, right, bottom, left in raw_locations:
            orig_top = max(0, min(h - 1, int(round(top * inv_scale))))
            orig_right = max(0, min(w - 1, int(round(right * inv_scale))))
            orig_bottom = max(0, min(h - 1, int(round(bottom * inv_scale))))
            orig_left = max(0, min(w - 1, int(round(left * inv_scale))))
            box_w = orig_right - orig_left
            box_h = orig_bottom - orig_top
            if box_w >= 50 and box_h >= 50:
                scaled_locations.append((orig_top, orig_right, orig_bottom, orig_left))
        locations = scaled_locations
    else:
        raw_locations = face_recognition.face_locations(
            rgb,
            model=model,
            number_of_times_to_upsample=0,
        )
        locations = [
            b for b in raw_locations
            if (b[1] - b[3]) >= 50 and (b[2] - b[0]) >= 50
        ]

    if not locations:
        return []

    # Sắp xếp theo diện tích bounding box giảm dần (khuôn mặt lớn nhất / gần camera nhất lên đầu)
    locations = sorted(locations, key=lambda b: (b[2] - b[0]) * (b[1] - b[3]), reverse=True)
    # Giới hạn tối đa 3 khuôn mặt nổi bật nhất trong 1 frame để đảm bảo hiệu năng CPU
    locations = locations[:3]

    engine = get_faiss_engine()
    has_faiss = engine.index is not None and engine.index.ntotal > 0

    # 2. Trích xuất encoding theo batch cho các khuôn mặt đã phát hiện
    try:
        face_encs = face_recognition.face_encodings(
            rgb,
            known_face_locations=locations,
            num_jitters=1,
        )
    except Exception:
        face_encs = []

    raw_results = []

    # 3. Duyệt và truy vấn FAISS cho từng khuôn mặt
    for idx, face_box in enumerate(locations):
        enc = face_encs[idx] if idx < len(face_encs) else None
        if enc is None or len(enc) == 0 or not has_faiss:
            raw_results.append({
                "box": face_box,
                "name": "Unknown",
                "code": "Unknown",
                "distance": 999.0,
                "status": "UNKNOWN",
                "candidate": None,
            })
            continue

        vec = np.asarray(enc, dtype=np.float32)
        search_res = engine.search_face(vec, top_k=5, match_threshold=match_thresh)

        status = search_res.get("status", "UNKNOWN")
        best_cand = search_res.get("candidate")
        dist = float(search_res.get("distance", 999.0))

        if status == "MATCH" and best_cand is not None and dist <= match_thresh:
            raw_results.append({
                "box": face_box,
                "name": best_cand["full_name"],
                "code": best_cand["employee_code"],
                "distance": dist,
                "status": "MATCH",
                "candidate": best_cand,
            })
        elif status == "AMBIGUOUS" and best_cand is not None:
            raw_results.append({
                "box": face_box,
                "name": f"AMBIGUOUS: {best_cand['full_name']}",
                "code": "AMBIGUOUS",
                "distance": dist,
                "status": "AMBIGUOUS",
                "candidate": best_cand,
                "second_candidate": search_res.get("second_candidate"),
            })
        else:
            raw_results.append({
                "box": face_box,
                "name": "Unknown",
                "code": "Unknown",
                "distance": dist,
                "status": "UNKNOWN",
                "candidate": None,
            })

    # 4. KHỬ TRÙNG LẶP (1-to-1 DEDUPLICATION):
    # Trong cùng 1 frame, một mã nhân viên chỉ được gán cho 1 khuôn mặt có khoảng cách nhỏ nhất
    assigned_codes = set()
    match_items = [r for r in raw_results if r["status"] == "MATCH"]
    match_items.sort(key=lambda x: x["distance"])

    for item in match_items:
        code = item["code"]
        if code not in assigned_codes:
            assigned_codes.add(code)
        else:
            # Mặt này có khoảng cách lớn hơn mặt kia -> đánh dấu Unknown
            item["status"] = "UNKNOWN"
            item["name"] = "Unknown"
            item["code"] = "Unknown"
            item["candidate"] = None

    return raw_results


def recognize_faces(
    frame: np.ndarray,
    known_encodings=None,
    known_names=None,
    known_codes=None,
    scale: int = 1,
    tolerance: Optional[float] = None,
    is_bgr: bool = True,
    scale_factor: float = 0.5,
):
    """
    Hàm tương thích ngược với API cũ.
    Trả về: (locations, names, codes, distances) nếu known_codes được yêu cầu,
    ngược lại trả về: (locations, names, distances).
    """
    detailed = recognize_faces_detailed(
        frame=frame,
        tolerance=tolerance,
        scale_factor=scale_factor,
        is_bgr=is_bgr,
    )

    if not detailed:
        if known_codes is not None:
            return [], [], [], []
        return [], [], []

    locations = [item["box"] for item in detailed]
    names = [item["name"] for item in detailed]
    codes = [item["code"] for item in detailed]
    distances = [item["distance"] for item in detailed]

    if known_codes is not None:
        return locations, names, codes, distances

    return locations, names, distances
