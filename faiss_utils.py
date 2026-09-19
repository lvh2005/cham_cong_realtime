import os
import threading
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np

import config
from database import load_all_active_face_embeddings


class FAISSEngine:
    """
    Bộ máy quản lý tìm kiếm khuôn mặt FAISS tối ưu cho 10.000+ nhân viên (30.000 - 50.000 vectors).
    - Sử dụng faiss.IndexFlatL2(128) bọc bởi faiss.IndexIDMap2.
    - ID vector trong FAISS chính là FaceID từ CSDL SQL Server.
    - Hỗ trợ Gom nhóm theo EmployeeID, phát hiện AMBIGUOUS và Duplicate Face.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(FAISSEngine, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self.dimension = 128
        self.index: Optional[faiss.IndexIDMap2] = None
        self.face_metadata: Dict[int, Dict[str, Any]] = {}  # {face_id: {"employee_id", "employee_code", "full_name"}}
        self.emp_to_face_ids: Dict[int, List[int]] = {}  # {employee_id: [face_id1, face_id2...]}
        self.engine_lock = threading.RLock()
        self._initialized = True

    def init_engine(self, force_rebuild: bool = False):
        """Khởi động bộ nhớ FAISS Index từ file hoặc rebuild từ SQL Server."""
        with self.engine_lock:
            if not force_rebuild and os.path.exists(config.FAISS_INDEX_PATH):
                try:
                    self.index = faiss.read_index(config.FAISS_INDEX_PATH)
                    self._refresh_metadata_cache()
                    # Kiểm tra tính đồng bộ
                    if self.index.ntotal == len(self.face_metadata):
                        return
                except Exception as ex:
                    print(f"[FAISS Warning] Lỗi đọc index file ({ex}), tiến hành rebuild từ SQL Server...")

            self.rebuild_from_database()

    def rebuild_from_database(self):
        """
        Nạp toàn bộ vector active từ Microsoft SQL Server và xây dựng lại toàn bộ FAISS Index.
        Không phụ thuộc vào file ảnh hay Cloudinary.
        """
        with self.engine_lock:
            raw_index = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIDMap2(raw_index)
            self.face_metadata.clear()
            self.emp_to_face_ids.clear()

            active_embeddings = load_all_active_face_embeddings()
            if not active_embeddings:
                self.save_index()
                return

            vectors = []
            ids = []

            for item in active_embeddings:
                f_id = item["face_id"]
                emp_id = item["employee_id"]
                code = item["employee_code"]
                name = item["full_name"]
                vec = np.asarray(item["embedding"], dtype=np.float32).flatten()

                # Đảm bảo chuẩn hóa L2
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm

                vectors.append(vec)
                ids.append(f_id)

                self.face_metadata[f_id] = {
                    "employee_id": emp_id,
                    "employee_code": code,
                    "full_name": name,
                }

                if emp_id not in self.emp_to_face_ids:
                    self.emp_to_face_ids[emp_id] = []
                self.emp_to_face_ids[emp_id].append(f_id)

            if vectors:
                vectors_matrix = np.ascontiguousarray(np.vstack(vectors), dtype=np.float32)
                ids_array = np.ascontiguousarray(np.array(ids, dtype=np.int64))
                self.index.add_with_ids(vectors_matrix, ids_array)

            self.save_index()

    def _refresh_metadata_cache(self):
        """Cập nhật lại từ điển metadata từ database mà không cần tạo lại index."""
        active_embeddings = load_all_active_face_embeddings()
        self.face_metadata.clear()
        self.emp_to_face_ids.clear()

        for item in active_embeddings:
            f_id = item["face_id"]
            emp_id = item["employee_id"]
            self.face_metadata[f_id] = {
                "employee_id": emp_id,
                "employee_code": item["employee_code"],
                "full_name": item["full_name"],
            }
            if emp_id not in self.emp_to_face_ids:
                self.emp_to_face_ids[emp_id] = []
            self.emp_to_face_ids[emp_id].append(f_id)

    def save_index(self):
        """Ghi FAISS index xuống đĩa an toàn."""
        with self.engine_lock:
            if self.index is not None:
                os.makedirs(os.path.dirname(config.FAISS_INDEX_PATH), exist_ok=True)
                faiss.write_index(self.index, config.FAISS_INDEX_PATH)

    def add_embeddings(
        self,
        face_ids: List[int],
        embeddings: List[np.ndarray],
        employee_id: int,
        employee_code: str,
        full_name: str,
    ):
        """Thêm các vector mới vào FAISS index kèm FaceID tương ứng."""
        with self.engine_lock:
            if self.index is None:
                self.init_engine()

            if not face_ids or not embeddings:
                return

            vectors = []
            ids = []

            for f_id, emb in zip(face_ids, embeddings):
                vec = np.asarray(emb, dtype=np.float32).flatten()
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm

                vectors.append(vec)
                ids.append(f_id)

                self.face_metadata[f_id] = {
                    "employee_id": employee_id,
                    "employee_code": employee_code,
                    "full_name": full_name,
                }
                if employee_id not in self.emp_to_face_ids:
                    self.emp_to_face_ids[employee_id] = []
                self.emp_to_face_ids[employee_id].append(f_id)

            vectors_matrix = np.ascontiguousarray(np.vstack(vectors), dtype=np.float32)
            ids_array = np.ascontiguousarray(np.array(ids, dtype=np.int64))
            self.index.add_with_ids(vectors_matrix, ids_array)
            self.save_index()

    def remove_employee(self, employee_id: int):
        """Xóa toàn bộ vector của một nhân viên khỏi FAISS (sử dụng khi Soft Delete)."""
        with self.engine_lock:
            if employee_id in self.emp_to_face_ids:
                face_ids = self.emp_to_face_ids[employee_id]
                if face_ids and self.index is not None:
                    ids_to_remove = np.array(face_ids, dtype=np.int64)
                    try:
                        self.index.remove_ids(ids_to_remove)
                        for fid in face_ids:
                            self.face_metadata.pop(fid, None)
                        self.emp_to_face_ids.pop(employee_id, None)
                        self.save_index()
                    except Exception:
                        self.rebuild_from_database()
            else:
                self.rebuild_from_database()

    def search_face(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        match_threshold: Optional[float] = None,
        ambiguous_margin: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Tìm kiếm khuôn mặt trong FAISS:
        1. Query Top 5 FaceID có khoảng cách Euclidean L2 nhỏ nhất.
        2. Gom nhóm kết quả theo EmployeeID (lấy min distance của từng nhân viên).
        3. Kiểm tra Match Threshold & Ambiguous Margin:
           - Nếu khoảng cách Top 1 > Match Threshold -> 'UNKNOWN'
           - Nếu chênh lệch (Top 2 - Top 1) < Ambiguous Margin -> 'AMBIGUOUS' (Hai người giống nhau)
           - Ngược lại -> 'MATCH'
        """
        with self.engine_lock:
            if self.index is None or self.index.ntotal == 0:
                return {
                    "status": "UNKNOWN",
                    "candidate": None,
                    "distance": 999.0,
                    "all_candidates": [],
                }

            thresh = match_threshold if match_threshold is not None else config.FACE_MATCH_THRESHOLD
            margin = ambiguous_margin if ambiguous_margin is not None else config.FACE_AMBIGUOUS_MARGIN

            vec = np.asarray(query_embedding, dtype=np.float32).flatten()
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            query_matrix = np.ascontiguousarray(np.expand_dims(vec, axis=0), dtype=np.float32)

            k = min(max(top_k, 5), self.index.ntotal)
            distances, face_ids = self.index.search(query_matrix, k)

            # Gom nhóm kết quả theo EmployeeID
            emp_candidates: Dict[int, Dict[str, Any]] = {}

            for dist, f_id in zip(distances[0], face_ids[0]):
                if f_id == -1 or f_id not in self.face_metadata:
                    continue

                meta = self.face_metadata[f_id]
                emp_id = meta["employee_id"]

                # Vì IndexFlatL2 trả về bình phương khoảng cách L2 nếu không căn bậc 2,
                # hoặc khoảng cách L2 trực tiếp (với vector chuẩn hóa: sqrt(d^2))
                l2_dist = float(np.sqrt(max(0.0, float(dist)))) if dist >= 0 else 0.0

                if emp_id not in emp_candidates or l2_dist < emp_candidates[emp_id]["distance"]:
                    emp_candidates[emp_id] = {
                        "employee_id": emp_id,
                        "employee_code": meta["employee_code"],
                        "full_name": meta["full_name"],
                        "distance": l2_dist,
                        "face_id": int(f_id),
                    }

            if not emp_candidates:
                return {
                    "status": "UNKNOWN",
                    "candidate": None,
                    "distance": 999.0,
                    "all_candidates": [],
                }

            # Sắp xếp các nhân viên theo khoảng cách nhỏ nhất
            sorted_candidates = sorted(emp_candidates.values(), key=lambda x: x["distance"])
            best = sorted_candidates[0]

            if best["distance"] > thresh:
                return {
                    "status": "UNKNOWN",
                    "candidate": None,
                    "distance": best["distance"],
                    "all_candidates": sorted_candidates,
                }

            # Kiểm tra trường hợp Ambiguous (Hai người khác nhau có khoảng cách quá gần)
            if len(sorted_candidates) > 1:
                second = sorted_candidates[1]
                diff = second["distance"] - best["distance"]
                if diff < margin and second["distance"] <= (thresh + 0.05):
                    return {
                        "status": "AMBIGUOUS",
                        "candidate": best,
                        "second_candidate": second,
                        "distance": best["distance"],
                        "margin": diff,
                        "all_candidates": sorted_candidates,
                    }

            return {
                "status": "MATCH",
                "candidate": best,
                "distance": best["distance"],
                "all_candidates": sorted_candidates,
            }

    def check_duplicate_face(
        self,
        embeddings: List[np.ndarray],
        duplicate_threshold: Optional[float] = None,
        exclude_employee_id: Optional[int] = None,
        exclude_employee_code: Optional[str] = None,
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Kiểm tra xem các vector khuôn mặt chụp khi đăng ký có trùng với người khác đã có trong hệ thống không.
        - Có thể truyền exclude_employee_id hoặc exclude_employee_code khi cập nhật lại mặt cho nhân viên cũ.
        Trả về: (is_duplicate: bool, existing_employee_info: Optional[dict])
        """
        with self.engine_lock:
            if self.index is None or self.index.ntotal == 0:
                return False, None

            dup_thresh = duplicate_threshold if duplicate_threshold is not None else config.FACE_DUPLICATE_THRESHOLD

            for emb in embeddings:
                res = self.search_face(emb, top_k=5, match_threshold=dup_thresh)
                if res["status"] in ("MATCH", "AMBIGUOUS"):
                    candidates = res.get("all_candidates", [])
                    if not candidates and res.get("candidate"):
                        candidates = [res["candidate"]]

                    for cand in candidates:
                        # Bỏ qua chính nhân viên đang được cập nhật
                        if exclude_employee_id is not None and cand.get("employee_id") == exclude_employee_id:
                            continue
                        if exclude_employee_code and cand.get("employee_code", "").strip().upper() == str(exclude_employee_code).strip().upper():
                            continue

                        if cand.get("distance", 999.0) <= dup_thresh:
                            return True, cand

            return False, None


# Singleton instance toàn cục
faiss_engine = FAISSEngine()


def get_faiss_engine() -> FAISSEngine:
    """Lấy singleton instance FAISSEngine."""
    return faiss_engine
