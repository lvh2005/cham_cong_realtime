import os
from flask import Blueprint, flash, redirect, render_template, url_for

import cloudinary_utils
import config
import database
import faiss_utils
from web_admin.routes.auth import login_required

system_bp = Blueprint("system", __name__)


@system_bp.route("/system")
@login_required
def index():
    """Trang thông tin trạng thái hệ thống và FAISS Index."""
    # 1. Kiểm tra SQL Server
    db_ok, db_msg = database.test_db_connection()
    db_info = {
        "status": "CONNECTED" if db_ok else "ERROR",
        "message": db_msg,
        "server": config.DB_SERVER,
        "database": config.DB_DATABASE,
        "driver": config.DB_DRIVER,
        "trusted": config.DB_TRUSTED_CONNECTION,
    }

    # 2. Kiểm tra Cloudinary
    cloud_ok = cloudinary_utils.is_cloudinary_configured()
    cloud_info = {
        "status": "CONFIGURED" if cloud_ok else "NOT CONFIGURED",
        "cloud_name": config.CLOUDINARY_CLOUD_NAME if cloud_ok else "(Chưa điền)",
        "folder": config.CLOUDINARY_FOLDER,
    }

    # 3. Kiểm tra FAISS Index
    engine = faiss_utils.get_faiss_engine()
    if engine.index is None:
        try:
            engine.init_engine()
        except Exception:
            pass

    index_exists = os.path.exists(config.FAISS_INDEX_PATH)
    file_size_kb = 0
    if index_exists:
        try:
            file_size_kb = round(os.path.getsize(config.FAISS_INDEX_PATH) / 1024, 2)
        except Exception:
            pass

    faiss_total = engine.index.ntotal if engine.index is not None else 0

    # 4. Thống kê CSDL liên quan FAISS
    stats = database.get_dashboard_stats()

    faiss_info = {
        "status": "READY" if engine.index is not None else "ERROR",
        "vectors_count": faiss_total,
        "dimension": engine.dimension,
        "index_type": "IndexIDMap2(IndexFlatL2)",
        "file_path": config.FAISS_INDEX_PATH,
        "file_exists": index_exists,
        "file_size_kb": file_size_kb,
        "db_active_embeddings": stats["total_embeddings"],
        "active_employees": stats["active_employees"],
        "match_threshold": config.FACE_MATCH_THRESHOLD,
        "duplicate_threshold": config.FACE_DUPLICATE_THRESHOLD,
        "ambiguous_margin": config.FACE_AMBIGUOUS_MARGIN,
    }

    return render_template(
        "system.html",
        db_info=db_info,
        cloud_info=cloud_info,
        faiss_info=faiss_info,
    )


@system_bp.route("/system/rebuild-faiss", methods=["POST"])
@login_required
def rebuild_faiss():
    """Tải lại toàn bộ vector active từ SQL Server và Rebuild FAISS Index."""
    try:
        engine = faiss_utils.get_faiss_engine()
        engine.rebuild_from_database()
        vector_count = engine.index.ntotal if engine.index is not None else 0
        flash(
            f"Rebuild FAISS Index thành công! Đã nạp {vector_count} vectors khuôn mặt từ Microsoft SQL Server.",
            "success",
        )
    except Exception as ex:
        flash(f"Lỗi khi Rebuild FAISS Index: {str(ex)}", "danger")

    return redirect(url_for("system.index"))
