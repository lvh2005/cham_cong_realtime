import ctypes
import os
import threading
import time
from typing import Optional

SOUNDS_DIR = os.path.join(os.path.dirname(__file__), "assets", "sounds")
os.makedirs(SOUNDS_DIR, exist_ok=True)


def _play_mp3_windows(file_path: str):
    """Phát file mp3 sử dụng Windows MCI API trong background thread (0ms lag, không popup)."""
    try:
        abs_path = os.path.abspath(file_path).replace("\\", "/")
        alias = f"voice_{int(time.time() * 1000)}"
        ctypes.windll.winmm.mciSendStringW(f'open "{abs_path}" type mpegvideo alias {alias}', None, 0, None)
        ctypes.windll.winmm.mciSendStringW(f'play {alias} wait', None, 0, None)
        ctypes.windll.winmm.mciSendStringW(f'close {alias}', None, 0, None)
    except Exception as e:
        print(f"Lỗi phát âm thanh MCI: {e}")


def _speak_pyttsx3_fallback(text: str):
    """Fallback sang pyttsx3 nếu không có mạng."""
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 140)
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"Lỗi pyttsx3 fallback: {e}")


def speak_async(text: str = "Xin cảm ơn!", employee_name: Optional[str] = None):
    """
    Phát giọng nói tiếng Việt 'Xin cảm ơn...' bất đồng bộ không chặn UI.
    Tự động cache file âm thanh để các lần sau phát tức thì 0ms.
    """
    def worker():
        full_text = text
        if employee_name and employee_name.strip():
            full_text = f"Xin cảm ơn {employee_name.strip()}!"

        # Đặt tên file cache an toàn
        safe_name = "".join(c for c in full_text if c.isalnum() or c in (" ", "_", "-")).strip().replace(" ", "_")
        cache_file = os.path.join(SOUNDS_DIR, f"{safe_name}.mp3")

        # 1. Nếu đã có file âm thanh trong cache -> phát ngay
        if os.path.exists(cache_file) and os.path.getsize(cache_file) > 1000:
            _play_mp3_windows(cache_file)
            return

        # 2. Tạo file âm thanh bằng gTTS tiếng Việt chuẩn
        try:
            from gtts import gTTS
            tts = gTTS(text=full_text, lang="vi", slow=False)
            tts.save(cache_file)
            _play_mp3_windows(cache_file)
        except Exception as ex:
            print(f"Không thể tải gTTS, chuyển sang engine offline: {ex}")
            # 3. Fallback pyttsx3
            _speak_pyttsx3_fallback(full_text)

    threading.Thread(target=worker, daemon=True).start()


if __name__ == "__main__":
    print("Đang thử nghiệm phát giọng nói...")
    speak_async("Xin cảm ơn Lê Vũ Hà!")
    time.sleep(3)
    print("Hoàn tất thử nghiệm.")
