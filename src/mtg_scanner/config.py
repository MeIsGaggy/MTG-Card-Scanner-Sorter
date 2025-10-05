# config.py
import os, json

SETTINGS_PATH = os.environ.get("SETTINGS_PATH", "./settings.json")
try:
    with open(SETTINGS_PATH, "r", encoding="utf-8") as _f:
        _SET = json.load(_f) or {}
except Exception:
    _SET = {}

def _get(name, default, cast=str):
    # priority: env > settings.json > default
    if name in os.environ:
        v = os.environ[name]
    elif name in _SET:
        v = _SET[name]
    else:
        v = default
    if cast is bool:
        if isinstance(v, bool): return v
        return str(v).lower() in ("1","true","on","yes")
    if cast is list:
        return list(v) if isinstance(v, (list,tuple)) else list(default)
    try:
        return cast(v)
    except Exception:
        return default

# =========================
# CAMERA
# =========================
CAMERA_DEVICE = _get("CAMERA_DEVICE", "/dev/video0", str)
REQ_WIDTH  = _get("REQ_WIDTH", 1920, int)
REQ_HEIGHT = _get("REQ_HEIGHT", 1080, int)
REQ_FPS    = _get("REQ_FPS", 30, int)
FOURCC_PRIMARY  = _get("FOURCC_PRIMARY", "MJPG", str)
FOURCC_FALLBACK = _get("FOURCC_FALLBACK", "YUYV", str)

# =========================
# PROCESSING / CARD CANVAS
# =========================
PROC_MAX_WIDTH = _get("PROC_MAX_WIDTH", 640, int)
CARD_W = _get("CARD_W", 1280, int)
CARD_H = _get("CARD_H", 1920, int)

MIN_CARD_AREA_RATIO = _get("MIN_CARD_AREA_RATIO", 0.015, float)
MAX_CARD_AREA_RATIO = _get("MAX_CARD_AREA_RATIO", 0.90, float)
BORDER_MARGIN_PCT   = _get("BORDER_MARGIN_PCT", 0.005, float)

CARD_ASPECT = _get("CARD_ASPECT", 63/88.0, float)
ASPECT_TOL  = _get("ASPECT_TOL",  0.22, float)
CONFIRM_FRAMES = _get("CONFIRM_FRAMES", 7, int)
STALE_FRAMES   = _get("STALE_FRAMES",   4, int)
MAX_ASSOC_DIST = _get("MAX_ASSOC_DIST", 80, int)
MAX_CARDS      = _get("MAX_CARDS", 1, int)

# Autoscan / Steady
STEADY_MIN_FRAMES    = _get("STEADY_MIN_FRAMES", 10, int)
AUTO_CAPTURE_WAIT_S  = _get("AUTO_CAPTURE_WAIT_S", 0.05, float)
AUTOSCAN_OCR_TIMEOUT = _get("AUTOSCAN_OCR_TIMEOUT", 60.0, float)
AUTOSCAN_SCRY_TIMEOUT= _get("AUTOSCAN_SCRY_TIMEOUT", 3.0, float)
AUTOSCAN_IMG_TIMEOUT = _get("AUTOSCAN_IMG_TIMEOUT", 4.0, float)
# Detection cadence/robustness
DETECT_EVERY_N_FRAMES = _get("DETECT_EVERY_N_FRAMES", 2, int)
RECTANGULARITY_MIN    = _get("RECTANGULARITY_MIN", 0.76, float)
USE_TESS_FOR_TITLES = bool(globals().get("USE_TESS_FOR_TITLES", False))
# =========================
# OCR + DEBUG
# =========================
OCR_BACKEND = _get("OCR_BACKEND", "rapidocr,tesseract", str)
OCR_ONLY_ON_SNAPSHOT = _get("OCR_ONLY_ON_SNAPSHOT", True, bool)
DEBUG_OCR = _get("DEBUG_OCR", 1, bool)
OCR_DEBUG_BOXES = _get("OCR_DEBUG_BOXES", True, bool)
SHOW_ROI_OVERLAY= _get("SHOW_ROI_OVERLAY", True, bool)

MIN_TITLE_LETTERS = _get("MIN_TITLE_LETTERS", 4, int)
TEXT_PRESENCE_MIN = _get("TEXT_PRESENCE_MIN", 0.020, float)
TITLE_ALLOW_TESS_FALLBACK = _get("TITLE_ALLOW_TESS_FALLBACK", 0, bool)

# =========================
# FOIL DETECTION
# =========================
FOIL_MIN_SCORE = _get("FOIL_MIN_SCORE", 1.25, float)
FOIL_ON_TH     = _get("FOIL_ON_TH",  1.25, float)
FOIL_OFF_TH    = _get("FOIL_OFF_TH", 1.00, float)
FOIL_DETECT    = _get("FOIL_DETECT", 1, bool)

# =========================
# MATCH / COMPARISON
# =========================
MATCH_ENABLE = _get("MATCH_ENABLE", 1, bool)
MATCH_W_HASH = _get("MATCH_W_HASH", 0.35, float)
MATCH_W_HIST = _get("MATCH_W_HIST", 0.25, float)
MATCH_W_ORB  = _get("MATCH_W_ORB",  0.40, float)
MATCH_TH     = _get("MATCH_TH",     0.50, float)
NAME_OK_TH   = _get("NAME_OK_TH",   60.0, float)
ALWAYS_SCAN_OK = _get("ALWAYS_SCAN_OK", 1, bool)
MATCH_USE_ART  = _get("MATCH_USE_ART", 1, bool)

# =========================
# SCRYFALL / PRICES
# =========================
SCRYFALL_TIMEOUT = _get("SCRYFALL_TIMEOUT", 25.0, float)
FX_URL = _get("FX_URL", "https://api.exchangerate.host/latest?base=USD&symbols=CAD", str)
FX_TTL_SEC = _get("FX_TTL_SEC", 43200, int)

# =========================
# MOONRAKER / KLIPPER
# =========================
MOONRAKER_URL = _get("MOONRAKER_URL", "ws://localhost:7125/websocket", str)
HTTP_POST_URL = _get("HTTP_POST_URL", "http://localhost:7125/printer/gcode/script", str)

# =========================
# ROIs (your exact old defaults)
# =========================
ROI_TITLE_TOP   = _get("ROI_TITLE_TOP",   [0.057, 0.031, 0.85, 0.12], list)
ROI_TITLE_ALT   = _get("ROI_TITLE_ALT",   [0.057, 0.60,  0.88, 0.72], list)

ROI_NUMBER_MAIN   = [0.045, 0.885, 0.300, 0.990]  # was [0.045, 0.910, 0.300, 0.975]
ROI_NUMBER_WIDE   = [0.030, 0.880, 0.330, 0.995]  # was [0.030, 0.900, 0.330, 0.985]
ROI_NUMBER_TALL   = [0.045, 0.875, 0.300, 0.995]  # was [0.045, 0.90,  0.300, 0.985]
ROI_NUMBER_NARROW = [0.045, 0.885, 0.240, 0.990]  # was [0.045, 0.910, 0.240, 0.975]

ROI_NUMBER        = ROI_NUMBER_MAIN  # legacy alias

ROI_ART = _get("ROI_ART", [0.065, 0.155, 0.935, 0.590], list)

# =========================
# DEBUG / PATHS
# =========================
DEBUG_LEVEL    = _get("DEBUG_LEVEL", 1, int)
DEBUG_SAVE_ROI = _get("DEBUG_SAVE_ROI", 0, bool)
DEBUG_DIR      = _get("DEBUG_DIR", "/tmp/mtg_dbg", str)
BAD_DIR        = _get("BAD_DIR",   "/tmp/mtg_bad", str)

# =========================
# PERSISTENCE (Review history)
# =========================
HISTORY_DIR       = _get("HISTORY_DIR", "./history", str)
HISTORY_IMG_DIR   = _get("HISTORY_IMG_DIR", os.path.join(HISTORY_DIR, "imgs"), str)
HISTORY_JSON_EXT  = _get("HISTORY_JSON_EXT", ".json", str)
JPEG_QUALITY_SNAP = _get("JPEG_QUALITY_SNAP", 82, int)
JPEG_QUALITY_CMP  = _get("JPEG_QUALITY_CMP",  80, int)
JPEG_QUALITY_THUMB= _get("JPEG_QUALITY_THUMB",70, int)

# =========================
# SERVER
# =========================
HOST = _get("HOST", "0.0.0.0", str)
PORT = _get("PORT", 5000, int)


# ==============================
# Performance tuning overrides
# ==============================
# Keep both so numbers OCR can still use Tesseract; title fallback is gated separately
OCR_BACKEND = "rapidocr,tesseract"

# Limit Tesseract fallback for title OCR (use RapidOCR primarily)
TITLE_ALLOW_TESS_FALLBACK = False

# Skip expensive blackhat variant unless explicitly enabled or foil is detected
OCR_ENABLE_BLACKHAT = False

# Heavier throttling / reduced workload
DETECT_EVERY_N_FRAMES = 2           # detect every 2nd frame
LIVE_OCR_ONLY_WHEN_STEADY = True     # only OCR when steady
LIVE_OCR_MIN_INTERVAL = 0.45         # seconds between live OCR passes
FOIL_EVERY_N_FRAMES = 10             # compute foil detection less often

# Downscale for detection stage
PROC_DOWNSCALE_MAX_W = 640

# Matching settings
MATCH_ORB_FEATURES = 600             # fewer ORB features -> faster
MATCH_W_HASH = 0.48
MATCH_W_HIST = 0.34
MATCH_W_ORB  = 0.18                  # rely less on ORB

# Optional: small safety margin for fast accept/reject remains in app