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
STEADY_MIN_FRAMES    = _get("STEADY_MIN_FRAMES", 12, int)
AUTO_CAPTURE_WAIT_S  = _get("AUTO_CAPTURE_WAIT_S", 0.35, float)
AUTOSCAN_OCR_TIMEOUT = _get("AUTOSCAN_OCR_TIMEOUT", 60.0, float)
AUTOSCAN_SCRY_TIMEOUT= _get("AUTOSCAN_SCRY_TIMEOUT", 60.0, float)
AUTOSCAN_IMG_TIMEOUT = _get("AUTOSCAN_IMG_TIMEOUT", 10.0, float)
# Detection cadence/robustness
DETECT_EVERY_N_FRAMES = _get("DETECT_EVERY_N_FRAMES", 1, int)
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
# AI / YOLOv5 Detection
# =========================
AI_DETECT_ENABLED = _get("AI_DETECT_ENABLED", True, bool)
AI_USE_FOR_CARDS  = _get("AI_USE_FOR_CARDS",  True, bool)
AI_USE_FOR_ROIS   = _get("AI_USE_FOR_ROIS",   True, bool)
# Point this at your trained weights (best.pt)
AI_MODEL_PATH     = _get("AI_MODEL_PATH", "/home/printpi/MTG_SCANNER_PROD/yolov5/runs/train/magic_vision2/weights/best.pt", str)
# Local yolov5 repo dir (for DetectMultiBackend)
YOLOV5_DIR        = _get("YOLOV5_DIR", "/home/printpi/MTG_SCANNER_PROD/yolov5", str)
AI_CONF_THRES     = _get("AI_CONF_THRES", 0.25, float)
AI_IOU_THRES      = _get("AI_IOU_THRES",  0.45, float)
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
DETECT_EVERY_N_FRAMES = 1           # detect every 2nd frame
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

# ========= AI / YOLO =========
AI_ENABLED        = True           # master on/off
AI_USE_FOR_CARDS  = True           # use AI to find the card box
AI_USE_FOR_ROIS   = True           # use AI to find title/set/etc ROIs
AI_MODEL_PATH     = "/home/printpi/MTG_SCANNER_PROD/yolov5/runs/train/magic_vision2/weights/best.pt"
YOLOV5_DIR        = "/home/printpi/MTG_SCANNER_PROD/yolov5"

AI_IMG_SIZE       = 640
AI_CONF_THRES     = 0.25
AI_IOU_THRES      = 0.45
AI_MAX_DETS       = 50

AI_CLASS_NAMES = ['1-Name', '2-Mana Value', '3-Set Symbol', '4-Card -', '5-Set Name']

AI_ROI_KEYMAP = {
    '1-Name'       : 'name',
    '2-Mana Value' : 'mana_value',
    '3-Set Symbol' : 'set_symbol',
    '4-Card -'     : 'card',
    '5-Set Name'   : 'set_name',
}
AI_CLASS_TO_KEY = AI_ROI_KEYMAP

AI_BOX_THICKNESS = 8
AI_LEGEND_FONT_SCALE = 2.5
AI_LEGEND_FONT_THICKNESS = 3
AI_LEGEND_LINE_THICKNESS = 8
AI_LEGEND_PAD = 10


AI_ONLY_MODE = True  # Only use AI ROIs; no static/fixed regions


# --- Icon matching (set symbol) ---
ICON_MATCH_ACCEPT = _get("ICON_MATCH_ACCEPT", 1.00, float)  # matchShapes score threshold (lower is better)
ICON_MATCH_SIZE   = _get("ICON_MATCH_SIZE",   96,   int)    # raster size for set SVG
ICON_MATCH_MARGIN = _get("ICON_MATCH_MARGIN", 0.02, float)  # require a clear margin vs 2nd-best
ICON_MATCH_REQUIRE_CLEAR_WIN = _get("ICON_MATCH_REQUIRE_CLEAR_WIN", True, bool)

# ---- AI name-ROI padding & OCR topline tuning ----
AI_NAME_PAD_X = 0.055  # widen left/right so first/last letters are not clipped
AI_NAME_PAD_Y = 0.010  # slight vertical headroom
AI_TOPLINE_FRAC_MIN = 0.14  # at least this fraction of ROI height considered 'top line' for join
AI_TOPLINE_MULT = 2.0      # or 2x the median box height, whichever is larger



# --- Set detection fallbacks & caching ---
SET_ICON_FALLBACK_ENABLE = _get("SET_ICON_FALLBACK_ENABLE", True, bool)   # use set icon → code when set code OCR fails
SET_NAME_FALLBACK_ENABLE = _get("SET_NAME_FALLBACK_ENABLE", True, bool)   # use set name → code when icon fails

# Cache for Scryfall /sets (code + icon_svg_uri), persisted across runs
SCRY_SETS_CACHE_PATH = _get("SCRY_SETS_CACHE_PATH", "./cache/scry_sets.json", str)
SCRY_SETS_CACHE_TTL_S = _get("SCRY_SETS_CACHE_TTL_S", 86400, int)  # 24h
