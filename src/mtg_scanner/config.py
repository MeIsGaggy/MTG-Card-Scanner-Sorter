# config.py
import os, json

SETTINGS_PATH = os.environ.get("SETTINGS_PATH", "./settings.json")
try:
    with open(SETTINGS_PATH, "r", encoding="utf-8") as _f:
        _SET = json.load(_f) or {}
except Exception:
    _SET = {}

def _get(name, default, cast=str):
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
STEADY_MIN_FRAMES    = _get("STEADY_MIN_FRAMES", 8, int)
AUTO_CAPTURE_WAIT_S  = _get("AUTO_CAPTURE_WAIT_S", 0.18, float)
AUTOSCAN_OCR_TIMEOUT = _get("AUTOSCAN_OCR_TIMEOUT", 20.0, float)
STEADY_RELAX_FRAMES  = _get("STEADY_RELAX_FRAMES", 4, int)
STEADY_PROMOTE_S     = _get("STEADY_PROMOTE_S", 0.9, float)
DETECT_EVERY_N_FRAMES = _get("DETECT_EVERY_N_FRAMES", 1, int)
RECTANGULARITY_MIN    = _get("RECTANGULARITY_MIN", 0.76, float)
CARD_DETECT_MIN_SCORE = _get("CARD_DETECT_MIN_SCORE", 0.42, float)
CARD_EDGE_MIN_STRENGTH = _get("CARD_EDGE_MIN_STRENGTH", 0.08, float)
CARD_EDGE_MIN_CONTRAST = _get("CARD_EDGE_MIN_CONTRAST", 0.020, float)
CARD_EDGE_RING_FRAC    = _get("CARD_EDGE_RING_FRAC", 0.012, float)
DETECT_QUAD_PAD_PCT    = _get("DETECT_QUAD_PAD_PCT", 0.035, float)
CARD_CLAHE_CLIP        = _get("CARD_CLAHE_CLIP", 2.4, float)
CARD_CLAHE_TILE        = _get("CARD_CLAHE_TILE", 8, int)
CARD_BG_BLUR           = _get("CARD_BG_BLUR", 41, int)
CARD_BLACKHAT_KERNEL   = _get("CARD_BLACKHAT_KERNEL", 33, int)
CARD_CHANNEL_THRESH_C  = _get("CARD_CHANNEL_THRESH_C", 4.0, float)
CARD_MASK_MIN_COMPONENT= _get("CARD_MASK_MIN_COMPONENT", 0.004, float)
CARD_EDGE_SNAP_FRAC    = _get("CARD_EDGE_SNAP_FRAC", 0.085, float)
CARD_EDGE_CONTRAST_WEIGHT = _get("CARD_EDGE_CONTRAST_WEIGHT", 0.65, float)
CARD_EDGE_CONTRAST_STEP   = _get("CARD_EDGE_CONTRAST_STEP", 3, int)
CARD_MASK_BORDER_TRIM_PCT = _get("CARD_MASK_BORDER_TRIM_PCT", 0.002, float)
CARD_CROP_LIVE = _get("CARD_CROP_LIVE", True, bool)
QUAD_STICKY_PX = _get("QUAD_STICKY_PX", 0.9, float)
QUAD_SMOOTH_MIN_ALPHA = _get("QUAD_SMOOTH_MIN_ALPHA", 0.28, float)
QUAD_SMOOTH_MAX_ALPHA = _get("QUAD_SMOOTH_MAX_ALPHA", 0.72, float)
CARD_RING_OUTSIDE_MIN_FRAC = _get("CARD_RING_OUTSIDE_MIN_FRAC", 0.18, float)
MANUAL_CROP_ENABLED = _get("MANUAL_CROP_ENABLED", True, bool)
MANUAL_CROP_QUAD = _get("MANUAL_CROP_QUAD",
    [[0.18, 0.10], [0.82, 0.10], [0.82, 0.92], [0.18, 0.92]], list)
USE_TESS_FOR_TITLES = _get("USE_TESS_FOR_TITLES", False, bool)

# =========================
# OCR + DEBUG
# =========================
OCR_BACKEND = _get("OCR_BACKEND", "tesseract", str)
OCR_ONLY_ON_SNAPSHOT = _get("OCR_ONLY_ON_SNAPSHOT", True, bool)
DEBUG_OCR = _get("DEBUG_OCR", 1, bool)
OCR_DEBUG_BOXES = _get("OCR_DEBUG_BOXES", True, bool)
SHOW_ROI_OVERLAY= _get("SHOW_ROI_OVERLAY", True, bool)
MIN_TITLE_LETTERS = _get("MIN_TITLE_LETTERS", 4, int)
TEXT_PRESENCE_MIN = _get("TEXT_PRESENCE_MIN", 0.020, float)
TITLE_ALLOW_TESS_FALLBACK = _get("TITLE_ALLOW_TESS_FALLBACK", 0, bool)
CARD_STREAM_LIVE = _get("CARD_STREAM_LIVE", True, bool)
LIVE_CROP_WHILE_PAUSED = _get("LIVE_CROP_WHILE_PAUSED", True, bool)
OCR_NUM_BUDGET_S = _get("OCR_NUM_BUDGET_S", 20.0, float)  # cap slow number OCR passes
OCR_NUM_HARD_CAP_S = _get("OCR_NUM_HARD_CAP_S", 0.0, float)  # absolute ceiling for number OCR (0 = no clamp)
OCR_NUM_FAST_CAP_S = _get("OCR_NUM_FAST_CAP_S", 3.5, float)  # max time spent in fast number loop
OCR_NUM_ALLOW_SLOW = _get("OCR_NUM_ALLOW_SLOW", True, bool)  # allow slow number OCR fallback
OCR_NUM_MAX_ROIS = _get("OCR_NUM_MAX_ROIS", 8, int)  # limit number ROI attempts (keeps EasyOCR passes bounded)
OCR_SKIP_NUMBER = _get("OCR_SKIP_NUMBER", False, bool)  # rely on Scryfall number; skip card-number OCR
OCR_NUM_TIMING_DEBUG = _get("OCR_NUM_TIMING_DEBUG", True, bool)  # extra timing logs for number OCR
OCR_NUM_MAX_ROI_W = _get("OCR_NUM_MAX_ROI_W", 960, int)  # clamp width of number ROIs to keep OCR fast
OCR_NUM_PREFIX_STRIP = _get("OCR_NUM_PREFIX_STRIP", True, bool)  # strip leading zeros before scoring digits

# =========================
# OCR FAST-PATH TOGGLES (easy to remove)
# =========================
OCR_FAST_PATH = _get("OCR_FAST_PATH", True, bool)
OCR_LAZY_SET_HINT = _get("OCR_LAZY_SET_HINT", True, bool)
OCR_TITLE_SINGLE_PASS = _get("OCR_TITLE_SINGLE_PASS", True, bool)
OCR_TITLE_EARLY_EXIT = _get("OCR_TITLE_EARLY_EXIT", True, bool)  # stop trying more variants once a good title hit is found
OCR_TITLE_EARLY_CONF = _get("OCR_TITLE_EARLY_CONF", 88.0, float)  # confidence threshold for early exit
OCR_TITLE_MAX_W = _get("OCR_TITLE_MAX_W", 640, int)
OCR_NUMBER_FASTPATH = _get("OCR_NUMBER_FASTPATH", True, bool)
REPROCESS_FORCE_FAST_OCR = _get("REPROCESS_FORCE_FAST_OCR", True, bool)

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
MATCH_REQUIRE_ORB = _get("MATCH_REQUIRE_ORB", 1, bool)
MATCH_ORB_FAIL_THRESHOLD = _get("MATCH_ORB_FAIL_THRESHOLD", 0.0, float)

# =========================
# SCRYFALL / PRICES
# =========================
SCRYFALL_TIMEOUT = _get("SCRYFALL_TIMEOUT", 6.0, float)
FAST_SCRY_TIMEOUT = _get("FAST_SCRY_TIMEOUT", 3.0, float)
FX_URL = _get("FX_URL", "https://api.exchangerate.host/latest?base=USD&symbols=CAD", str)
FX_TTL_SEC = _get("FX_TTL_SEC", 43200, int)

# =========================
# STREAM (UI live/crop feed)
# =========================
STREAM_ENABLED       = _get("STREAM_ENABLED", True, bool)
STREAM_FPS           = _get("STREAM_FPS", REQ_FPS, int)  # match camera fps by default
STREAM_JPEG_QUALITY  = _get("STREAM_JPEG_QUALITY", 95, int)
STREAM_WARP_DOWNSCALE = _get("STREAM_WARP_DOWNSCALE", 1.0, float)  # <1.0 downsizes input before warping (stream only)
STREAM_CARD_SCALE    = _get("STREAM_CARD_SCALE", 1.0, float)       # <1.0 sends a smaller crop for UI only
SNAPSHOT_CARD_SCALE  = _get("SNAPSHOT_CARD_SCALE", 1.25, float)    # >1.0 upsamples the saved/ocr snapshot for clarity

# =========================
# MOONRAKER / KLIPPER
# =========================
MOONRAKER_URL = _get("MOONRAKER_URL", "ws://localhost:7125/websocket", str)
HTTP_POST_URL = _get("HTTP_POST_URL", "http://localhost:7125/printer/gcode/script", str)

# =========================
# STACKS / SORTER
# =========================
CARD_THICKNESS_MM = _get("CARD_THICKNESS_MM", 0.305, float)
REMEASURE_EVERY   = _get("REMEASURE_EVERY", 5, int)  # cards per re-probe (0 = disable)

# =========================
# DEBUG
# =========================
DEBUG_LEVEL    = _get("DEBUG_LEVEL", 1, int)

# =========================
# PERSISTENCE (Review history)
# =========================
HISTORY_DIR       = _get("HISTORY_DIR", "./history", str)
HISTORY_IMG_DIR   = _get("HISTORY_IMG_DIR", os.path.join(HISTORY_DIR, "imgs"), str)
HISTORY_JSON_EXT  = _get("HISTORY_JSON_EXT", ".json", str)
JPEG_QUALITY_SNAP = _get("JPEG_QUALITY_SNAP", 95, int)
JPEG_QUALITY_CMP  = _get("JPEG_QUALITY_CMP",  95, int)
JPEG_QUALITY_THUMB= _get("JPEG_QUALITY_THUMB", 95, int)
HISTORY_API_DEFAULT_LIMIT = _get("HISTORY_API_DEFAULT_LIMIT", 200, int)
HISTORY_API_MAX_LIMIT     = _get("HISTORY_API_MAX_LIMIT", 2000, int)  # 0 = unlimited

# =========================
# SERVER
# =========================
HOST = _get("HOST", "0.0.0.0", str)
PORT = _get("PORT", 5000, int)

# ==============================
# Performance / Tuning (UI-adjustable; env → settings.json → defaults)
# ==============================
PROC_DOWNSCALE_MAX_W = _get("PROC_DOWNSCALE_MAX_W", 640, int)
LIVE_OCR_ONLY_WHEN_STEADY = _get("LIVE_OCR_ONLY_WHEN_STEADY", True, bool)
LIVE_OCR_MIN_INTERVAL = _get("LIVE_OCR_MIN_INTERVAL", 0.45, float)
FOIL_EVERY_N_FRAMES = _get("FOIL_EVERY_N_FRAMES", 10, int)
OCR_ENABLE_BLACKHAT = _get("OCR_ENABLE_BLACKHAT", True, bool)
USE_OPENCL = _get("USE_OPENCL", True, bool)

MATCH_ORB_FEATURES = _get("MATCH_ORB_FEATURES", 60, int)
MATCH_FAST_ACCEPT_DELTA = _get("MATCH_FAST_ACCEPT_DELTA", 0.08, float)
MATCH_FAST_REJECT_DELTA = _get("MATCH_FAST_REJECT_DELTA", 0.12, float)
HASH_STABILITY_BITS = _get("HASH_STABILITY_BITS", 4, int)

# ========= AI / YOLO =========
AI_ENABLED        = _get("AI_ENABLED", True, bool)           # master on/off
AI_USE_FOR_CARDS  = _get("AI_USE_FOR_CARDS", True, bool)     # use AI to find the card box
AI_USE_FOR_ROIS   = _get("AI_USE_FOR_ROIS", True, bool)      # use AI to find title/set/etc ROIs
AI_ROIS_SNAPSHOT_ONLY = _get("AI_ROIS_SNAPSHOT_ONLY", True, bool)  # run AI ROI detection only on captured snapshots (not every live frame)
AI_MODEL_PATH     = _get("AI_MODEL_PATH", "/home/printpi/MTG_SCANNER_PROD/yolov5/runs/train/magic_vision2/weights/best.pt", str)
YOLOV5_DIR        = _get("YOLOV5_DIR", "/home/printpi/MTG_SCANNER_PROD/yolov5", str)
AI_IMG_SIZE       = _get("AI_IMG_SIZE", 640, int)
AI_CONF_THRES     = _get("AI_CONF_THRES", 0.25, float)
AI_IOU_THRES      = _get("AI_IOU_THRES", 0.45, float)
AI_MAX_DETS       = _get("AI_MAX_DETS", 50, int)
AI_CLASS_NAMES = _get("AI_CLASS_NAMES",
    ['1-Name', '2-Mana Value', '3-Set Symbol', '4-Card -', '5-Set Name'], list)
AI_ROI_KEYMAP = {
    '1-Name'       : 'name',
    '2-Mana Value' : 'mana_value',
    '3-Set Symbol' : 'set_symbol',
    '4-Card -'     : 'card',
    '5-Set Name'   : 'set_name',
}
AI_CLASS_TO_KEY = AI_ROI_KEYMAP
AI_BOX_THICKNESS = _get("AI_BOX_THICKNESS", 8, int)
AI_LEGEND_FONT_SCALE = _get("AI_LEGEND_FONT_SCALE", 2.5, float)
AI_LEGEND_FONT_THICKNESS = _get("AI_LEGEND_FONT_THICKNESS", 3, int)
AI_LEGEND_LINE_THICKNESS = _get("AI_LEGEND_LINE_THICKNESS", 8, int)
AI_LEGEND_PAD = _get("AI_LEGEND_PAD", 10, int)
AI_ONLY_MODE = _get("AI_ONLY_MODE", True, bool)
AI_CARD_MIN_CONF      = _get("AI_CARD_MIN_CONF",      0.15, float)
AI_SETNAME_MIN_CONF   = _get("AI_SETNAME_MIN_CONF",   0.30, float)
AI_SETNAME_MIN_Y      = _get("AI_SETNAME_MIN_Y",      0.70, float)
AI_SETNAME_MAX_Y      = _get("AI_SETNAME_MAX_Y",      0.99, float)
AI_SETNAME_MIN_WIDTH  = _get("AI_SETNAME_MIN_WIDTH",  0.16, float)
AI_SETNAME_MIN_HEIGHT = _get("AI_SETNAME_MIN_HEIGHT", 0.05, float)
AI_CARD_MIN_AREA      = _get("AI_CARD_MIN_AREA",      0.10, float)
AI_CARD_MIN_AREA_STRICT = _get("AI_CARD_MIN_AREA_STRICT", 0.12, float)
AI_CARD_MAX_TOP       = _get("AI_CARD_MAX_TOP",       0.55, float)
AI_NAME_PAD_X = _get("AI_NAME_PAD_X", 0.055, float)  # widen left/right so first/last letters are not clipped
AI_NAME_PAD_Y = _get("AI_NAME_PAD_Y", 0.010, float)  # slight vertical headroom
AI_TOPLINE_FRAC_MIN = _get("AI_TOPLINE_FRAC_MIN", 0.14, float)  # at least this fraction of ROI height considered 'top line' for join
AI_TOPLINE_MULT = _get("AI_TOPLINE_MULT", 2.0, float)
SET_ICON_FALLBACK_ENABLE = _get("SET_ICON_FALLBACK_ENABLE", False, bool)   # use set icon → code when set code OCR fails
SET_NAME_FALLBACK_ENABLE = _get("SET_NAME_FALLBACK_ENABLE", False, bool)
ICON_MATCH_BUDGET_S = _get("ICON_MATCH_BUDGET_S", 3.0, float)
ICON_MATCH_HTTP_TIMEOUT = _get("ICON_MATCH_HTTP_TIMEOUT", 3.0, float)
ICON_MATCH_MAX_SETS = _get("ICON_MATCH_MAX_SETS", 800, int)
SCRY_SETS_CACHE_PATH = _get("SCRY_SETS_CACHE_PATH", "./cache/scry_sets.json", str)
SCRY_SETS_CACHE_TTL_S = _get("SCRY_SETS_CACHE_TTL_S", 86400, int)
REQUIRE_PRINTER_ACK = _get('REQUIRE_PRINTER_ACK', False, bool)
OCR_SKIP_NUMBER = _get("OCR_SKIP_NUMBER", False, bool)
OCR_FAST_NUMBER_CONF_EXIT = _get("OCR_FAST_NUMBER_CONF_EXIT", 70.0, float)

# =========================
# EXPORTS & DISK CACHES
# =========================
EXPORT_DIR          = _get("EXPORT_DIR", "./exports", str)
SC_IMG_CACHE_DIR    = _get("SC_IMG_CACHE_DIR", "./.scry_img_cache", str)
SCRY_IMG_LRU_MAX    = _get("SCRY_IMG_LRU_MAX", 150, int)   # card image LRU (in-memory)
SCRY_FEATS_LRU_MAX  = _get("SCRY_FEATS_LRU_MAX", 300, int) # features LRU (in-memory)

# =========================
# AUTOSCAN / STEADY (helpers)
# =========================
FAST_OCR_MODE            = _get("FAST_OCR_MODE", True, bool)
# Reprocess speed knobs
REPROCESS_MAX_WIDTH      = _get("REPROCESS_MAX_WIDTH", 640, int)          # downscale snapshot width before re-OCR (0=off)
REPROCESS_SKIP_AI_ROIS   = _get("REPROCESS_SKIP_AI_ROIS", True, bool)    # skip AI ROI refresh during reprocess
REPROCESS_SKIP_FOIL_DETECT = _get("REPROCESS_SKIP_FOIL_DETECT", True, bool)  # skip foil detect during reprocess

# =========================
# ORIENTATION
# =========================
AUTO_ORIENT = _get("AUTO_ORIENT", True, bool)

# =========================
# AI ROI (full-card padding)
# =========================
AI_CARD_PAD_LEFT  = _get("AI_CARD_PAD_LEFT", 0.10, float)
AI_CARD_PAD_RIGHT = _get("AI_CARD_PAD_RIGHT", 0.02, float)
AI_CARD_PAD_Y     = _get("AI_CARD_PAD_Y", 0.10, float)
TRACK_ALPHA = _get("TRACK_ALPHA", 0.5, float)
TRACK_BETA = _get("TRACK_BETA", 0.25, float)
TRACK_DEADBAND_PX = _get("TRACK_DEADBAND_PX", 1.2, float)
LOCK_IOU_THRESH = _get("LOCK_IOU_THRESH", 0.6, float)
ACQUIRE_FRAMES = _get("ACQUIRE_FRAMES", 3, int)
DROP_MISS_FRAMES = _get("DROP_MISS_FRAMES", 6, int)
PREDICT_HOLD = _get("PREDICT_HOLD", 8, int)
STEADY_SPEED_PX = _get("STEADY_SPEED_PX", 1.05, float)
WARP_EXPAND_PCT = _get("WARP_EXPAND_PCT", 0.01, float)
WARP_CROP_PCT = _get("WARP_CROP_PCT", 0.0, float)
DETECT_MAX_FPS = _get("DETECT_MAX_FPS", 60, int)
APPEARANCE_ALIGN_ENABLE = _get("APPEARANCE_ALIGN_ENABLE", True, bool)
APPEARANCE_AB_STRENGTH = _get("APPEARANCE_AB_STRENGTH", 0.55, float)
APPEARANCE_SAT_STRENGTH = _get("APPEARANCE_SAT_STRENGTH", 0.65, float)
APPEARANCE_GAMMA_CLAMP = _get("APPEARANCE_GAMMA_CLAMP", [0.65, 1.6], list)
TONE_ALIGN_ENABLE = _get("TONE_ALIGN_ENABLE", False, bool)
TRACK_BOX_BLEND = _get("TRACK_BOX_BLEND", 0.55, float)
