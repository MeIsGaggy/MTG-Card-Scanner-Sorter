# --- bootstrap logging stub (before any imports) ---
try:
    _BOOT_LOG
except NameError:
    import time as _time
    _BOOT_LOG = []
    def _dbg(tag, msg, level=1):
        try:
            # Suppress time-debug logs with PERF tag even during bootstrap
            if str(tag).strip().upper().startswith('PERF'):
                return
            _BOOT_LOG.append((level, _time.time(), str(tag), str(msg)))
        except Exception:
            pass
            try:
                print(f"[{tag}] {msg}")
            except Exception:
                pass
    def _dbg2(tag, msg):
        _dbg(tag, msg, 2)
# --- end bootstrap stub ---

#!/usr/bin/env python3
import time as _t
_BOOT_LOG = []

import os
from .update import get_current_version, check_for_update_async


REPO = os.environ.get("MTG_SCANNER_REPO")
if REPO:
    def _on_update(res):
        _dbg("UPDATE", f"Latest {res['latest']} available at {res['html_url']}" if res["is_update"]
                      else f"Up to date (v{res['current']})")
    check_for_update_async(REPO, get_current_version(), on_result=_on_update)

try:
    import os, json, re, threading, time, signal, io, base64, shutil, glob, difflib, uuid
    from collections import OrderedDict, deque
    from contextlib import contextmanager
    from werkzeug.serving import make_server, WSGIRequestHandler
    import cv2, numpy as np, requests
    from flask import Flask, Response, jsonify, request, send_file, render_template, send_from_directory
    from rapidfuzz import process, fuzz
    import websocket
    WSGIRequestHandler.log_request = lambda *a, **k: None
    _dbg("IMPORTS", "Successfully imorted all modules")
except ImportError as e:
    _dbg("IMPORTS", "Failed to import modules: {e}")

try:
    from .config import *
except ImportError:
    from config import *
    _dbg("CONFIG", "Successfully imorted config.py")
except ImportError as e:
    _dbg("CONFIG", "Successfully imorted all modules.")

# Force Tesseract-only backend (skip Paddle/RapidOCR/EasyOCR)
OCR_BACKEND = "tesseract"

try:
    _TORCH_PIN_WARNED = False
    import torch
    from torch.utils.data import DataLoader as _TorchDataLoader
    if not torch.cuda.is_available():
        _orig_dl_init = _TorchDataLoader.__init__
        def _patched_dl_init(self, *args, **kwargs):
            if kwargs.pop("pin_memory", False):
                global _TORCH_PIN_WARNED
                if not _TORCH_PIN_WARNED:
                    _dbg("TORCH", "pin_memory disabled (no accelerator).")
                    _TORCH_PIN_WARNED = True
            _orig_dl_init(self, *args, **kwargs)
        _TorchDataLoader.__init__ = _patched_dl_init
except Exception as _torch_err:
    _dbg("TORCH", f"Pin-memory patch skipped: {_torch_err}")
try:
    RAPIDOCR_AVAILABLE = False
    RapidOCR = None
    _dbg("RAPID OCR", "RapidOCR import skipped (disabled)")
except Exception as e:
    RapidOCR, RAPIDOCR_AVAILABLE = None, False
    _dbg("RAPID OCR", f"RapidOCR disabled: {e}")
try:
    import pytesseract
    from pytesseract import Output as _TessOutput
    TESS_AVAILABLE = True
    _dbg("TESSERACT", "pytesseract available; using Tesseract backend")
except Exception as _tess_err:
    pytesseract = None
    _TessOutput = None
    TESS_AVAILABLE = False
    _dbg("TESSERACT", f"Tesseract disabled: {_tess_err}")
try:
    FUZZ_AVAILABLE = True
    _dbg("RAPID FUZZ", "RapidFuzz is available")
except Exception:
    FUZZ_AVAILABLE = False
    process = fuzz = None
    _dbg("RAPID FUZZ", "RapidFuzz not available; fuzzy matching disabled.")

# Force-disable RapidOCR (user preference: Tesseract-only)
RapidOCR = None
RAPIDOCR_AVAILABLE = False
_dbg("RAPID OCR", "RapidOCR disabled; using Tesseract")

CORES = os.cpu_count() or 4
os.environ.setdefault("OMP_NUM_THREADS", str(CORES - 2))
os.environ.setdefault("ORT_NUM_THREADS", str(CORES - 2))
os.environ.setdefault("OMP_WAIT_POLICY", "ACTIVE")
try:
    cv2.setNumThreads(CORES); cv2.setUseOptimized(True); os.nice(-5)
except Exception:
    pass

PRIMARY_PROVIDER_LABEL = "Tesseract"

DFC_FACE_TO_COMBINED = {}
DFC_FACE_TO_COMBINED_NORM = {}

_scryfall_once_lock = threading.Lock()
_scryfall_once = {} 

SETTINGS_PATH = os.environ.get("SETTINGS_PATH", "./settings.json")

def _norm_key(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower()).strip()

# -------- Performance toggles (overridable from config.py) --------
MATCH_FAST_ACCEPT_DELTA = globals().get("MATCH_FAST_ACCEPT_DELTA", 0.08)  # accept early if (hash+hist) beats MATCH_TH by this
MATCH_FAST_REJECT_DELTA = globals().get("MATCH_FAST_REJECT_DELTA", 0.12)  # reject early if (hash+hist) is under by this
MATCH_ORB_FEATURES      = globals().get("MATCH_ORB_FEATURES", 900)        # fewer ORB features = faster
HASH_STABILITY_BITS     = globals().get("HASH_STABILITY_BITS", 4)         # min changed bits to re-OCR
PROC_DOWNSCALE_MAX_W    = int(globals().get("PROC_DOWNSCALE_MAX_W", globals().get("PROC_MAX_WIDTH", 640)))  # still honor your config
SC_IMG_CACHE_DIR        = globals().get("SC_IMG_CACHE_DIR", "./.scry_img_cache") # disk cache for card images
USE_OPENCL              = globals().get("USE_OPENCL", True)

# PERF: new runtime tuning knobs (can be overridden in config.py)
LIVE_OCR_ONLY_WHEN_STEADY = globals().get("LIVE_OCR_ONLY_WHEN_STEADY", True)  # only OCR live frames when steady
LIVE_OCR_MIN_INTERVAL     = globals().get("LIVE_OCR_MIN_INTERVAL", 0.35)      # seconds between live OCR passes
FAST_OCR_MODE            = globals().get("FAST_OCR_MODE", True)
FAST_REPROCESS_SCRY_TIMEOUT = globals().get("FAST_REPROCESS_SCRY_TIMEOUT", 30.0)
REPROCESS_TIMER_DEBUG = bool(globals().get("REPROCESS_TIMER_DEBUG", True))
REPROCESS_FORCE_FAST_OCR = bool(globals().get("REPROCESS_FORCE_FAST_OCR", True))
CARD_STREAM_LIVE = bool(globals().get("CARD_STREAM_LIVE", True))
LIVE_CROP_WHILE_PAUSED = bool(globals().get("LIVE_CROP_WHILE_PAUSED", True))
LIVE_CARD_CROP = bool(globals().get("CARD_CROP_LIVE", True))
COMPARE_DEBUG = bool(globals().get("COMPARE_DEBUG", True))  # emit debug saves/logs when compare visuals look wrong
ART_MIN_FRAC = float(globals().get("ART_MIN_FRAC", 0.35))   # minimum art-crop size vs full card before falling back

if FAST_OCR_MODE:
    _dbg("PERF TUNING", "FAST_OCR_MODE enabled: faster OCR with possible accuracy tradeoffs")

def _fast_mode() -> bool:
    """Read runtime fast OCR mode from saved settings (fallback to global)."""
    try:
        cur = _settings_load() or {}
        return bool(cur.get("FAST_OCR_MODE", FAST_OCR_MODE))
    except Exception:
        return bool(FAST_OCR_MODE)

FOIL_EVERY_N_FRAMES       = globals().get("FOIL_EVERY_N_FRAMES", 30)           # compute foil detection every N frames
FOIL_SPEC_VAL_THRESHOLD   = int(globals().get("FOIL_SPEC_VAL_THRESHOLD", 212)) # brighter than this is treated as glare
FOIL_SPEC_SAT_THRESHOLD   = int(globals().get("FOIL_SPEC_SAT_THRESHOLD", 88))  # low saturation threshold for glare masking
FOIL_SPEC_DILATE          = int(globals().get("FOIL_SPEC_DILATE", 3))          # expand glare mask to catch halos
FOIL_CLAHE_CLIP           = float(globals().get("FOIL_CLAHE_CLIP", 3.8))       # contrast limit for foil band CLAHE
FOIL_CLAHE_TILE           = int(globals().get("FOIL_CLAHE_TILE", 8))           # tile size for CLAHE
FOIL_SAT_SCALE            = float(globals().get("FOIL_SAT_SCALE", 0.55))       # post-processing saturation dampening
FOIL_GAMMA                = float(globals().get("FOIL_GAMMA", 0.92))           # gamma applied to value channel to lift ink
FOIL_UNSHARP_AMOUNT       = float(globals().get("FOIL_UNSHARP_AMOUNT", 0.32))  # unsharp mask strength for edge pop

try:
    if USE_OPENCL and hasattr(cv2, "ocl"):
        cv2.ocl.setUseOpenCL(True)
except Exception:
    pass

try:
    _K1 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    _K2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
except Exception:
    _K1 = _K2 = None

try:
    _ORB = cv2.ORB_create(nfeatures=MATCH_ORB_FEATURES, fastThreshold=10, scaleFactor=1.2)
except Exception:
    _ORB = None
try:
    _BF  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
except Exception:
    _BF = None

try:
    os.makedirs(SC_IMG_CACHE_DIR, exist_ok=True)
except Exception:
    pass

log_lock = threading.Lock()
LOG_RING = deque(maxlen=1200)
LOG_SEQ = 0


def _push_log(tag: str, msg: str, level: int = 1):
    """Store a console line for the UI, with a severity level."""
    global LOG_SEQ
    try:
        t = str(tag or "")
        m = str(msg or "")
        lvl = int(level or 1) 

        U = (t + " " + m).upper()
        if ("ERROR" in U) or ("EXCEPTION" in U) or ("TRACEBACK" in U):
            lvl = max(lvl, 4)
        elif "WARN" in U:                     
            lvl = max(lvl, 3)
        elif ("SCAN_OK" in U) or ("SUCCESS" in U) or ("CONNECTED" in U):
            lvl = min(lvl, 0)              

        with log_lock:
            LOG_SEQ += 1
            LOG_RING.append({
                "id":  LOG_SEQ,
                "ts":  time.time(),
                "tag": t,
                "msg": m,
                "lvl": int(lvl),
            })
    except Exception:
        pass

def _scryfall_lookup_once(name, number_raw, set_hint, seq):
    """Ensure only one thread performs a Scryfall lookup for the same (seq,name,number,set_hint)."""
    key = (seq, name or "", number_raw or "", set_hint or "")
    now = time.time()
    with _scryfall_once_lock:
        entry = _scryfall_once.get(key)
        if entry is None:
            evt = threading.Event()
            _scryfall_once[key] = {"evt": evt, "result": None, "ts": now}
            owner = True
        else:
            evt = entry["evt"]
            owner = False

    if owner:
        try:
            res = _scryfall_lookup(name, number_raw, set_hint=set_hint or "")
            with _scryfall_once_lock:
                _scryfall_once[key]["result"] = res
            return res
        finally:
            evt.set()
            with _scryfall_once_lock:
                stale = [k for k,v in _scryfall_once.items() if now - v["ts"] > 60]
                for k in stale:
                    _scryfall_once.pop(k, None)
    else:
        evt.wait(timeout=5.0)
        with _scryfall_once_lock:
            entry = _scryfall_once.get(key)
            return entry["result"] if entry else None

# =======================================
# SECTION: Logging (colorized, unified)
# =======================================
# Provides a single logger used across the app:
#   _dbg(tag, msg, level=1)   -> normal
#   _dbg2(tag, msg)           -> verbose (level=2)
# Features:
#   - Boot buffering: logs before initialization are captured and replayed.
#   - Colorized [TAG] only (no 'INFO/OK/ERROR' words), color encodes severity.
#   - Severity inference from content and 'level' for UI and console.
import sys as _sys, time as _time

try:
    _BOOT_LOG
except NameError:
    _BOOT_LOG = []                         # (level, ts, tag, msg)
_LOG_READY = False                     # flip after full init

# ANSI palette (simple; cross-platform terminals handle these now)
_ANSI = {
    "reset": "\033[0m",
    "ok":    "\033[32m",   # green
    "info":  "\033[36m",   # cyan
    "warn":  "\033[33m",   # yellow
    "err":   "\033[31m",   # red
}
def _supports_color():
    mode = str(os.environ.get("LOG_COLOR_MODE", "auto")).lower()
    if mode == "always": return True
    if mode == "never":  return False
    try:
        return _sys.stdout.isatty()
    except Exception:
        return False

def _infer_severity(tag: str, msg: str, level: int) -> str:
    T = f"{tag or ''} {msg or ''}".upper()
    if ("ERROR" in T) or ("EXCEPTION" in T) or ("TRACEBACK" in T) or level >= 4:
        return "err"
    if ("WARN" in T) or ("DEPRECATED" in T):
        return "warn"
    if ("SCAN_OK" in T) or ("READY" in T) or ("CONNECTED" in T) or ("LOADED" in T) or ("SUCCESS" in T) or level <= 0:
        return "ok"
    return "info"

# Ring buffer for UI (populated by _push_log)
from collections import deque
log_lock = threading.Lock()
LOG_RING = deque(maxlen=1200)
LOG_SEQ = 0

def _push_log(tag: str, msg: str, level: int = 1):
    """Store a line for the UI. Also infers severity."""
    global LOG_SEQ
    try:
        sev = _infer_severity(tag, msg, level)
        with log_lock:
            LOG_SEQ += 1
            LOG_RING.append({
                "id":  LOG_SEQ,
                "ts":  _time.time(),
                "tag": str(tag or ""),
                "msg": str(msg or ""),
                "lvl": int(level or 1),
                "sev": sev,
            })
    except Exception:
        pass

def _print_console(tag: str, msg: str, level: int = 1):
    """Console print with colored [TAG] only; no extra severity words."""
    try:
        sev = _infer_severity(tag, msg, level)
        use_color = _supports_color()
        tag_str = f"[{tag}]"
        if use_color:
            color = _ANSI.get(sev, _ANSI["info"])
            out = f"{color}{tag_str}{_ANSI['reset']} {msg}"
        else:
            out = f"{tag_str} {msg}"
        print(out)
    except Exception:
        # As a last resort, avoid crashing logger
        try:
            print(f"[{tag}] {msg}")
        except Exception:
            pass

def _buffer_or_emit(tag, msg, level=1):
    if not _LOG_READY:
        _BOOT_LOG.append((level, _time.time(), str(tag), str(msg)))
    else:
        _push_log(tag, msg, level)
        # console print based on DEBUG_LEVEL
        try:
            if (level == 1 and DEBUG_LEVEL > 0) or (level == 2 and DEBUG_LEVEL > 1) or (level <= 0):
                _print_console(tag, msg, level)
        except Exception:
            pass

def _dbg(tag, msg, level=1):
    _buffer_or_emit(tag, msg, level)

def _dbg2(tag, msg):
    _buffer_or_emit(tag, msg, level=2)

def _logger_ready():
    """Replay boot logs and mark the logger ready."""
    global _LOG_READY
    _LOG_READY = True
    for (level, ts, tag, msg) in list(_BOOT_LOG):
        _push_log(tag, msg, level)
        try:
            if (level == 1 and DEBUG_LEVEL > 0) or (level == 2 and DEBUG_LEVEL > 1) or (level <= 0):
                _print_console(tag, msg, level)
        except Exception:
            pass
    _BOOT_LOG.clear()

def _save_roi_debug(label: str, img):
    """ROI debug saving disabled."""
    return

# =========================
#MARK: GLOBALS
# =========================
app = Flask(__name__, static_folder='static', template_folder='templates')
_logger_ready()

APP_VERSION = get_current_version()
UPDATE_INFO = {'current': APP_VERSION, 'latest': APP_VERSION, 'is_update': False, 'html_url': ''}

_repo = os.environ.get("MTG_SCANNER_REPO")
if _repo:
    def _on_update(res):
        global UPDATE_INFO
        UPDATE_INFO = res
        try:
            _dbg("UPDATE", f"latest {res.get('latest')} available={res.get('is_update')} {res.get('html_url','')}")
        except Exception:
            print("[UPDATE]", res)
    try:
        check_for_update_async(_repo, APP_VERSION, on_result=_on_update)
    except Exception as _e:
        print("[UPDATE] check failed:", _e)

@app.context_processor
def inject_version():

    return {"APP_VERSION": APP_VERSION}

@app.get("/")
def index():
    return render_template("index.html")

APP_DIR = os.path.dirname(os.path.abspath(__file__))

video_lock = threading.Lock()
ocr_lock = threading.Lock()
scan_lock = threading.Lock()
printer_lock = threading.Lock()
cardinfo_lock = threading.Lock()
compare_lock = threading.Lock()
preview_card = None
# compare image lock/state
_last_cmp_img = None     # BGR image for /compare.jpg
_last_cmp_stats = {}     # metric breakdown for debug (not currently shown as JSON)
_last_cmp_seq = 0        # monotonic change token for cache validation

def _update_compare_cache(img, stats=None):
    """Update compare cache and bump a change token for clients."""
    global _last_cmp_img, _last_cmp_stats, _last_cmp_seq
    with compare_lock:
        _last_cmp_img = img
        if stats is not None:
            _last_cmp_stats = stats
        _last_cmp_seq = max(_last_cmp_seq + 1, int(time.time() * 1000))
        return _last_cmp_seq

def _get_compare_seq():
    with compare_lock:
        return int(_last_cmp_seq or 0)

def _debug_save_compare(img, details=None, note=""):
    # Compare-image debug saving disabled.
    return

def _repair_cmp_visual(entry: dict):
    """
    Decode a compare visual from history, and transparently rebuild it if it's missing or too small.
    Returns (img, stats) where img is BGR or None.
    """
    cmp_img = _decode_jpg(entry.get("cmp_jpg"))
    cmp_stats = dict(entry.get("cmp_stats") or {})
    needs_rebuild = False
    try:
        if cmp_img is not None:
            h, w = cmp_img.shape[:2]
            if h >= 120 and w >= 120:
                # If art mode is enabled now but this visual was generated without art, rebuild.
                if MATCH_USE_ART and not bool(cmp_stats.get("use_art")):
                    needs_rebuild = True
                else:
                    return cmp_img, cmp_stats
            else:
                try:
                    _dbg("COMPARE DBG", f"cmp_jpg too small h={h} w={w} id={entry.get('id')}")
                    _debug_save_compare(cmp_img, entry.get("cmp_stats"), note="small_cmp")
                except Exception:
                    pass
        else:
            needs_rebuild = True
    except Exception:
        cmp_img = None
        needs_rebuild = True

    # Attempt a rebuild from stored snapshot + scry card.
    if needs_rebuild:
        try:
            snap = _decode_jpg(entry.get("snap_jpg"))
            if snap is None and entry.get("id"):
                # Late fallback: load from disk if present
                _, snap_p, thmb_p, cmp_p = _entry_paths(int(entry["id"]))
                snap = _decode_jpg(_load_image(snap_p))
            scry = entry.get("scry")
            scry_img = _fetch_scry_image(scry) if scry else None
            if snap is not None and scry_img is not None:
                _, _, details = compare_snapshot_to_scryfall(
                    snap, scry_img, return_details=True, scry_card=scry
                )
                if details:
                    cmp_img = _render_compare_visual(details)
                    cmp_stats = _cmp_stats_sanitized(details)
                    # Opportunistically persist the rebuilt visual so future loads are fast
                    try:
                        _ensure_history_dirs()
                        entry["cmp_jpg"] = _jpeg_bytes(cmp_img, JPEG_QUALITY_CMP) if cmp_img is not None else None
                        entry["cmp_stats"] = cmp_stats
                        # Only persist if we actually rebuilt something usable
                        if cmp_img is not None and cmp_img.size > 0:
                            _persist_entry_to_disk(entry)
                        if cmp_img is not None and (cmp_img.shape[0] < 400 or cmp_img.shape[1] < 800):
                            _dbg("COMPARE DBG", f"rebuilt visual still small h={cmp_img.shape[0]} w={cmp_img.shape[1]} id={entry.get('id')}")
                            _debug_save_compare(cmp_img, cmp_stats, note="rebuilt_small")
                    except Exception:
                        pass
        except Exception:
            cmp_img = None
    return cmp_img, cmp_stats

shutdown_evt = threading.Event()

# Cache for frozen stream frames (encoded JPG bytes)
STREAM_CACHE = {'live': None, 'card': None}

# --- Live frame handoff and detection throttle ---
try:
    from collections import deque as _deque_type  # may already be imported
except Exception:
    pass
_frame_q = deque(maxlen=1)           # latest-only queue for detector
try:
    DETECT_MAX_FPS  # if present in config
except NameError:
    DETECT_MAX_FPS = int(globals().get("DETECT_MAX_FPS", REQ_FPS))
_last_detect_ts = 0.0

# Protect shared tracking state between the video (preview) and detector
tracks_lock = threading.Lock()


cap, output_frame, latest_frame_raw, current_card_crop, current_card_quad, scanned_card = None, None, None, None, None, None

# ====== Stiff Card Tracker (fighter-jet style) ======
# Alpha-Beta filter per-corner + hysteresis lock/unlock + deadband + hold-on-miss
TRACK_ALPHA = float(globals().get("TRACK_ALPHA", 0.35))   # position blend (lower = stiffer)
TRACK_BETA  = float(globals().get("TRACK_BETA", 0.15))    # velocity blend
TRACK_DEADBAND_PX = float(globals().get("TRACK_DEADBAND_PX", 2.5))  # px: ignore sub-pixel jitter
LOCK_IOU_THRESH   = float(globals().get("LOCK_IOU_THRESH", 0.55))   # bbox IoU to keep lock
ACQUIRE_FRAMES    = int(globals().get("ACQUIRE_FRAMES", 6))         # stable frames to acquire
DROP_MISS_FRAMES  = int(globals().get("DROP_MISS_FRAMES", 10))      # misses before drop lock
PREDICT_HOLD      = int(globals().get("PREDICT_HOLD", 8))           # predict-only frames allowed
STEADY_SPEED_PX   = float(globals().get("STEADY_SPEED_PX", 0.6))    # avg corner speed threshold
TRACK_BOX_BLEND   = float(globals().get("TRACK_BOX_BLEND", 0.55))   # weight of previous track quad when merging with new detections

import numpy as _np

def _quad_to_bbox(q):
    q = _np.asarray(q, dtype=_np.float32)
    xs, ys = q[:,0], q[:,1]
    return _np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=_np.float32)

def _bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter + 1e-6
    return float(inter/ua)

class CardTracker:
    def __init__(self):
        self.locked = False
        self.steady = False
        self.q = None               # filtered quad (4x2)
        self.v = None               # velocity per corner (4x2)
        self.missed = 0
        self.seen = 0
        self.frames_since_update = 0
        self.bbox = None

    def reset(self):
        self.__init__()

    def predict(self):
        # simple constant-velocity prediction
        if self.q is None or self.v is None: 
            return self.q
        self.q = self.q + self.v
        self.frames_since_update += 1
        return self.q

    def update(self, meas_quad):
        """Supply measurement quad (4x2) or None if not available this frame.
        Returns (stable_quad or None, locked, steady).
        """
        if meas_quad is None:
            # no measurement: predict a few frames, then drop
            if self.locked and self.frames_since_update < PREDICT_HOLD:
                self.predict()
                self.missed += 1
            else:
                self.reset()
            self._refresh_flags()
            return (self.q, self.locked, self.steady)

        m = _np.asarray(meas_quad, dtype=_np.float32)
        if m.shape != (4,2):
            self._refresh_flags()
            return (self.q, self.locked, self.steady)

        if self.q is None:
            # initialize
            self.q = m.copy()
            self.v = _np.zeros_like(m)
            self.frames_since_update = 0
            self.seen = 1
            self.missed = 0
            self.bbox = _quad_to_bbox(self.q)
            # not locked yet; require few frames
            self._refresh_flags()
            return (self.q, self.locked, self.steady)

        # gating: require IoU overlap to accept measurement if already locked
        iou_ok = True
        if self.locked:
            iou_ok = _bbox_iou(self.bbox, _quad_to_bbox(m)) >= LOCK_IOU_THRESH

        # alpha-beta per corner with deadband
        if iou_ok:
            diff = m - self.q
            # deadband: ignore tiny changes
            mag = _np.linalg.norm(diff, axis=1, keepdims=True)
            mask = (mag >= TRACK_DEADBAND_PX).astype(_np.float32)
            corr = mask * diff
            # velocity + position updates
            self.v = (1.0 - TRACK_BETA) * self.v + TRACK_BETA * corr
            self.q = (1.0 - TRACK_ALPHA) * self.q + TRACK_ALPHA * (self.q + corr)
            self.frames_since_update = 0
            self.missed = 0
            self.seen += 1
            self.bbox = _quad_to_bbox(self.q)
        else:
            # reject: just predict; count as miss
            self.predict()
            self.missed += 1

        self._refresh_flags()
        return (self.q, self.locked, self.steady)

    def _refresh_flags(self):
        # lock after enough consecutive accepted updates
        if self.seen >= ACQUIRE_FRAMES and self.missed == 0:
            self.locked = True
        # drop lock if too many misses
        if self.missed >= DROP_MISS_FRAMES:
            self.reset()
            return
        # steady when average corner speed is tiny and recently updated
        if self.q is not None and self.v is not None and self.frames_since_update == 0:
            speed = float(_np.mean(_np.linalg.norm(self.v, axis=1)))
            self.steady = speed <= STEADY_SPEED_PX
        else:
            self.steady = False

# global tracker instance
card_tracker = CardTracker()


# --- Warp framing controls ---
# Expand the quad outward slightly before warping (de-zooms). 0.0 = none.
WARP_EXPAND_PCT = float(globals().get("WARP_EXPAND_PCT", 0.01))
# Extra protective padding so the final warp always keeps a slight border.
WARP_EDGE_PAD_PCT = float(globals().get("WARP_EDGE_PAD_PCT", 0.018))
# Additional crop after warp (keep at 0.0 to avoid auto-zoom).
WARP_CROP_PCT   = float(globals().get("WARP_CROP_PCT", 0.0))
# Edge trim applied by refine_card_quad’s warp (kept tiny so crop view is not zoomed)
REFINE_WARP_EDGE_CROP_PCT = float(globals().get("REFINE_WARP_EDGE_CROP_PCT", 0.004))
# Guardrail: reject refined quads that jump too far from the detector box
REFINE_MAX_DELTA_PX = float(globals().get("REFINE_MAX_DELTA_PX", 22.0))

# --- Quad smoothing thresholds ---
QUAD_FREEZE_THRESH = float(globals().get("QUAD_FREEZE_THRESH", 1.6))   # px; under this reuse last crop
QUAD_BLEND_THRESH  = float(globals().get("QUAD_BLEND_THRESH", 4.2))    # px; under this blend crops
QUAD_STICKY_PX     = float(globals().get("QUAD_STICKY_PX", 0.9))
QUAD_SMOOTH_MIN_ALPHA = float(globals().get("QUAD_SMOOTH_MIN_ALPHA", 0.28))
QUAD_SMOOTH_MAX_ALPHA = float(globals().get("QUAD_SMOOTH_MAX_ALPHA", 0.72))

last_scan_ts = 0.0

tracks = OrderedDict()

ws_ref = {"ws": None, "connected": False}
printer_state = {
    "awaiting": False, "job_id": None, "last_ready_line": "", "last_decision": None,
    "ws_connected": False, "last_error": None,
    "scan_captured": False, "ready_since": 0.0, "ack_received_at": 0.0,
    "messages_seen": 0, "ack_ok_jobs": set()
}
# Track deck heights and estimated cards remaining for each stack (1-indexed).
stack_state = {
    "start_height": [None, None],
    "current_height": [None, None],
    "start_cards": [None, None],
    "remaining_cards": [None, None],
    "processed": [0, 0],  # cards scanned/placed since baseline per stack
    "last_remeasure_at": None,
    "remeasure_count": 0,
    "thickness": float(globals().get("CARD_THICKNESS_MM", 0.305)),
    "remeasure_every": int(globals().get("REMEASURE_EVERY", 0) or 0),
}

# Give UI something sane on first /api/state call
scanner_state = {
    "locked": False,
    "locked_frames": 0,
    "locked_area": 0.0,
    "steady": False,
    "steady_relaxed": False,
    "steady_promoted": False,
    "foil": False,
    "foil_score": 0.0,
}
ocr_state = {"provider": PRIMARY_PROVIDER_LABEL}
cardinfo_state = {}

# Manual crop (user-set quad instead of live detection)
manual_crop_lock = threading.Lock()
def _default_manual_quad():
    try:
        return np.array(MANUAL_CROP_QUAD, dtype=np.float32).reshape(4, 2)
    except Exception:
        return np.array([[0.18, 0.10], [0.82, 0.10], [0.82, 0.92], [0.18, 0.92]], dtype=np.float32)

def _sanitize_manual_quad(q):
    try:
        arr = np.array(q, dtype=np.float32)
        if arr.shape != (4, 2):
            return _default_manual_quad()
        arr[:, 0] = np.clip(arr[:, 0], 0.0, 1.0)
        arr[:, 1] = np.clip(arr[:, 1], 0.0, 1.0)
        return order_points(arr.astype(np.float32))
    except Exception:
        return _default_manual_quad()

manual_crop_enabled = bool(MANUAL_CROP_ENABLED)
manual_crop_quad = _sanitize_manual_quad(MANUAL_CROP_QUAD)

def _set_manual_crop(quad=None, enabled=None):
    """Update manual crop quad (normalized) and toggle in-memory + persisted settings."""
    global manual_crop_enabled, manual_crop_quad
    with manual_crop_lock:
        if quad is not None:
            manual_crop_quad = _sanitize_manual_quad(quad)
        if enabled is not None:
            manual_crop_enabled = bool(enabled)
        return manual_crop_quad.copy(), manual_crop_enabled

_rapidocr_engine = None
_HTTP = requests.Session()
_HTTP.headers.update({"User-Agent": "mtg-scanner/1.1"})
_scry_cache, _fx_cache = {}, {"rate": None, "exp": 0}
_scry_img_cache = {}

def _printer_ready_for_detection():
    if not DETECT_ONLY_WHEN_READY:
        return True
    with printer_lock:
        return bool(printer_state.get("awaiting", False))

def _stack_cards_from_height(height, thickness):
    try:
        h = float(height)
        t = float(thickness)
        if t <= 0 or h < 0:
            return None
        return max(0, int(round(h / t)))
    except Exception:
        return None

def _update_stack_config(thickness=None, remeasure_every=None):
    with printer_lock:
        if thickness is not None:
            try:
                new_thick = float(thickness)
                stack_state["thickness"] = new_thick
                # Recompute remaining cards with the updated thickness if heights are known.
                for idx, h in enumerate(stack_state.get("current_height", [])):
                    if h is None:
                        continue
                    rem = _stack_cards_from_height(h, new_thick)
                    start_cards = None
                    try:
                        start_cards = stack_state.get("start_cards", [None, None])[idx]
                    except Exception:
                        start_cards = None
                    if start_cards is not None and rem is not None:
                        rem = min(rem, start_cards)
                    stack_state.get("remaining_cards", [None, None])[idx] = rem
            except Exception:
                pass
        if remeasure_every is not None:
            try:
                stack_state["remeasure_every"] = int(max(0, remeasure_every))
            except Exception:
                pass

def _update_stack_state(h1=None, h2=None, start1=None, start2=None, count1=None, count2=None, source=None):
    """Update stack height + card estimates and bump remeasure counters."""
    now = time.time()
    updated = False
    with printer_lock:
        if not isinstance(stack_state.get("processed"), list):
            stack_state["processed"] = [0, 0]
        while len(stack_state["processed"]) < 2:
            stack_state["processed"].append(0)

        thick = float(stack_state.get("thickness") or globals().get("CARD_THICKNESS_MM", 0.305) or 0.305)
        def cards_from_height(v):
            return _stack_cards_from_height(v, thick)

        def clamp_remaining(idx, rem_val):
            if rem_val is None:
                return None
            start_cards = stack_state["start_cards"][idx]
            if start_cards is not None:
                rem_val = min(rem_val, start_cards)
            return max(0, int(rem_val))

        # Heuristic tolerance to decide whether a reading matches an existing stack height.
        tol_mm = max(0.5, 2.5 * thick)

        def is_close(val, ref):
            return (val is not None) and (ref is not None) and (abs(val - ref) <= tol_mm)

        def _dedupe_identical(v1, v2):
            """
            If both heights are effectively identical but we already know the stacks
            are at different levels, keep the value only for the stack it most
            likely belongs to. This prevents a single remeasure height from
            overwriting both stacks.
            """
            orig = (v1, v2)
            if v1 is None or v2 is None:
                return v1, v2
            if abs(v1 - v2) > tol_mm * 0.25:
                return v1, v2
            refs = stack_state.get("current_height") or [None, None]
            if refs[0] is None or refs[1] is None:
                refs = stack_state.get("start_height") or [None, None]
            if refs[0] is None or refs[1] is None:
                return v1, v2
            d0 = abs(v1 - refs[0]); d1 = abs(v1 - refs[1])
            # If one stack is clearly closer, keep it there and drop the other.
            if d0 + tol_mm * 0.25 < d1:
                try: _dbg("STACK MAP", f"dedupe_identical -> keep stack0 {v1} drop stack1 (refs={refs})")
                except Exception: pass
                return v1, None
            if d1 + tol_mm * 0.25 < d0:
                try: _dbg("STACK MAP", f"dedupe_identical -> drop stack0 keep stack1 {v2} (refs={refs})")
                except Exception: pass
                return None, v2
            if orig != (v1, v2):
                try: _dbg("STACK MAP", f"dedupe_identical unchanged orig={orig} refs={refs}")
                except Exception: pass
            return v1, v2

        def _ambig_same_height(v1, v2):
            """
            If both new heights are the same AND both stacks currently read the same,
            avoid overwriting both by defaulting the update to stack 0 only.
            """
            orig = (v1, v2)
            if v1 is None or v2 is None:
                return v1, v2
            if abs(v1 - v2) > tol_mm * 0.25:
                return v1, v2
            refs = stack_state.get("current_height") or [None, None]
            if refs[0] is None or refs[1] is None:
                refs = stack_state.get("start_height") or [None, None]
            if refs[0] is None or refs[1] is None:
                return v1, v2
            if abs(refs[0] - refs[1]) <= tol_mm * 0.25:
                try: _dbg("STACK MAP", f"ambig_same_height -> keep stack0 {v1} drop stack1 (refs={refs})")
                except Exception: pass
                return v1, None
            if orig != (v1, v2):
                try: _dbg("STACK MAP", f"ambig_same_height unchanged orig={orig} refs={refs}")
                except Exception: pass
            return v1, v2

        def _maybe_swap_for_best_match(v1, v2):
            """
            If both heights are present and both stacks already have baselines,
            swap them when that better matches the known stacks. This avoids
            accidentally applying stack1's remeasure to both stacks.
            """
            if v1 is None or v2 is None:
                return v1, v2
            sh = stack_state.get("current_height") or [None, None]
            if sh[0] is None or sh[1] is None:
                sh = stack_state.get("start_height") or [None, None]
            if sh[0] is None or sh[1] is None:
                return v1, v2
            err_as_is = abs(v1 - sh[0]) + abs(v2 - sh[1])
            err_swap = abs(v2 - sh[0]) + abs(v1 - sh[1])
            if err_swap + tol_mm * 0.5 < err_as_is:
                try: _dbg("STACK MAP", f"swap h1/h2 -> ({v2},{v1}) current={sh} tol={tol_mm}")
                except Exception: pass
                return v2, v1
            return v1, v2

        def _update_remaining(idx, height_val, count_val):
            """Refresh current height + remaining with learning on start_cards."""
            if height_val is None and count_val is None:
                return
            proc = int(stack_state["processed"][idx] or 0)
            # Use the measured drop from the saved start height to re-sync how many
            # cards we've actually pulled from this stack. This keeps remeasures
            # from one stack from polluting thickness estimates for the other.
            if height_val is not None:
                try:
                    shv = stack_state["start_height"][idx]
                    if shv is not None:
                        drop = float(shv) - float(height_val)
                        est_cards_used = max(0, int(round(drop / thick))) if drop > 0 else 0
                        if abs(est_cards_used - proc) >= 1:
                            proc = est_cards_used
                            stack_state["processed"][idx] = proc
                except Exception:
                    pass
            est_height_cards = cards_from_height(height_val) if height_val is not None else None
            if height_val is not None:
                stack_state["current_height"][idx] = float(height_val)
            # Learn a better start_cards estimate: what start size would yield the current height after 'proc' cards.
            try:
                est_start = None
                if est_height_cards is not None:
                    est_start = proc + est_height_cards
                if est_start is not None:
                    prev = stack_state["start_cards"][idx]
                    if prev is None or est_start > prev:
                        stack_state["start_cards"][idx] = max(0, int(round(est_start)))
            except Exception:
                pass
            remaining_by_flow = None
            if stack_state["start_cards"][idx] is not None:
                remaining_by_flow = max(0, int(stack_state["start_cards"][idx] - proc))
            candidates = []
            for v in (count_val, est_height_cards, remaining_by_flow):
                if v is None:
                    continue
                try:
                    candidates.append(int(round(v)))
                except Exception:
                    pass
            if candidates:
                stack_state["remaining_cards"][idx] = max(0, max(candidates))
            else:
                stack_state["remaining_cards"][idx] = None

        # If we receive explicit start heights, install them.
        # If both provided and identical, map to the closest known stack to avoid overwriting both.
        if start1 is not None and start2 is not None and abs(start1 - start2) <= tol_mm:
            # prefer the stack whose baseline is closer; if only one baseline exists, use it.
            dist1 = abs(start1 - stack_state["start_height"][0]) if stack_state["start_height"][0] is not None else None
            dist2 = abs(start1 - stack_state["start_height"][1]) if stack_state["start_height"][1] is not None else None
            if dist1 is not None and dist2 is not None:
                if dist1 <= dist2:
                    start2 = None
                else:
                    start1 = None
            elif dist1 is not None:
                start2 = None
            elif dist2 is not None:
                start1 = None

        if start1 is not None:
            stack_state["start_height"][0] = float(start1)
            stack_state["start_cards"][0] = cards_from_height(start1)
            stack_state["processed"][0] = 0
            stack_state["remeasure_count"] = 0
            stack_state["current_height"][0] = float(start1)
            stack_state["remaining_cards"][0] = clamp_remaining(0, cards_from_height(start1))
            updated = True
        if start2 is not None:
            stack_state["start_height"][1] = float(start2)
            stack_state["start_cards"][1] = cards_from_height(start2)
            stack_state["processed"][1] = 0
            stack_state["remeasure_count"] = 0
            stack_state["current_height"][1] = float(start2)
            stack_state["remaining_cards"][1] = clamp_remaining(1, cards_from_height(start2))
            updated = True

        # First measurement can serve as the start height if none is set.
        if h1 is not None and stack_state["start_height"][0] is None and stack_state["start_height"][1] is None:
            stack_state["start_height"][0] = float(h1)
            stack_state["start_cards"][0] = cards_from_height(h1)
        if h2 is not None and stack_state["start_height"][1] is None and stack_state["start_height"][0] is None:
            stack_state["start_height"][1] = float(h2)
            stack_state["start_cards"][1] = cards_from_height(h2)

        # If only one start height is known, map the other reading to the missing stack,
        # and avoid overwriting the known one if the reading is a repeat.
        if stack_state["start_height"][0] is not None and stack_state["start_height"][1] is None and h1 is not None and h2 is not None:
            if is_close(h1, stack_state["start_height"][0]) and not is_close(h2, stack_state["start_height"][0]):
                # h1 matches stack1; treat h2 as stack2
                stack_state["start_height"][1] = float(h2)
                stack_state["start_cards"][1] = cards_from_height(h2)
                stack_state["current_height"][1] = float(h2)
                stack_state["remaining_cards"][1] = clamp_remaining(1, cards_from_height(h2))
                updated = True
                h1 = None  # don't overwrite stack1 current with this line
            elif is_close(h2, stack_state["start_height"][0]) and not is_close(h1, stack_state["start_height"][0]):
                # h2 matches stack1; treat h1 as stack2
                stack_state["start_height"][1] = float(h1)
                stack_state["start_cards"][1] = cards_from_height(h1)
                stack_state["current_height"][1] = float(h1)
                stack_state["remaining_cards"][1] = clamp_remaining(1, cards_from_height(h1))
                updated = True
                h2 = None  # don't overwrite stack1 current with this line
            else:
                # Default: use the value farther from stack1 as stack2
                if abs(h2 - stack_state["start_height"][0]) >= abs(h1 - stack_state["start_height"][0]):
                    stack_state["start_height"][1] = float(h2)
                    stack_state["start_cards"][1] = cards_from_height(h2)
                    stack_state["current_height"][1] = float(h2)
                    stack_state["remaining_cards"][1] = clamp_remaining(1, cards_from_height(h2))
                    h2 = None
                else:
                    stack_state["start_height"][1] = float(h1)
                    stack_state["start_cards"][1] = cards_from_height(h1)
                    stack_state["current_height"][1] = float(h1)
                    stack_state["remaining_cards"][1] = clamp_remaining(1, cards_from_height(h1))
                    h1 = None
                stack_state["processed"][1] = 0
                stack_state["remeasure_count"] = 0
                updated = True

        if stack_state["start_height"][1] is not None and stack_state["start_height"][0] is None and h1 is not None and h2 is not None:
            if is_close(h1, stack_state["start_height"][1]) and not is_close(h2, stack_state["start_height"][1]):
                stack_state["start_height"][0] = float(h2)
                stack_state["start_cards"][0] = cards_from_height(h2)
                stack_state["current_height"][0] = float(h2)
                stack_state["remaining_cards"][0] = clamp_remaining(0, cards_from_height(h2))
                updated = True
                h1 = None
            elif is_close(h2, stack_state["start_height"][1]) and not is_close(h1, stack_state["start_height"][1]):
                stack_state["start_height"][0] = float(h1)
                stack_state["start_cards"][0] = cards_from_height(h1)
                stack_state["current_height"][0] = float(h1)
                stack_state["remaining_cards"][0] = clamp_remaining(0, cards_from_height(h1))
                updated = True
                h2 = None
            else:
                if abs(h2 - stack_state["start_height"][1]) >= abs(h1 - stack_state["start_height"][1]):
                    stack_state["start_height"][0] = float(h2)
                    stack_state["start_cards"][0] = cards_from_height(h2)
                    stack_state["current_height"][0] = float(h2)
                    stack_state["remaining_cards"][0] = clamp_remaining(0, cards_from_height(h2))
                    h2 = None
                else:
                    stack_state["start_height"][0] = float(h1)
                    stack_state["start_cards"][0] = cards_from_height(h1)
                    stack_state["current_height"][0] = float(h1)
                    stack_state["remaining_cards"][0] = clamp_remaining(0, cards_from_height(h1))
                    h1 = None
                stack_state["processed"][0] = 0
                stack_state["remeasure_count"] = 0
                updated = True

        # Apply current heights + counts using whatever readings remain mapped.
        h1, h2 = _ambig_same_height(h1, h2)
        h1, h2 = _dedupe_identical(h1, h2)
        h1, h2 = _maybe_swap_for_best_match(h1, h2)
        try:
            _dbg("STACK MAP", f"apply heights h1={h1} h2={h2} start={stack_state.get('start_height')} cur={stack_state.get('current_height')} proc={stack_state.get('processed')}")
        except Exception:
            pass
        if h1 is not None:
            _update_remaining(0, h1, count1)
            updated = True
        elif count1 is not None:
            _update_remaining(0, None, count1)
            updated = True
        if h2 is not None:
            _update_remaining(1, h2, count2)
            updated = True
        elif count2 is not None:
            _update_remaining(1, None, count2)
            updated = True

        # If we have start height(s) and current heights and know how many cards were processed, refine thickness.
        try:
            proc = stack_state.get("processed", [0, 0])
            sh = stack_state.get("start_height", [None, None])
            ch = stack_state.get("current_height", [None, None])
            for idx in (0, 1):
                cards_done = proc[idx] if isinstance(proc, (list, tuple)) else 0
                if cards_done and sh[idx] is not None and ch[idx] is not None and sh[idx] > ch[idx]:
                    implied = (sh[idx] - ch[idx]) / float(cards_done)
                    if implied > 0:
                        # smooth to avoid jumps
                        thick = 0.7 * thick + 0.3 * implied
                        stack_state["thickness"] = thick
        except Exception:
            pass

        if updated:
            stack_state["last_remeasure_at"] = now
            stack_state["remeasure_count"] = int(stack_state.get("remeasure_count", 0) or 0) + 1
            stack_state["last_source"] = source or "gcode"

def _stack_state_snapshot():
    with printer_lock:
        snap = dict(stack_state)
        for k in ("start_height","current_height","start_cards","remaining_cards","processed"):
            snap[k] = list(stack_state.get(k, [None, None]))
        return snap

def _stack_cards_processed(n=1, stack_idx=0):
    try:
        n = int(max(0, n))
    except Exception:
        n = 0
    if n <= 0:
        return
    with printer_lock:
        proc = stack_state.get("processed", [0, 0])
        if not isinstance(proc, list):
            proc = [0, 0]
        while len(proc) < 2:
            proc.append(0)
        proc[stack_idx] = int(proc[stack_idx] or 0) + n
        stack_state["processed"] = proc

@app.post("/api/stacks/measure")
def api_stacks_measure():
    """Manual hook to record stack heights (mm) and optional start heights."""
    data = request.get_json(force=True, silent=True) or {}
    h1 = data.get("h1") if isinstance(data.get("h1"), (int, float)) else None
    h2 = data.get("h2") if isinstance(data.get("h2"), (int, float)) else None
    start = bool(data.get("start", False))
    if start:
        _update_stack_state(start1=h1, start2=h2, source="api")
    _update_stack_state(h1=h1, h2=h2, source="api")
    return jsonify({"ok": True, "stacks": _stack_state_snapshot()})

SCRYFALL_CARD_NAMES = []
SCRYFALL_CARD_NAMES_LOWER = set()
SCRYFALL_SET_CODES = set()

# Where to save exported decklists (on the device running this app)
EXPORT_DIR = globals().get("EXPORT_DIR", "./exports")

# --- Autoscan timing defaults (safe fallbacks if not provided via config) ---
try:
    AUTO_CAPTURE_WAIT_S = float(globals().get("AUTO_CAPTURE_WAIT_S", 0.18))
except Exception:
    AUTO_CAPTURE_WAIT_S = 0.18
try:
    AUTOSCAN_OCR_TIMEOUT = float(globals().get("AUTOSCAN_OCR_TIMEOUT", 2.0))
except Exception:
    AUTOSCAN_OCR_TIMEOUT = 2.0
try:
    ALWAYS_SCAN_OK = bool(globals().get("ALWAYS_SCAN_OK", True))
except Exception:
    ALWAYS_SCAN_OK = True
try:
    STEADY_RELAX_FRAMES = int(globals().get("STEADY_RELAX_FRAMES", 6))
except Exception:
    STEADY_RELAX_FRAMES = 6
try:
    STEADY_PROMOTE_S = float(globals().get("STEADY_PROMOTE_S", 0.9))
except Exception:
    STEADY_PROMOTE_S = 0.9
try:
    DETECT_ONLY_WHEN_READY = bool(globals().get("DETECT_ONLY_WHEN_READY", False))
except Exception:
    DETECT_ONLY_WHEN_READY = False
try:
    MATCH_REQUIRE_ORB = bool(globals().get("MATCH_REQUIRE_ORB", True))
except Exception:
    MATCH_REQUIRE_ORB = True
try:
    MATCH_ORB_FAIL_THRESHOLD = float(globals().get("MATCH_ORB_FAIL_THRESHOLD", 0.0))
except Exception:
    MATCH_ORB_FAIL_THRESHOLD = 0.0

# --- AI name ROI padding + topline join tuning (read from config if present) ---
# --- Number-band padding (left-biased) ---
_AI_CARD_PAD_LEFT   = float(globals().get("AI_CARD_PAD_LEFT", 0.0))
_AI_CARD_PAD_RIGHT  = float(globals().get("AI_CARD_PAD_RIGHT", 0.02))
_AI_CARD_PAD_Y      = float(globals().get("AI_CARD_PAD_Y", 0.1))

_AI_NAME_PAD_X = float(globals().get("AI_NAME_PAD_X", 0.045))
_AI_NAME_PAD_Y = float(globals().get("AI_NAME_PAD_Y", 0.010))
_AI_TOPLINE_FRAC_MIN = float(globals().get("AI_TOPLINE_FRAC_MIN", 0.12))
_AI_TOPLINE_MULT = float(globals().get("AI_TOPLINE_MULT", 1.8))
_AI_CARD_MIN_CONF = max(0.18, float(globals().get("AI_CARD_MIN_CONF", 0.36)))
_AI_SETNAME_MIN_CONF = max(0.12, float(globals().get("AI_SETNAME_MIN_CONF", 0.30)))
_AI_SETNAME_MIN_Y = float(globals().get("AI_SETNAME_MIN_Y", 0.70))
_AI_SETNAME_MAX_Y = float(globals().get("AI_SETNAME_MAX_Y", 0.99))
_AI_SETNAME_MIN_WIDTH = float(globals().get("AI_SETNAME_MIN_WIDTH", 0.05))
_AI_SETNAME_MIN_HEIGHT = float(globals().get("AI_SETNAME_MIN_HEIGHT", 0.02))
_AI_CARD_MIN_AREA = float(globals().get("AI_CARD_MIN_AREA", 0.32))
_AI_CARD_MIN_AREA_STRICT = max(0.12, float(globals().get("AI_CARD_MIN_AREA_STRICT", 0.20)))
_AI_CARD_MAX_TOP = float(globals().get("AI_CARD_MAX_TOP", 0.20))
_FAST_NUM_CONF_EXIT = float(globals().get("OCR_FAST_NUMBER_CONF_EXIT", 58.0))
_AI_ROI_STRICT = False  # allow legacy band/set fallbacks when AI boxes miss

os.makedirs(EXPORT_DIR, exist_ok=True)
# Bad list (in-memory ring + files on disk)
bad_cards = deque(maxlen=500)

try:
    HISTORY_API_DEFAULT_LIMIT = int(globals().get("HISTORY_API_DEFAULT_LIMIT", 200))
except Exception:
    HISTORY_API_DEFAULT_LIMIT = 200
try:
    HISTORY_API_MAX_LIMIT = int(globals().get("HISTORY_API_MAX_LIMIT", 2000))
except Exception:
    HISTORY_API_MAX_LIMIT = 2000

# ======= Review / History (persistent + in-memory) =======
scan_history = []          # newest-first
scan_id_seq = 0
current_loaded_scan_id = None
history_lock = threading.Lock()
_badlist_lock = threading.Lock()
# --- post-snapshot pause control (processing pause separate from streaming pause)
PROC_PAUSED = False
PROC_PAUSE_SEQ = None  # snapshot_seq this pause belongs to

STREAM_STATE = {
    "enabled": bool(globals().get("STREAM_ENABLED", True)),          # on/off
    "fps":     int(globals().get("STREAM_FPS", REQ_FPS)),            # 1–60, default to camera fps
    "quality": int(globals().get("STREAM_JPEG_QUALITY", 72)),        # 30–95

    'paused': False,
    'paused_reason': ''
}

STREAM_WARP_DOWNSCALE = float(globals().get("STREAM_WARP_DOWNSCALE", 1.0))
STREAM_CARD_SCALE = float(globals().get("STREAM_CARD_SCALE", 1.0))
SNAPSHOT_CARD_SCALE = float(globals().get("SNAPSHOT_CARD_SCALE", 1.25))

AUTO_ORIENT = globals().get("AUTO_ORIENT", True)

_httpd_ref = {"srv": None}

snapshot_seq = 0
_SCRY_IMG_LRU_MAX = 150
_scry_img_cache = OrderedDict()
_SCRY_FEATS_LRU_MAX = 300
_scry_img_feats = OrderedDict()

def _lru_del_excess(d, maxn):
    while len(d) > maxn:
        d.popitem(last=False)

def _scry_cache_key(card_obj, scry_bgr=None, use_art=False, tgt=None):
    # stable base: scryfall id or image url or in-memory pointer
    base = (card_obj or {}).get("id") or (((card_obj or {}).get("image_uris") or {}).get("normal"))
    if not base and scry_bgr is not None:
        base = id(scry_bgr)
    return (base, bool(use_art), tuple(tgt or ()))


def _get_cached_reference_feats(card_obj, scry_bgr, use_art=False, tgt=(640,900)):
    key = _scry_cache_key(card_obj, scry_bgr, use_art=use_art, tgt=tgt)
    feats = _lru_get(_scry_img_feats, key)
    if feats is not None:
        # ensure compatibility in case code changes
        if feats.get("use_art") == bool(use_art) and feats.get("tgt") == tuple(tgt):
            return feats

    if scry_bgr is None or scry_bgr.size == 0:
        return None

    B0 = _center_crop(scry_bgr, 0.98)
    if use_art:  # << use the parameter, not the global
        # Avoid stale AI detections when cropping Scryfall reference art.
        B0 = _crop_art(B0, allow_ai=False)

    B_hist = cv2.resize(B0, tgt, interpolation=cv2.INTER_AREA)
    B_orb  = cv2.resize(_normalize_illum(B0), tgt, interpolation=cv2.INTER_AREA)

    ah2 = _ahash64(B_orb);  dh2 = _dhash64(B_orb)

    hsvB = cv2.cvtColor(B_hist, cv2.COLOR_BGR2HSV)
    hB = cv2.calcHist([hsvB],[0],None,[32],[0,180]); sB = cv2.calcHist([hsvB],[1],None,[32],[0,256])
    cv2.normalize(hB, hB); cv2.normalize(sB, sB)

    kB = dB = None
    if _ORB is not None:
        gB = cv2.cvtColor(B_orb, cv2.COLOR_BGR2GRAY)
        kB, dB = _ORB.detectAndCompute(gB, None)

    feats = {
        "tgt": tuple(tgt),
        "use_art": bool(use_art),
        "B_hist": B_hist,
        "B_orb":  B_orb,
        "ah2": ah2, "dh2": dh2,
        "hB": hB,  "sB": sB,
        "kB": kB,  "dB": dB,
    }
    _lru_put(_scry_img_feats, key, feats)
    _lru_del_excess(_scry_img_feats, _SCRY_FEATS_LRU_MAX)
    return feats


def _orient_score(img):
    # cheap “is this upright?” score (no OCR needed)
    top = _crop_title_roi_top(img)
    alt = _crop_title_roi_alt(img)
    band = None
    try:
        bands = _ai_card_number_rois(img)
        if bands:
            band = bands[0][1]
    except Exception:
        band = None
    if band is None:
        band = _slice_frac(_card_crop_or_full(img), 0.68, 1.00)
    try:
        top_bin = _prep_roi_for_ocr(top) if top is not None else None
        alt_bin = _prep_roi_for_ocr(alt) if alt is not None else None
    except Exception:
        return 0.0
    s = 0.0
    s += 1.0 if (top_bin is not None and _has_text_quick(top_bin)) else 0.0
    s += 0.6 if (alt_bin is not None and _has_text_quick(alt_bin)) else 0.0
    # digits band — super cheap ink check
    s += 0.5 if (band is not None and _ink_present_quick(band, thresh=0.01)) else 0.0
    return s

def _auto_orient_upright(bgr):
    if not AUTO_ORIENT or bgr is None or bgr.size == 0:
        return bgr
    candidates = [bgr, np.rot90(bgr, 1), np.rot90(bgr, 2), np.rot90(bgr, 3)]
    scores = [ _orient_score(c) for c in candidates ]
    return candidates[int(np.argmax(scores))]

def _lru_get(d, k):
    v = d.get(k)
    if v is not None: d.move_to_end(k)
    return v

def _lru_put(d, k, v):
    d[k] = v; d.move_to_end(k)
    while len(d) > _SCRY_IMG_LRU_MAX:
        d.popitem(last=False)

def _ensure_history_dirs():
    try:
        os.makedirs(HISTORY_DIR, exist_ok=True)
        os.makedirs(HISTORY_IMG_DIR, exist_ok=True)
    except Exception as e:
        _dbg("HISTORY ERROR", f"Could not make directory: {e}")

HISTORY_BAD_DIR = os.path.join(HISTORY_DIR, "bad")

def _quarantine_history(meta_path, sid=None, reason="parse_error"):
    """Move a corrupt history entry (json + associated images) into HISTORY_BAD_DIR."""
    try:
        os.makedirs(HISTORY_BAD_DIR, exist_ok=True)
        if sid is None:
            try:
                base = os.path.basename(meta_path)
                sid = int(os.path.splitext(base)[0])
            except Exception:
                sid = None
        _, snap_p, thmb_p, cmp_p = _entry_paths(sid) if sid is not None else (meta_path, None, None, None)
        for p in (meta_path, snap_p, thmb_p, cmp_p):
            if p and os.path.exists(p):
                dst = os.path.join(HISTORY_BAD_DIR, f"{os.path.basename(p)}.{reason}")
                try:
                    shutil.move(p, dst)
                except Exception:
                    pass
    except Exception as e:
        _dbg("HISTORY ERROR", f"quarantine failed {meta_path}: {e}")

def _jpeg_bytes(bgr, q=80):
    if bgr is None or getattr(bgr, "size", 0) == 0:
        return b""
    ok, enc = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, q])
    return enc.tobytes() if ok else b""

def _decode_jpg(bts):
    if not bts:
        return None
    data = np.frombuffer(bts, np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

def _make_thumb(bgr, w=140):
    if bgr is None or getattr(bgr, "size", 0) == 0:
        return b""
    h, ww = bgr.shape[:2]
    if ww <= 0:
        return _jpeg_bytes(bgr, JPEG_QUALITY_THUMB)
    hh = int(w * (h / float(ww)))
    return _jpeg_bytes(cv2.resize(bgr, (w, hh), interpolation=cv2.INTER_AREA), JPEG_QUALITY_THUMB)

def _entry_paths(sid: int):
    base = os.path.join(HISTORY_DIR, f"{sid}")
    meta = base + HISTORY_JSON_EXT
    snap = os.path.join(HISTORY_IMG_DIR, f"{sid}_snap.jpg")
    thmb = os.path.join(HISTORY_IMG_DIR, f"{sid}_thumb.jpg")
    cmp  = os.path.join(HISTORY_IMG_DIR, f"{sid}_cmp.jpg")
    return meta, snap, thmb, cmp

def _save_image(path, bts):
    if not bts:
        return
    try:
        with open(path, "wb") as f:
            f.write(bts)
    except Exception as e:
        _dbg("HISTORY ERROR", f"save image failed {path}: {e}")

def _load_image(path):
    try:
        with open(path, "rb") as f:
            return f.read()
    except Exception:
        return b""

def _persist_entry_to_disk(entry: dict):
    """Write per-scan JSON + images. Excludes raw bytes from JSON."""
    meta_path, snap_path, thmb_path, cmp_path = _entry_paths(entry["id"])
    meta = {k: v for k, v in entry.items() if k not in ("snap_jpg", "thumb", "cmp_jpg")}
    try:
        with open(meta_path, "w") as f:
            json.dump(meta, f)
    except Exception as e:
        _dbg("HISTORY ERROR", f"save meta failed {meta_path}: {e}")
    _save_image(snap_path, entry.get("snap_jpg", b""))
    _save_image(thmb_path, entry.get("thumb", b""))
    _save_image(cmp_path,  entry.get("cmp_jpg", b""))

def _delete_entry_from_disk(sid: int):
    meta, snap, thmb, cmp = _entry_paths(sid)
    for p in (meta, snap, thmb, cmp):
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception as e:
            _dbg("HISTORY ERROR", f"delete failed {p}: {e}")

def _load_history_from_disk():
    """Load all *.json entries from HISTORY_DIR, newest-first."""
    items = []
    for meta in glob.glob(os.path.join(HISTORY_DIR, f"*{HISTORY_JSON_EXT}")):
        try:
            with open(meta, "r") as f:
                e = json.load(f)
            sid = int(e.get("id"))
            _, snap_p, thmb_p, cmp_p = _entry_paths(sid)
            e["snap_jpg"] = _load_image(snap_p)
            e["thumb"]    = _load_image(thmb_p)
            e["cmp_jpg"]  = _load_image(cmp_p)
            items.append(e)
        except Exception as e:
            _dbg("HISTORY ERROR", f"load meta failed {meta}: {e}")
            try:
                _quarantine_history(meta, reason="badjson")
            except Exception:
                pass
    items.sort(key=lambda x: x.get("ts", 0), reverse=True)
    return items
def _canon_from_entry(e: dict):
    """Return canonical (name, set_code, collector_number) for an entry."""
    scry = e.get("scry") or {}
    name = (scry.get("name") or e.get("name") or "").strip()
    set_code = (scry.get("set") or e.get("set_hint") or "").strip().lower()
    cn_raw = (scry.get("collector_number") or e.get("number_raw") or e.get("number") or "").strip()
    cn = _parse_collector_for_display(cn_raw)  # "123" or "123/456" → "123"
    return name, (set_code or ""), (cn or "")

def _aggregate_history(only_status: str = "pass"):
    """
    Build counts keyed by (name, set, cn).
    only_status: 'pass' | 'pass_no_review' | 'review' | 'fail' | 'all'
    """
    counts = {}
    for e in list(scan_history):
        st = (e.get("status") or "fail").lower()
        if only_status == "pass":
            if st != "pass":
                continue
            if bool(e.get("flagged")):
                continue
        elif only_status == "pass_no_review":
            if st != "pass":
                continue
            if bool(e.get("flagged")) or (st == "review"):
                continue
        elif only_status == "review":
            if st != "review":
                continue
        elif only_status == "fail":
            if st != "fail":
                continue
        nm, set_code, cn = _canon_from_entry(e)
        if not nm:
            continue
        key = (nm, set_code, cn)
        counts[key] = counts.get(key, 0) + 1
    return counts

def _deckline(name: str, count: int, set_code: str, cn: str, fmt: str):
    """Archidekt/Moxfield both accept: 'N Name (SET) CN' or just 'N Name'."""
    set_segment = f" ({set_code.upper()})" if set_code else ""
    cn_segment  = f" {cn}" if cn else ""
    return f"{count} {name}{set_segment}{cn_segment}".strip()


def _build_review_reasons(ocr, scry, match_score, match_ok, snapshot_ok=True, scry_img_ok=True):
    reasons = []
    match_mode = (scry or {}).get("_match_mode")
    if match_mode in ("cn_set_only", "cn_only"):
        reasons.append("Card name not reliably read — identified by set + collector number.")
    try:
        ms = float(match_score)
    except Exception:
        ms = 0.0
    if match_ok is False:
        try:
            th = float(MATCH_TH)
        except Exception:
            th = 0.0
        reasons.append(f"Scryfall candidate did not visually match (score {ms:.2f} below threshold {th:.2f}).")
    if not scry:
        reasons.append("No Scryfall result; manual selection required.")
    try:
        nm_ok = bool((ocr or {}).get("name") or (ocr or {}).get("name_raw"))
        set_ok = bool((ocr or {}).get("set_hint"))
        cn_raw = (ocr or {}).get("number_raw") or (ocr or {}).get("number") or ""
        try:
            cn_ok = bool(_parse_collector_for_display(cn_raw or ""))
        except Exception:
            cn_ok = bool((cn_raw or "").strip())
        if not (nm_ok and set_ok and cn_ok):
            missing = []
            if not nm_ok: missing.append("name")
            if not set_ok: missing.append("set")
            if not cn_ok: missing.append("card number")
            reasons.append("Scryfall was queried without full data: missing " + ", ".join(missing) + ".")
    except Exception:
        pass
    if not snapshot_ok:
        reasons.append("Snapshot missing; visual comparison unavailable.")
    if (scry or {}) and not scry_img_ok:
        reasons.append("Scryfall image missing; visual comparison unavailable.")
    return reasons


def _derive_review_outcome(
    ocr,
    scry,
    match_score,
    match_ok,
    snapshot_ok=True,
    scry_img_ok=True,
    match_mode=None,
    allow_auto_name_fix=True,
):
    """
    Normalize pass/review/fail along with flags and reasons so live scan and reprocess stay consistent.
    """
    match_mode = match_mode or (scry or {}).get("_match_mode")
    ocr_out = dict(ocr or {})
    filled_from_scry = []
    # If Scryfall has data we missed, fill it in so entries don't save as "(unnamed)".
    if scry:
        if not ocr_out.get("name") and scry.get("name"):
            ocr_out["name"] = scry.get("name")
            filled_from_scry.append("name")
        if not ocr_out.get("set_hint") and scry.get("set"):
            ocr_out["set_hint"] = scry.get("set")
            filled_from_scry.append("set")
        if not ocr_out.get("number_raw") and scry.get("collector_number"):
            ocr_out["number_raw"] = str(scry.get("collector_number") or "")
            ocr_out["number"] = _parse_collector_for_display(ocr_out["number_raw"]) or ocr_out.get("number","")
            filled_from_scry.append("number")

    review_reasons = _build_review_reasons(
        ocr_out, scry, match_score, match_ok, snapshot_ok=snapshot_ok, scry_img_ok=scry_img_ok
    )

    auto_name_fix = False
    if (
        allow_auto_name_fix
        and (match_mode == "cn_set_only")
        and bool(match_ok)
        and scry
        and scry.get("name")
    ):
        ocr_out["name"] = scry.get("name") or ocr_out.get("name") or ""
        auto_name_fix = True
        review_reasons.append("Name auto-corrected from OCR to Scryfall (cn_set_only + visual match passed).")

    try:
        has_name = bool((ocr_out or {}).get("name") or (ocr_out or {}).get("name_raw"))
        has_set = bool((ocr_out or {}).get("set_hint"))
        cn_raw = (ocr_out or {}).get("number_raw") or (ocr_out or {}).get("number") or ""
        try:
            has_cn = bool(_parse_collector_for_display(cn_raw or ""))
        except Exception:
            has_cn = bool((cn_raw or "").strip())
    except Exception:
        has_name = has_set = has_cn = False

    missing_inputs = not (has_name and has_set and has_cn)
    missing_images = not snapshot_ok or not scry_img_ok
    no_scry = not bool(scry)
    visual_fail = match_ok is False
    needs_review = missing_inputs or missing_images or no_scry or auto_name_fix

    status = "fail" if visual_fail else ("review" if (needs_review or match_ok is None) else "pass")
    flagged = status != "pass"
    review_level = (
        "critical"
        if visual_fail
        else ("warning" if (missing_inputs or missing_images or no_scry or auto_name_fix) else "info")
    )

    inputs_present = {
        "name": has_name,
        "set": has_set,
        "number": has_cn,
    }

    return {
        "ocr": ocr_out,
        "status": status,
        "flagged": flagged,
        "review_level": review_level,
        "review_reasons": review_reasons,
        "auto_name_fix": bool(auto_name_fix),
        "inputs_present": inputs_present,
    }


def save_scan_entry(snap_img, ocr, scry, cmp_details, match_score, match_ok, review_decision=None):
    """Create a review record and persist it."""
    global scan_id_seq, scan_history
    sid = int(time.time()*1000) + (scan_id_seq % 1000)
    scan_id_seq += 1

    cmp_jpg = None
    if cmp_details:
        vis = _render_compare_visual(cmp_details)
        if vis is not None:
            cmp_jpg = _jpeg_bytes(vis, JPEG_QUALITY_CMP)

    # --- Build review reasons & normalize pass/review/fail in one place ---
    match_mode = (scry or {}).get("_match_mode")
    snapshot_ok = snap_img is not None
    scry_img_ok = bool(cmp_details)
    decision = review_decision or _derive_review_outcome(
        ocr,
        scry,
        match_score,
        match_ok,
        snapshot_ok=snapshot_ok,
        scry_img_ok=scry_img_ok,
        match_mode=match_mode,
        allow_auto_name_fix=True,
    )
    ocr = decision["ocr"]
    review_reasons = decision["review_reasons"]
    auto_name_fix = decision["auto_name_fix"]
    flagged = decision["flagged"]
    status = decision["status"]

    try:
        ms = float(match_score)
    except Exception:
        ms = 0.0
    # Strip out any non-JSON-serializable values (e.g. numpy arrays) from cmp_details
    def _cmp_stats_sanitized(src):
        out = {}
        for k, v in (src or {}).items():
            if k == "orb_dbg":
                continue
            # skip numpy arrays / images
            if hasattr(v, "shape"):
                continue
            out[k] = v
        return out
    entry = {
        "id": sid,
        "ts": time.time(),
        "name": (ocr or {}).get("name") or (ocr or {}).get("name_raw") or "",
        "name_conf": float((ocr or {}).get("name_conf", 0.0)),
        "number": (ocr or {}).get("number") or "",
        "number_raw": (ocr or {}).get("number_raw") or "",
        "set_hint": (ocr or {}).get("set_hint") or "",
        "foil": bool((ocr or {}).get("foil", False)),
        "inputs_present": decision["inputs_present"],
        "match_score": round(float(ms), 3),
        "match_ok": bool(match_ok) if match_ok is not None else None,
        "flagged": bool(flagged),
        "status": status,
        "scry": scry,
        "cmp_stats": _cmp_stats_sanitized(cmp_details),
        "snap_jpg": _jpeg_bytes(snap_img, JPEG_QUALITY_SNAP),
        "thumb": _make_thumb(snap_img, 120),
        "cmp_jpg": cmp_jpg,

        # NEW: rich review context
        "review_reasons": review_reasons,
        "auto_name_fix": bool(auto_name_fix),
        "review_level": decision.get("review_level", "info"),
        "scry_match_mode": match_mode or "none",
    }

    with history_lock:
        scan_history.insert(0, entry)
        _persist_entry_to_disk(entry)
    return entry

# =========================
#MARK: UTILITIES
# =========================
def _get_rapid_engine():
    """Stubbed out RapidOCR engine (disabled)."""
    return None

def order_points(pts):
    rect = np.zeros((4,2), dtype='float32')
    s = pts.sum(axis=1)
    rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
    d = np.diff(pts, axis=1)
    rect[1], rect[3] = pts[np.argmin(d)], pts[np.argmax(d)]
    return rect


def _expand_quad(quad, pct, img_w=None, img_h=None):
    try:
        import numpy as _np
        q = _np.asarray(quad, dtype=_np.float32)
        if q.shape != (4,2): 
            return quad
        if pct <= 0.0:
            return q
        cx, cy = _np.mean(q, axis=0)
        v = q - _np.array([cx, cy], dtype=_np.float32)
        q2 = _np.array([cx, cy], dtype=_np.float32) + (1.0 + pct) * v
        # clamp to image bounds if provided
        if img_w is not None and img_h is not None:
            q2[:,0] = _np.clip(q2[:,0], 0, img_w-1)
            q2[:,1] = _np.clip(q2[:,1], 0, img_h-1)
        return q2.astype('float32')
    except Exception:
        return quad

def warp_card(frame, quad):
    import numpy as np
    h, w = frame.shape[:2]
    # expand a touch to avoid zoomed look
    pad_pct = max(0.0, float(WARP_EXPAND_PCT) + float(WARP_EDGE_PAD_PCT))
    src_q = _expand_quad(np.asarray(quad, dtype=np.float32), pad_pct, w, h)
    dst = np.array([[0,0],[CARD_W-1,0],[CARD_W-1,CARD_H-1],[0,CARD_H-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_q.astype('float32'), dst)
    warped = cv2.warpPerspective(frame, M, (CARD_W, CARD_H))
    # optional crop (keep at 0.0 to avoid auto-zoom)
    if WARP_CROP_PCT > 0:
        px = int(CARD_W * WARP_CROP_PCT); py = int(CARD_H * WARP_CROP_PCT)
        if px or py:
            warped = warped[py:CARD_H-py, px:CARD_W-px]
            warped = cv2.resize(warped, (CARD_W, CARD_H), interpolation=cv2.INTER_AREA)
    warped = _auto_orient_upright(warped)
    return warped

def _stabilize_quad(new_quad, prev_quad=None):
    """
    Force a consistent point order and blend with the previous quad to suppress jitter/twist.
    """
    try:
        q = order_points(np.asarray(new_quad, dtype=np.float32))
    except Exception:
        return new_quad
    if prev_quad is None:
        return q
    try:
        prev = order_points(np.asarray(prev_quad, dtype=np.float32))
        delta = float(np.mean(np.linalg.norm(q - prev, axis=1)))
        sticky_px = max(0.0, float(QUAD_STICKY_PX))
        if delta <= sticky_px:
            return prev.copy()
        alpha_min = max(0.0, min(1.0, float(QUAD_SMOOTH_MIN_ALPHA)))
        alpha_max = max(alpha_min, min(1.0, float(QUAD_SMOOTH_MAX_ALPHA)))
        spread = max(1.0, float(QUAD_BLEND_THRESH) - sticky_px)
        t = min(1.0, max(0.0, (delta - sticky_px) / spread))
        alpha = alpha_min + (alpha_max - alpha_min) * t
        return (alpha * q + (1.0 - alpha) * prev).astype(np.float32)
    except Exception:
        return q

def _quad_delta(a, b):
    try:
        pa = np.asarray(a, dtype=np.float32)
        pb = np.asarray(b, dtype=np.float32)
        if pa.shape != (4,2) or pb.shape != (4,2):
            return None
        return float(np.max(np.linalg.norm(pa - pb, axis=1)))
    except Exception:
        return None

def _quad_area(q):
    try:
        pts = np.asarray(q, dtype=np.float32)
        if pts.shape != (4, 2):
            return 0.0
        x = pts[:, 0]; y = pts[:, 1]
        return 0.5 * float(abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))
    except Exception:
        return 0.0

def _manual_quad_px(frame_shape):
    try:
        h, w = frame_shape[:2]
    except Exception:
        return None
    with manual_crop_lock:
        q = np.array(manual_crop_quad, dtype=np.float32)
    if q.shape != (4, 2):
        return None
    qp = q.copy()
    qp[:,0] = np.clip(qp[:,0], 0.0, 1.0) * max(w - 1, 1)
    qp[:,1] = np.clip(qp[:,1], 0.0, 1.0) * max(h - 1, 1)
    try:
        return order_points(qp.astype(np.float32))
    except Exception:
        return qp

def refine_card_quad(frame, init_quad=None, prev_quad=None):
    try:
        H, W = frame.shape[:2]
        if init_quad is None:
            return None, None
        q = np.array(init_quad, dtype=np.float32)
        x0 = max(0, int(np.floor(q[:,0].min())))
        y0 = max(0, int(np.floor(q[:,1].min())))
        x1 = min(W, int(np.ceil(q[:,0].max())))
        y1 = min(H, int(np.ceil(q[:,1].max())))
        pad_frac = float(globals().get("REFINE_PAD_RATIO", 0.09))
        pad = int(pad_frac * max(x1-x0, y1-y0))
        x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
        x1 = min(W, x1 + pad); y1 = min(H, y1 + pad)
        roi = frame[y0:y1, x0:x1]
        if roi.size == 0:
            return None, None

        g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        g = cv2.GaussianBlur(g, (5,5), 0)
        v = float(np.median(g))
        lo = int(max(0, 0.66*v)); hi = int(min(255, 1.33*v))
        e = cv2.Canny(g, lo, hi)
        e = cv2.morphologyEx(e, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=2)
        cnts,_ = cv2.findContours(e, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best = None; best_score = -1.0
        area_img = float(roi.shape[0] * roi.shape[1])
        min_frac = float(globals().get("REFINE_MIN_CONTOUR_FRAC", 0.02))
        ar_target = CARD_H / max(CARD_W, 1e-6)
        ar_tol = float(globals().get("REFINE_ASPECT_TOL", 0.35))
        for c in cnts:
            a = cv2.contourArea(c)
            if a < min_frac * area_img:
                continue
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02*peri, True)
            if len(approx) < 4:
                approx = cv2.convexHull(c)
            if len(approx) < 4:
                continue
            rect = cv2.minAreaRect(approx)
            box = order_points(cv2.boxPoints(rect).astype(np.float32))
            # back to full-frame coords
            box[:,0] += x0; box[:,1] += y0

            # aspect check (H/W ~ 1.40)
            w = (np.linalg.norm(box[1]-box[0]) + np.linalg.norm(box[2]-box[3])) * 0.5
            h = (np.linalg.norm(box[3]-box[0]) + np.linalg.norm(box[2]-box[1])) * 0.5
            ar = h / max(w, 1e-6)
            if abs(ar - ar_target) > ar_tol:
                continue
            score = a / area_img
            if score > best_score:
                best, best_score = box, score

        if best is None:
            return None, None

        # temporal EMA if previous exists
        if prev_quad is not None:
            prev = np.array(prev_quad, dtype=np.float32)
            if prev.shape == best.shape:
                # adapt smoothing: tiny deltas => dampen jitter, big deltas => follow quickly
                delta = float(np.mean(np.linalg.norm(best - prev, axis=1)))
                alpha = 0.25 + min(0.55, delta / 12.0)  # clamp to 0.25..0.80
                alpha = min(max(alpha, 0.25), 0.80)
                best = (alpha*best + (1.0-alpha)*prev).astype(np.float32)

        # Reject jumps that drift too far from the detector quad; fall back to tracker quad instead.
        try:
            max_delta = max(5.0, float(REFINE_MAX_DELTA_PX))
        except Exception:
            max_delta = 22.0
        d_init = _quad_delta(best, q)
        if d_init is not None and d_init > max_delta:
            return None, None

        # produce warp as well for immediate OCR
        M = cv2.getPerspectiveTransform(best.astype('float32'),
                                        np.array([[0,0],[CARD_W-1,0],[CARD_W-1,CARD_H-1],[0,CARD_H-1]], dtype='float32'))
        warp = cv2.warpPerspective(frame, M, (CARD_W, CARD_H))
        px = int(CARD_W * REFINE_WARP_EDGE_CROP_PCT)
        py = int(CARD_H * REFINE_WARP_EDGE_CROP_PCT)
        if px or py:
            warp = warp[py:CARD_H-py, px:CARD_W-px]
            warp = cv2.resize(warp, (CARD_W, CARD_H), interpolation=cv2.INTER_AREA)
        warp = _auto_orient_upright(warp)
        return best, warp
    except Exception:
        return None, None


def _roi_rel_pad(img, roi, pad_x=0.02, pad_y=0.00):
    h, w = img.shape[:2]
    x0,y0,x1,y1 = roi
    x0 = max(0.0, x0 - pad_x); x1 = min(1.0, x1 + pad_x)
    y0 = max(0.0, y0 - pad_y); y1 = min(1.0, y1 + pad_y)
    X0, Y0, X1, Y1 = int(x0*w), int(y0*h), int(x1*w), int(y1*h)
    if X1<=X0 or Y1<=Y0: return np.zeros((1,1,3), dtype=np.uint8)
    return img[Y0:Y1, X0:X1]

def _ai_name_roi(img):
    """Return the name ROI, refreshing AI detections if needed."""
    roi = _ai_crop_exact(img, "name") if _ai_enabled() else None
    if _ai_enabled() and (roi is None or getattr(roi, "size", 0) == 0):
        try:
            _update_ai_rois(img)
            roi = _ai_crop_exact(img, "name")
        except Exception:
            roi = None
    if roi is not None and getattr(roi, "size", 0) > 0:
        return roi
    # Fallback: approximate the name band from the detected card crop (or full image)
    card = _ai_crop(img, "card") if _ai_enabled() else None
    base = card if card is not None and getattr(card, "size", 0) > 0 else img
    try:
        return base[:max(4, int(0.20 * base.shape[0])), :]
    except Exception:
        return None


def _card_crop_or_full(img):
    """Prefer the AI card crop; otherwise use the full image."""
    if img is None or getattr(img, "size", 0) == 0:
        return None
    if _ai_enabled():
        nb = _ai_get_norm_box("card")
        if nb is not None:
            area = max(0.0, (nb[2] - nb[0]) * (nb[3] - nb[1]))
            if area >= _AI_CARD_MIN_AREA_STRICT:
                roi = _ai_crop(img, "card")
                if roi is not None and getattr(roi, "size", 0) > 0:
                    return roi
    # fallback to full image
    return img


def _slice_frac(img, y0_frac, y1_frac, x0_frac=0.0, x1_frac=1.0):
    """Return a fractional slice of an image; clamps to bounds."""
    if img is None or getattr(img, "size", 0) == 0:
        return None
    h, w = img.shape[:2]
    y0 = int(max(0.0, min(1.0, y0_frac)) * h)
    y1 = int(max(0.0, min(1.0, y1_frac)) * h)
    x0 = int(max(0.0, min(1.0, x0_frac)) * w)
    x1 = int(max(0.0, min(1.0, x1_frac)) * w)
    if x1 <= x0 or y1 <= y0:
        return None
    return img[y0:y1, x0:x1]

# === AI-aware ROI helpers (override) ===
def _crop_title_roi_top(img):
    """Prefer AI 'name' box; fallback to top band of the card crop."""
    roi = _ai_name_roi(img)
    if roi is not None and getattr(roi, "size", 0) > 0:
        return roi
    band = _slice_frac(_card_crop_or_full(img), 0.02, 0.18)
    return band if band is not None else np.zeros((1, 1, 3), dtype=np.uint8)

def _crop_title_roi_alt(img):
    """Alternate title band fallback from the mid-upper portion of the card."""
    roi = _ai_name_roi(img)
    if roi is not None and getattr(roi, "size", 0) > 0:
        return roi
    band = _slice_frac(_card_crop_or_full(img), 0.50, 0.72)
    return band if band is not None else np.zeros((1, 1, 3), dtype=np.uint8)


def _roi_rel(img, roi):
    h,w = img.shape[:2]
    x0r,y0r,x1r,y1r = roi
    x0,y0,x1,y1 = int(x0r*w), int(y0r*h), int(x1r*w), int(y1r*h)
    if x1<=x0 or y1<=y0:
        return np.zeros((1,1,3), dtype=np.uint8)
    return img[y0:y1, x0:x1]

# ---- Foil detection helpers
def _colorfulness(bgr: np.ndarray) -> float:
    if bgr is None or bgr.size == 0:
        return 0.0
    b, g, r = cv2.split(bgr.astype(np.float32))
    rg = np.abs(r - g)
    yb = np.abs(0.5 * (r + g) - b)
    std_rg, std_yb = rg.std(), yb.std()
    mean_rg, mean_yb = rg.mean(), yb.mean()
    return float(np.sqrt(std_rg**2 + std_yb**2) + 0.3 * np.sqrt(mean_rg**2 + mean_yb**2))

def _foil_feats(roi_bgr: np.ndarray):
    if roi_bgr is None or roi_bgr.size == 0:
        return 0.0, 0, 0.0, 0.0
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    spec = (s < 40) & (v > 210)
    spec_ratio = float(spec.mean())
    sat_mask = (s > 95) & (v > 110)
    sat_count = int(sat_mask.sum())
    hue_div = 0
    if sat_count > 0:
        hh = h[sat_mask]
        hist, _ = np.histogram(hh, bins=36, range=(0, 180))
        thresh = max(1, int(0.007 * sat_count))
        hue_div = int((hist >= thresh).sum())
    sat_frac = float(sat_count) / float(h.size)
    cf = _colorfulness(roi_bgr)
    return spec_ratio, hue_div, sat_frac, cf

def _pad_roi_for_ocr(roi_bgr, pad_px: int = 4):
    """
    Add a small constant border around an ROI to give the OCR engine
    some background context, without changing the saved debug crop.
    """
    try:
        import cv2 as _cv2, numpy as _np
        if roi_bgr is None or getattr(roi_bgr, "size", 0) == 0:
            return roi_bgr
        p = int(max(1, pad_px))
        h, w = roi_bgr.shape[:2]
        if h <= 2 or w <= 2:
            return roi_bgr
        if getattr(roi_bgr, "ndim", 0) == 2:
            val = int(_np.median(roi_bgr))
        else:
            flat = roi_bgr.reshape(-1, roi_bgr.shape[2])
            med = _np.median(flat, axis=0)
            val = [int(med[i]) for i in range(len(med))]
        return _cv2.copyMakeBorder(roi_bgr, p, p, p, p, _cv2.BORDER_CONSTANT, value=val)
    except Exception:
        return roi_bgr

def _detect_foil_card(card_bgr: np.ndarray):
    rois = []
    try:
        name_roi = _ai_name_roi(card_bgr)
        if name_roi is not None and getattr(name_roi, "size", 0) > 0:
            rois.append(name_roi)
    except Exception:
        pass
    card_crop = _card_crop_or_full(card_bgr)
    if card_crop is not None and getattr(card_crop, "size", 0) > 0:
        alt_band = _slice_frac(card_crop, 0.50, 0.72)
        if alt_band is not None and getattr(alt_band, "size", 0) > 0:
            rois.append(alt_band)
    try:
        for _, r in _ai_card_number_rois(card_bgr):
            if r is not None and getattr(r, "size", 0) > 0:
                rois.append(r)
                break
    except Exception:
        pass
    if not rois and card_crop is not None:
        band = _slice_frac(card_crop, 0.68, 1.0)
        if band is not None and getattr(band, "size", 0) > 0:
            rois.append(band)
    if not rois:
        rois.append(card_bgr)
    specs, hues, sats, cfs = [], [], [], []
    for r in rois:
        spec, hdiv, sfrac, cf = _foil_feats(r)
        specs.append(spec); hues.append(hdiv); sats.append(sfrac); cfs.append(cf)
    max_spec = max(specs or [0.0])
    max_hdiv = max(hues or [0])
    mean_cf  = float(np.mean(cfs)) if cfs else 0.0
    max_sat  = max(sats or [0.0])
    score = 0.0
    score += 1.05 if max_spec > 0.012 else 0.0
    score += 0.85 if max_hdiv >= 8 else 0.0
    score += 0.55 if mean_cf > 22.0 else 0.0
    score += 0.35 if max_sat > 0.20 else 0.0
    return (score >= FOIL_MIN_SCORE), float(score)

def _prep_roi_for_rapid_bgr(roi_bgr):
    if roi_bgr is None or roi_bgr.size == 0:
        return np.zeros((1,1,3), dtype=np.uint8)
    bgr = cv2.resize(roi_bgr, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    bgr = cv2.bilateralFilter(bgr, 7, 45, 45)
    blur = cv2.GaussianBlur(bgr, (0,0), 2.0)
    bgr = cv2.addWeighted(bgr, 1.6, blur, -0.6, 0)
    return bgr

def _prep_roi_for_ocr(roi_bgr):
    g = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    g = cv2.bilateralFilter(g, 7, 35, 35)
    blurred = cv2.GaussianBlur(g, (0,0), 3)
    sharpened = cv2.addWeighted(g, 1.5, blurred, -0.5, 0)
    return cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 4)

def _prep_number_crisp(gray: np.ndarray):
    """Produce a thick, high-contrast binary for skinny printed collector digits."""
    if gray is None or getattr(gray, "size", 0) == 0:
        return None
    try:
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(gray, None, fx=3.2, fy=3.2, interpolation=cv2.INTER_CUBIC)
        g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(6,6))
        g = clahe.apply(g)
        # keep the strokes open — avoid heavy closing that can turn "5" into "6"
        blur = cv2.GaussianBlur(g, (3, 3), 0)
        sharp = cv2.addWeighted(g, 1.55, blur, -0.55, 0)
        th = cv2.adaptiveThreshold(
            sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
            15, 2
        )
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k)
        return cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
    except Exception:
        return None

def _iter_set_hint_rois(roi_bgr):
    """Yield progressively tighter crops where the printed set code usually lives."""
    if roi_bgr is None or getattr(roi_bgr, "size", 0) == 0:
        return
    yield roi_bgr
    try:
        h, w = roi_bgr.shape[:2]
        if h <= 0 or w <= 0:
            return
        # Bottom-left quadrant: covers set code + collector number label
        y0 = int(0.55 * h)
        x1 = int(0.55 * w)
        if (x1 - 0) > 6 and (h - y0) > 6:
            yield roi_bgr[y0:h, 0:x1]
        # Tight region around the actual three-letter code (left of card number)
        y1 = h
        y0b = max(0, int(0.68 * h))
        x0b = max(0, int(0.02 * w))
        x1b = min(w, int(0.42 * w))
        if (x1b - x0b) > 6 and (y1 - y0b) > 6:
            yield roi_bgr[y0b:y1, x0b:x1b]
        # Whole bottom strip (helps when entire left edge is cropped)
        strip_y0 = max(0, int(0.72 * h))
        if (h - strip_y0) > 6:
            yield roi_bgr[strip_y0:h, 0:w]
        # Left half of lower third — catches cases where only part of the band is visible
        lower_y0 = max(0, int(0.63 * h))
        lower_x1 = max(6, int(0.5 * w))
        if (h - lower_y0) > 6:
            yield roi_bgr[lower_y0:h, 0:lower_x1]
    except Exception:
        pass

def _ai_band_roi(card_bgr):
    """Deprecated helper kept for compatibility; no longer used."""
    return None

def _ai_card_number_rois(card_bgr):
    """Use the AI 'card' ROI exactly for the collector number (no padding)."""
    rois = []
    if not _ai_enabled() or card_bgr is None or getattr(card_bgr, "size", 0) == 0:
        return rois
    # Relaxed crop so we don't drop low-conf card boxes during number OCR.
    roi = _ai_crop_relaxed(card_bgr, "card")
    if roi is not None and getattr(roi, "size", 0) > 0:
        rois.append(("ai-num", roi))
    return rois

def _ai_set_code_roi(card_bgr):
    """Return the exact AI 'set_name' ROI for set-code OCR (no padding)."""
    if not _ai_enabled() or card_bgr is None or getattr(card_bgr, "size", 0) == 0:
        return None
<<<<<<< ours
    # Try relaxed first so we still get a box when YOLO confidence is low.
    roi = _ai_crop_asym_relaxed(
        card_bgr,
        "set_name",
        pad_left=_AI_SET_PAD_LEFT,
        pad_right=_AI_SET_PAD_RIGHT,
        pad_top=_AI_SET_PAD_TOP,
        pad_bottom=_AI_SET_PAD_BOTTOM,
    )
    if roi is None or getattr(roi, "size", 0) == 0:
        roi = _ai_crop_asym(
            card_bgr,
            "set_name",
            pad_left=_AI_SET_PAD_LEFT,
            pad_right=_AI_SET_PAD_RIGHT,
            pad_top=_AI_SET_PAD_TOP,
            pad_bottom=_AI_SET_PAD_BOTTOM,
        )
    if roi is None or getattr(roi, "size", 0) == 0:
        roi = _ai_crop(card_bgr, "set_name", pad_x=_AI_SET_PAD_RIGHT, pad_y=_AI_SET_PAD_TOP)
    if roi is None or getattr(roi, "size", 0) == 0:
        roi = _ai_crop_exact(card_bgr, "set_name")
    if roi is None or getattr(roi, "size", 0) == 0:
        try:
            if DEBUG_OCR and _AI_SET_LOGS["count"] < 4:
                _AI_SET_LOGS["count"] += 1
                _dbg("OCR SET ROI", "missing set_name box; falling back later")
        except Exception:
            pass
=======
    roi = _ai_crop_exact(card_bgr, "set_name")
    if roi is None or getattr(roi, "size", 0) == 0:
        return None
>>>>>>> theirs
    return roi

def _iter_number_rois(card_bgr):
    """Yield candidate ROIs that may contain the collector number."""
    if card_bgr is None or getattr(card_bgr, "size", 0) == 0:
        return
    yielded = 0
    try:
        max_rois_cfg = int(globals().get("OCR_NUM_MAX_ROIS", 4))
    except Exception:
        max_rois_cfg = 4
    max_rois = 1 if _AI_ROI_STRICT else max(6, max_rois_cfg)
    card_roi = _card_crop_or_full(card_bgr)
    if _ai_enabled():
        try:
            if DEBUG_OCR and _AI_NUM_LOGS["count"] < 3:
                with _ai_lock:
                    keys = list((_ai_last.get("boxes") or {}).keys())
                _dbg("AI NUM BOXES", f"keys={keys}")
                _AI_NUM_LOGS["count"] += 1
        except Exception:
            pass
        ai_card = _ai_crop_relaxed(card_bgr, "card", pad_x=_AI_CARD_PAD_RIGHT, pad_y=_AI_CARD_PAD_Y)
        if ai_card is not None and getattr(ai_card, "size", 0) > 0:
            try:
                # Tight bottom-right slice where collector numbers live.
                br = _slice_frac(ai_card, 0.70, 0.99, 0.58, 0.99)
                if br is not None and getattr(br, "size", 0) > 0:
                    yielded += 1
                    yield ("ai-card-bottom-right", br)
                    if yielded >= max_rois:
                        return
                # Wider bottom band from AI card crop.
                ai_band = _slice_frac(ai_card, 0.68, 1.0)
                if ai_band is not None and getattr(ai_band, "size", 0) > 0:
                    yielded += 1
                    yield ("ai-card-band", ai_band)
                    if yielded >= max_rois:
                        return
                    try:
                        h, w = ai_band.shape[:2]
                        right = ai_band[:, int(0.55 * w):]
                        if right is not None and getattr(right, "size", 0) > 0:
                            yielded += 1
                            yield ("ai-card-band-right", right)
                            if yielded >= max_rois:
                                return
                        tight = ai_band[:, int(0.70 * w):]
                        if tight is not None and getattr(tight, "size", 0) > 0:
                            yielded += 1
                            yield ("ai-card-band-tight", tight)
                            if yielded >= max_rois:
                                return
                    except Exception:
                        pass
            except Exception:
                pass
        # Last resort AI regions: raw card box and set-name band.
        for label, ai_roi in _ai_card_number_rois(card_bgr):
            if ai_roi is None or getattr(ai_roi, "size", 0) == 0:
                continue
            yielded += 1
            yield (label, ai_roi)
            if yielded >= max_rois:
                return
        roi = _ai_crop(card_bgr, "set_name", pad_x=_AI_CARD_PAD_RIGHT, pad_y=_AI_CARD_PAD_Y)
        if roi is not None and getattr(roi, "size", 0) > 0:
            yielded += 1
            yield ("ai-set-name", roi)
            if yielded >= max_rois:
                return
    if _AI_ROI_STRICT:
        return
    if card_roi is not None and getattr(card_roi, "size", 0) > 0:
        band_roi = _slice_frac(card_roi, 0.68, 1.0)
        if band_roi is not None and getattr(band_roi, "size", 0) > 0:
            try:
                # Focused right-side slice where digits usually live
                h, w = band_roi.shape[:2]
                right = band_roi[:, int(0.55 * w):]
                if right is not None and getattr(right, "size", 0) > 0:
                    yielded += 1
                    yield ("band-right", right)
                    if yielded >= max_rois:
                        return
                tight = band_roi[:, int(0.72 * w):]
                if tight is not None and getattr(tight, "size", 0) > 0:
                    yielded += 1
                    yield ("band-tight", tight)
                    if yielded >= max_rois:
                        return
            except Exception:
                pass
            yielded += 1
            yield ("card-band", band_roi)
            if yielded >= max_rois:
                return
            # Use the same sub-ROI strategy as set-code OCR to focus on the printed band
            idx = 0
            for sub in _iter_set_hint_rois(band_roi):
                if sub is None or getattr(sub, "size", 0) == 0:
                    continue
                yielded += 1
                yield (f"band-sub-{idx}", sub)
                idx += 1
                if yielded >= max_rois:
                    return
    if yielded == 0:
        # last resort: full card crop
        _dbg("OCR NUM", "No number ROIs detected; using full card fallback")
        fallback = card_roi if card_roi is not None else card_bgr
        yield ("fallback-card", fallback)

_SET_CODE_LOOKALIKES = str.maketrans({
    "0": "o",
    "1": "l",
    "5": "s",
    "8": "b",
    "|": "l",
    "§": "s",
})

def _normalize_set_code_token(tok: str) -> str:
    t = (tok or "").strip().lower()
    if not t:
        return ""
    t = re.sub(r"[^a-z0-9]", "", t.translate(_SET_CODE_LOOKALIKES))
    return t

def _closest_set_code(tok: str) -> str:
    if not tok:
        return ""
    norm = _normalize_set_code_token(tok)
    if not norm:
        return ""
    if norm in SCRYFALL_SET_CODES:
        return norm
    try:
        codes = list(SCRYFALL_SET_CODES or [])
        if not codes:
            return ""
        # 1-char substitution (helps D↔O/0 typos) for 3-letter codes
        if len(norm) == 3:
            confusable = {
                "o": ("o", "0", "d"),
                "0": ("0", "o", "d"),
                "d": ("d", "o", "0"),
            }
            for i, ch in enumerate(norm):
                repls = confusable.get(ch, (ch,))
                for r in repls:
                    cand = norm[:i] + r + norm[i+1:]
                    if cand in SCRYFALL_SET_CODES:
                        return cand
            for code in codes:
                if len(code) != 3:
                    continue
                try:
                    if sum(a != b for a, b in zip(code, norm)) == 1:
                        return code
                except Exception:
                    continue
        # Looser fuzzy match fallback
        matches = difflib.get_close_matches(norm, codes, n=1, cutoff=0.60)
        return matches[0] if matches else ""
    except Exception:
        return ""

def _fast_read_set_hint(roi_bgr):
    """Prefer the most common OCR hit across several tight set-band crops."""
    votes: dict[str, int] = {}
    best_raw = ""
    try:
        for sub in _iter_set_hint_rois(roi_bgr):
            if sub is None or sub.size == 0:
                continue
            b = _prep_roi_for_ocr(sub)
            raw, _, _ = _read_text_general(b)
            if not raw:
                continue
            txt = raw.upper()
            # Strongly prefer 3-letter all-cap tokens (set codes are usually 3 letters)
            primary = re.findall(r'([A-Z]{3})', txt)
            fallback = re.findall(r'([A-Z0-9]{2,5})', txt)
            for token in primary + fallback:
                code = _normalize_set_code_token(token)
                if not code:
                    continue
                votes[code] = votes.get(code, 0) + 1
                if not best_raw:
                    best_raw = code
                if code in SCRYFALL_SET_CODES and votes[code] >= 2 and len(code) == 3:
                    return code
    except Exception:
        pass
    if votes:
        best_code = ""
        best_count = 0
        for code, count in votes.items():
            if code in SCRYFALL_SET_CODES and len(code) == 3 and count > best_count:
                best_code, best_count = code, count
        if best_code:
            return best_code
        best_raw = max(votes.items(), key=lambda kv: kv[1])[0]
    return _closest_set_code(best_raw)


def _fast_read_number(roi_bgr, source_tag="", deadline=None):
    try:
        if roi_bgr is None or roi_bgr.size == 0:
            return "", 0.0
        if deadline is not None and time.perf_counter() >= deadline:
            return "", 0.0
        if deadline is not None and (deadline - time.perf_counter()) < 0.15:
            return "", 0.0

        _num_timing_debug = bool(globals().get("OCR_NUM_TIMING_DEBUG", True))
        _fast_start = time.perf_counter()

        def _time_left():
            if deadline is None:
                return True
            return time.perf_counter() < deadline

        # Light preprocess: upsample skinny digits to help OCR.
        roi_bgr = _pad_roi_for_ocr(roi_bgr, pad_px=2)
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        # Upscale to make strokes thicker for Tesseract.
        gray = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        g = cv2.GaussianBlur(gray, (3, 3), 0)
        _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        tokens = []
        digit_counts = {}
        crisp_bgr = _prep_number_crisp(gray) if _use_tesseract_backend() else None

        def _collect(raw_txt, bonus=0.0):
            if not raw_txt:
                return
            if not _time_left():
                return
            for tok, conf in _num_tokens_from_text(raw_txt):
                tokens.append((tok, conf + bonus))
                s = str(tok)
                if s.isdigit():
                    digit_counts[s] = digit_counts.get(s, 0) + 1

        def _finalize():
            best_tok, best_conf = "", 0.0
            for tok, conf in tokens:
                best_tok, best_conf = _prefer_collector_candidate(best_tok, best_conf, tok, conf)
            if best_tok:
                occ = sum(1 for t, _ in tokens if t == best_tok)
                best_conf = _boost_collector_conf(best_tok, best_conf, occ)
                trimmed = _trim_digit_candidate(best_tok, digit_counts)
                if trimmed != best_tok and str(trimmed).isdigit():
                    best_tok = trimmed
                    best_conf = max(best_conf, _FAST_NUM_CONF_EXIT)
            return best_tok, best_conf

        def _strong(tok, conf):
            if not tok:
                return False
            s = str(tok)
            return conf >= _FAST_NUM_CONF_EXIT and s.isdigit() and 1 <= len(s) <= 4

        def _strip_leading_zeros(tok: str):
            try:
                s = str(tok)
                if s.isdigit() and len(s) > 1 and s.startswith("0"):
                    return str(int(s))
                return tok
            except Exception:
                return tok

        def _finish(tok, conf, stage="done"):
            return tok, conf

        def _run_paddle_variant(img_bgr, stage, bonus=0.0, maxw_val=380):
            if img_bgr is None or getattr(img_bgr, "size", 0) == 0:
                return None
            if not _time_left():
                return None
            try:
                txt, conf = _tess_text_from_bgr(img_bgr, maxw=maxw_val, allowlist="0123456789/", deadline=deadline, det=False)
            except Exception:
                return None
            if not txt:
                return None
            txt = re.sub(r"[^0-9/ ]", " ", txt)
            _collect(txt, bonus + max(0.0, min(conf, 100.0)) * 0.10)
            tok, conf_best = _finalize()
            if _strong(tok, conf_best):
                return tok, conf_best, stage
            return None

        if _use_tesseract_backend() and _time_left():
            hit = _run_paddle_variant(roi_bgr, "paddle")
            if hit:
                tok, conf, stage = hit
                _dbg("OCR NUM FAST", f"roi={source_tag or '?'} tok='{tok}' conf={conf:.1f} ({stage})")
                return _finish(tok, conf, stage=stage)
            if crisp_bgr is not None and _time_left():
                hit = _run_paddle_variant(crisp_bgr, "paddle-crisp", bonus=1.2, maxw_val=520)
                if hit:
                    tok, conf, stage = hit
                    _dbg("OCR NUM FAST", f"roi={source_tag or '?'} tok='{tok}' conf={conf:.1f} ({stage})")
                    return _finish(tok, conf, stage=stage)
            if _time_left():
                try:
                    bin_bgr = cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)
                except Exception:
                    bin_bgr = None
                hit = _run_paddle_variant(bin_bgr, "paddle-bin", bonus=0.5)
                if hit:
                    tok, conf, stage = hit
                    _dbg("OCR NUM FAST", f"roi={source_tag or '?'} tok='{tok}' conf={conf:.1f} ({stage})")
                    return _finish(tok, conf, stage=stage)

        # Direct Tesseract pass with a strict allowlist to catch thin digits
        if _time_left() and TESS_AVAILABLE:
            try:
                rgb = cv2.cvtColor(b, cv2.COLOR_GRAY2RGB)
                data = pytesseract.image_to_data(
                    rgb,
                    output_type=_TessOutput.DICT,
                    config="--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789/",
                )
                txts = []
                for txt, conf_str in zip(data.get("text", []), data.get("conf", [])):
                    txt = (txt or "").strip()
                    if not txt:
                        continue
                    try:
                        conf_val = float(conf_str)
                    except Exception:
                        conf_val = 0.0
                    txts.append((txt, conf_val))
                if txts:
                    combined = " ".join(t for t, _ in txts)
                    avg_conf = np.mean([c for _, c in txts]) if txts else 0.0
                    _collect(combined, bonus=max(0.0, avg_conf) * 0.10)
            except Exception:
                pass

        best_tok, best_conf = _finalize()
        if best_tok:
            best_tok = _strip_leading_zeros(best_tok)
            _dbg("OCR NUM FAST", f"roi={source_tag or '?'} tok='{best_tok}' conf={best_conf:.1f}")
        return _finish(best_tok, best_conf)
    except Exception:
        return "", 0.0

def _collector_candidate_score(token: str, conf: float) -> float:
    """Rough quality score so we can decide between fast/slow OCR results."""
    if not token:
        return -1.0
    tok = str(token)
    conf_bonus = max(0.0, min(float(conf or 0.0), 100.0)) / 50.0
    try:
        num_val = int(re.sub(r"\D", "", tok) or "0")
    except Exception:
        num_val = 0
    if "/" in tok:
        parts = re.findall(r"(\d+)", tok)
        num_str = parts[0] if parts else ""
        denom_str = parts[1] if len(parts) > 1 else ""
        digits = len(num_str)
        slash_bonus = 0.3
        penalty = 0.0
        try:
            num = int(num_str or "0")
            denom = int(denom_str or "0")
        except Exception:
            num = denom = 0
        if denom and num and denom >= (num * 2):
            penalty += 0.8
        if denom >= 300:
            penalty += 0.8
        if denom >= 500:
            penalty += 1.2
        if num >= 300:
            penalty += 0.6
        if num < 10:
            penalty += 0.5
        if denom and num > denom:
            penalty += 1.5
            if num >= 100 and denom <= 50:
                penalty += 1.0
        length_score = 4.5 - abs(digits - 3) * 1.4
        # strongly prefer the numerator alone in 3–4 digit slash cases
        if 2 <= len(num_str) <= 4:
            penalty += 2.0
        return length_score + slash_bonus + conf_bonus - penalty
    digits = len(re.findall(r"\d", tok))
    if digits == 0:
        return -1.0
    length_score = 4.8 - abs(digits - 3) * 1.5
    if digits >= 4:
        length_score -= 0.8 * (digits - 3)
    if digits == 3:
        length_score -= 0.5
    if digits == 2:
        length_score += 0.6
    # Repeated digits are a common false-positive when glare doubles strokes.
    if tok.isdigit() and len(tok) == 2 and tok[0] == tok[1]:
        length_score -= 0.35
    if tok.isdigit() and len(tok) == 3 and tok.count(tok[0]) == 3:
        length_score -= 0.55
    if tok.isdigit() and len(tok) == 3 and tok[0] == tok[2] and tok[1] == "0":
        length_score -= 0.7
    if tok.isdigit() and len(tok) <= 2 and tok.endswith("0"):
        length_score -= 1.0
    if tok.isdigit() and len(tok) >= 3 and tok.startswith("0"):
        length_score += 0.4
    if tok.isdigit() and len(tok) >= 3 and tok.endswith("0"):
        length_score -= 0.8
    penalty = 0.0
    if num_val >= 400:
        penalty += (num_val - 380) / 240.0
    if num_val >= 800:
        penalty += 1.2
    if 1 <= num_val <= 200:
        length_score += 1.2
    return length_score + conf_bonus - penalty

def _boost_collector_conf(token: str, conf: float, repeats: int = 1) -> float:
    base = float(conf or 0.0)
    if not token:
        return base
    tok = str(token)
    digits = len(re.findall(r"\d", tok))
    if "/" not in tok and digits >= 2:
        base = max(base, 60.0 + 3.0 * max(0, digits - 2))
        base += 3.0 * max(0, repeats - 1)
    else:
        base += 1.5 * max(0, repeats - 1)
    return min(100.0, base)

def _trim_digit_candidate(token, counts: dict[str, int] | None):
    s = str(token or "")
    if bool(globals().get("OCR_NUM_PREFIX_STRIP", True)):
        try:
            if s.isdigit() and len(s) > 1:
                s = s.lstrip("0") or "0"
        except Exception:
            s = str(token or "")
    if not s.isdigit() or len(s) <= 2 or not counts:
        return s
    current = counts.get(s, 0)
    for trim_len in range(len(s) - 1, 1, -1):
        prefix = s[:trim_len]
        if prefix.isdigit() and counts.get(prefix, 0) >= current:
            trimmed = prefix.lstrip("0") or "0"
            return trimmed
    return s

def _prefer_collector_candidate(cur_token, cur_conf, new_token, new_conf):
    """Return whichever collector number candidate looks more like a valid value."""
    if _collector_candidate_score(new_token, new_conf) > _collector_candidate_score(cur_token, cur_conf):
        return new_token, new_conf
    return cur_token, cur_conf


def _read_text_general(bin_img):
    inverted = 255 - bin_img
    if _use_tesseract_backend():
        try:
            bgr = inverted
            if inverted is not None and getattr(inverted, "ndim", 0) == 2:
                bgr = cv2.cvtColor(inverted, cv2.COLOR_GRAY2BGR)
            txt, conf = _tess_text_from_bgr(bgr, det=False)
            if txt:
                return txt, conf, "Tesseract"
        except Exception:
            pass
    return "", 0.0, "none"

def _has_text_quick(bin_img) -> bool:
    if bin_img is None or bin_img.size == 0: return False
    ink = cv2.countNonZero(bin_img) / float(bin_img.size)
    return ink >= TEXT_PRESENCE_MIN

_clean_title_text = lambda t: re.sub(r"\s+", " ", re.sub(r"[^A-Za-z'\- ]+", " ", t or "")).strip()

# --- New: stopwords & common singletons so multiword (with lowercase words) is OK
STOPWORDS = set("""
of the a an and to for nor but or into onto over under with without from in on at by as
""".split())

COMMON_SINGLETONS = set("""
when target each until unless choose create add draw exile destroy return attack block land
creature sorcery instant artifact enchantment planeswalker you your
""".split())

def _is_probable_title(t: str) -> bool:
    if not t: return False
    letters = re.sub(r"[^A-Za-z]", "", t)
    if len(letters) < MIN_TITLE_LETTERS: return False
    words = [w for w in t.split() if w]
    if len(words) == 1:
        w = words[0]
        if len(w) < 4:
            return False
        # reject obvious non-title singletons unless they are actual card names
        if w.lower() in COMMON_SINGLETONS and _score_against_lexicon(t) < 0.98:
            return False
    # capitalization ratio ignoring stopwords
    core = [w for w in words if w.lower() not in STOPWORDS]
    pool = core if core else words
    cap_ratio = sum(1 for w in pool if w[:1].isupper()) / max(1, len(pool))
    return cap_ratio >= 0.45  # was 0.60; relax to accept normal titles like "Watcher of the Wayside"

def _letters_ratio(s: str) -> float:
    if not s: return 0.0
    letters = sum(ch.isalpha() for ch in s)
    return letters / max(1, len(s))

def _score_against_lexicon(s: str) -> float:
    if not s or not SCRYFALL_CARD_NAMES:
        return 0.0
    sl = s.lower().strip()
    if sl in SCRYFALL_CARD_NAMES_LOWER:
        return 1.0
    if FUZZ_AVAILABLE:
        _cand, sc, _ = process.extractOne(s, SCRYFALL_CARD_NAMES, scorer=fuzz.WRatio, processor=_norm_name_for_match)
        return sc / 100.0
    return 0.0

def _maybe_fix_reversed_two_words(t: str) -> str:
    parts = [w for w in re.split(r"\s+", t.strip()) if w]
    if len(parts) != 2:
        return t
    if not FUZZ_AVAILABLE or not SCRYFALL_CARD_NAMES:
        return t
    orig_sc = _score_against_lexicon(t)
    rev = parts[1] + " " + parts[0]
    rev_sc = _score_against_lexicon(rev)
    return rev if (rev_sc - orig_sc) >= 0.05 else t

def _read_lines_for_title(roi_src_bgr, blacklist_re=None, band_name="", foil=False):
    """
    Build many candidates from several preprocessors and both OCR engines.
    Returns list of (text, conf, provider).
    """
    lines = []
    if roi_src_bgr is None or roi_src_bgr.size == 0:
        return lines

    eng = None  # RapidOCR disabled

    def maybe_add(txt, conf, prov, relax=False):
        t = _clean_title_text(txt)
        if not t:
            return
        if blacklist_re and blacklist_re.search(t):
            return
        if _letters_ratio(t) < 0.60:
            return
        if relax:
            words = [w for w in t.split() if w]
            if len(words) >= 2 or _score_against_lexicon(t) >= 0.90:
                t2 = _maybe_fix_reversed_two_words(t)
                bonus = _score_against_lexicon(t2) * 15.0
                lines.append((t2, float(conf) + 8.0 + bonus, prov))
            return
        if not _is_probable_title(t):
            return
        t2 = _maybe_fix_reversed_two_words(t)
        bonus = _score_against_lexicon(t2) * 15.0
        lines.append((t2, float(conf) + bonus, prov))

    # RapidOCR helpers removed (disabled)

    # Tesseract-only configuration
    def add_tess(bin_img, tag, psm=6):
        """Run Tesseract on a binary title ROI with a letter-heavy allowlist to reduce D↔O confusions."""
        if not TESS_AVAILABLE or bin_img is None or getattr(bin_img, "size", 0) == 0:
            return
        try:
            rgb = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2RGB) if bin_img.ndim == 2 else cv2.cvtColor(bin_img, cv2.COLOR_BGR2RGB)
            cfg = f"--psm {int(psm)} --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789,'- "
            data = pytesseract.image_to_data(rgb, output_type=_TessOutput.DICT, config=cfg)
            texts = []
            confs = []
            for txt, conf_str in zip(data.get("text", []), data.get("conf", [])):
                txt = (txt or "").strip()
                if not txt:
                    continue
                try:
                    conf_val = float(conf_str)
                except Exception:
                    conf_val = -1.0
                if conf_val < 0:
                    continue
                texts.append(txt)
                confs.append(conf_val / 100.0)
            if not texts:
                return
            line = " ".join(texts).strip()
            max_conf = max(confs) if confs else 0.0
            conf = max(0.0, min(100.0, 100.0 * float(max_conf)))
            maybe_add(line, conf, f"Tesseract:{tag}")
        except Exception:
            return

    def add_paddle(img_bgr, tag, relax=False, bonus=0.0):
        if not _use_tesseract_backend():
            return
        if img_bgr is None or getattr(img_bgr, "size", 0) == 0:
            return
        try:
            src = img_bgr
            if img_bgr.ndim == 2:
                src = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
            txt, conf = _tess_text_from_bgr(src, maxw=int(_FAST_CFG.get("MAXW", 640)), det=False)
            if txt:
                maybe_add(txt, conf + bonus, f"Tesseract:{tag}", relax=relax)
        except Exception:
            pass

    # ---------- Build preprocess variants ----------
    roi_color = roi_src_bgr
    if foil:
        roi_color = _enhance_for_foil(roi_color)
    add_paddle(roi_color, "color", relax=False, bonus=2.0 if foil else 0.0)

    # binary → Tess
    bin_img = _prep_roi_for_ocr(roi_src_bgr)
    add_tess(bin_img, "bin-psm6", psm=6)
    add_tess(bin_img, "bin-psm7", psm=7)
    try:
        bin_bgr = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)
        add_paddle(bin_bgr, "bin", relax=True)
    except Exception:
        pass

    # blackhat (great on shiny / low ink)
    if globals().get('OCR_ENABLE_BLACKHAT', False):
        g = cv2.cvtColor(roi_src_bgr, cv2.COLOR_BGR2GRAY)
        bh = _blackhat_bin(g)
        add_tess(bh, "blackhat-psm7", psm=7)
        try:
            bh_bgr = cv2.cvtColor(bh, cv2.COLOR_GRAY2BGR)
            add_paddle(bh_bgr, "blackhat", relax=True, bonus=1.5)
        except Exception:
            pass

    return lines

# ========================
#MARK: Collector number helpers / OCR
# =========================
def _normalize_num_chars(txt: str) -> str:
    # Map common OCR lookalikes to digits; use \u escapes to avoid mojibake
    return (txt or "").translate({
        ord('O'): '0',
        ord('o'): '0',
        ord('|'): '1',
        ord('I'): '1',
        ord('l'): '1',
        ord('\u4E28'): '1',  # 丨 (vertical stroke) — was shown as "ä¸¨"
        ord('\u00B9'): '1',  # ¹ (superscript one) — was shown as "Â¹"
        ord('\u20AC'): '0',  # € (Euro) sometimes mis-OCRs as zero
    })


def _collapse_numeric_spans(txt: str):
    """Yield plausible collector number tokens, even when the slash is missing."""
    t = _normalize_num_chars(txt)
    seen = set()

    def emit(val: str):
        val = (val or "").strip()
        if not val or val in seen:
            return None
        seen.add(val)
        return val

    def try_split_combo(digits: str):
        """Heuristic: digits like 0096250 are often 0096/250 without the slash."""
        digits = digits.strip()
        n = len(digits)
        if n < 5:
            return
        max_left = min(4, n - 1)
        for cut in range(max_left, 1, -1):
            left = digits[:cut]
            right = digits[cut:]
            if not left or not right:
                continue
            if not (1 <= len(right) <= 4):
                continue
            try:
                L = int(left)
                R = int(right)
            except ValueError:
                continue
            if L == 0 or R == 0:
                continue
            out = emit(f"{L}/{R}")
            if out:
                yield out

    for m in re.finditer(r'(?<!\d)([0-9\s|IlOo]{1,5})\s*/\s*([0-9\s|IlOo]{1,5})(?!\d)', t):
        L = re.sub(r'\D', '', m.group(1))[:4]
        R = re.sub(r'\D', '', m.group(2))[:4]
        if L and R:
            out = emit(f"{int(L)}/{int(R)}")
            if out:
                yield out

    for m in re.finditer(r'(?<!\d)([0-9\s|IlOo]{2,10})(?!\d)', t):
        digits = re.sub(r'\D', '', m.group(1))
        if len(digits) < 2:
            continue
        if 2 <= len(digits) <= 4:
            out = emit(str(int(digits)))
            if out:
                yield out
            continue
        for out in try_split_combo(digits):
            yield out
        head = digits[:4]
        tail = digits[-4:]
        if len(head) == 4:
            out = emit(str(int(head)))
            if out:
                yield out
        if len(tail) == 4:
            out = emit(str(int(tail)))
            if out:
                yield out

    if seen:
        return

    singles = re.findall(r'\d', t)
    if len(singles) >= 2:
        digits = ''.join(singles)
        if len(digits) <= 4:
            out = emit(str(int(digits)))
            if out:
                yield out
        else:
            for out in try_split_combo(digits):
                yield out
            head = digits[:4]
            tail = digits[-4:]
            if len(head) == 4:
                out = emit(str(int(head)))
                if out:
                    yield out
            if len(tail) == 4:
                out = emit(str(int(tail)))
                if out:
                    yield out

def _ink_present_quick(gray_or_bin: np.ndarray, thresh: float = 0.012) -> bool:
    if gray_or_bin is None or gray_or_bin.size == 0:
        return False
    img = gray_or_bin
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    if len(np.unique(img)) <= 8:
        ink = cv2.countNonZero(img) / float(img.size)
        return ink >= thresh
    _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink = cv2.countNonZero(th) / float(th.size)
    return ink >= thresh

def _num_tokens_from_text(raw: str):
    if not raw:
        return
    text = _normalize_num_chars(raw)
    toks = list(_collapse_numeric_spans(text))
    if not toks:
        return
    best = {}
    for t in toks:
        best[t] = max(best.get(t, 0.0), 45.0 if "/" in t else 40.0)
    for t in list(best.keys()):
        if len(t) >= 3 and t.endswith("0") and t[:-1] in best:
            del best[t]
    for t, base_conf in list(best.items()):
        if "/" not in t:
            continue
        parts = re.findall(r"(\d+)", t)
        if len(parts) < 2:
            continue
        num_str, denom_str = parts[0], parts[1]
        salvage = set()
        if len(num_str) == 1 and len(denom_str) >= 2:
            salvage.add(num_str + denom_str[:1])
        if len(num_str) >= 3 and num_str.endswith("00"):
            salvage.add(num_str[:-2])
        if len(num_str) >= 4 and num_str.endswith("0"):
            salvage.add(num_str[:-1])
        for sv in salvage:
            sv = sv.lstrip("0") or "0"
            if not (1 <= len(sv) <= 4):
                continue
            try:
                sv_int = str(int(sv))
            except Exception:
                continue
            best[sv_int] = max(best.get(sv_int, 0.0), base_conf - 5.0)
    def prio(k: str):
        if "/" in k:
            return (3, 0, 10**6)
        digits = int(re.sub(r"\D", "", k) or 0)
        closeness = -abs(len(k) - 3)
        return (2, closeness, digits)
    for tok in sorted(best.keys(), key=prio, reverse=True):
        yield tok, best[tok]

def _read_collector_number(img, foil: bool = False):
    if bool(globals().get("OCR_SKIP_NUMBER", False)):
        _dbg("OCR NUM", "Collector number OCR skipped (config)")
        return "", 0.0
    if not bool(globals().get("OCR_NUM_ALLOW_SLOW", True)):
        return "", 0.0
    best_tok, best_conf = "", 0.0
    candidate_hits = {}
    start_ts = time.perf_counter()
    budget_s = max(0.5, float(globals().get("OCR_NUM_BUDGET_S", 2.0)))
    hard_cap = float(globals().get("OCR_NUM_HARD_CAP_S", 5.0))
    num_timing_debug = bool(globals().get("OCR_NUM_TIMING_DEBUG", True))
    roi_times = [] if num_timing_debug else None
    if hard_cap > 0:
        if budget_s > hard_cap:
            _dbg("OCR NUM", f"Clamping number OCR budget to {hard_cap:.1f}s (was {budget_s:.1f}s)")
        budget_s = min(budget_s, hard_cap)
    deadline = start_ts + budget_s

    def _time_left():
        return time.perf_counter() < deadline

    if num_timing_debug:
        try:
            _dbg(
                "OCR NUM CFG",
                f"skip={bool(globals().get('OCR_SKIP_NUMBER', False))} allow_slow={globals().get('OCR_NUM_ALLOW_SLOW', True)} "
                f"max_rois={globals().get('OCR_NUM_MAX_ROIS', 'n/a')} budget={budget_s:.2f}s hard_cap={hard_cap:.2f}s"
            )
        except Exception:
            pass

    def _update_tokens(raw_text, bonus=0.0):
        nonlocal best_tok, best_conf
        if not raw_text:
            return False
        prev = (best_tok, best_conf)
        for tok, conf in _num_tokens_from_text(raw_text):
            if tok:
                candidate_hits[tok] = candidate_hits.get(tok, 0) + 1
            best_tok, best_conf = _prefer_collector_candidate(best_tok, best_conf, tok, conf + bonus)
        return (best_tok, best_conf) != prev

    def _scan_roi(roi_bgr, label="", bonus=0.0):
        if roi_bgr is None or getattr(roi_bgr, "size", 0) == 0:
            return False
        t0 = time.perf_counter()
        roi_deadline = min(deadline, t0 + 0.90)  # per-ROI cap to avoid burning all time on one band
        def _roi_time_left():
            return time.perf_counter() < roi_deadline and _time_left()
        if not _roi_time_left():
            return False
        if roi_deadline - time.perf_counter() <= 0.05:
            return False
        try:
            # Hard clamp ROI size so a huge band doesn't blow the budget
            try:
                h0, w0 = roi_bgr.shape[:2]
                maxw = int(globals().get("OCR_NUM_MAX_ROI_W", 720))
                if w0 > maxw and maxw > 0:
                    scale = maxw / float(w0)
                    roi_bgr = cv2.resize(roi_bgr, (int(w0*scale), int(h0*scale)), interpolation=cv2.INTER_AREA)
            except Exception:
                pass
            src = _enhance_for_foil(roi_bgr) if foil else roi_bgr
            gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
            if not _ink_present_quick(gray):
                if num_timing_debug:
                    try:
                        _dbg("OCR NUM SKIP", f"{label or 'roi'} no-ink roi_shape={getattr(roi_bgr,'shape',None)}")
                    except Exception:
                        pass
                return False
            if not _roi_time_left():
                return False
            g = cv2.resize(gray, None, fx=2.2, fy=2.2, interpolation=cv2.INTER_CUBIC)
            g = cv2.GaussianBlur(g, (3, 3), 0)
            if not _roi_time_left():
                return False
            bin_img_inv = cv2.adaptiveThreshold(
                g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                17 if foil else 23, 3
            )
            bin_img = cv2.adaptiveThreshold(
                g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                17 if foil else 23, 3
            )
            changed = False
            use_tess = _use_tesseract_backend()
            crisp_bgr = _prep_number_crisp(gray) if use_tess else None

            def _paddle_num(img_bgr, allow_bonus=0.0, maxw_val=380, stage_hint="paddle"):
                nonlocal changed
                if img_bgr is None or getattr(img_bgr, "size", 0) == 0:
                    return False
                if not _roi_time_left():
                    return False
                try:
                    dl = roi_deadline if deadline is None else min(deadline, roi_deadline)
                    txt_raw, conf_raw = _tess_text_from_bgr(img_bgr, maxw=maxw_val, allowlist="0123456789/", deadline=dl)
                    if txt_raw:
                        txt_raw = re.sub(r"[^0-9/ ]", " ", txt_raw)
                        changed = _update_tokens(txt_raw, bonus + allow_bonus + max(0.0, min(conf_raw, 100.0)) * 0.10) or changed
                        if best_tok and best_conf >= max(_FAST_NUM_CONF_EXIT, 65.0):
                            _dbg("OCR NUM SLOW", f"{label or 'roi'} tok='{best_tok}' conf={best_conf:.1f} ({stage_hint})")
                            return True
                except Exception:
                    return False
                return False

            if use_tess and _roi_time_left():
                _paddle_num(src, stage_hint="paddle")
            if use_tess and _roi_time_left() and crisp_bgr is not None:
                _paddle_num(crisp_bgr, allow_bonus=1.2, maxw_val=520, stage_hint="paddle-crisp")
            if use_tess and _roi_time_left():
                try:
                    bin_bgr = cv2.cvtColor(bin_img_inv, cv2.COLOR_GRAY2BGR)
                except Exception:
                    bin_bgr = None
                _paddle_num(bin_bgr, allow_bonus=0.0, maxw_val=380, stage_hint="paddle-bin")
            if use_tess and _roi_time_left():
                try:
                    _, otsu_inv = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                    otsu_bgr = cv2.cvtColor(otsu_inv, cv2.COLOR_GRAY2BGR)
                except Exception:
                    otsu_bgr = None
                _paddle_num(otsu_bgr, allow_bonus=0.0, maxw_val=380, stage_hint="paddle-otsu")
            if use_tess and _roi_time_left():
                try:
                    bin_pos = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)
                except Exception:
                    bin_pos = None
                _paddle_num(bin_pos, allow_bonus=0.5, maxw_val=420, stage_hint="paddle-bin-pos")
            if use_tess and _roi_time_left():
                try:
                    _, otsu_pos = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    otsu_pos_bgr = cv2.cvtColor(otsu_pos, cv2.COLOR_GRAY2BGR)
                except Exception:
                    otsu_pos_bgr = None
                _paddle_num(otsu_pos_bgr, allow_bonus=0.5, maxw_val=420, stage_hint="paddle-otsu-pos")
            if use_tess and _roi_time_left():
                try:
                    bh = _blackhat_bin(gray)
                    bh_bgr = cv2.cvtColor(bh, cv2.COLOR_GRAY2BGR)
                except Exception:
                    bh_bgr = None
                _paddle_num(bh_bgr, allow_bonus=1.0, maxw_val=420, stage_hint="paddle-blackhat")
            if (not changed) and use_tess and _roi_time_left():
                _paddle_num(src, stage_hint="paddle-repeat")
            if roi_times is not None:
                try:
                    roi_times.append((label or "roi", time.perf_counter() - t0, changed, str(best_tok), float(best_conf)))
                except Exception:
                    pass
            if changed and best_tok:
                _dbg("OCR NUM SLOW", f"{label or 'roi'} tok='{best_tok}' conf={best_conf:.1f}")
            return changed
        except Exception:
            return False

    rois_list = list(_iter_number_rois(img))
    if num_timing_debug:
        try:
            desc = "; ".join([f"{lbl}:{getattr(roi,'shape',None)}" for lbl, roi in rois_list])
            _dbg("OCR NUM ROIS", desc or "none")
        except Exception:
            pass
    for label, roi in rois_list:
        if not _time_left():
            break
        bonus = 4.0 if label.startswith("ai-") else 2.0 if "band" in label else 0.5
        _scan_roi(roi, label=label, bonus=bonus)
        if best_tok and best_conf >= max(_FAST_NUM_CONF_EXIT, 72.0):
            break
        if best_tok and label.startswith("ai-"):
            # Early exit on confident AI-guided hit
            break
        if best_tok and best_conf >= max(_FAST_NUM_CONF_EXIT, 70.0):
            # Good enough: stop burning time on extra rois
            break
    if not _time_left() and not best_tok:
        _dbg("OCR NUM", f"Number OCR budget hit after {budget_s:.1f}s")
    if best_tok:
        hits = candidate_hits.get(best_tok, 1)
        best_conf = _boost_collector_conf(best_tok, best_conf, hits)
        digit_counts = {str(k): v for k, v in candidate_hits.items() if str(k).isdigit()}
        trimmed = _trim_digit_candidate(best_tok, digit_counts)
        if trimmed != best_tok:
            best_tok = trimmed
            best_conf = max(best_conf, _FAST_NUM_CONF_EXIT)
        # If we saw a slash but the numerator is small and the denominator is much larger, prefer the numerator.
        if "/" in str(best_tok):
            parts = re.findall(r"(\d+)", str(best_tok))
            if len(parts) >= 2:
                try:
                    num = int(parts[0]); denom = int(parts[1])
                except Exception:
                    num = denom = 0
                if num > 0 and denom > 0 and denom >= (num * 2):
                    best_tok = str(num)
                    best_conf = max(best_conf, _FAST_NUM_CONF_EXIT)
        # If we have trailing zero inflation, prefer the trimmed form when it scores better.
        if str(best_tok).isdigit() and len(str(best_tok)) >= 3 and str(best_tok).endswith("0"):
            trimmed_zero = str(best_tok)[:-1].lstrip("0") or "0"
            if trimmed_zero and trimmed_zero != str(best_tok):
                cur_score = _collector_candidate_score(str(best_tok), best_conf)
                trim_score = _collector_candidate_score(trimmed_zero, max(best_conf - 3.0, 0.0))
                if trim_score > cur_score:
                    best_tok = trimmed_zero
                    best_conf = max(best_conf, _FAST_NUM_CONF_EXIT)
        # Vote-based fallback: favor the most common short digit token if confidence candidates disagree.
        try:
            vote_candidates = [(cnt, tok) for tok, cnt in candidate_hits.items()
                               if str(tok).isdigit() and 1 <= len(str(tok)) <= 3]
            if vote_candidates:
                vote_candidates.sort(key=lambda t: (-t[0], len(str(t[1])), int(str(t[1])) if str(t[1]).isdigit() else 9999))
                top_count, top_tok = vote_candidates[0]
                if top_count >= 2:
                    top_score = _collector_candidate_score(str(top_tok), best_conf)
                    cur_score = _collector_candidate_score(str(best_tok), best_conf)
                    if top_score > cur_score:
                        best_tok = str(top_tok)
                        best_conf = max(best_conf, _FAST_NUM_CONF_EXIT)
        except Exception:
            pass
        _dbg("OCR NUM RESULT", f"collector='{best_tok}' conf={best_conf:.1f}")
    if num_timing_debug and roi_times:
        try:
            summary = ", ".join(f"{lbl}:{dur:.3f}s{'*' if ch else ''}" for lbl, dur, ch, _, _ in roi_times)
            _dbg("OCR NUM TIMING", f"slow_rois={summary}")
        except Exception:
            pass
    try:
        _dbg("PERF OCR", f"Number OCR time={time.perf_counter() - start_ts:.3f}s best='{best_tok}' conf={best_conf:.1f}")
    except Exception:
        pass
    if not best_tok:
        try:
            if candidate_hits:
                top_cand = sorted(candidate_hits.items(), key=lambda kv: kv[1], reverse=True)[0]
                _dbg("OCR NUM", f"No committed number; top_seen={top_cand[0]} count={top_cand[1]}")
            else:
                _dbg("OCR NUM", "Collector number OCR produced no tokens")
        except Exception:
            _dbg("OCR NUM", "Collector number OCR produced no tokens")
    return best_tok, best_conf


def _parse_collector_for_display(raw: str) -> str:
    if not raw:
        return ""
    s = str(raw).strip()
    m = re.search(r'(\d{1,4})\s*/\s*(\d{1,4})', s)
    if m:
        return f"{int(m.group(1))}/{int(m.group(2))}"
    nums = [int(n) for n in re.findall(r'(\d{1,4})', s)]
    nums = [n for n in nums if 1 <= n <= 9999]
    return str(max(nums)) if nums else ""

def _normalize_cn_for_search(raw: str) -> str:
    if not raw:
        return ""
    s = str(raw).strip()
    m = re.search(r'(\d{1,4})\s*/\s*(\d{1,4})', s)
    if m:
        return str(int(m.group(1)))
    nums = [int(n) for n in re.findall(r'(\d{1,4})', s)]
    nums = [n for n in nums if 1 <= n <= 9999]
    return str(max(nums)) if nums else ""

def _norm_name_for_match(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", (s or "").lower()).strip()

def _maybe_correct_ocr_name(text: str, conf: float) -> str:
    if not text:
        return text
    original = text.strip()
    tl = original.lower()
    if tl in SCRYFALL_CARD_NAMES_LOWER:
        for n in SCRYFALL_CARD_NAMES:
            if n.lower() == tl:
                return n
        return original
    if not FUZZ_AVAILABLE or not SCRYFALL_CARD_NAMES:
        return original
    # NEW: strong whole-string match
    cand, score, _ = process.extractOne(
        original, SCRYFALL_CARD_NAMES, scorer=fuzz.WRatio, processor=_norm_name_for_match
    )
    if cand and score >= 90:
        return cand
    # NEW: partial expansion (prefix/substring -> full title)
    for c, sc, _ in process.extract(original, SCRYFALL_CARD_NAMES, scorer=fuzz.partial_ratio, limit=5, processor=_norm_name_for_match):
        if sc >= 95:
            return c
    # Existing two-word swap fallback
    toks = re.findall(r"[A-Za-z']+", original)
    if len(toks) == 2:
        swapped = f"{toks[1]} {toks[0]}"
        sl = swapped.lower()
        if sl in SCRYFALL_CARD_NAMES_LOWER:
            for n in SCRYFALL_CARD_NAMES:
                if n.lower() == sl:
                    return n
            return swapped
        cand_orig, score_orig, _ = process.extractOne(
            original, SCRYFALL_CARD_NAMES, scorer=fuzz.WRatio, processor=_norm_name_for_match
        )
        cand_swap, score_swap, _ = process.extractOne(
            swapped, SCRYFALL_CARD_NAMES, scorer=fuzz.WRatio, processor=_norm_name_for_match
        )
        if cand_swap and (score_swap - (score_orig or 0)) >= 3 and score_swap >= 90:
            return cand_swap
    if cand:
        orig_tokens = [t for t in re.split(r"\s+", original) if t]
        if score >= 90 and cand.lower().split()[0] == (orig_tokens[0].lower() if orig_tokens else ""):
            return cand
        if _norm_name_for_match(cand).startswith(_norm_name_for_match(original)) and \
           (len(cand) - len(original)) <= 2 and score >= 86:
            return cand
    return original

# =========================
#MARK: DETECTION
# =========================
CARD_EDGE_MIN_STRENGTH = float(globals().get("CARD_EDGE_MIN_STRENGTH", 0.08))
CARD_EDGE_MIN_CONTRAST = float(globals().get("CARD_EDGE_MIN_CONTRAST", 0.020))
CARD_DETECT_MIN_SCORE  = float(globals().get("CARD_DETECT_MIN_SCORE", 0.42))
CARD_EDGE_RING_FRAC    = float(globals().get("CARD_EDGE_RING_FRAC", 0.012))
CARD_CLAHE_CLIP        = float(globals().get("CARD_CLAHE_CLIP", 2.4))
CARD_CLAHE_TILE        = int(globals().get("CARD_CLAHE_TILE", 8))
CARD_BG_BLUR           = int(globals().get("CARD_BG_BLUR", 41))
CARD_BLACKHAT_KERNEL   = int(globals().get("CARD_BLACKHAT_KERNEL", 33))
CARD_CHANNEL_THRESH_C  = float(globals().get("CARD_CHANNEL_THRESH_C", 4.0))
CARD_MASK_MIN_COMPONENT= float(globals().get("CARD_MASK_MIN_COMPONENT", 0.004))
CARD_EDGE_SNAP_FRAC    = float(globals().get("CARD_EDGE_SNAP_FRAC", 0.085))
CARD_EDGE_CONTRAST_WEIGHT = float(globals().get("CARD_EDGE_CONTRAST_WEIGHT", 0.65))
CARD_EDGE_CONTRAST_STEP   = int(globals().get("CARD_EDGE_CONTRAST_STEP", 3))
CARD_MASK_BORDER_TRIM_PCT = float(globals().get("CARD_MASK_BORDER_TRIM_PCT", 0.002))
CARD_RING_OUTSIDE_MIN_FRAC = float(globals().get("CARD_RING_OUTSIDE_MIN_FRAC", 0.18))

def _odd_kernel(value: float, limit: int | None = None, min_size: int = 3) -> int:
    """Return a clamped odd kernel size (limit derived from current frame)."""
    try:
        k = int(round(value))
    except Exception:
        k = min_size
    k = max(min_size, k)
    if k % 2 == 0:
        k += 1
    if limit is not None:
        limit = max(min_size, int(limit))
        if limit % 2 == 0:
            limit -= 1
        k = min(k, limit)
    if k % 2 == 0:
        k = max(min_size, k - 1)
    return max(min_size, k)

def _adaptive_block_size(shape, preferred: int = 31) -> int:
    h, w = shape[:2]
    limit = max(3, min(h, w))
    return _odd_kernel(preferred, limit=limit)

def _multi_channel_card_mask(bgr_small: np.ndarray) -> np.ndarray | None:
    """Adaptive threshold per color channel similar to tmikonen's detector."""
    if bgr_small is None or bgr_small.size == 0:
        return None
    try:
        chans = cv2.split(bgr_small)
    except Exception:
        return None
    if not chans:
        return None
    tile = max(1, CARD_CLAHE_TILE)
    clahe = cv2.createCLAHE(clipLimit=max(0.5, CARD_CLAHE_CLIP), tileGridSize=(tile, tile))
    block = _adaptive_block_size(chans[0].shape, preferred=31)
    mask = np.zeros_like(chans[0], dtype=np.uint8)
    for ch in chans:
        eq = clahe.apply(ch)
        try:
            th = cv2.adaptiveThreshold(eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, block, CARD_CHANNEL_THRESH_C)
        except Exception:
            continue
        mask = cv2.max(mask, th)
    return mask

def _dark_card_mask(gray: np.ndarray) -> np.ndarray | None:
    """Black-hat transform to emphasize dark cards on bright mats."""
    if gray is None or gray.size == 0:
        return None
    h, w = gray.shape[:2]
    k = _odd_kernel(CARD_BLACKHAT_KERNEL, limit=min(h, w))
    try:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    except Exception:
        return None
    try:
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    except Exception:
        return None
    if blackhat is None:
        return None
    blackhat = cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX)
    _, mask = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.medianBlur(mask, 5)
    return mask

def _fix_rect_aspect(rect, target_aspect=CARD_ASPECT, grow_limit=1.06):
    # rect: (center(x,y), (w,h), angle)
    (cx, cy), (w, h), ang = rect
    if w <= 0 or h <= 0:
        return rect

    flip = (w > h)                # treat wr/hr with wr <= hr
    wr, hr = (h, w) if flip else (w, h)
    r = max(1e-6, wr / hr)
    if abs(r - target_aspect) < 1e-3:
        return rect

    # keep area, change aspect
    area = wr * hr
    wr_t = (area * target_aspect) ** 0.5
    hr_t = (area / target_aspect) ** 0.5

    # avoid big jumps when the contour jitters
    if wr_t > wr * grow_limit or hr_t > hr * grow_limit:
        alpha = 0.30              # blend 30% toward target
        wr_t = wr*(1-alpha) + wr_t*alpha
        hr_t = hr*(1-alpha) + hr_t*alpha

    new_w, new_h = (hr_t, wr_t) if flip else (wr_t, hr_t)
    return ((cx, cy), (float(new_w), float(new_h)), float(ang))

def _illum_norm_gray(bgr_small: np.ndarray) -> np.ndarray:
    """Normalize lighting using LAB+CLAHE (per tmikonen) before division normalization."""
    if bgr_small is None or bgr_small.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    try:
        lab = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        tile = max(1, CARD_CLAHE_TILE)
        clahe = cv2.createCLAHE(clipLimit=max(0.5, CARD_CLAHE_CLIP), tileGridSize=(tile, tile))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        eq = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        g = cv2.cvtColor(eq, cv2.COLOR_BGR2GRAY)
    except Exception:
        g = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2GRAY)

    blur_sz = _odd_kernel(CARD_BG_BLUR, limit=min(g.shape[:2]))
    bg = cv2.GaussianBlur(g, (blur_sz, blur_sz), 0)
    bg = np.maximum(bg, 1)
    norm = cv2.divide(g, bg, scale=255)
    return norm.astype(np.uint8)

def _shadow_robust_mask(bgr_small: np.ndarray, drop_border: bool = True, norm_gray=None) -> np.ndarray:
    """
    Keep the card, reject white-sheet background + soft shadows.
    Pipeline:
      - illumination-normalized gray
      - adaptive inverse threshold (card ~ white)
      - add dilated Canny edges (stabilize borders)
      - drop connected components that touch the image border
      - close/open + trim 1% frame to avoid edge locks
      - optional reuse of a precomputed normalized gray frame
    """
    g = norm_gray if norm_gray is not None else _illum_norm_gray(bgr_small)

    # Card is darker than paper => after norm, invert binary so card = 255
    th = cv2.adaptiveThreshold(
        g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 2
    )

    edges = cv2.Canny(g, 36, 110)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), 1)

    mask = cv2.max(th, edges)

    chan_mask = _multi_channel_card_mask(bgr_small)
    if chan_mask is not None:
        mask = cv2.max(mask, chan_mask)
    dark_mask = _dark_card_mask(g)
    if dark_mask is not None:
        mask = cv2.max(mask, dark_mask)

    h, w = mask.shape[:2]
    if drop_border:
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        keep = np.zeros_like(mask)
        for i in range(1, num):
            x, y, ww, hh, area = stats[i]
            touches = (x <= 0) or (y <= 0) or (x + ww >= w - 1) or (y + hh >= h - 1)
            if touches:
                continue
            if area < CARD_MASK_MIN_COMPONENT * (w * h):  # drop tiny flecks
                continue
            keep[labels == i] = 255
        mask = keep

    k1 = _K1 if _K1 is not None else cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    k2 = _K2 if _K2 is not None else cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k1, 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k2, 1)

    trim_pct = max(0.0, CARD_MASK_BORDER_TRIM_PCT)
    pad = int(round(trim_pct * min(h, w)))
    if pad > 0:
        pad = max(1, pad)
        mask[:pad, :] = 0; mask[-pad:, :] = 0; mask[:, :pad] = 0; mask[:, -pad:] = 0
    return mask

def _rectangularity(cnt, rect) -> float:
    """Contour area divided by its minAreaRect box area."""
    area = cv2.contourArea(cnt)
    (w, h) = rect[1]
    box_area = max(1.0, float(w) * float(h))
    return float(area) / box_area

def _quad_from_angle_bins(pts: np.ndarray):
    """Pick one representative point from each quadrant around the center."""
    if pts is None or len(pts) < 4:
        return None
    center = pts.mean(axis=0)
    chosen = [None, None, None, None]
    dists = [0.0, 0.0, 0.0, 0.0]
    for p in pts:
        ang = (np.degrees(np.arctan2(p[1] - center[1], p[0] - center[0])) + 360.0) % 360.0
        idx = int(ang // 90) % 4
        d = float(np.linalg.norm(p - center))
        if chosen[idx] is None or d > dists[idx]:
            chosen[idx] = p
            dists[idx] = d
    if any(c is None for c in chosen):
        return None
    return np.asarray(chosen, dtype=np.float32)

def _approx_card_quad(cnt):
    """Approximate a convex card-like contour with a clean quadrilateral."""
    if cnt is None or len(cnt) < 4:
        return None
    try:
        hull = cv2.convexHull(cnt)
    except Exception:
        return None
    if hull is None or len(hull) < 4:
        return None
    hull = hull.reshape(-1, 2)
    perim = cv2.arcLength(hull, True)
    if perim <= 0:
        return None
    # Inspired by https://tmikonen.github.io/quantitatively/2020-01-01-magic-card-detector/
    for frac in np.linspace(0.005, 0.06, 12):
        approx = cv2.approxPolyDP(hull, frac * perim, True)
        if approx is None:
            continue
        if len(approx) == 4:
            quad = approx.reshape(-1, 2).astype(np.float32)
            return order_points(quad)
    approx = cv2.approxPolyDP(hull, 0.06 * perim, True)
    if approx is not None and len(approx) >= 4:
        quad = _quad_from_angle_bins(approx.reshape(-1, 2))
        if quad is not None and len(quad) == 4:
            return order_points(quad)
    return None

def _line_strength(edge_map: np.ndarray, p0, p1, samples: int = 64) -> float:
    """Average edge magnitude sampled along a segment."""
    if edge_map is None or edge_map.size == 0:
        return 0.0
    try:
        xs = np.linspace(float(p0[0]), float(p1[0]), samples)
        ys = np.linspace(float(p0[1]), float(p1[1]), samples)
        xs = np.clip(np.round(xs).astype(np.int32), 0, edge_map.shape[1] - 1)
        ys = np.clip(np.round(ys).astype(np.int32), 0, edge_map.shape[0] - 1)
        vals = edge_map[ys, xs]
        return float(vals.mean()) if vals.size else 0.0
    except Exception:
        return 0.0

def _line_mean(gray: np.ndarray, p0, p1, samples: int = 64) -> float:
    """Average intensity sampled along a segment."""
    if gray is None or gray.size == 0:
        return 0.0
    try:
        xs = np.linspace(float(p0[0]), float(p1[0]), samples)
        ys = np.linspace(float(p0[1]), float(p1[1]), samples)
        xs = np.clip(np.round(xs).astype(np.int32), 0, gray.shape[1] - 1)
        ys = np.clip(np.round(ys).astype(np.int32), 0, gray.shape[0] - 1)
        vals = gray[ys, xs]
        return float(vals.mean()) if vals.size else 0.0
    except Exception:
        return 0.0

def _snap_quad_edges(quad, edge_map: np.ndarray, max_inset: int = 12, gray=None) -> np.ndarray:
    """Slide each edge inward until edge response is strongest."""
    if quad is None or edge_map is None or edge_map.size == 0:
        return quad
    try:
        q = order_points(np.asarray(quad, dtype=np.float32))
    except Exception:
        return quad
    center = q.mean(axis=0)
    inset = max(1, int(max_inset))
    outset = max(1, int(round(inset * 1.4)))
    contrast_step = max(1.0, float(CARD_EDGE_CONTRAST_STEP))
    contrast_w = max(0.0, float(CARD_EDGE_CONTRAST_WEIGHT))
    for idx in range(4):
        nxt = (idx + 1) % 4
        p0 = q[idx].copy()
        p1 = q[nxt].copy()
        edge_vec = p1 - p0
        normal = np.array([-edge_vec[1], edge_vec[0]], dtype=np.float32)
        L = np.linalg.norm(normal)
        if L < 1e-3:
            continue
        normal /= L
        if np.dot(normal, center - p0) < 0:
            normal = -normal
        best_d = 0.0
        best_score = -1.0
        for shift in range(-outset, inset + 1):
            a = p0 + normal * shift
            b = p1 + normal * shift
            if not (
                0 <= a[0] < edge_map.shape[1] and 0 <= a[1] < edge_map.shape[0] and
                0 <= b[0] < edge_map.shape[1] and 0 <= b[1] < edge_map.shape[0]
            ):
                continue
            edge_score = _line_strength(edge_map, a, b, samples=72)
            contrast_score = 0.0
            if gray is not None and contrast_w > 0.0:
                inner_shift = min(inset, shift + contrast_step)
                outer_shift = max(-outset, shift - contrast_step)
                inner_a = p0 + normal * inner_shift
                inner_b = p1 + normal * inner_shift
                outer_a = p0 + normal * outer_shift
                outer_b = p1 + normal * outer_shift
                inner_mean = _line_mean(gray, inner_a, inner_b, samples=48)
                outer_mean = _line_mean(gray, outer_a, outer_b, samples=48)
                contrast_score = max(0.0, outer_mean - inner_mean) / 255.0
            score = edge_score + contrast_w * contrast_score
            if score > (best_score + 1e-3):
                best_score = score
                best_d = float(shift)
        q[idx] = p0 + normal * best_d
        q[nxt] = p1 + normal * best_d
    return q

def detect_cards(frame):
    if frame is None or frame.size == 0:
        return []
    sub = frame

    # --- downscale within ROI for speed ---
    maxw = PROC_DOWNSCALE_MAX_W
    scale = maxw / sub.shape[1] if sub.shape[1] > maxw else 1.0
    small = cv2.resize(sub, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    norm_gray = _illum_norm_gray(small)
    mask = _shadow_robust_mask(small, drop_border=True, norm_gray=norm_gray)
    if not np.any(mask):
        mask = _shadow_robust_mask(small, drop_border=False, norm_gray=norm_gray)
    mask_ref = mask.copy()
    edge_map = cv2.Canny(norm_gray, 36, 110)
    edge_map = cv2.dilate(edge_map, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), 1)
    try:
        grad_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    except Exception:
        try:
            grad_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        except Exception:
            grad_kernel = None
    if grad_kernel is not None:
        mask_outline = cv2.morphologyEx(mask_ref, cv2.MORPH_GRADIENT, grad_kernel)
        edge_map = cv2.max(edge_map, mask_outline)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return []

    Hs, Ws = small.shape[:2]
    img_area = float(Hs * Ws)
    diag = max(Hs, Ws)
    ring_px = max(2, int(round(CARD_EDGE_RING_FRAC * diag)))
    ring_px = min(ring_px, max(2, diag // 2))
    inner_px = max(1, ring_px // 2)
    outer_px = max(inner_px + 1, int(round(ring_px * 1.5)))
    inner_px = min(inner_px, 45)
    outer_px = min(max(outer_px, inner_px + 1), 61)
    edge_thickness = max(2, inner_px)
    outer_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (outer_px, outer_px))
    inner_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, inner_px), max(1, inner_px)))
    perim_mask = np.zeros_like(edge_map, dtype=np.uint8)
    fill_mask = np.zeros_like(edge_map, dtype=np.uint8)

    best = None
    best_score = -1.0
    best_quad_small = None

    aspect_target = CARD_ASPECT
    aspect_tol    = ASPECT_TOL
    min_area      = MIN_CARD_AREA_RATIO
    max_area      = max(MAX_CARD_AREA_RATIO, 0.995)
    rect_min      = max(RECTANGULARITY_MIN, 0.60)

    for cnt in cnts:
        area = cv2.contourArea(cnt)
        area_norm = area / img_area
        if area_norm < min_area or area_norm > max_area:
            continue

        rect = cv2.minAreaRect(cnt)
        (w, h) = rect[1]
        if w <= 0 or h <= 0:
            continue

        aspect = min(w, h) / max(w, h)
        tol_local = aspect_tol
        if not (aspect_target - tol_local <= aspect <= aspect_target + tol_local):
            continue

        rectness = _rectangularity(cnt, rect)
        if rectness < rect_min:
            continue

        eps = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps, True)
        if len(approx) >= 4:
            hull = cv2.convexHull(approx).reshape(-1, 2)
            if hull.shape[0] >= 4:
                rect = cv2.minAreaRect(hull.astype(np.float32))

        box = order_points(cv2.boxPoints(rect).astype(np.float32))
        candidate_quad = _approx_card_quad(cnt)
        use_quad = candidate_quad if candidate_quad is not None else box

        aspect = min(rect[1][0], rect[1][1]) / max(rect[1][0], rect[1][1])
        aspect_score = 1.0 - min(1.0, abs(aspect - aspect_target) / (aspect_target + 1e-6))
        box_i = np.round(use_quad).astype(np.int32)
        perim_mask.fill(0)
        cv2.polylines(perim_mask, [box_i], True, 255, edge_thickness)
        if cv2.countNonZero(perim_mask) == 0:
            continue
        edge_strength = cv2.mean(edge_map, mask=perim_mask)[0] / 255.0
        if edge_strength < CARD_EDGE_MIN_STRENGTH:
            continue

        fill_mask.fill(0)
        cv2.fillPoly(fill_mask, [box_i], 255)
        inner_mask = fill_mask if inner_px <= 1 else cv2.erode(fill_mask, inner_kernel, iterations=1)
        if cv2.countNonZero(inner_mask) == 0:
            inner_mask = fill_mask.copy()
        outer_mask = cv2.dilate(fill_mask, outer_kernel, iterations=1)
        ring_mask = cv2.subtract(outer_mask, fill_mask)
        contrast = CARD_EDGE_MIN_CONTRAST
        ring_area = cv2.countNonZero(ring_mask)
        if ring_area == 0:
            ring_mask = cv2.dilate(fill_mask, outer_kernel, iterations=2)
            ring_mask = cv2.subtract(ring_mask, fill_mask)
            ring_area = cv2.countNonZero(ring_mask)
        if ring_area > 0:
            ring_inside = cv2.bitwise_and(mask_ref, ring_mask)
            inside_frac = cv2.countNonZero(ring_inside) / float(max(1, ring_area))
            outside_frac = 1.0 - inside_frac
            if outside_frac < CARD_RING_OUTSIDE_MIN_FRAC:
                continue
            inner_mean = cv2.mean(norm_gray, mask=inner_mask)[0]
            outer_mean = cv2.mean(norm_gray, mask=ring_mask)[0]
            contrast = abs(inner_mean - outer_mean) / 255.0
            if contrast < CARD_EDGE_MIN_CONTRAST:
                continue

        edge_score = min(1.0, edge_strength)
        contrast_score = min(1.0, contrast / max(0.01, CARD_EDGE_MIN_CONTRAST * 2.0))
        score = (
            0.45 * area_norm +
            0.25 * rectness +
            0.15 * aspect_score +
            0.10 * edge_score +
            0.05 * contrast_score
        )
        if score > best_score:
            best_score = score
            best = rect
            best_quad_small = use_quad.copy()

    if best is None:
        return []
    if best_score < CARD_DETECT_MIN_SCORE:
        return []

    # lock to exact aspect, blend with hull-based quad, then unscale + offset back into full-frame coords
    best = _fix_rect_aspect(best, target_aspect=aspect_target)
    rect_quad = order_points(cv2.boxPoints(best).astype(np.float32))
    if best_quad_small is None:
        quad_small = rect_quad
    else:
        quad_small = (0.8 * best_quad_small + 0.2 * rect_quad).astype(np.float32)
    snap_frac = max(0.01, float(CARD_EDGE_SNAP_FRAC))
    snap_px = max(4, int(round(snap_frac * diag)))
    quad_small = _snap_quad_edges(quad_small, edge_map, max_inset=snap_px, gray=norm_gray)
    pad_pct = max(0.0, float(globals().get("DETECT_QUAD_PAD_PCT", 0.0)))
    if pad_pct > 0.0:
        quad_small = _expand_quad(quad_small, pad_pct, img_w=Ws, img_h=Hs)
    try:
        snap_rect = cv2.minAreaRect(quad_small.astype(np.float32))
        best = snap_rect
    except Exception:
        pass
    quad_full = (quad_small / scale).astype(np.float32)

    (bw, bh) = best[1]
    full_area = float(bw * bh) / (scale * scale)

    return [{'box': quad_full, 'area': full_area}]

# =========================
#MARK: OCR / FOIL / SET HINT
# =========================
def _blackhat_bin(gray: np.ndarray) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (19,19))
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k)
    bh = cv2.normalize(bh, None, 0, 255, cv2.NORM_MINMAX)
    _, th = cv2.threshold(bh, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return th

def _apply_gamma_u8(channel: np.ndarray, gamma: float) -> np.ndarray:
    if channel is None or channel.size == 0 or gamma <= 0.0:
        return channel
    if abs(gamma - 1.0) < 1e-3:
        return channel
    ch = channel.astype(np.float32) / 255.0
    ch = np.power(np.clip(ch, 0.0, 1.0), gamma)
    return np.clip(ch * 255.0, 0, 255).astype(np.uint8)

def _reduce_specular_highlights(bgr: np.ndarray) -> np.ndarray:
    if bgr is None or bgr.size == 0:
        return bgr
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    spec_mask = ((v > FOIL_SPEC_VAL_THRESHOLD) & (s < FOIL_SPEC_SAT_THRESHOLD)).astype(np.uint8)
    if FOIL_SPEC_DILATE > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (FOIL_SPEC_DILATE, FOIL_SPEC_DILATE))
        spec_mask = cv2.dilate(spec_mask, k, iterations=1)
    if np.any(spec_mask):
        mask = cv2.GaussianBlur(spec_mask * 255, (5,5), 0).astype(bool)
        v_blur = cv2.GaussianBlur(v, (5,5), 0)
        v_suppressed = ((v.astype(np.float32) * 0.35) + (v_blur.astype(np.float32) * 0.65)).astype(np.uint8)
        v = np.where(mask, v_suppressed, v)
        s = np.where(mask, (s.astype(np.float32) * 0.55).clip(0,255).astype(np.uint8), s)
    hsv2 = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR)

def _foil_match_normalize(bgr: np.ndarray) -> np.ndarray:
    """Specular + saturation suppression to make foil shots match better against flat references."""
    if bgr is None or bgr.size == 0:
        return bgr
    try:
        img = _reduce_specular_highlights(bgr)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = (s.astype(np.float32) * 0.45).clip(0, 255).astype(np.uint8)
        v = _apply_gamma_u8(v, 0.92)
        return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)
    except Exception:
        return bgr

def _enhance_for_foil(roi_bgr: np.ndarray) -> np.ndarray:
    if roi_bgr is None or roi_bgr.size == 0:
        return np.zeros((1,1,3), dtype=np.uint8)
    bgr = cv2.bilateralFilter(roi_bgr, 5, 35, 35)
    bgr = _reduce_specular_highlights(bgr)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=FOIL_CLAHE_CLIP, tileGridSize=(FOIL_CLAHE_TILE, FOIL_CLAHE_TILE))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    bgr = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = (s.astype(np.float32) * FOIL_SAT_SCALE).clip(0,255).astype(np.uint8)
    v = _apply_gamma_u8(v, FOIL_GAMMA)
    hsv2 = cv2.merge([h, s, v])
    bgr = cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR)
    if FOIL_UNSHARP_AMOUNT > 0.0:
        blur = cv2.GaussianBlur(bgr, (0,0), sigmaX=1.1)
        bgr = cv2.addWeighted(bgr, 1.0 + FOIL_UNSHARP_AMOUNT, blur, -FOIL_UNSHARP_AMOUNT, 0)
    return bgr

def _orig_ocr_from_card_upright(img):
    roi_top = _crop_title_roi_top(img)
    roi_alt = _crop_title_roi_alt(img)

    foil = False
    foil_score = 0.0
    if FOIL_DETECT:
        try:
            foil, foil_score = _detect_foil_card(img)
        except Exception:
            foil, foil_score = False, 0.0
    _dbg("FOIL DETECTION", f"status={foil} score={foil_score:.2f}")

    top_bin = _prep_roi_for_ocr(roi_top)
    alt_bin = _prep_roi_for_ocr(roi_alt)
    top_has = _has_text_quick(top_bin)
    alt_has = _has_text_quick(alt_bin)
    prefer_top = top_has or not alt_has

    lines_top = _read_lines_for_title(roi_top, None, band_name="TOP", foil=foil) if roi_top is not None else []
    lines_alt = _read_lines_for_title(roi_alt, None, band_name="ALT", foil=foil) if roi_alt is not None else []

    def pick_from(primary, secondary):
        if primary:   return _pick_best_title_line(primary)
        if secondary: return _pick_best_title_line(secondary)
        return "", 0.0, "none"

    if prefer_top:
        chosen_name, chosen_conf, chosen_prov = pick_from(lines_top, lines_alt)
    else:
        chosen_name, chosen_conf, chosen_prov = pick_from(lines_alt, lines_top)

    num_raw, num_conf = _read_collector_number(img, foil=foil)
    band_roi = _slice_frac(_card_crop_or_full(img), 0.68, 1.0)
    set_hint = _read_set_hint(band_roi) or ""

    return {
        "name_raw": chosen_name,
        "name": chosen_name,
        "name_conf": round(float(chosen_conf), 1),
        "number_raw": num_raw,
        "number": _parse_collector_for_display(num_raw),
        "number_conf": round(float(num_conf), 1) if _parse_collector_for_display(num_raw) else 0.0,
        "provider": (chosen_prov if chosen_name else "none"),
        "set_hint": set_hint,
        "foil": bool(foil),
        "foil_score": float(foil_score),
    }
# --- Set icon matching helpers (fallback when no collector number / set code) ---
try:
    import cairosvg as _cairosvg  # optional for local SVG->PNG rasterization
    _HAVE_CAIROSVG = True
except Exception:
    _cairosvg = None
    _HAVE_CAIROSVG = False

_set_icon_cache = {}
_set_icon_cache_lock = threading.Lock()

def _symbol_mask_from_rgba(_img):
    """Return a clean 0/255 mask for a set icon image (handles RGBA or BGR)."""
    try:
        if _img is None or getattr(_img, "size", 0) == 0:
            return None
        import numpy as _np
        import cv2
        if _img.ndim == 3 and _img.shape[2] == 4:
            alpha = _img[:, :, 3]
            mask = (alpha > 0).astype(_np.uint8) * 255
        else:
            gray = cv2.cvtColor(_img, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if mask.mean() > 127:  # keep symbol white on black
                mask = 255 - mask
        # Keep largest blob
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        biggest = max(cnts, key=cv2.contourArea)
        clean = _np.zeros_like(mask)
        cv2.drawContours(clean, [biggest], -1, 255, thickness=cv2.FILLED)
        return clean
    except Exception:
        return None

def _fetch_set_icon_mask(set_code: str, svg_url: str, size: int = 96, timeout=None):
    """Fetch and rasterize a set icon (SVG) to a binary mask, with caching."""
    key = (str(set_code or "").lower(), int(size))
    with _set_icon_cache_lock:
        if key in _set_icon_cache:
            return _set_icon_cache[key]
    import numpy as _np, cv2
    png_bytes = b""
    to = timeout or SCRYFALL_TIMEOUT
    try:
        if svg_url:
            if _HAVE_CAIROSVG:
                try:
                    _svg_text = (_HTTP.get(svg_url, timeout=to).text or "")
                    if _svg_text:
                        png_bytes = _cairosvg.svg2png(bytestring=_svg_text.encode("utf-8"),
                                                      output_width=size, output_height=size)
                except Exception:
                    png_bytes = b""
            if not png_bytes:
                # Scryfall sometimes supports PNG query on the svg endpoint
                try:
                    r = _HTTP.get(svg_url, params={"format": "png", "size": str(size)}, timeout=to)
                    if r.status_code == 200:
                        png_bytes = r.content or b""
                except Exception:
                    png_bytes = b""
    except Exception:
        png_bytes = b""
    if not png_bytes:
        return None
    arr = _np.frombuffer(png_bytes, _np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    m = _symbol_mask_from_rgba(img)
    with _set_icon_cache_lock:
        _set_icon_cache[key] = m
    return m


def _match_set_symbol_to_code(card_bgr, name_hint: str = ""):
    """Infer set code by matching the detected set icon against Scryfall set icons.
       Returns (set_code, confidence) or ("", 0.0). Uses multi-cue scoring + margin check.
    """
    try:
        import cv2
        import numpy as _np
        budget = float(globals().get("ICON_MATCH_BUDGET_S", 3.0))
        if budget <= 0.0:
            return "", 0.0
        deadline = time.time() + budget
        http_timeout = min(
            float(globals().get("ICON_MATCH_HTTP_TIMEOUT", 3.0)),
            float(globals().get("SCRYFALL_TIMEOUT", 25.0)),
            max(0.25, budget)
        )

        def _time_left():
            return time.time() < deadline

        if _scry_offline_now():
            return "", 0.0
        # 1) crop set_symbol ROI via AI
        roi = _ai_crop(card_bgr, "set_symbol") if _ai_enabled() else None
        if roi is None or getattr(roi, "size", 0) == 0:
            return "", 0.0
        mask_roi = _symbol_mask_from_rgba(roi)
        if mask_roi is None or cv2.countNonZero(mask_roi) < 50:
            return "", 0.0
        cnts, _ = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return "", 0.0
        cnt_roi = max(cnts, key=cv2.contourArea)
        # Normalize ROI contour size
        mask_roi = cv2.resize(mask_roi, (int(globals().get("ICON_MATCH_SIZE", 96)), int(globals().get("ICON_MATCH_SIZE", 96))), interpolation=cv2.INTER_AREA)
        cnts, _ = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt_roi = max(cnts, key=cv2.contourArea) if cnts else cnt_roi

        # Helper: hu distance
        def _hu_distance(cnt_a, cnt_b):
            try:
                hu_a = cv2.HuMoments(cv2.moments(cnt_a)).flatten()
                hu_b = cv2.HuMoments(cv2.moments(cnt_b)).flatten()
                # stability: log transform
                hu_a = _np.sign(hu_a) * _np.log1p(_np.abs(hu_a))
                hu_b = _np.sign(hu_b) * _np.log1p(_np.abs(hu_b))
                return float(_np.mean(_np.abs(hu_a - hu_b)))
            except Exception:
                return 0.5  # neutral

        def _quick_scry(url, params=None):
            if not _time_left() or _scry_offline_now():
                return None
            return _scryfall_request(url, params or {}, tries=1, timeout=http_timeout)

        # 2) Candidate sets from prints of the fuzzy name (keeps this fast)
        set_codes = []
        if name_hint and _time_left():
            try:
                fuzzy = _quick_scry("https://api.scryfall.com/cards/named", {"fuzzy": name_hint})
                prints_uri = (fuzzy or {}).get("prints_search_uri")
                if prints_uri and _time_left():
                    prints = _quick_scry(prints_uri, {"order": "released"})
                    if prints and isinstance(prints, dict):
                        plist = prints.get("data") or []
                        # unique set codes only
                        set_codes = sorted({(p.get("set") or "").lower() for p in plist if p.get("set")})
            except Exception:
                set_codes = []
        if not set_codes:
            # Fall back to cached set list only (skip network to avoid OCR stalls)
            sets = _get_all_scry_sets(cache_only=True) or []
            # Prefer modern releases by reversing (Scryfall returns oldest first)
            set_codes = [s.get("code","").lower() for s in reversed(sets) if s.get("code")]
        if not set_codes or not _time_left():
            return "", 0.0

        # 3) Score each candidate with multiple cues
        scored = []
        size = int(globals().get("ICON_MATCH_SIZE", 96))
        max_sets = int(globals().get("ICON_MATCH_MAX_SETS", 800)) or 800
        for code in set_codes[:max_sets]:  # hard cap for safety
            if not _time_left():
                break
            try:
                S = _quick_scry(f"https://api.scryfall.com/sets/{code}", {})
                svg_url = (S or {}).get("icon_svg_uri")
                if not svg_url:
                    continue
                m = _fetch_set_icon_mask(code, svg_url, size=size, timeout=http_timeout)
                if m is None or cv2.countNonZero(m) < 30:
                    continue
                icnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not icnts:
                    continue
                cnt_icon = max(icnts, key=cv2.contourArea)

                # Basic matchShapes (lower is better)
                ms = float(cv2.matchShapes(cnt_roi, cnt_icon, cv2.CONTOURS_MATCH_I1, 0.0))

                # Fill ratio cue (shape compactness)
                x,y,w,h = cv2.boundingRect(cnt_icon)
                fr_icon = cv2.contourArea(cnt_icon) / max(1.0, float(w*h))
                x2,y2,w2,h2 = cv2.boundingRect(cnt_roi)
                fr_roi = cv2.contourArea(cnt_roi) / max(1.0, float(w2*h2))
                fr_diff = abs(fr_icon - fr_roi)

                # Hu invariant distance
                hu = _hu_distance(cnt_roi, cnt_icon)

                # Combined score (tuned weights)
                score = ms + 0.12*fr_diff + 0.08*hu

                scored.append((score, code, ms, fr_diff, hu))
            except Exception:
                continue

        if not scored:
            return "", 0.0

        scored.sort(key=lambda t: t[0])
        best_score, best_code, ms, fr_diff, hu = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else (best_score + 1.0)

        accept_th = float(globals().get("ICON_MATCH_ACCEPT", 1.00))
        margin = float(globals().get("ICON_MATCH_MARGIN", 0.02))
        require_margin = bool(globals().get("ICON_MATCH_REQUIRE_CLEAR_WIN", True))

        # Accept only if below threshold and (optionally) clearly better than 2nd best
        if best_score < accept_th and (not require_margin or (second_score - best_score) >= margin):
            conf = 1.0 / (1.0 + best_score)
            _dbg("SET_ICON", f"Matched set icon → {best_code.upper()} (score={best_score:.3f}, Δ={second_score-best_score:.3f})")
            return best_code, float(conf)

        # Otherwise, be conservative: no guess
        if scored:
            _dbg("SET_ICON", f"No clear icon winner (best={best_score:.3f}, second={second_score:.3f}); skipping.")
        return "", 0.0
    except Exception as _e:
        _dbg("SET_ICON", f"icon match failed: {_e}")
        return "", 0.0
# --- end set icon helpers ---

# --- Fallback: derive set code from OCR'd "set_name" ROI if icon match fails ---
_scry_sets_cache = {"ts": 0, "data": []}

def _get_all_scry_sets(cache_only: bool = False, tries: int = 2, timeout=None):
    """Fetch (and cache) Scryfall set list; persist to disk for offline use.
       When cache_only=True, only in-memory/disk caches are used (no network)."""
    now = time.time()
    ttl = int(globals().get("SCRY_SETS_CACHE_TTL_S", 86400))
    cache_path = str(globals().get("SCRY_SETS_CACHE_PATH", "./cache/scry_sets.json"))
    # 1) Use in-memory cache if fresh
    if _scry_sets_cache["data"] and (now - _scry_sets_cache["ts"] < 3600):
        return _scry_sets_cache["data"]
    # 2) Try disk cache
    try:
        import os, json
        if os.path.exists(cache_path):
            st = os.stat(cache_path)
            if (now - st.st_mtime) < ttl:
                with open(cache_path, "r", encoding="utf-8") as f:
                    items = json.load(f) or []
                    _scry_sets_cache.update({"ts": now, "data": items})
                    return items
    except Exception:
        pass
    if cache_only:
        return _scry_sets_cache["data"]
    # 3) Fetch from Scryfall
    try:
        res = _scryfall_request("https://api.scryfall.com/sets", {}, tries=tries, timeout=timeout)
        items = (res or {}).get("data") or []
        _scry_sets_cache.update({"ts": now, "data": items})
        try:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(items, f)
        except Exception:
            pass
        return items
    except Exception:
        # 4) Fallback to stale disk cache if present
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                items = json.load(f) or []
                _scry_sets_cache.update({"ts": now, "data": items})
                return items
        except Exception:
            return _scry_sets_cache["data"]


def _read_set_name_roi(card_bgr):
    """OCR the YOLO 'set_name' ROI to text."""
    try:
        roi = _ai_crop_exact(card_bgr, "set_name") if _ai_enabled() else None
        if roi is None or getattr(roi, "size", 0) == 0:
            return ""
        # Reuse the general OCR pipeline
        bin_img = _prep_roi_for_ocr(roi)
        txt, _, _ = _read_text_general(bin_img)
        return (txt or "").strip()
    except Exception:
        return ""

def _set_code_from_set_name(name_raw):
    """Fuzzy-match a human set name to a Scryfall set CODE."""
    try:
        nm = (name_raw or "").strip()
        if not nm:
            return ""
        sets = _get_all_scry_sets(cache_only=False, tries=1, timeout=min(float(globals().get("SCRYFALL_TIMEOUT", 25.0)), 20.0))
        if not sets:
            return ""
        # Build searchable names
        candidates = [(s.get("name",""), s.get("code","")) for s in sets if s.get("code")]
        try:
            from rapidfuzz import process, fuzz
            # Strong match on full name first
            best = process.extractOne(nm, [c[0] for c in candidates], scorer=fuzz.WRatio, score_cutoff=82)
            if best:
                idx = best[2]
                return candidates[idx][1].lower()
            # Looser fallback
            best = process.extractOne(nm, [c[0] for c in candidates], scorer=fuzz.token_set_ratio, score_cutoff=75)
            if best:
                idx = best[2]
                return candidates[idx][1].lower()
        except Exception:
            # Simple fallback: direct equality ignoring case
            for n, code in candidates:
                if n.lower() == nm.lower():
                    return code.lower()
        return ""
    except Exception:
        return ""
# --- end set-name fallback ---

def _read_set_code_quick(roi_bgr):
    """Lightweight set-code read that avoids the slower fallback stack."""
    try:
        if roi_bgr is None or getattr(roi_bgr, "size", 0) == 0:
            return ""
        code = _fast_read_set_hint(roi_bgr)
        if code:
            return code
        bin_img = _prep_roi_for_ocr(roi_bgr)
        txt, _, _ = _read_text_general(bin_img)
        for tok in re.findall(r"[A-Za-z0-9]{2,5}", txt or ""):
            norm = _normalize_set_code_token(tok)
            if norm in SCRYFALL_SET_CODES:
                return norm
        return ""
    except Exception:
        return ""

def _resolve_set_hint(card_bgr, initial_hint="", name_hint="", allow_band_scan=True):
    """Consolidate set-hint extraction (band OCR, icon match, set-name OCR)."""
    hint = (initial_hint or "").strip().lower()
    # Normalize an incoming hint to the closest known code if it's off by a confusable char
    try:
        if hint and hint not in SCRYFALL_SET_CODES:
            corr = _closest_set_code(hint)
            if corr:
                hint = corr
            else:
                hint = ""
        # Common promo prefixes (t/p/s) – strip and retry
        if hint and hint not in SCRYFALL_SET_CODES and len(hint) == 4 and hint[0] in ("t", "p", "s"):
            base = hint[1:]
            if base in SCRYFALL_SET_CODES:
                hint = base
    except Exception:
        pass
    _set_name_cache = {"done": False, "code": ""}

    def _get_set_name_code():
        if _set_name_cache["done"]:
            return _set_name_cache["code"]
        _set_name_cache["done"] = True
        if not globals().get("SET_NAME_FALLBACK_ENABLE", True):
            _set_name_cache["code"] = ""
            return ""
        try:
            nm = _read_set_name_roi(card_bgr)
            code2 = _set_code_from_set_name(nm)
            _set_name_cache["code"] = (code2 or "").strip().lower()
            if _set_name_cache["code"]:
                _dbg("SET_NAME", f"set_hint<-name = {_set_name_cache['code']}")
        except Exception as _e:
            _set_name_cache["code"] = ""
            _dbg("SET_NAME", f"name->code fallback failed: {_e}")
        return _set_name_cache["code"]

    # 1) Direct OCR of the "set_name" ROI (where the three-letter code usually lives)
    try:
        if not hint:
            roi = _ai_crop_exact(card_bgr, "set_name") if _ai_enabled() else None
            if roi is not None and getattr(roi, "size", 0) > 0:
                hint = (_read_set_code_quick(roi) or "").strip().lower()
                if hint:
                    _dbg("SET_HINT", f"set_hint<-set_name_roi = {hint}")
                    return hint
    except Exception:
        pass
    # 2) OCR the full collector band (optional when laziness disabled)
    try:
        if (not hint) and allow_band_scan:
            roi = _card_crop_or_full(card_bgr)
            band = _slice_frac(roi, 0.68, 1.0)
            if band is not None and getattr(band, "size", 0) > 0:
                hint = (_read_set_hint(band) or "").strip().lower()
                if hint:
                    _dbg("SET_HINT", f"set_hint<-band = {hint}")
                    return hint
    except Exception:
        pass
    if hint:
        code2 = _get_set_name_code()
        if code2 and code2 != hint:
            _dbg("SET_NAME", f"band hint override: {hint} -> {code2}")
            return code2
        return hint
    # 3) OCR the printed set name and map to a code (cheap text lookup)
    code2 = _get_set_name_code()
    if code2:
        return code2
    # 4) Only if everything else failed, run expensive set-icon matching
    try:
        if globals().get("SET_ICON_FALLBACK_ENABLE", True):
            code, score = _match_set_symbol_to_code(card_bgr, name_hint=(name_hint or ""))
            if code:
                hint = code.strip().lower()
                _dbg("SET_ICON", f"set_hint<-icon = {hint}")
                return hint
    except Exception as _e:
        _dbg("SET_ICON", f"fallback failed: {_e}")
    return hint

def ocr_from_card_upright(img):
    # Use original OCR pipeline (now AI-aware title + numbers via overrides)
    res = _orig_ocr_from_card_upright(img)
    # Replace set_hint using AI "card" region if available
    try:
        roi = _card_crop_or_full(img)
        band = _slice_frac(roi, 0.68, 1.0)
        sh = _read_set_hint(band) or ""
        if sh:
            res["set_hint"] = sh
    except Exception:
        pass
    # Fallback: if no collector number AND no set_hint yet, try set-icon matching
    try:
        if (not res.get("set_hint")) and globals().get("SET_ICON_FALLBACK_ENABLE", True):
            code, score = _match_set_symbol_to_code(img, name_hint=(res.get("name") or res.get("name_raw") or ""))
            if code:
                res["set_hint"] = code; _dbg("SET_ICON", f"set_hint<-icon = {code}")
    except Exception as _e:
        _dbg("SET_ICON", f"fallback failed: {_e}")
    # Fallback A: use set icon to infer set code (if we still lack set_hint)
    try:
        if not res.get("set_hint"):
            code, score = _match_set_symbol_to_code(img, name_hint=(res.get("name") or res.get("name_raw") or ""))
            if code:
                res["set_hint"] = code; _dbg("SET_ICON", f"set_hint<-icon = {code}")
    except Exception as _e:
        _dbg("SET_ICON", f"icon fallback failed: {_e}")
    # Fallback B: OCR the "set_name" ROI and map to a set code
    try:
        if (not res.get("set_hint")) and globals().get("SET_NAME_FALLBACK_ENABLE", True):
            _setnm = _read_set_name_roi(img)
            _code2 = _set_code_from_set_name(_setnm)
            if _code2:
                res["set_hint"] = _code2; _dbg("SET_NAME", f"set_hint<-name = {_code2}")
    except Exception as _e:
        _dbg("SET_NAME", f"name->code fallback failed: {_e}")

    
    return res


def _pick_best_title_line(lines):
    if not lines:
        return "", 0.0, "none"
    def score(item):
        t, conf, prov = item
        words = t.split()
        cap_ratio = sum(1 for w in words if w[:1].isupper()) / max(1, len(words))
        prov_bias = 3.0 if "topline" in (prov or "").lower() else 0.0
        length_bias = min(6, len(words)) * 0.8  # prefer multiword titles a bit
        return conf + prov_bias + length_bias + cap_ratio*5.0
    return max(lines, key=score)

# =========================
#MARK: SCRYFALL
# =========================
_SCRY_OFFLINE = {"until": 0.0}

def _scry_offline_now():
    return time.time() < _SCRY_OFFLINE["until"]

def _scryfall_request(url, params=None, tries=2, timeout=None):
    if _scry_offline_now():
        return None
    params = params or {}
    to = timeout if timeout is not None else SCRYFALL_TIMEOUT
    for i in range(tries):
        try:
            r = _HTTP.get(url, params=params, timeout=to)
            if r.status_code == 200:
                return r.json()
        except Exception as e:
            if i+1 == tries:
                _dbg("SCRYFALL ERROR", f"Request failed after {tries} tries: {e}")
            _SCRY_OFFLINE["until"] = time.time() + 60.0  # 1 minute backoff
        time.sleep(0.15)
    return None

def _scry_fix_mismatch(choice, name, number_raw, set_hint):
    """
    If we have a set hint and/or collector number and the chosen print
    doesn't match, try to fetch the correct printing explicitly.
    Returns a (possibly new) choice or the original one if nothing better is found.
    """
    try:
        nm_orig = (name or "").strip()
        set_hint = (set_hint or "").lower()

        # normalize collector number like the lookup does (strip leading zeros)
        import re as _re
        def norm_cn(cn):
            if not cn:
                return ""
            m = _re.match(r"0*(\d{1,4})$", str(cn))
            return m.group(1) if m else ""

        cn_norm = _normalize_cn_for_search(number_raw)

        # If nothing to enforce, or the current choice already matches, keep it.
        if not set_hint and not cn_norm:
            return choice
        try:
            if choice and (not set_hint or (choice.get("set","") == set_hint)) and \
               (not cn_norm or norm_cn(choice.get("collector_number")) == cn_norm):
                return choice
        except Exception:
            pass

        # small fetch helper that mirrors the rest of the code
        def fetch(url, params=None):
            r = _scryfall_request(url, params or {})
            return r if r and r.get("object") != "error" else None

        # Strategy 1: strict name + set + cn
        if nm_orig and set_hint and cn_norm:
            data = fetch("https://api.scryfall.com/cards/search", {
                "q": f'!"{nm_orig}" cn:{cn_norm} set:{set_hint}',
                "unique": "prints", "order": "released", "dir": "desc"
            })
            if data and data.get("data"):
                card = data["data"][0]
                try: card["_match_mode"] = "postfix_enforce_name_set_cn"
                except Exception: pass
                return card

        # Strategy 2: set + cn (no name)
        if set_hint and cn_norm:
            data = fetch("https://api.scryfall.com/cards/search", {
                "q": f"cn:{cn_norm} set:{set_hint}",
                "unique": "prints", "order": "released", "dir": "desc"
            })
            if data and data.get("data"):
                card = data["data"][0]
                try: card["_match_mode"] = "postfix_enforce_set_cn"
                except Exception: pass
                return card

        # Strategy 3: name + set (ignore cn)
        if nm_orig and set_hint:
            data = fetch("https://api.scryfall.com/cards/search", {
                "q": f'!"{nm_orig}" set:{set_hint}',
                "unique": "prints", "order": "released", "dir": "desc"
            })
            if data and data.get("data"):
                card = data["data"][0]
                try: card["_match_mode"] = "postfix_enforce_name_set"
                except Exception: pass
                return card

        return choice
    except Exception:
        return choice

def _scryfall_lookup(name, number_raw, set_hint=""):
    nm_orig = (name or "").strip()
    cn_norm = _normalize_cn_for_search(number_raw)
    set_hint = (set_hint or "").lower()

    # helper: annotate chosen card with how we matched
    def _ret(card, how):
        try:
            card["_match_mode"] = how
        except Exception:
            pass
        return card

    # If we have no name, skip the name queries but DO try cn+set / cn fallbacks.
    if not nm_orig:
        def fetch(url, params=None):
            r = _scryfall_request(url, params or {})
            return r if r and r.get("object") != "error" else None

        if cn_norm and set_hint:
            data = fetch("https://api.scryfall.com/cards/search",
                         {"q": f"cn:{cn_norm} set:{set_hint}", "unique":"prints","order":"released","dir":"desc"})
            if data and data.get("data"):
                return _ret(data["data"][0], "cn_set_only")

        if cn_norm:
            data = fetch("https://api.scryfall.com/cards/search",
                         {"q": f"cn:{cn_norm}", "unique":"prints","order":"released","dir":"desc"})
            if data and data.get("data"):
                return _ret(data["data"][0], "cn_only")
        return None

    # Prepare keys
    nm_orig = (name or "").strip()
    cn_norm = _normalize_cn_for_search(number_raw)
    set_hint = (set_hint or "").lower()

    # Expand candidate names: original + MDFC combined (if OCR saw only a face)
    candidates = [nm_orig]
    nm_l = nm_orig.lower()
    if nm_l in DFC_FACE_TO_COMBINED:
        candidates.append(DFC_FACE_TO_COMBINED[nm_l])
    else:
        nk = _norm_key(nm_orig)
        if nk in DFC_FACE_TO_COMBINED_NORM:
            candidates.append(DFC_FACE_TO_COMBINED_NORM[nk])

    def _try_named(nm):
        matched_by = "none"
        choice = None

        def fetch(url, params=None):
            r = _scryfall_request(url, params or {})
            return r if r and r.get("object") != "error" else None

        def norm_cn(cn):
            if not cn: return ""
            m = re.match(r"0*(\d{1,4})$", str(cn))
            return m.group(1) if m else ""

        # 1) exact by name, refine by prints
        exact = fetch("https://api.scryfall.com/cards/named", {"exact": nm})
        if exact:
            prints_uri = exact.get("prints_search_uri")
            if prints_uri:
                prints = fetch(prints_uri, {"order": "released", "dir": "desc", "unique": "prints"})
                if prints and prints.get("data"):
                    plist = prints["data"]
                    cand = None
                    if set_hint:
                        cand = next((c for c in plist if (c.get("set","") == set_hint)), None)
                        if cand and cn_norm and norm_cn(cand.get("collector_number")) != cn_norm:
                            cand = None
                    if not cand and cn_norm:
                        cand = next((c for c in plist if norm_cn(c.get("collector_number")) == cn_norm), None)
                    if cand:
                        return _ret(cand, "prints_refine"), "prints_refine"
            if not choice:
                return _ret(exact, "exact"), "exact"

        # 2) explicit search by name + cn (+ set if present)
        if cn_norm:
            q = f'!"{nm}" cn:{cn_norm}'
            if cn_norm:
                short_or_single = (len(nm.split()) == 1 or len(nm) <= 5)
                if short_or_single:
                    q = f"name:{nm} cn:{cn_norm}"
                    if set_hint:
                        q += f" set:{set_hint}"
                    data = fetch("https://api.scryfall.com/cards/search",
                                {"q": q, "unique": "prints", "order": "released", "dir": "desc"})
                    if data and data.get("data"):
                        return _ret(data["data"][0], "search_name_field_cn"), "search_name_field_cn"
            if set_hint: q += f" set:{set_hint}"
            data = fetch("https://api.scryfall.com/cards/search",
                         {"q": q, "unique": "prints", "order": "released", "dir": "desc"})
            if data and data.get("data"):
                return _ret(data["data"][0], "search_name_cn"), "search_name_cn"

        # 3) fuzzy by name (then refine by prints if possible)
        fuzzy = fetch("https://api.scryfall.com/cards/named", {"fuzzy": nm})
        if fuzzy:
            prints_uri = fuzzy.get("prints_search_uri")
            if prints_uri:
                prints = fetch(prints_uri, {"order": "released", "dir": "desc", "unique": "prints"})
                if prints and prints.get("data"):
                    plist = prints["data"]
                    cand = None
                    if set_hint:
                        cand = next((c for c in plist if (c.get("set","") == set_hint)), None)
                        if cand and cn_norm and norm_cn(cand.get("collector_number")) != cn_norm:
                            cand = None
                    if not cand and cn_norm:
                        cand = next((c for c in plist if norm_cn(c.get("collector_number")) == cn_norm), None)
                    if cand:
                        return _ret(cand, "fuzzy_prints_refine"), "fuzzy_prints_refine"
            return _ret(fuzzy, "fuzzy"), "fuzzy"

        return None, "none"

    # in-memory request cache (keyed by candidate)
    for nm in candidates:
        cache_key = (nm.lower(), cn_norm, set_hint)
        if cache_key in _scry_cache:
            return _scry_cache[cache_key]
        choice, how = _try_named(nm)
        _dbg("SCRYFALL", f"Card Name='{nm}', Set='{set_hint or '-'}', Card Number='{cn_norm or '-'}', Matching Algorithm='{how}'")
        if choice is not None:
            _scry_cache[cache_key] = choice
            return choice

    # FINAL fallbacks: allow cn+set (no name), then cn only
    def fetch(url, params=None):
        r = _scryfall_request(url, params or {})
        return r if r and r.get("object") != "error" else None

    if cn_norm and set_hint:
        q = f"cn:{cn_norm} set:{set_hint}"
        data = fetch("https://api.scryfall.com/cards/search",
                     {"q": q, "unique": "prints", "order": "released", "dir": "desc"})
        if data and data.get("data"):
            choice = _ret(data["data"][0], "cn_set_only")
            _scry_cache[(nm_orig.lower(), cn_norm, set_hint)] = choice
            _dbg("SCRYFALL", f"nm='{nm_orig}', set_hint='{set_hint}', cn_norm='{cn_norm}', matched_by='cn_set_only'")
            return choice

    if cn_norm:
        q = f"cn:{cn_norm}"
        data = fetch("https://api.scryfall.com/cards/search",
                     {"q": q, "unique": "prints", "order": "released", "dir": "desc"})
        if data and data.get("data"):
            choice = _ret(data["data"][0], "cn_only")
            _scry_cache[(nm_orig.lower(), cn_norm, set_hint)] = choice
            _dbg("SCRYFALL", f"nm='{nm_orig}', set_hint='{set_hint or '-'}', cn_norm='{cn_norm}', matched_by='cn_only'")
            return choice

    _scry_cache[(nm_orig.lower(), cn_norm, set_hint)] = None
    return None

def _scry_img_disk_path(card_obj):
    # stable key from scryfall id if present; otherwise from normal image url
    key = card_obj.get("id") or ((card_obj.get("image_uris") or {}).get("normal")) or ""
    if not key:
        return None
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", key)[:96]
    return os.path.join(SC_IMG_CACHE_DIR, f"{safe}.jpg")

def _fetch_scry_image(card_obj):
    if not card_obj:
        return None

    # Stable cache key
    key = card_obj.get("id") or ((card_obj.get("image_uris") or {}).get("normal")) or ""
    if not key:
        return None

    # In-memory LRU cache first
    cached = _lru_get(_scry_img_cache, key)
    if cached is not None:
        return cached

    # Disk cache next
    disk_path = _scry_img_disk_path(card_obj)
    if disk_path and os.path.exists(disk_path):
        try:
            with open(disk_path, "rb") as f:
                data = np.frombuffer(f.read(), np.uint8)
            bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if bgr is not None:
                _lru_put(_scry_img_cache, key, bgr)
                return bgr
        except Exception:
            pass

    # Pick best URL
    url = None
    iu = card_obj.get("image_uris")
    if iu and iu.get("normal"):
        url = iu["normal"]
    elif card_obj.get("card_faces"):
        cf0 = card_obj["card_faces"][0]
        if cf0.get("image_uris") and cf0["image_uris"].get("normal"):
            url = cf0["image_uris"]["normal"]
    if not url:
        return None

    # Download and persist
    try:
        r = _HTTP.get(url, timeout=SCRYFALL_TIMEOUT)
        if r.status_code == 200:
            data = np.frombuffer(r.content, np.uint8)
            bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if bgr is not None:
                _lru_put(_scry_img_cache, key, bgr)
                if disk_path:
                    try:
                        with open(disk_path, "wb") as f:
                            f.write(r.content)
                    except Exception:
                        pass
                return bgr
    except Exception as e:
        _dbg("SCRYFALL ERROR", f"download failed: {e}")
    return None

# =========================
#MARK: MATCHING
# =========================
def _center_crop(img, pct=0.98):
    h, w = img.shape[:2]
    dw = int(w * (1-pct) * 0.5); dh = int(h * (1-pct) * 0.5)
    return img[dh:h-dh, dw:w-dw]

def _ahash64(img):
    if img is None or img.size == 0:
        return 0
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (8, 8), interpolation=cv2.INTER_AREA)
    m = float(g.mean())
    bits = (g > m).flatten()
    v = 0
    for i, b in enumerate(bits):
        if b:
            v |= (1 << i)
    return v


def _dhash64(img):
    gray = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (9,8), interpolation=cv2.INTER_AREA)
    diff = gray[:,1:] > gray[:,:-1]
    return sum(1<<i for i,bit in enumerate(diff.flatten()) if bit)

def _hamming64(a, b):
    return bin((a ^ b) & ((1<<64)-1)).count("1")

def _normalize_illum(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l2 = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l2,a,b]), cv2.COLOR_LAB2BGR)


def _orb_sim_v2_cached(A_orb, ref_feats):
    if _ORB is None or _BF is None:
        return 0.0, [], None
    gA = cv2.cvtColor(A_orb, cv2.COLOR_BGR2GRAY)
    kA, dA = _ORB.detectAndCompute(gA, None)
    dB = ref_feats.get("dB"); kB = ref_feats.get("kB")
    if dA is None or dB is None or len(kA or []) < 8 or len(kB or []) < 8:
        denom = max(len(kA) if kA else 0, len(kB) if kB else 0, 1)
        return float(0.0), [], None
    knn = _BF.knnMatch(dA, dB, k=2)
    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good.append(m)
    if len(good) < 6:
        denom = max(len(kA), len(kB), 1)
        return float(len(good)) / float(denom), good, None
    ptsA = np.float32([kA[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    ptsB = np.float32([kB[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, 3.0)
    inliers = int(mask.sum()) if mask is not None else 0
    denom = max(len(kA), len(kB), 1)
    return float(inliers) / float(denom), good, (kA, kB, good, mask)

def _orb_sim_v2(bgrA, bgrB):
    if _ORB is None or _BF is None:
        return 0.0, [], None
    gA = cv2.cvtColor(bgrA, cv2.COLOR_BGR2GRAY)
    gB = cv2.cvtColor(bgrB, cv2.COLOR_BGR2GRAY)
    kA, dA = _ORB.detectAndCompute(gA, None)
    kB, dB = _ORB.detectAndCompute(gB, None)
    if dA is None or dB is None or len(kA) < 8 or len(kB) < 8:
        denom = max(len(kA) if kA else 0, len(kB) if kB else 0, 1)
        return float(0.0), [], None

    knn = _BF.knnMatch(dA, dB, k=2)
    good = []
    for pair in knn:
        if len(pair) < 2: 
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good.append(m)
    if len(good) < 6:
        denom = max(len(kA), len(kB), 1)
        return float(len(good)) / float(denom), good, None

    ptsA = np.float32([kA[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    ptsB = np.float32([kB[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, 3.0)
    inliers = int(mask.sum()) if mask is not None else 0
    denom = max(len(kA), len(kB), 1)
    return float(inliers) / float(denom), good, (kA, kB, good, mask)



def compare_snapshot_to_scryfall(snap_bgr, scry_bgr, return_details=False, scry_card=None):
    if not MATCH_ENABLE or snap_bgr is None or scry_bgr is None:
        return (0.0, True, {}) if return_details else (0.0, True)

    tgt = (min(640, CARD_W), min(900, CARD_H))

    # 1) crops
    A0 = _center_crop(snap_bgr, 0.98); B0 = _center_crop(scry_bgr, 0.98)
    if MATCH_USE_ART:
        # Use the same static art-window crop for both images so the visual comparison lines up 1:1.
        A0 = _crop_art(A0, allow_ai=False); B0 = _crop_art(B0, allow_ai=False)

    ref_feats = _get_cached_reference_feats(scry_card, scry_bgr, use_art=MATCH_USE_ART, tgt=tgt) or {}
    def _score_pair(A_img, B_img, use_ref=True, hist_mode="hs", label="base"):
        A_orb = cv2.resize(_normalize_illum(A_img), tgt, interpolation=cv2.INTER_AREA)
        A_hist = cv2.resize(A_img, tgt, interpolation=cv2.INTER_AREA)

        # Hash similarity
        ah = _ahash64(A_orb)
        dh = _dhash64(A_orb)
        ah2 = dh2 = None
        if use_ref:
            ah2 = ref_feats.get("ah2"); dh2 = ref_feats.get("dh2")
        if ah2 is None or dh2 is None:
            B_orb_tmp = cv2.resize(_normalize_illum(B_img), tgt, interpolation=cv2.INTER_AREA)
            ah2 = _ahash64(B_orb_tmp); dh2 = _dhash64(B_orb_tmp)
        sim_hash = 1.0 - min(_hamming64(ah, ah2), _hamming64(dh, dh2)) / 64.0

        # Histogram similarity
        if hist_mode == "hs":
            hsvA = cv2.cvtColor(A_hist, cv2.COLOR_BGR2HSV)
            hA = cv2.calcHist([hsvA],[0],None,[32],[0,180]); sA = cv2.calcHist([hsvA],[1],None,[32],[0,256])
            cv2.normalize(hA, hA); cv2.normalize(sA, sA)
            hB = sB = None
            if use_ref:
                hB = ref_feats.get("hB"); sB = ref_feats.get("sB")
            if hB is None or sB is None:
                hsvB = cv2.cvtColor(cv2.resize(B_img, tgt, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV)
                hB = cv2.calcHist([hsvB],[0],None,[32],[0,180]); sB = cv2.calcHist([hsvB],[1],None,[32],[0,256])
                cv2.normalize(hB, hB); cv2.normalize(sB, sB)
            cH = cv2.compareHist(hA, hB, cv2.HISTCMP_CORREL)
            cS = cv2.compareHist(sA, sB, cv2.HISTCMP_CORREL)
            sim_hist = float((cH + 1.0) * 0.5 * 0.6 + (cS + 1.0) * 0.5 * 0.4)
        else:  # luminance-only for foil-heavy shots
            gA = cv2.cvtColor(A_hist, cv2.COLOR_BGR2GRAY)
            gB = cv2.cvtColor(cv2.resize(B_img, tgt, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
            hA = cv2.calcHist([gA],[0],None,[32],[0,256]); hB = cv2.calcHist([gB],[0],None,[32],[0,256])
            cv2.normalize(hA, hA); cv2.normalize(hB, hB)
            sim_hist = float((cv2.compareHist(hA, hB, cv2.HISTCMP_CORREL) + 1.0) * 0.5)

        # ORB (use cached reference keypoints/descriptors if available)
        orb_dbg = None
        orb_B_vis = None
        ref_kits_ready = use_ref and ref_feats.get("kB") is not None and ref_feats.get("dB") is not None
        if ref_kits_ready:
            sim_orb, good, orb_dbg = _orb_sim_v2_cached(A_orb, ref_feats)
            orb_B_vis = ref_feats.get("B_orb")
        else:
            B_orb = cv2.resize(_normalize_illum(B_img), tgt, interpolation=cv2.INTER_AREA)
            sim_orb, good, orb_dbg = _orb_sim_v2(A_orb, B_orb)
            orb_B_vis = B_orb

        score = MATCH_W_HASH * sim_hash + MATCH_W_HIST * sim_hist + MATCH_W_ORB * sim_orb
        orb_gate = bool(MATCH_REQUIRE_ORB and sim_orb <= (MATCH_ORB_FAIL_THRESHOLD + 1e-6))
        ok = False if orb_gate else bool(score >= MATCH_TH)
        return {
            "label": label,
            "sim_hash": float(sim_hash),
            "sim_hist": float(sim_hist),
            "sim_orb":  float(sim_orb),
            "score":    float(score),
            "ok":       ok,
            "orb_gate": orb_gate,
            "orb_dbg": {"A": A_orb, "B": orb_B_vis, "orb_data": orb_dbg},
        }

    base = _score_pair(A0, B0, use_ref=True, hist_mode="hs", label="base")

    # Optional foil-tolerant path: specular suppression + luminance histogram.
    alt_paths = []
    use_alt = (not base["ok"]) or return_details
    if use_alt:
        A_foil = _foil_match_normalize(A0)
        B_foil = _foil_match_normalize(B0)
        foil_res = _score_pair(A_foil, B_foil, use_ref=False, hist_mode="luma", label="foil_norm")
        alt_paths.append({k: foil_res[k] for k in ("label","sim_hash","sim_hist","sim_orb","score","ok","orb_gate")})
    else:
        foil_res = None

    best = base
    if foil_res:
        better = False
        if foil_res["ok"] and not base["ok"]:
            better = True
        elif foil_res["score"] > base["score"] + 0.02:
            better = True
        if better:
            best = foil_res

    if not return_details:
        return float(best["score"]), bool(best["ok"])

    details = {
        "sim_hash": float(best["sim_hash"]),
        "sim_hist": float(best["sim_hist"]),
        "sim_orb":  float(best["sim_orb"]),
        "score":    float(best["score"]),
        "ok":       bool(best["ok"]),
        "orb_gate": bool(best["orb_gate"]),
        "weights": {"hash": MATCH_W_HASH, "hist": MATCH_W_HIST, "orb": MATCH_W_ORB},
        "orb_dbg": best.get("orb_dbg", {}),
        "snap": A0,   # center-cropped (and art-cropped if MATCH_USE_ART)
        "scry": B0,
        # Keep full originals so visualization can fall back if crops are tiny
        "snap_full": snap_bgr,
        "scry_full": scry_bgr,
        "use_art": bool(MATCH_USE_ART),
        "match_path": best.get("label", "base"),
    }
    if alt_paths:
        details["alt_paths"] = alt_paths
    return float(best["score"]), bool(best["ok"]), details


def _cmp_stats_sanitized(src):
    """Strip numpy arrays and debug-only fields from compare stats so they are JSON-safe."""
    out = {}
    for k, v in (src or {}).items():
        if k in ("orb_dbg", "snap", "scry", "snap_full", "scry_full"):
            continue
        if hasattr(v, "shape") or hasattr(v, "dtype"):
            continue
        out[k] = v
    return out


def _crop_art(img, allow_ai=True):
    if img is None or img.size == 0:
        return img
    # Avoid reusing stale AI detections on reference images; only trust AI when explicitly allowed.
    card = _ai_crop(img, "card") if (allow_ai and _ai_enabled()) else None
    base = card if card is not None and getattr(card, "size", 0) > 0 else img
    h, w = base.shape[:2]
    # Approximate the printed art box using fixed fractions of the card crop.
    x0, y0, x1, y1 = (0.065, 0.155, 0.935, 0.590)
    X0, Y0, X1, Y1 = int(x0*w), int(y0*h), int(x1*w), int(y1*h)
    art = base[Y0:Y1, X0:X1] if (X1 > X0 and Y1 > Y0) else None
    # Guardrail: if the art crop is too small relative to the card, fall back to full card crop.
    try:
        min_side = min(h, w)
        if art is None or art.shape[0] < ART_MIN_FRAC * min_side or art.shape[1] < ART_MIN_FRAC * min_side:
            if COMPARE_DEBUG:
                _dbg("COMPARE DBG", f"art crop too small; falling back. art={None if art is None else art.shape} base={base.shape}")
            return base
    except Exception:
        return base
    return art

def _render_compare_visual(details):
    try:
        # Prefer the original crops for visualization (full card or art-only),
        # fall back to the ORB-normalized images if needed.
        # NOTE: Avoid using `or` with numpy arrays; their truthiness raises ValueError.
        A = details.get("snap")
        if A is None:
            A = (details.get("orb_dbg") or {}).get("A")
        B = details.get("scry")
        if B is None:
            B = (details.get("orb_dbg") or {}).get("B")
        use_art = bool(details.get("use_art"))
        # Force art crops from the originals when art mode is enabled, using the same
        # static art-window crop for both sides so they align visually.
        if use_art:
            snap_full = details.get("snap_full")
            scry_full = details.get("scry_full")
            try:
                if snap_full is not None and getattr(snap_full, "size", 0) > 0:
                    A = _crop_art(_center_crop(snap_full, 0.98), allow_ai=False)
                elif A is not None and getattr(A, "size", 0) > 0:
                    A = _crop_art(_center_crop(A, 0.98), allow_ai=False)
            except Exception:
                pass
            try:
                if scry_full is not None and getattr(scry_full, "size", 0) > 0:
                    B = _crop_art(_center_crop(scry_full, 0.98), allow_ai=False)
                elif B is not None and getattr(B, "size", 0) > 0:
                    B = _crop_art(_center_crop(B, 0.98), allow_ai=False)
            except Exception:
                pass
        if not use_art:
            if A is not None and A.shape[0] < 300:
                A_full = details.get("snap_full")
                if A_full is not None and A_full.shape[0] > A.shape[0]:
                    A = A_full
            if B is not None and B.shape[0] < 300:
                B_full = details.get("scry_full")
                if B_full is not None and B_full.shape[0] > B.shape[0]:
                    B = B_full
        else:
            # Art mode: still avoid pathological tiny crops by falling back to full card if under absolute floor.
            if A is not None and A.shape[0] < 180:
                A_full = details.get("snap_full")
                if A_full is not None and A_full.shape[0] > A.shape[0]:
                    if COMPARE_DEBUG:
                        _dbg("COMPARE DBG", f"art crop A very small ({A.shape}); falling back to full card {A_full.shape}")
                    A = A_full
            if B is not None and B.shape[0] < 180:
                B_full = details.get("scry_full")
                if B_full is not None and B_full.shape[0] > B.shape[0]:
                    if COMPARE_DEBUG:
                        _dbg("COMPARE DBG", f"art crop B very small ({B.shape}); falling back to full card {B_full.shape}")
                    B = B_full
        orb_dbg = details.get("orb_dbg") or {}
        orb_data = orb_dbg.get("orb_data")
        if A is None or B is None:
            try:
                _dbg("COMPARE VIS", f"missing A/B in details; keys={list(details.keys())}")
            except Exception:
                pass
            return None
        # Resize B to match A's height for a side-by-side view.
        hA, wA = A.shape[:2]
        hB, wB = B.shape[:2]
        try:
            _dbg("COMPARE VIS", f"A_shape={A.shape} B_shape={B.shape} use_art={use_art} orb_data={'y' if orb_data is not None else 'n'}")
        except Exception:
            pass
        if hB != hA:
            scale = hA / float(max(hB, 1))
            B = cv2.resize(B, (int(wB * scale), hA), interpolation=cv2.INTER_AREA)
        if orb_data is not None and orb_dbg.get("A") is not None and orb_dbg.get("B") is not None:
            # Draw match lines using the ORB keypoints, but overlay on the color crops.
            kA, kB, good, mask = orb_data
            matchesMask = mask.ravel().tolist() if mask is not None else None
            flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            vis = cv2.drawMatches(orb_dbg["A"], kA, orb_dbg["B"], kB, good, None,
                                  matchesMask=matchesMask, flags=flags)
        else:
            vis = cv2.hconcat([A, B])
            if COMPARE_DEBUG:
                try:
                    _dbg("COMPARE DBG", f"orb lines missing; orb_data={orb_data is not None} len_good={len(orb_data[2]) if orb_data is not None else 0} kA={len((orb_data or [None,None,[ ]])[0] or []) if orb_data else 0} kB={len((orb_data or [None,None,[ ]])[1] or []) if orb_data else 0} use_art={use_art}")
                except Exception:
                    pass
        t = f"HASH {details['sim_hash']:.2f} • HIST {details['sim_hist']:.2f} • ORB {details['sim_orb']:.2f}  ⇒  SCORE {details['score']:.3f}  ({'MATCH' if details['ok'] else 'REVIEW'})"
        pad = 36
        canvas = np.zeros((vis.shape[0]+pad, vis.shape[1], 3), dtype=np.uint8)
        canvas[:,:] = (18,18,24)
        canvas[pad:pad+vis.shape[0], :vis.shape[1]] = vis
        cv2.putText(canvas, t, (10,24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230,230,230), 1, cv2.LINE_AA)
        # Debug: if the visual is suspiciously small, save it for inspection.
        try:
            if COMPARE_DEBUG and (canvas.shape[0] < 400 or canvas.shape[1] < 800):
                _dbg("COMPARE DBG", f"tiny compare visual h={canvas.shape[0]} w={canvas.shape[1]} A={A.shape if A is not None else None} B={B.shape if B is not None else None}")
                _debug_save_compare(canvas, details, note="tiny_render")
        except Exception:
            pass
        return canvas
    except Exception:
        return None

def _publish_compare_visual(details):
    img = _render_compare_visual(details)
    if img is None:
        return
    _update_compare_cache(img, {k: v for k, v in details.items() if k != "orb_dbg"})

# =========================
#MARK: THREADS / WORKERS
# =========================

# =========================
#MARK: AI / YOLOv5 ROI DETECTION
# =========================
_ai_lock = threading.Lock()
_ai_last = {"time": 0.0, "boxes": {}, "raw": []}

@contextmanager
def _preserve_ai_rois():
    """Snapshot AI ROI boxes and restore after a non-live operation (e.g., reprocess)."""
    with _ai_lock:
        saved = {
            "time": _ai_last.get("time", 0.0),
            "boxes": dict(_ai_last.get("boxes", {})),
            "raw": list(_ai_last.get("raw", [])),
        }
    try:
        yield
    finally:
        with _ai_lock:
            _ai_last["time"] = saved.get("time", 0.0)
            _ai_last["boxes"] = dict(saved.get("boxes", {}))
            _ai_last["raw"] = list(saved.get("raw", []))
_ai_model_loaded = False

# Map keys -> colors (BGR) for overlay
_AI_COLORS = {
    "name": (255, 0, 0),          # blue
    "mana_value": (0, 165, 255),  # orange
    "set_symbol": (0, 255, 0),    # green
    "card": (0, 0, 255),          # red
    "set_name": (255, 0, 255),    # purple
}

_AI_FALLBACK_KEYMAP = {
    "1-name": "name",
    "name": "name",
    "card name": "name",
    "title": "name",
    "2-mana value": "mana_value",
    "mana value": "mana_value",
    "3-set symbol": "set_symbol",
    "set symbol": "set_symbol",
    "4-card": "card",
    "4-card -": "card",
    "card": "card",
    "5-set name": "set_name",
    "set name": "set_name",
    "set_name": "set_name",
}

def _ai_enabled():
    try:
        return bool(globals().get("AI_ENABLED", False) and globals().get("AI_USE_FOR_ROIS", False))
    except Exception:
        return False

def _ai_class_to_roi_key(label: str):
    """Map a YOLO class label to our internal ROI key, with robust fallbacks."""
    base = str(label or "").strip()
    try:
        # exact mapping from config if provided
        if isinstance(AI_CLASS_TO_KEY, dict) and base in AI_CLASS_TO_KEY:
            return AI_CLASS_TO_KEY[base]
    except Exception:
        pass
    try:
        norm = re.sub(r"[^a-z0-9]+", " ", base.lower()).strip()
    except Exception:
        norm = str(base or "").lower().strip()
    # normalized mapping from config (handles "1-Name" vs "1 name" etc.)
    try:
        if isinstance(AI_CLASS_TO_KEY, dict):
            for raw, mapped in AI_CLASS_TO_KEY.items():
                try:
                    if norm == re.sub(r"[^a-z0-9]+", " ", str(raw or "").lower()).strip():
                        return mapped
                except Exception:
                    continue
    except Exception:
        pass
    return _AI_FALLBACK_KEYMAP.get(norm)

def _ai_load():
    """Lazy-load YOLOv5 model."""
    global _ai_model_loaded, _ai_model, _ai_fn, _ai_letterbox, _ai_scale_boxes, _ai_device, _ai_names, _ai_stride
    if _ai_model_loaded:
        return True
    if not _ai_enabled():
        return False
    try:
        import sys
        sys.path.insert(0, str(YOLOV5_DIR))
        import torch
        from models.common import DetectMultiBackend
        from utils.torch_utils import select_device
        from utils.general import non_max_suppression, scale_boxes
        from utils.augmentations import letterbox
        device = select_device("")
        model = DetectMultiBackend(AI_MODEL_PATH, device=device, dnn=False, data=None, fp16=False)
        _ai_stride = int(getattr(model, "stride", 32))
        _ai_names = getattr(model, "names", None) or AI_CLASS_NAMES
        _ai_device = device
        _ai_model = model
        _ai_fn = non_max_suppression
        _ai_scale_boxes = scale_boxes
        _ai_letterbox = letterbox
        _ai_model_loaded = True
        _dbg("AI", f"Loaded YOLOv5 model: {_ai_names}")
        return True
    except Exception as e:
        _dbg("AI ERROR", f"Failed to load YOLOv5: {e}")
        _ai_model_loaded = False
        return False

def _ai_detect_boxes(bgr):
    """Run detection on the given card crop (BGR). Returns dict key->norm_box."""
    if bgr is None or bgr.size == 0:
        return {}
    if not _ai_load():
        return {}
    try:
        import torch
        _perf_dbg = bool(globals().get("PERF_TIMING_DEBUG", True))
        _t0 = time.perf_counter()
        img0 = bgr
        # Letterbox
        im = _ai_letterbox(img0, AI_IMG_SIZE, stride=_ai_stride, auto=True)[0]
        im = im.transpose((2, 0, 1))[::-1]  # BGR->RGB, to CHW
        im = np.ascontiguousarray(im)
        im = torch.from_numpy(im).to(_ai_device)
        im = im.float()  # uint8 to fp32
        im /= 255.0
        if im.ndimension() == 3:
            im = im.unsqueeze(0)
        pred = _ai_model(im, augment=False, visualize=False)
        pred = _ai_fn(pred, AI_CONF_THRES, AI_IOU_THRES, max_det=int(globals().get("AI_MAX_DETS", 50)))[0]
        h0, w0 = img0.shape[:2]
        out = {}
        raw = []
        if pred is not None and len(pred):
            pred[:, :4] = _ai_scale_boxes(im.shape[2:], pred[:, :4], img0.shape).round()
            for *xyxy, conf, cls in pred.tolist():
                x0, y0, x1, y1 = xyxy
                x0 = max(0, min(w0 - 1, int(x0))); x1 = max(0, min(w0 - 1, int(x1)))
                y0 = max(0, min(h0 - 1, int(y0))); y1 = max(0, min(h0 - 1, int(y1)))
                if x1 <= x0 or y1 <= y0:
                    continue
                # Normalize
                nb = [x0 / w0, y0 / h0, x1 / w0, y1 / h0]
                # Map to our key
                name = str(_ai_names[int(cls)]) if int(cls) < len(_ai_names) else str(int(cls))
                key = _ai_class_to_roi_key(name)
                raw.append({"name": name, "key": key, "conf": float(conf), "box": nb})
                if key:
                    # Keep highest confidence per key
                    if key not in out or float(conf) > out[key][4]:
                        out[key] = [nb[0], nb[1], nb[2], nb[3], float(conf)]
        with _ai_lock:
            _ai_last["time"] = time.time()
            _ai_last["boxes"] = out
            _ai_last["raw"] = raw
        if _perf_dbg:
            try:
                _dbg("PERF AI", f"det={time.perf_counter() - _t0:.3f}s boxes={len(out)} size={img0.shape[1]}x{img0.shape[0]}")
            except Exception:
                pass
        return out
    except Exception as e:
        _dbg("AI ERROR", f"Inference failed: {e}")
        return {}

def _ai_get_norm_box(key):
    with _ai_lock:
        b = _ai_last.get("boxes", {}).get(key)
    if b is None:
        return None
    import numpy as np
    arr = np.asarray(b, dtype=float).reshape(-1)
    if arr.size < 4:
        return None
    x0, y0, x1, y1 = arr[:4]
    if x1 <= x0 or y1 <= y0:
        return None
    conf = float(arr[4]) if arr.size >= 5 else 1.0
    if key == "card":
        if conf < _AI_CARD_MIN_CONF:
            return None
        area = max(0.0, (x1 - x0) * (y1 - y0))
        if (not _AI_ROI_STRICT) and area < _AI_CARD_MIN_AREA:
            return None
        if (not _AI_ROI_STRICT) and y0 > _AI_CARD_MAX_TOP:
            return None
    elif key == "set_name":
        # For set-name / set-code OCR we trust the YOLO box
        # as-is and only enforce a minimal confidence.
        if conf < _AI_SETNAME_MIN_CONF:
            return None
    return [float(x0), float(y0), float(x1), float(y1)]

def _ai_get_norm_box_relaxed(key):
    """Return the raw box without confidence/area clamps (used for number OCR)."""
    with _ai_lock:
        b = _ai_last.get("boxes", {}).get(key)
    if b is None:
        return None
    import numpy as np
    arr = np.asarray(b, dtype=float).reshape(-1)
    if arr.size < 4:
        return None
    x0, y0, x1, y1 = arr[:4]
    if x1 <= x0 or y1 <= y0:
        return None
    return [float(x0), float(y0), float(x1), float(y1)]

def _ai_crop(img, key, pad_x=0.0, pad_y=0.0):
    b = _ai_get_norm_box(key)
    if b is None:
        return None
    # expand the normalized box a bit to avoid clipping first/last letters
    x0,y0,x1,y1 = b
    x0 = max(0.0, x0 - float(pad_x)); x1 = min(1.0, x1 + float(pad_x))
    y0 = max(0.0, y0 - float(pad_y)); y1 = min(1.0, y1 + float(pad_y))
    return _roi_rel(img, (x0,y0,x1,y1))

def _ai_crop_exact(img, key):
    """
    Crop using the exact YOLO box for the given key (no padding).
    """
    b = _ai_get_norm_box(key)
    if b is None:
        return None
    x0, y0, x1, y1 = [float(v) for v in b[:4]]
    return _roi_rel(img, (x0, y0, x1, y1))



def _ai_crop_asym(img, key, pad_left=0.0, pad_right=0.0, pad_top=0.0, pad_bottom=0.0):
    b = _ai_get_norm_box(key)
    if b is None:
        return None
    x0,y0,x1,y1 = [float(v) for v in b]
    x0 = max(0.0, x0 - float(pad_left));  x1 = min(1.0, x1 + float(pad_right))
    y0 = max(0.0, y0 - float(pad_top));   y1 = min(1.0, y1 + float(pad_bottom))
    return _roi_rel(img, (x0,y0,x1,y1))

def _ai_crop_asym_relaxed(img, key, pad_left=0.0, pad_right=0.0, pad_top=0.0, pad_bottom=0.0):
    """
    Asymmetric crop that ignores confidence/area clamps (useful when YOLO is unsure).
    """
    b = _ai_get_norm_box_relaxed(key)
    if b is None:
        return None
    x0, y0, x1, y1 = [float(v) for v in b[:4]]
    x0 = max(0.0, x0 - float(pad_left));  x1 = min(1.0, x1 + float(pad_right))
    y0 = max(0.0, y0 - float(pad_top));   y1 = min(1.0, y1 + float(pad_bottom))
    return _roi_rel(img, (x0, y0, x1, y1))

def _ai_crop_relaxed(img, key, pad_x=0.0, pad_y=0.0):
    """
    Crop using the YOLO box for the given key but without rejecting low-conf/offset cards.
    """
    b = _ai_get_norm_box_relaxed(key)
    if b is None:
        return None
    x0, y0, x1, y1 = [float(v) for v in b[:4]]
    x0 = max(0.0, x0 - float(pad_x)); x1 = min(1.0, x1 + float(pad_x))
    y0 = max(0.0, y0 - float(pad_y)); y1 = min(1.0, y1 + float(pad_y))
    return _roi_rel(img, (x0, y0, x1, y1))

def _update_ai_rois(card_bgr):
    if not _ai_enabled():
        return
    _ai_detect_boxes(card_bgr)

def _draw_ai_overlay(img):
    if img is None or img.size == 0:
        return
    with _ai_lock:
        boxes = dict(_ai_last.get("boxes", {}))
        raw = list(_ai_last.get("raw", []))
    if not boxes and not raw:
        return
    h, w = img.shape[:2]

    box_th   = int(globals().get("AI_BOX_THICKNESS", 5))
    font_s   = float(globals().get("AI_LEGEND_FONT_SCALE", 0.7))
    font_th  = int(globals().get("AI_LEGEND_FONT_THICKNESS", 1))
    line_th  = int(globals().get("AI_LEGEND_LINE_THICKNESS", 5))
    pad      = int(globals().get("AI_LEGEND_PAD", 10))

    # draw boxes
    for key, val in boxes.items():
        x0n = float(val[0]); y0n = float(val[1]); x1n = float(val[2]); y1n = float(val[3])
        x0 = int(x0n*w); y0 = int(y0n*h); x1 = int(x1n*w); y1 = int(y1n*h)
        color = _AI_COLORS.get(key, (255,255,255))
        cv2.rectangle(img, (x0, y0), (x1, y1), color, box_th)
        try:
            conf = float(val[4]) if len(val) >= 5 else 0.0
            label = f"{key} {conf*100:.0f}%"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_s, font_th)
        except Exception:
            pass

    # legend with dynamic spacing
    x, y = pad, pad
    legend = [("name","1-Name"),("mana_value","2-Mana Value"),
              ("set_symbol","3-Set Symbol"),("card","4-Card -"),
              ("set_name","5-Set Name")]
    for key, label in legend:
        color = _AI_COLORS.get(key, (255,255,255))
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_s, font_th)
        mid = y + th // 2
        cv2.line(img, (x, mid), (x+24, mid), color, line_th)
        cv2.putText(img, label, (x+30, y + th),
                    cv2.FONT_HERSHEY_SIMPLEX, font_s, (220,220,220), font_th, cv2.LINE_AA)
        y += th + max(8, int(0.5*th))   # <— spacing scales with font size

def _open_camera():
    try:
        cap = cv2.VideoCapture(int(CAMERA_DEVICE), cv2.CAP_V4L2)
    except ValueError:
        cap = cv2.VideoCapture(CAMERA_DEVICE, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*FOURCC_PRIMARY))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, REQ_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, REQ_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, REQ_FPS)
    # PERF: keep camera queue short to avoid backlog when CPU spikes
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    if not cap.isOpened() or cap.get(cv2.CAP_PROP_FPS) == 0:
        cap.release()
        cap = cv2.VideoCapture(int(CAMERA_DEVICE) if str(CAMERA_DEVICE).isdigit() else CAMERA_DEVICE, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*FOURCC_FALLBACK))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, REQ_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, REQ_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, REQ_FPS)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
    return cap if cap.isOpened() else None

def video_thread():
    global cap, output_frame, latest_frame_raw, current_card_crop, scanner_state
    cap = _open_camera()
    if not cap:
        _dbg("CAMERA ERROR", f"Could not open camera {CAMERA_DEVICE}")
        return
    _dbg("CAMERA INFO", f"{int(cap.get(3))}x{int(cap.get(4))} @ {cap.get(5):.1f}fps")
    scanner_state.update({'locked': False, 'locked_frames': 0, 'locked_area': 0, 'steady': False, 'steady_relaxed': False, 'steady_promoted': False})
    frame_i = 0

    try:
        while not shutdown_evt.is_set():
            if PROC_PAUSED and not LIVE_CROP_WHILE_PAUSED:
                time.sleep(0.03)
                continue

            ok, frame = cap.read()
            if not ok:
                time.sleep(0.02)
                continue
            # hand off latest frame to detector without blocking
            try:
                _frame_q.clear(); _frame_q.append(frame)
            except Exception:
                pass

            # Draw using last-known tracks; detector updates them asynchronously
            out = frame.copy()

            # Compute lock purely for overlay; state is set by detector
            try:
                with tracks_lock:
                    locked = next((t for t in tracks.values() if t.get('seen', 0) >= CONFIRM_FRAMES), None)
                if locked:
                    quad = None
                    try:
                        with video_lock:
                            quad = current_card_quad
                    except Exception:
                        quad = None
                    if quad is None:
                        quad = locked['box']
                    cv2.polylines(out, [quad.astype(int)], True, (0, 255, 0), 2)
            except Exception:
                pass

            with video_lock:
                latest_frame_raw = frame
                output_frame = out
    finally:
        if cap:
            cap.release()
            print("Camera released.")


def detect_worker():
    """Run card detection/tracking on the latest frame without ever blocking capture/preview.
    Throttled to DETECT_MAX_FPS and, unless LIVE_CROP_WHILE_PAUSED is enabled, skips work while PROC_PAUSED is active.
    Updates: tracks (with tracks_lock), current_card_crop, scanner_state.
    """
    global _last_detect_ts, current_card_crop, current_card_quad, scanner_state
    frame_i = 0
    ready_last = True
    manual_locked_frames = 0
    while not shutdown_evt.is_set():
        # pacing: time-gate
        now = time.time()
        min_dt = 1.0 / max(1, int(DETECT_MAX_FPS))
        if (now - _last_detect_ts) < min_dt:
            time.sleep(0.001)
            continue

        if PROC_PAUSED and not LIVE_CROP_WHILE_PAUSED:
            time.sleep(0.02)
            continue

        if not _printer_ready_for_detection():
            if ready_last:
                with tracks_lock:
                    tracks.clear()
                try:
                    card_tracker.reset()
                except Exception:
                    pass
                with video_lock:
                    scanner_state.update({'locked': False, 'locked_frames': 0, 'locked_area': 0.0,
                                          'steady': False, 'steady_relaxed': False, 'steady_promoted': False})
                    current_card_crop = None
                    current_card_quad = None
                manual_locked_frames = 0
            ready_last = False
            time.sleep(0.025)
            continue
        ready_last = True

        # get latest frame if available
        if not _frame_q:
            time.sleep(0.005)
            continue
        frame = _frame_q[-1]
        _last_detect_ts = time.time()

        # Manual crop path: bypass detection entirely and use saved quad
        if manual_crop_enabled:
            with tracks_lock:
                tracks.clear()
            manual_q = _manual_quad_px(frame.shape) if frame is not None else None
            manual_crop = warp_card(frame, manual_q) if manual_q is not None else None
            foil_now = scanner_state.get("foil")
            foil_score = scanner_state.get("foil_score")
            if FOIL_DETECT and manual_crop is not None and (frame_i % max(1, int(FOIL_EVERY_N_FRAMES)) == 0):
                try:
                    isf, score = _detect_foil_card(manual_crop)
                    prev = float(scanner_state.get("foil_score", 0.0))
                    sm = 0.7 * prev + 0.3 * score if prev else score
                    foil_now = (sm >= FOIL_ON_TH) or (bool(scanner_state.get("foil", False)) and sm >= FOIL_OFF_TH)
                    foil_score = round(sm, 2)
                except Exception:
                    pass
            # Update ROI detector so OCR uses the correct regions even in manual mode
            if _ai_enabled() and manual_crop is not None and not bool(globals().get("AI_ROIS_SNAPSHOT_ONLY", False)):
                try:
                    _update_ai_rois(manual_crop)
                except Exception:
                    pass
            if manual_q is None or manual_crop is None:
                manual_locked_frames = 0
            else:
                manual_locked_frames += 1
            with video_lock:
                current_card_quad = manual_q
                current_card_crop = manual_crop
                scanner_state.update({
                    'locked': manual_q is not None,
                    'locked_frames': manual_locked_frames if manual_q is not None else 0,
                    'locked_area': _quad_area(manual_q) if manual_q is not None else 0.0,
                    'steady': bool(manual_q is not None),
                    'steady_relaxed': False,
                    'steady_promoted': False,
                    'foil': foil_now,
                    'foil_score': foil_score
                })
            frame_i += 1
            continue

        # run detection (respect existing cadence param too)
        dets = []
        try:
            # Optionally skip based on DETECT_EVERY_N_FRAMES count
            if (frame_i % max(1, int(DETECT_EVERY_N_FRAMES))) == 0:
                dets = detect_cards(frame)
                # clear last detect error if any
                scanner_state.pop('detect_error', None)
        except Exception as e:
            dets = []
            scanner_state['detect_error'] = str(e)

        # Associate to tracks
        with tracks_lock:
            for _, t in tracks.items():
                t['updated'] = False

            for d in dets:
                d['centroid'] = d['box'].mean(axis=0)
                best_id, best_dist = None, MAX_ASSOC_DIST
                for tid, t in list(tracks.items()):
                    dist = float(np.hypot(*(d['centroid'] - t['centroid'])))
                    if dist < best_dist:
                        best_dist, best_id = dist, tid
                if best_id is not None:
                    t = tracks[best_id]
                    if 'first_seen_ts' not in t:
                        t['first_seen_ts'] = time.time()
                    blend = min(0.99, max(0.0, float(TRACK_BOX_BLEND)))
                    t['box'] = (t['box'] * blend + d['box'] * (1.0 - blend)).astype(np.float32)
                    t['centroid'] = t['box'].mean(axis=0)
                    t['area'] = d.get('area', t.get('area', 0))
                    t['seen'] = t.get('seen', 0) + 1
                    t['missed'] = 0
                    t['updated'] = True
                else:
                    tracks[len(tracks)] = {
                        **d,
                        'box': d['box'].astype(np.float32),
                        'centroid': d['box'].mean(axis=0),
                        'area': d.get('area', 0),
                        'seen': 1,
                        'missed': 0,
                        'updated': True,
                        'first_seen_ts': time.time(),
                    }

            # drop stale
            stale_ids = [tid for tid, t in list(tracks.items()) if not t['updated']]
            for tid in stale_ids:
                t = tracks.get(tid)
                if not t:
                    continue
                t['missed'] = t.get('missed', 0) + 1
                if t['missed'] > STALE_FRAMES:
                    del tracks[tid]

            # Find locked
            locked = next((t for t in tracks.values() if t.get('seen', 0) >= CONFIRM_FRAMES), None)

        # Update crop + scanner state
        if locked is not None:
            tracker_quad = None
            tracker_steady = True
            try:
                tracker_quad, _, tracker_steady = card_tracker.update(locked['box'])
            except Exception:
                tracker_quad = None
                tracker_steady = True

            init_quad = tracker_quad if tracker_quad is not None else locked['box']
            ai_crop = None
            try:
                # refine the quad for cropping without affecting locking/steady logic
                prev_q = None
                prev_crop = None
                with video_lock:
                    prev_q = current_card_quad
                    prev_crop = current_card_crop
                refined_q, refined_warp = refine_card_quad(frame, init_quad=init_quad, prev_quad=prev_q)
                if refined_q is not None and refined_warp is not None:
                    if tracker_quad is not None:
                        tracker_arr = np.asarray(tracker_quad, dtype=np.float32)
                        refined_q = (0.65 * refined_q + 0.35 * tracker_arr).astype(np.float32)
                        refined_warp = warp_card(frame, refined_q)
                    crop = refined_warp
                    ai_crop = crop
                    use_quad = refined_q
                else:
                    fallback_quad = prev_q if prev_q is not None else init_quad
                    if fallback_quad is not None:
                        crop = warp_card(frame, fallback_quad)
                        ai_crop = crop
                        use_quad = fallback_quad
                    else:
                        crop = None
                        use_quad = None
            except Exception:
                crop = None
                ai_crop = None
                use_quad = None
            if use_quad is not None:
                stabilized = _stabilize_quad(use_quad, prev_q)
                if stabilized is not None:
                    use_quad = stabilized
                    crop = warp_card(frame, use_quad) if frame is not None else crop
                    if ai_crop is None:
                        ai_crop = crop
                delta = _quad_delta(use_quad, prev_q)
                if (not LIVE_CARD_CROP) and delta is not None and prev_crop is not None and crop is not None:
                    try:
                        if delta <= QUAD_FREEZE_THRESH:
                            crop = prev_crop.copy()
                        elif delta <= QUAD_BLEND_THRESH and prev_crop.shape == crop.shape:
                            crop = cv2.addWeighted(prev_crop, 0.6, crop, 0.4, 0)
                    except Exception:
                        pass

            with video_lock:
                current_card_crop = crop
                current_card_quad = use_quad
                locked_frames = int(locked.get('seen', 0))
                base_steady = locked_frames >= STEADY_MIN_FRAMES
                relax_ready = (
                    base_steady and STEADY_RELAX_FRAMES > 0 and
                    locked_frames >= (STEADY_MIN_FRAMES + STEADY_RELAX_FRAMES)
                )
                promote_ready = False
                promote_window = float(STEADY_PROMOTE_S or 0.0)
                if not tracker_steady and promote_window > 0.0:
                    try:
                        first_seen_ts = float(locked.get('first_seen_ts') or 0.0)
                    except Exception:
                        first_seen_ts = 0.0
                    if first_seen_ts > 0.0 and (time.time() - first_seen_ts) >= promote_window:
                        promote_ready = True
                steady = (base_steady and (bool(tracker_steady) or relax_ready)) or promote_ready
                relaxed_flag = bool((relax_ready or promote_ready) and not tracker_steady)
                scanner_state.update({
                    'locked': True,
                    'locked_frames': locked_frames,
                    'locked_area': locked.get('area', 0.0),
                    'steady': steady,
                    'steady_relaxed': relaxed_flag,
                    'steady_promoted': bool(promote_ready and not tracker_steady)
                })

            # Update YOLOv5 ROI detector (throttled)
            try:
                if _ai_enabled() and ai_crop is not None and not bool(globals().get("AI_ROIS_SNAPSHOT_ONLY", False)):
                    ai_every = int(globals().get('DETECT_AI_EVERY_N_FRAMES', 3)) or 3
                    if (frame_i % ai_every) == 0:
                        _update_ai_rois(ai_crop)
            except Exception:
                pass
            # optional foil detection (throttled)
            if FOIL_DETECT and (frame_i % max(1, int(FOIL_EVERY_N_FRAMES)) == 0) and crop is not None:
                try:
                    isf, score = _detect_foil_card(crop)
                    prev = float(scanner_state.get("foil_score", 0.0))
                    sm = 0.7 * prev + 0.3 * score if prev else score
                    was = bool(scanner_state.get("foil", False))
                    nowf = (sm >= FOIL_ON_TH) or (was and sm >= FOIL_OFF_TH)
                    scanner_state.update({"foil": nowf, "foil_score": round(sm, 2)})
                except Exception:
                    pass
            elif not FOIL_DETECT:
                scanner_state.pop("foil", None)
                scanner_state.pop("foil_score", None)
        else:
            try:
                card_tracker.update(None)
            except Exception:
                pass
            with video_lock:
                current_card_crop = None
                current_card_quad = None
                scanner_state.update({'locked': False, 'locked_frames': 0, 'locked_area': 0.0, 'steady': False, 'steady_relaxed': False, 'steady_promoted': False})

        frame_i += 1
def ocr_worker():
    _get_rapid_engine()
    last_hash = None
    last_seq  = -1
    # PERF: throttle live OCR so it doesn't hammer CPU every frame
    last_live_ocr_ts = 0.0
    while not shutdown_evt.is_set():
        time.sleep(0.05)
        snap = None
        local_seq = None
        is_snapshot = False
        with scan_lock:
            if scanned_card is not None:
                snap = scanned_card.copy()
                local_seq = snapshot_seq
                is_snapshot = True
        if snap is None and not OCR_ONLY_ON_SNAPSHOT:
            with video_lock:
                if current_card_crop is not None:
                    # Only consider live OCR when allowed (steady, interval)
                    if LIVE_OCR_ONLY_WHEN_STEADY:
                        if not bool(scanner_state.get("steady", False)):
                            # not steady: skip OCR this cycle
                            continue
                    now = time.time()
                    if (now - last_live_ocr_ts) < float(LIVE_OCR_MIN_INTERVAL):
                        # too soon: skip
                        continue
                    snap = current_card_crop.copy()
                    is_snapshot = False

        if snap is None:
            continue

        if local_seq is not None and local_seq != last_seq:
            last_hash = None
            last_seq  = local_seq
        cur = _ahash64(snap)
        if last_hash is not None and bin(cur ^ last_hash).count("1") < HASH_STABILITY_BITS:
            # small change; skip re-OCR
            if not is_snapshot:
                last_live_ocr_ts = time.time()  # still advance timer so we don't spin
            continue
        last_hash = cur

        _clear_ocr()
        _perf_start = time.perf_counter()
        prev_from_snapshot = getattr(_OCR_CONTEXT, "from_snapshot", False)
        _OCR_CONTEXT.from_snapshot = bool(is_snapshot)
        try:
            res = ocr_from_card_upright(snap)
            res["name"] = _maybe_correct_ocr_name(res["name_raw"], res["name_conf"])
            if DEBUG_OCR:
                if res["name"] != res["name_raw"]:
                    _dbg("OCR INFO", f"'{res['name_raw']}' -> '{res['name']}'")
                _dbg("OCR INFO", f"name='{res['name']}'(c:{res['name_conf']:.1f}) num='{res['number']}' raw='{res.get('number_raw','')}'(c:{res['number_conf']:.1f}) set='{res.get('set_hint','')}' foil={res.get('foil')} ({res.get('foil_score',0):.2f})")
            with ocr_lock:
                ocr_state.update(res)
                ocr_state["updated_at"] = time.time()
                ocr_state["seq"] = local_seq
        except Exception as e:
            with ocr_lock:
                ocr_state["last_error"] = str(e)
        finally:
            _OCR_CONTEXT.from_snapshot = prev_from_snapshot
            try:
                _dbg("PERF OCR", f"OCR worker total={time.perf_counter() - _perf_start:.3f}s seq={local_seq} snapshot={is_snapshot}")
            except Exception:
                pass
            # PERF: mark last run for live OCR throttling
            if not is_snapshot:
                last_live_ocr_ts = time.time()
        
        try:
            global PROC_PAUSED, PROC_PAUSE_SEQ
            snap_seq = None
            with ocr_lock:
                snap_seq = ocr_state.get("seq")
            number     = (ocr_state.get("number") or "").strip()
            if PROC_PAUSED and (PROC_PAUSE_SEQ is not None) and (snap_seq == PROC_PAUSE_SEQ):
                PROC_PAUSED = False
                PROC_PAUSE_SEQ = None
        except Exception:
            pass

def _clear_ocr(preserve_seq: bool = False):
    with ocr_lock:
        saved_seq = ocr_state.get("seq") if preserve_seq else None
        ocr_state.clear()
        ocr_state.update({
            "name": "", "name_raw": "", "name_conf": 0.0,
            "number": "", "number_raw": "", "number_conf": 0.0,
            "set_hint": "", "updated_at": time.time(),
            "provider": PRIMARY_PROVIDER_LABEL,
            "foil": False, "foil_score": 0.0,
            "match_score": 0.0, "match_ok": None, "flagged": False,
            "seq": saved_seq if preserve_seq else None,
        })
    with cardinfo_lock:
        cardinfo_state.clear()

def _scan_in_progress():
    """True only while a snapshot is actively being processed."""
    if PROC_PAUSED:
        return True
    # Keep a short guard window after a capture, but don't block history actions
    # just because a cached snapshot exists.
    with scan_lock:
        snap_present = scanned_card is not None
        snap_age = time.time() - float(last_scan_ts or 0.0)
    return bool(snap_present and snap_age < 1.0)

def _get_usd_to_cad():
    now = time.time()
    if _fx_cache.get("rate") and now < _fx_cache.get("exp", 0):
        return _fx_cache["rate"]
    try:
        r = requests.get(FX_URL, timeout=5)
        rate = float(r.json()["rates"]["CAD"])
        _fx_cache.update({"rate": rate, "exp": now + FX_TTL_SEC})
        return rate
    except Exception:
        return _fx_cache.get("rate", 1.3)

def _apply_scry_collector_number(choice):
    """Normalize/propagate collector numbers coming from Scryfall."""
    raw = ""
    disp = ""
    match_mode = ""
    choice_set = ""
    try:
        raw = str((choice or {}).get("collector_number") or "").strip()
    except Exception:
        raw = ""
    try:
        match_mode = str((choice or {}).get("_match_mode") or "")
    except Exception:
        match_mode = ""
    try:
        choice_set = str((choice or {}).get("set") or "").strip().lower()
    except Exception:
        choice_set = ""
    if raw:
        try:
            disp = _parse_collector_for_display(raw)
        except Exception:
            disp = raw.strip()
    if not disp:
        return "", ""
    try:
        with ocr_lock:
            cur_num = (ocr_state.get("number") or "").strip()
            cur_conf = float(ocr_state.get("number_conf") or 0.0)
            cur_disp = ""
            try:
                cur_disp = _parse_collector_for_display(cur_num) if cur_num else ""
            except Exception:
                cur_disp = cur_num
            digit_only = bool(re.fullmatch(r"\d{1,4}", cur_num or ""))
            frac_form  = bool(re.fullmatch(r"\d{1,4}/\d{1,4}", cur_num or ""))
            ocr_set_hint = str((ocr_state.get("set_hint") or "")).strip().lower()
            set_match = bool(choice_set and ocr_set_hint and choice_set == ocr_set_hint)
            mm = match_mode.lower()
            cn_modes = (
                "cn_set_only", "cn_only", "search_name_cn", "search_name_field_cn",
                "prints_refine", "fuzzy_prints_refine",
                "postfix_enforce_name_set_cn", "postfix_enforce_set_cn", "postfix_enforce_name_set"
            )
            trusted_by_mode = any(k in mm for k in cn_modes)
            trust_scry_num = trusted_by_mode or set_match
            mismatch = bool(cur_disp and disp and cur_disp != disp)
            needs_override = False
            weak_conf = (
                (digit_only and cur_conf < max(70.0, _FAST_NUM_CONF_EXIT))
                or (frac_form and cur_conf < 90.0)
                or (not (digit_only or frac_form))
            )
            if not cur_num or weak_conf:
                needs_override = True
            if mismatch:
                if trust_scry_num:
                    needs_override = True
                elif cur_conf < 75.0:
                    needs_override = True
            if set_match and not cur_num:
                needs_override = True
            if needs_override:
                fast_exit = float(globals().get("OCR_FAST_NUMBER_CONF_EXIT", 58.0))
                boost_floor = 90.0 if trust_scry_num else 78.0
                if mismatch and trust_scry_num:
                    boost_floor = max(boost_floor, 92.0)
                new_conf = max(cur_conf, fast_exit, boost_floor if (mismatch or trust_scry_num or not cur_num) else 0.0)
                ocr_state["number"] = disp
                ocr_state["number_raw"] = raw or disp
                ocr_state["number_conf"] = new_conf
                ocr_state["updated_at"] = max(ocr_state.get("updated_at", 0), time.time())
                try:
                    _dbg(
                        "OCR NUM",
                        f"Scryfall override -> '{disp}' (mode={match_mode or 'unknown'}, set_match={set_match}, was='{cur_num or '-'}' conf={cur_conf:.1f})"
                    )
                except Exception:
                    pass
    except Exception:
        pass
    return raw or disp, disp

def _set_choice(choice, seq=None, updated_at=None, usd_to_cad=None):
    """Install a Scryfall choice into cardinfo_state and optionally override OCR number."""
    try:
        rate = usd_to_cad if usd_to_cad is not None else _get_usd_to_cad()
    except Exception:
        rate = None
    ts = updated_at if updated_at is not None else time.time()
    with cardinfo_lock:
        cardinfo_state["scry"] = choice
        if rate:
            cardinfo_state["fx"] = {"rate": rate}
        cardinfo_state["last_updated"] = ts
    try:
        _apply_scry_collector_number(choice)
    except Exception:
        pass

def cardinfo_worker():
    last_key = None
    while not shutdown_evt.is_set():
        time.sleep(0.2)

        # Snapshot OCR fields
        with ocr_lock:
            name       = (ocr_state.get("name") or "").strip()
            number_raw = (ocr_state.get("number_raw") or "").strip()
            number     = (ocr_state.get("number") or "").strip()
            set_hint   = (ocr_state.get("set_hint") or "").strip().lower()
            updated_at = ocr_state.get("updated_at", 0)
            seq        = ocr_state.get("seq")

        # Only work once per OCR update; require at least name or number
        if not (name or number_raw) or updated_at == cardinfo_state.get("last_updated", 0):
            continue
        # FAST set+number lookup (most restrictive, fastest)
        if _fast_mode() and name and (number_raw or number) and set_hint:
            try:
                _old_timeout = globals().get("SCRYFALL_TIMEOUT", 4.0)
                globals()["SCRYFALL_TIMEOUT"] = float(FAST_SCRY_TIMEOUT)
                choice = _scryfall_lookup_once(name, number_raw or number, set_hint, seq)
            except Exception:
                choice = None
            finally:
                globals()["SCRYFALL_TIMEOUT"] = _old_timeout
            if choice is not None:
                _set_choice(choice, seq, updated_at=updated_at)
                continue

        key = (name, number_raw, set_hint, seq)
        if key == last_key:
            continue
        last_key = key

        choice = None

        # 1) Prefer OCR-provided set hint
        if set_hint:
            try:
                choice = _scryfall_lookup_once(name, number_raw, set_hint, seq)
            except Exception:
                choice = None

        # 2) If still no choice, try reading the set band from ROI
        if choice is None:
            roi_img = None
            with scan_lock:
                if scanned_card is not None:
                    roi_img = _slice_frac(_card_crop_or_full(scanned_card), 0.68, 1.0)
            if roi_img is None:
                with video_lock:
                    if current_card_crop is not None:
                        roi_img = _slice_frac(_card_crop_or_full(current_card_crop), 0.68, 1.0)
            if roi_img is not None:
                sh = _read_set_hint(roi_img)
                if sh:
                    try:
                        choice = _scryfall_lookup_once(name, number_raw, sh, seq)
                    except Exception:
                        choice = None

        # 3) Final fallback: no set constraint
        if choice is None:
            try:
                choice = _scryfall_lookup_once(name, number_raw, "", seq)
            except Exception:
                choice = None

        try:
            choice = _scry_fix_mismatch(choice, name, number_raw, set_hint)
        except Exception:
            pass

        usd_to_cad = _get_usd_to_cad()

        # Warm image cache (ignore failures)
        try:
            _ = _fetch_scry_image(choice)
        except Exception:
            pass

        # Publish
        _set_choice(choice, seq, updated_at=updated_at, usd_to_cad=usd_to_cad)

# =========================
#MARK: MOONRAKER WS + AUTOSCAN
# =========================

def _send_gcode(script, timeout=None):
    # Use a longer timeout for long-running moves/homing/macros; Moonraker blocks until completion.
    try:
        s = (script or "").strip().upper()
        long = s.startswith(("G28", "RUN_SORTER_INTERACTIVE", "SORTER_HOME", "SORTER_CYCLE_INTERACTIVE", "MOVE_TO_SCAN", "PICKUP_CARD"))
        to = timeout if timeout is not None else (120 if long else 60)
        _HTTP.post(HTTP_POST_URL, json={"script": script}, timeout=to)
    except Exception as e:
        _dbg("WEBSOCKET ERROR", f"gcode post failed: {e}")

def _try_parse_stack_measure(line: str) -> bool:
    """Parse gcode responses for stack heights/counts and update estimates."""
    try:
        if not line or not isinstance(line, str):
            return False
        text = line.strip()
        if not re.search(r"(STACK|DECK|REMEASURE|START_HEIGHT|HEIGHT)", text, re.IGNORECASE):
            return False
        status_line = bool(re.search(r"(REMEASURE|REMEASURED|STACKS)", text, re.IGNORECASE))
        tokens = re.findall(r'([A-Za-z0-9_]+)\s*[:=]\s*([+-]?\d+(?:\.\d+)?)(?:\s*mm)?', text, re.IGNORECASE)
        h1 = h2 = s1 = s2 = c1 = c2 = thick = None
        parsed_keyed = False
        for k, v in tokens:
            key = k.upper()
            val = float(v)
            if key in ("H1","STACK1","CUR1","CURRENT1","HEIGHT1","HGT1","S1H"):
                h1 = val; parsed_keyed = True
            elif key in ("H2","STACK2","CUR2","CURRENT2","HEIGHT2","HGT2","S2H"):
                h2 = val; parsed_keyed = True
            elif key in ("START1","BASE1","START_1","S1"):
                if status_line or key == "S1":
                    h1 = val
                else:
                    s1 = val
                    if h1 is None:
                        h1 = val
                parsed_keyed = True
            elif key in ("START2","BASE2","START_2","S2"):
                if status_line or key == "S2":
                    h2 = val
                else:
                    s2 = val
                    if h2 is None:
                        h2 = val
                parsed_keyed = True
            elif key in ("CARDS1","COUNT1","N1","LEFT1","REM1","REMAIN1"):
                c1 = int(round(val)); parsed_keyed = True
            elif key in ("CARDS2","COUNT2","N2","LEFT2","REM2","REMAIN2"):
                c2 = int(round(val)); parsed_keyed = True
            elif key in ("THICK","THICKNESS","CARDTHICKNESS","CARD_THICKNESS","T","NEW_T"):
                thick = val; parsed_keyed = True
        # Explicit patterns for "Start1 height = X" / "Start2 height = Y"
        try:
            m = re.search(r"START[\s_]*1[^0-9]*([+-]?\d+(?:\.\d+)?)", text, re.IGNORECASE)
            if m:
                v = float(m.group(1))
                if status_line:
                    if h1 is None:
                        h1 = v
                else:
                    s1 = v
                    if h1 is None:
                        h1 = v
                parsed_keyed = True
            m = re.search(r"START[\s_]*2[^0-9]*([+-]?\d+(?:\.\d+)?)", text, re.IGNORECASE)
            if m:
                v = float(m.group(1))
                if status_line:
                    if h2 is None:
                        h2 = v
                else:
                    s2 = v
                    if h2 is None:
                        h2 = v
                parsed_keyed = True
        except Exception:
            pass
        # Fallback simple patterns like "STACK1: 42.5"
        if h1 is None:
            m = re.search(r"STACK1[^0-9\-+]*([+-]?\d+(?:\.\d+)?)", text, re.IGNORECASE)
            if m: h1 = float(m.group(1))
        if h2 is None:
            m = re.search(r"STACK2[^0-9\-+]*([+-]?\d+(?:\.\d+)?)", text, re.IGNORECASE)
            if m: h2 = float(m.group(1))
        if thick is not None:
            _update_stack_config(thickness=thick)
        try:
            _dbg("STACKS", f"parsed line h1={h1} h2={h2} s=({s1},{s2}) c=({c1},{c2}) thick={thick} raw='{text}'")
        except Exception:
            pass
        if any(v is not None for v in (h1, h2, s1, s2, c1, c2)):
            _update_stack_state(h1=h1, h2=h2, start1=s1, start2=s2, count1=c1, count2=c2, source="gcode")
            try:
                _dbg("STACKS", f"Parsed heights h1={h1} h2={h2} start=({s1},{s2}) cards=({c1},{c2})")
                stack_state["last_line"] = text
            except Exception:
                pass
            return True
        else:
            try:
                _dbg("STACKS", f"Stack line seen but no numbers parsed (ignored): '{text}'")
                stack_state["last_line"] = text
            except Exception:
                pass
    except Exception as e:
        try:
            _dbg("STACKS", f"parse error: {e}")
        except Exception:
            pass
    return False

def _send_ack_ok(job): _send_gcode(f"M118 ACK_OK job={job}")
def _send_scan_ok(job): _send_gcode(f"SCAN_OK job={int(job or 0)}")
def _send_scan_fail(job): _send_gcode("SCAN_FAIL")

def _parse_ready_line(line):
    m = re.search(r"READY_TO_SCAN\s+job=(\d+)", line)
    return int(m.group(1)) if m else None

def ws_thread():
    if websocket is None:
        _dbg("WEBSOCKET ERROR", "websocket-client not installed; skipping Websocket.")
        return
    def on_open(ws):
        with printer_lock:
            ws_ref["connected"] = True; printer_state["ws_connected"] = True
        _dbg("WEBSOCKET","connected to Moonraker")
    def on_message(ws, message):
        with printer_lock:
            printer_state["messages_seen"] += 1
        try:
            obj = json.loads(message)
        except Exception:
            return
        method = obj.get("method","")
        params = obj.get("params",[])
        if method == "notify_gcode_response" and params:
            line = str(params[0]).strip()
            try:
                _try_parse_stack_measure(line)
            except Exception:
                pass
            if "READY_TO_SCAN" in line:
                global PROC_PAUSED, PROC_PAUSE_SEQ, scanned_card
                global PROC_PAUSED, PROC_PAUSE_SEQ, scanned_card
                job = _parse_ready_line(line)
                if job is None: return
                with printer_lock:
                    printer_state["last_ready_line"] = line
                    printer_state["awaiting"] = True
                    printer_state["job_id"] = job
                    _send_ack_ok(job)
                    _dbg("WEBSOCKET", f"READY_TO_SCAN -> ACK_OK (job={job})")
# Auto-pause streams while the printer requests a scan
                STREAM_STATE['paused'] = False
                STREAM_STATE['paused_reason'] = f'ready_to_scan job={job}'
                PROC_PAUSED = False
                PROC_PAUSE_SEQ = None
                with scan_lock:
                    scanned_card = None
                try:
                    _clear_ocr()
                except Exception:
                    pass
                with printer_lock:
                    printer_state['scan_captured'] = False
                PROC_PAUSED = False
                PROC_PAUSE_SEQ = None
                with scan_lock:
                    scanned_card = None
                try:
                    _clear_ocr()
                except Exception:
                    pass
            elif "ACK_SCAN_OK" in line:
                m = re.search(r"ACK_SCAN_OK\s+job=(\d+)", line)
                ack_job = int(m.group(1)) if m else None
                with printer_lock:
                    printer_state["ack_received_at"] = time.time()
                    printer_state["last_ack_job"] = ack_job
                _dbg("WEBSOCKET", f"Printer ACKed SCAN_OK echo: {line}")
                _post_job_cleanup(printer_state.get('last_ack_job'))
            elif "ACK_SCAN_FAIL" in line:
                _dbg("WEBSOCKET", f"Printer ACKed SCAN_FAIL {line}")
    def on_close(ws, code, reason):
        with printer_lock:
            ws_ref["connected"] = False; printer_state["ws_connected"] = False
        _dbg("WEBSOCKET","closed")
    def on_error(ws, error):
        with printer_lock:
            printer_state["last_error"] = str(error)
        _dbg("WEBSOCKET ERROR", f"error: {error}")
    
    while not shutdown_evt.is_set():
        try:
            ws = websocket.WebSocketApp(
                MOONRAKER_URL, on_open=on_open, on_message=on_message,
                on_error=on_error, on_close=on_close
            )
            ws_ref["ws"] = ws
            ws.run_forever(ping_interval=25, ping_timeout=10)
        except Exception as e:
            _dbg("WEBSOCKET ERROR", f"connect failed: {e}")
        if shutdown_evt.is_set(): break
        time.sleep(2.0)  # backoff

def autoscan_manager():
    while not shutdown_evt.is_set():
        time.sleep(0.05)

        # 1) Wait for a READY_TO_SCAN job
        with printer_lock:
            pending = bool(printer_state.get("awaiting", False))
            job = printer_state.get("job_id")
        if not pending:
            continue

        job_for_cycle = int(job or 0)
        _scan_perf_start = time.time()

        # 2) Small grace period to let the stream settle, then capture (manual crop if available)
        _dbg("AUTOSCAN", f"ready -> capture after grace delay (job={job})")
        settle_deadline = time.time() + AUTO_CAPTURE_WAIT_S
        still_pending = pending
        while not shutdown_evt.is_set() and time.time() < settle_deadline:
            with printer_lock:
                still_pending = bool(printer_state.get("awaiting", False))
            if not still_pending:
                break
            time.sleep(0.02)
        if not still_pending:
            _dbg("AUTOSCAN", f"aborting capture; pending={still_pending} (job={job})",)
            continue

        # 3) Capture a snapshot (using manual crop when available)
        ok = capture_scanned_card_from_live(force_manual=True)

        with scan_lock:
            cur_seq = snapshot_seq
        _capture_done = time.time()
        
        if not ok:
            _dbg("AUTO-SNAPSHOT WARNING", f"Auto-snap failed for job number:{job} (no card?)")

        # 4) Wait briefly for OCR results tied to THIS snapshot seq
        deadline = time.time() + AUTOSCAN_OCR_TIMEOUT
        nm = ""; cn_raw = ""; shint = ""; ocr_updated_at = 0.0
        got_ocr = False
        wait_start = time.time()
        t_ocr_ready = wait_start
        while time.time() < deadline:
            with ocr_lock:
                seq_ok  = (ocr_state.get("seq") == cur_seq)
                nm      = (ocr_state.get("name") or "").strip()
                cn_raw  = (ocr_state.get("number_raw") or "").strip()
                shint   = (ocr_state.get("set_hint") or "").strip().lower()
                ocr_updated_at = float(ocr_state.get("updated_at", 0))
                got_ocr = seq_ok and bool(nm or cn_raw)
            if got_ocr:
                t_ocr_ready = time.time()
                break
            time.sleep(0.02)
        if not got_ocr:
            try:
                with ocr_lock:
                    last_seq = ocr_state.get("seq")
                    last_name = (ocr_state.get("name") or ocr_state.get("name_raw") or "").strip()
                    last_num = (ocr_state.get("number_raw") or ocr_state.get("number") or "").strip()
                    last_updated_ts = float(ocr_state.get("updated_at", 0.0) or 0.0)
                age = time.time() - last_updated_ts if last_updated_ts else -1.0
                _dbg("OCR WARNING", f"OCR timeout ({AUTOSCAN_OCR_TIMEOUT:.1f}s) for job={job} (seq={cur_seq}) last_seq={last_seq} last_age={age:.2f}s last_name='{last_name}' last_num='{last_num}'")
            except Exception:
                _dbg("OCR WARNING", f"OCR timeout ({AUTOSCAN_OCR_TIMEOUT:.1f}s) for job={job} (seq={cur_seq})")
        else:
            try:
                _dbg("PERF OCR", f"OCR wait {time.time() - wait_start:.3f}s seq={cur_seq} name='{nm}' num='{cn_raw}'")
            except Exception:
                pass
        try:
            if bool(globals().get("PERF_TIMING_DEBUG", True)):
                _dbg("PERF SCAN", f"capture={_capture_done - _scan_perf_start:.3f}s ocr_wait={t_ocr_ready - wait_start:.3f}s seq={cur_seq}")
        except Exception:
            pass

        # 5) Prefer the cardinfo_worker result for this OCR update
        choice = None
        with cardinfo_lock:
            if cardinfo_state.get("last_updated") == ocr_updated_at:
                choice = cardinfo_state.get("scry")

        # 6) Single, de-duplicated fallback lookup (no repeats)
        if choice is None and (nm or cn_raw):
            try:
                # First try with OCR’s set hint (if any)…
                # If set hint is missing and we have a detected set-icon, infer it now
                try:
                    if not shint:
                        with scan_lock:
                            _img_for_icon = scanned_card.copy() if scanned_card is not None else None
                        if _img_for_icon is not None:
                            _code,_sc = _match_set_symbol_to_code(_img_for_icon, name_hint=nm)
                            if _code:
                                shint = _code
                except Exception:
                    pass
                choice = _scryfall_lookup_once(nm, cn_raw, shint, cur_seq)
                # …then fall back to no set constraint
                if choice is None and shint:
                    choice = _scryfall_lookup_once(nm, cn_raw, "", cur_seq)
            except Exception:
                choice = None
        try:
            choice = _scry_fix_mismatch(choice, nm, cn_raw, shint)
        except Exception:
            pass
        # Publish cardinfo immediately so the UI shows data without waiting for the worker
        if choice is not None:
            try:
                rate = _fx_cache.get("rate") or _get_usd_to_cad()
            except Exception:
                rate = None
            with cardinfo_lock:
                cardinfo_state["scry"] = choice
                if rate:
                    cardinfo_state["fx"] = {"rate": rate}
                cardinfo_state["last_updated"] = ocr_updated_at or time.time()

        # 7) Grab snapshot & Scryfall image (cache is warmed by cardinfo_worker when possible)
        with scan_lock:
            snap_img = scanned_card.copy() if scanned_card is not None else None
        scry_img = _fetch_scry_image(choice) if choice is not None else None
        scry_cn_raw = ""
        scry_cn_disp = ""
        if choice is not None:
            scry_cn_raw, scry_cn_disp = _apply_scry_collector_number(choice)

        # 8) Visual compare (always compute details so UI/history have an image)
        match_score, match_ok, cmp_details = 0.0, False, None
        if scry_img is not None and snap_img is not None:
            # fast path: get the score + pass/fail
            match_score, match_ok = compare_snapshot_to_scryfall(
                snap_img, scry_img, return_details=False, scry_card=choice
            )
            # ALWAYS build a details image for the UI/history, even when it passed
            _, _, cmp_details = compare_snapshot_to_scryfall(
                snap_img, scry_img, return_details=True, scry_card=choice
            )
            # publish the visual so /compare.jpg shows something immediately
            if cmp_details:
                _publish_compare_visual(cmp_details)



        snapshot_ok = snap_img is not None
        scry_img_ok = scry_img is not None
        visual_ready = bool(choice and snapshot_ok and scry_img_ok)

        with ocr_lock:
            ocr_snapshot = {
                "name": ocr_state.get("name",""),
                "name_raw": ocr_state.get("name",""),
                "name_conf": ocr_state.get("name_conf",0.0),
                "number": ocr_state.get("number",""),
                "number_raw": ocr_state.get("number_raw",""),
                "set_hint": ocr_state.get("set_hint",""),
                "foil": ocr_state.get("foil", False),
            }
        decision = _derive_review_outcome(
            ocr_snapshot,
            choice,
            match_score,
            match_ok,
            snapshot_ok=snapshot_ok,
            scry_img_ok=scry_img_ok,
            match_mode=(choice or {}).get("_match_mode"),
            allow_auto_name_fix=True,
        )
        ocr_snapshot = decision["ocr"]
        flagged = decision["flagged"]

        if decision["status"] == "fail":
            _dbg("COMPARE", f"FLAGGED job={job} score={match_score:.3f} (visual threshold {MATCH_TH})")
        elif decision["status"] == "review":
            why = "; ".join(decision.get("review_reasons") or [])
            _dbg("COMPARE", f"REVIEW job={job} reasons={why or 'unspecified'} score={match_score:.3f}")
        else:
            _dbg("COMPARE", f"VISUAL MATCH job={job} score={match_score:.3f} (>= {MATCH_TH})")

        # 9) Publish quick summary to OCR state for UI
        with ocr_lock:
            ocr_state["match_score"] = round(float(match_score), 3)
            ocr_state["match_ok"] = bool(match_ok)
            ocr_state["flagged"] = bool(flagged)
            ocr_state["status"] = decision.get("status")
            # Reflect any auto name fix in the UI immediately
            if decision.get("auto_name_fix") and ocr_snapshot.get("name"):
                ocr_state["name"] = ocr_snapshot.get("name")

        # 10) Persist review entry & resume processing/stream
        try:
            save_scan_entry(
                snap_img if snap_img is not None else np.zeros((10,10,3), np.uint8),
                ocr_snapshot,
                choice,
                cmp_details,
                match_score,
                match_ok,
                review_decision=decision,
            )

            # Resume processing for this snapshot
            global PROC_PAUSED, PROC_PAUSE_SEQ
            PROC_PAUSED = False
            PROC_PAUSE_SEQ = None
            STREAM_STATE['paused'] = False
            STREAM_STATE['paused_reason'] = ''

        except Exception as e:
            _dbg("HISTORY ERROR", f"save_scan_entry failed: {e}")

        try:
            if bool(globals().get("PERF_TIMING_DEBUG", True)):
                _dbg("PERF SCAN", f"total={time.time() - _scan_perf_start:.3f}s scry_compare={time.time() - t_ocr_ready:.3f}s seq={cur_seq}")
        except Exception:
            pass

        # 11) Notify printer
        if ALWAYS_SCAN_OK or not flagged:
            _send_scan_ok(job_for_cycle)
            _dbg("WEBSOCKET", f"Sent SCAN_OK (job={job_for_cycle})")
            # Fallback cleanup if printer won't ACK
            if not bool(globals().get('REQUIRE_PRINTER_ACK', True)):
                try:
                    _post_job_cleanup(job_for_cycle)
                except Exception as _e:
                    _dbg('AUTOSCAN', f'post_job_cleanup (no-ack OK) error: {_e}', level=2)
            STREAM_STATE['paused'] = False
            STREAM_STATE['paused_reason'] = ''
            with printer_lock:
                printer_state["last_decision"] = "SCAN_OK (flagged)" if flagged else "SCAN_OK"
        else:
            _send_scan_fail(job_for_cycle)
            _dbg("WEBSOCKET", f"Sent SCAN_FAIL (job={job_for_cycle})")
            # Fallback cleanup if printer won't ACK
            if not bool(globals().get('REQUIRE_PRINTER_ACK', True)):
                try:
                    _post_job_cleanup(job_for_cycle)
                except Exception as _e:
                    _dbg('AUTOSCAN', f'post_job_cleanup (no-ack FAIL) error: {_e}', level=2)
            with printer_lock:
                printer_state["last_decision"] = "SCAN_FAIL"

        # 12) (moved) Cleanup happens on ACK; do not clear awaiting here.

# =========================
#MARK: STREAM / UI
# =========================

def _draw_ocr_debug_boxes(img):
    """Draw what the AI sees (detected ROI boxes)."""
    if globals().get("SHOW_ROI_OVERLAY", False) and _ai_enabled():
        try:
            _draw_ai_overlay(img)
        except Exception as e:
            _dbg("AI OVERLAY ERROR", f"{e}")
    # Legacy fallback removed; rely on AI overlay only.


def _stream_frames(kind='card'):
    while not shutdown_evt.is_set():
        if not STREAM_STATE.get("enabled", True):
            # heartbeat to keep <img> alive
            blank = np.zeros((1,1,3), dtype=np.uint8)
            ok, enc = cv2.imencode('.png', blank)
            tiny = enc.tobytes() if ok else b''
            yield (b'--frame\r\nContent-Type: image/png\r\nContent-Length: ' +
                   str(len(tiny)).encode() + b'\r\n\r\n' + tiny + b'\r\n')
            time.sleep(0.25)
            continue

        paused = bool(STREAM_STATE.get("paused", False))

        # Serve cached frame while paused
        if paused:
            cached = STREAM_CACHE.get(kind)
            if cached:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ' +
                       str(len(cached)).encode() + b'\r\n\r\n' + cached + b'\r\n')
                time.sleep(max(0.03, 1.0 / max(1, int(STREAM_STATE.get("fps", 25)))))
                continue

        # Pick source frame
        img = None
        if kind == 'live':
            with video_lock:
                img = output_frame.copy() if output_frame is not None else None
        else:
            preview = None
            snap = None
            with scan_lock:
                if preview_card is not None:
                    preview = preview_card.copy()
                if scanned_card is not None:
                    snap = scanned_card.copy()
            if preview is not None:
                img = preview
            elif snap is not None:
                img = snap
            else:
                img = None

        if img is None:
            time.sleep(0.03)
            continue

        if kind == 'card' and globals().get("SHOW_ROI_OVERLAY", False):
            _draw_ocr_debug_boxes(img)

        q = int(STREAM_STATE.get("quality", 80))
        ok, enc = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, q])
        if not ok:
            time.sleep(0.03)
            continue
        frame = enc.tobytes()

        STREAM_CACHE[kind] = frame

        yield (b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ' +
               str(len(frame)).encode() + b'\r\n\r\n' + frame + b'\r\n')

        fps = max(1, int(STREAM_STATE.get("fps", 25)))
        time.sleep(max(0.01, 1.0 / fps))

@app.route('/live')
def live(): return Response(_stream_frames('live'), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/card')
def card_stream():
    return Response(_stream_frames('card'), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.get('/api/stream')
def api_stream_get():
    return jsonify({"ok": True, **STREAM_STATE})

@app.post('/api/stream')
def api_stream_set():
    data = request.get_json(silent=True) or {}
    # allowed keys: enabled, quality, fps, paused
    for key in ('enabled','quality','fps','paused'):
        if key in data:
            STREAM_STATE[key] = data[key]
    # clamp values
    STREAM_STATE['fps'] = max(1, min(60, int(STREAM_STATE.get('fps', REQ_FPS))))
    STREAM_STATE['quality'] = max(40, min(95, int(STREAM_STATE.get('quality', 80))))
    return jsonify({"ok": True, **STREAM_STATE})

@app.route('/compare.jpg')
def compare_jpg():
    with compare_lock:
        img = _last_cmp_img
        cmp_seq = _last_cmp_seq

    etag = f'W/"cmp-{cmp_seq}"'
    inm = request.headers.get("If-None-Match", "")
    if img is not None and etag and etag in inm:
        resp = Response(status=304)
        resp.headers["ETag"] = etag
        resp.headers["Cache-Control"] = "private, max-age=3, must-revalidate"
        return resp

    if img is None:
        empty = np.zeros((1,1,3), dtype=np.uint8)
        ok, enc = cv2.imencode('.png', empty)
        resp = Response(enc.tobytes(), mimetype='image/png')
    else:
        ok, enc = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY_CMP])
        resp = Response(enc.tobytes(), mimetype='image/jpeg')

    resp.headers["ETag"] = etag
    resp.headers["Cache-Control"] = "private, max-age=3, must-revalidate"
    return resp

# ---------------------------
# UI file routes
# ---------------------------


@app.get("/api/logs")
def api_logs():
    """Poll console logs.
    Query params:
      after=<id>  return lines with id > after (default 0)
      limit=<n>   max lines to return (1..1000, default 200)
    """
    try:
        after = int(request.args.get("after") or 0)
    except Exception:
        after = 0
    try:
        limit = int(request.args.get("limit") or 200)
    except Exception:
        limit = 200
    limit = max(1, min(limit, 1000))

    with log_lock:
        items = [e for e in LOG_RING if e["id"] > after][-limit:]
        nxt = LOG_SEQ

    out = []
    for e in items:
        sev = e.get("sev")
        if not sev:
            try:
                lvl_i = int(e.get("lvl", 1))
            except Exception:
                lvl_i = 1
            sev = {0:"ok", 1:"info", 2:"info", 3:"warn", 4:"err"}.get(lvl_i, "info")
        out.append({
            "id":   int(e["id"]),
            "ts":   float(e["ts"]),
            "tag":  e["tag"],
            "msg":  e["msg"],
            "lvl":  int(e.get("lvl", 1)),
            "sev":  sev,
            "line": f"[{e['tag']}] {e['msg']}",
        })
    return jsonify({"ok": True, "items": out, "next": int(nxt)})



@app.post("/api/logs/clear")
def api_logs_clear():
    with log_lock:
        LOG_RING.clear()
    return jsonify({"ok": True})

@app.get("/ui.js")
def serve_ui_js():
    path = os.path.join(APP_DIR, "ui.js")
    return send_file(path, mimetype="application/javascript") if os.path.exists(path) \
           else Response("// ui.js not found", mimetype="application/javascript", status=404)


@app.get("/favicon.ico")
def favicon():
    # quiet the browser’s automatic favicon request
    return Response(b"", mimetype="image/x-icon")

# in app.py (your Flask file), replace /api/state with:
@app.route('/api/state')
def api_state():
    state = {}

    with printer_lock:
        p = dict(printer_state)
        p["ack_ok_jobs"] = list(p.get("ack_ok_jobs", []))
    with ocr_lock:
        ocr_snap = dict(ocr_state)
    with video_lock:
        scan_snap = dict(scanner_state)
    with scan_lock:
        snap_present = scanned_card is not None
        snap_age = int(time.time() - last_scan_ts) if snap_present else 0

    state.update(p)
    state.update(ocr_snap)   # OCR first…
    state.update(scan_snap)  # …then live scanner values win

    # Always provide stable defaults so the UI pills render immediately
    for k, v in {
        "locked": False,
        "locked_frames": 0,
        "locked_area": 0.0,
        "steady": False,
        "foil": False,
        "foil_score": 0.0,
        "provider": PRIMARY_PROVIDER_LABEL,
        "name": "",
        "name_raw": "",
        "name_conf": 0.0,
        "number": "",
        "number_raw": "",
        "number_conf": 0.0,
        "set_hint": "",
        "match_score": 0.0,
        "match_ok": None,
        "flagged": False,
        "status": "",
        "review_level": "info",
        "review_reasons": [],
    }.items():
        state.setdefault(k, v)

    state["live_foil"] = scan_snap.get("foil")
    state["live_foil_score"] = scan_snap.get("foil_score")
    state["ocr_foil"] = ocr_snap.get("foil")
    state["ocr_foil_score"] = ocr_snap.get("foil_score")

    state.update({
        "stream": dict(STREAM_STATE),
        "snapshot_present": snap_present,
        "snapshot_age_sec": snap_age,
        "cmp_seq": _get_compare_seq(),
        "match": {
            "ok": ocr_snap.get("match_ok"),
            "score": ocr_snap.get("match_score"),
        },
        "stacks": _stack_state_snapshot(),
    })
    state["bad_count"] = len(bad_cards)
    state["loaded_scan_id"] = current_loaded_scan_id
    return jsonify(state)

def card(): return Response(_stream_frames('card'), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.after_request
def _no_cache(resp):
    p = request.path
    if resp.headers.get("Cache-Control"):
        return resp
    if p.startswith('/api/scan/') and p.endswith('/thumb'):
        resp.headers['Cache-Control'] = 'public, max-age=31536000, immutable'
    else:
        resp.headers['Cache-Control'] = 'no-store'
    return resp

@app.route('/api/carddata')
def api_carddata():
    with cardinfo_lock:
        payload = dict(cardinfo_state)
    last_updated = float(payload.get("last_updated") or 0.0)
    etag = f'W/"card-{int(last_updated * 1000)}"'
    inm = request.headers.get("If-None-Match", "")
    since = request.args.get("since")
    cache_headers = {"ETag": etag, "Cache-Control": "private, max-age=2, must-revalidate"}

    try:
        if since is not None and float(since) >= last_updated:
            resp = Response(status=304)
            for k, v in cache_headers.items():
                resp.headers[k] = v
            return resp
    except Exception:
        pass

    if etag and etag in inm:
        resp = Response(status=304)
        for k, v in cache_headers.items():
            resp.headers[k] = v
        return resp

    resp = jsonify(payload)
    for k, v in cache_headers.items():
        resp.headers[k] = v
    return resp

@app.route('/api/badlist')
def api_bad():
    return jsonify({"items": list(bad_cards), "count": len(bad_cards)})


def _post_job_cleanup(job=None):
    """Reset state so the scanner is ready for the next job cycle."""
    try:
        # Treat each completed job as one card processed from stack 1 (input).
        _stack_cards_processed(1, 0)
        with printer_lock:
            cur = printer_state.get('job_id')
            if job is None or cur == job:
                printer_state['awaiting'] = False
                printer_state['job_id'] = None
        with video_lock:
            scanner_state['steady'] = False
        STREAM_STATE['paused'] = False
        STREAM_STATE['paused_reason'] = None
        try:
            with scan_lock:
                globals().get('scanned_card')
                scanned_card = None
        except Exception:
            pass
        try:
            printer_state.get('ack_ok_jobs', set()).discard(job)
        except Exception:
            pass
        _dbg('AUTOSCAN', f'post_job_cleanup done (job={job})')
    except Exception as e:
        _dbg('AUTOSCAN', f'post_job_cleanup error: {e}')


def capture_scanned_card_from_live(force_manual: bool = False):
    global scanned_card, last_scan_ts, snapshot_seq, preview_card, current_loaded_scan_id
    global PROC_PAUSED, PROC_PAUSE_SEQ

    snap = None
    with video_lock:
        frame = latest_frame_raw.copy() if latest_frame_raw is not None else None
        live_crop = current_card_crop.copy() if current_card_crop is not None else None

    # Prefer manual crop from the latest raw frame so we honor the user-set quad
    if frame is not None and (force_manual or manual_crop_enabled):
        q = _manual_quad_px(frame.shape)
        if q is not None:
            try:
                snap = warp_card(frame, q)
            except Exception:
                snap = None

    # Fallback to the last detected/warped crop if manual failed or is disabled
    if snap is None:
        snap = live_crop

    if snap is None:
        return False

    # Optional upsample for better OCR clarity (UI/processing snapshot only)
    try:
        scale = max(1.0, float(SNAPSHOT_CARD_SCALE))
        if scale > 1.001 and snap is not None:
            h, w = snap.shape[:2]
            tw = int(w * scale); th = int(h * scale)
            tw = max(2, tw); th = max(2, th)
            snap = cv2.resize(snap, (tw, th), interpolation=cv2.INTER_CUBIC)
    except Exception:
        pass

    # Install the new snapshot
    with scan_lock:
        preview_card = None            # ensure preview doesn't mask the new snap
        scanned_card = snap.copy()
        last_scan_ts = time.time()
        snapshot_seq += 1
        current_loaded_scan_id = None  # drop any history selection while a live scan is processed
        # Pause processing until OCR + Scryfall complete for this snapshot
        PROC_PAUSED = True
        PROC_PAUSE_SEQ = snapshot_seq

    # While paused, /card serves STREAM_CACHE['card'].
    # Overwrite it *now* so the UI shows the NEW snapshot instead of the previous one.
    try:
        STREAM_CACHE['card'] = _jpeg_bytes(
            scanned_card,
            int(STREAM_STATE.get('quality', 80))
        )
    except Exception:
        pass

    # Pause the outgoing stream (it will now serve the fresh cached frame)
    STREAM_STATE['paused'] = False
    STREAM_STATE['paused_reason'] = f'snapshot seq={snapshot_seq}'

    with printer_lock:
        printer_state["scan_captured"] = True
    return True

from flask import send_from_directory

@app.post("/api/scans/export")
def api_scans_export():
    """
    Export aggregated history to a decklist file on the device.
    Body JSON:
      fmt: 'archidekt' | 'moxfield' | 'simple'   (default 'archidekt')
      status: 'pass' | 'pass_no_review' | 'review' | 'fail' | 'all' (default 'pass')
      filename: optional, else auto timestamp
    Returns JSON with file path and a /exports/... URL.
    """
    data = request.get_json(silent=True) or {}
    fmt = (data.get("fmt") or "archidekt").lower()
    status = (data.get("status") or "pass").lower()
    if status == "flagged":
        status = "review"
    allowed_status = {"pass", "pass_no_review", "all", "review", "fail"}
    if status not in allowed_status:
        status = "pass"

    # aggregate
    counts = _aggregate_history(only_status=status)
    if not counts:
        return jsonify({"ok": False, "error": "no items to export"}), 400

    # make lines (sorted for stability)
    lines = []
    for (name, set_code, cn), c in sorted(counts.items(), key=lambda x: (x[0][0].lower(), x[0][1], x[0][2])):
        lines.append(_deckline(name, c, set_code, cn, fmt))

    text = "\n".join(lines) + "\n"

    # filename
    ts = time.strftime("%Y%m%d_%H%M%S")
    base = (data.get("filename") or f"deck_export_{ts}_{fmt}").strip()
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", base)[:80]
    fname = f"{safe}.txt"
    fpath = os.path.join(EXPORT_DIR, fname)

    # write to disk on the device
    try:
        os.makedirs(EXPORT_DIR, exist_ok=True)
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(text)
    except Exception as e:
        return jsonify({"ok": False, "error": f"write failed: {e}"}), 500

    return jsonify({
        "ok": True,
        "count": sum(counts.values()),
        "unique": len(counts),
        "filename": fname,
        "saved_path": os.path.abspath(fpath),
        "url": f"/exports/{fname}",
        "fmt": fmt,
        "status": status
    })

@app.get('/api/scans/export/download')
def api_scans_export_download():
    """
    Exports current (filtered) history.
      fmt=txt|csv     default: txt
      fields=<comma separated header names>  (CSV only; order preserved)

    All other query params are the same filters as /api/scans
    (status, score_min, since/ts_from/ts_to, q, set, foil, sort_by, sort_dir, limit/offset…).
    """
    a = request.args
    fmt = (a.get("fmt") or a.get("format") or "txt").lower()

    # collect filtered rows
    F = _parse_scan_filters(a)
    with history_lock:
        rows = list(scan_history)
    rows = _apply_scan_filters(rows, F)

    # aggregate identical cards -> counts + attach a representative scry object & timestamps
    aggs = {}
    for e in rows:
        scry = e.get("scry") or {}
        name = (scry.get("name") or e.get("name") or "").strip()
        setc = (scry.get("set") or "").upper().strip()
        cn   = str((scry.get("collector_number") or e.get("number") or "")).strip()
        foil = bool(e.get("foil", False))
        key = (name, setc, cn, foil)
        if key not in aggs:
            aggs[key] = {
                "count": 0, "name": name, "setc": setc, "cn": cn, "foil": foil,
                "scry": scry, "first_ts": float(e.get("ts", 0)), "last_ts": float(e.get("ts", 0))
            }
        ag = aggs[key]
        ag["count"] += 1
        ts = float(e.get("ts", 0))
        ag["first_ts"] = min(ag["first_ts"], ts)
        ag["last_ts"]  = max(ag["last_ts"], ts)

    ts = time.strftime("%Y%m%d-%H%M%S")
    fn_base = f"collection-{ts}"

    # ----- CSV (Selectable fields) -----
    if fmt == "csv":
        import csv, io

        # Map incoming aliases -> canonical Archidekt-style headers (order preserved)
        _aliases = {
            # common synonyms
            "Card Name": "Name",
            "Set": "Edition Code",
            "Edition": "Edition Code",
            "Set Code": "Edition Code",
            "CollectorNumber": "Collector Number",
            "Collector number": "Collector Number",
            "Collector #": "Collector Number",
            "Foil": "Finish",
            "Finish": "Finish",
            "MV": "Mana Value",
            "Mana value": "Mana Value",
            "Color identities": "Identities",
            "Color identity": "Identities",
            "Card types": "Types",
            "Oracle ID": "Scryfall Oracle ID",
        }

        # Full list of headers we know how to fill (empty if unavailable)
        def _split_typeline(tl: str):
            SUPER = {"Basic", "Legendary", "Snow", "World", "Ongoing", "Elite", "Host"}
            TYPES = {
                "Artifact","Battle","Creature","Enchantment","Instant","Land","Planeswalker","Sorcery","Tribal",
                "Conspiracy","Dungeon","Plane","Phenomenon","Scheme","Vanguard","Attraction","Sticker","Siege"
            }
            tl = (tl or "").strip()
            left, right = (tl.split("—", 1) + [""])[:2] if "—" in tl else (tl, "")
            words = [w for w in left.split() if w]
            sups = [w for w in words if w in SUPER]
            tys  = [w for w in words if w in TYPES]
            return {
                "Super-types": " ".join(sups),
                "Types": " ".join(tys),
                "Sub-types": right.strip(),
            }

        def _first(lst): 
            try:
                return (lst or [None])[0] or ""
            except Exception:
                return ""

        def _get(ag, key, default=""):
            return (ag.get("scry") or {}).get(key, default)

        def _price(ag, which):
            try:
                prices = (ag.get("scry") or {}).get("prices") or {}
                foil = bool(ag.get("foil"))
                if which == "usd":
                    v = prices.get("usd_foil" if foil else "usd")
                elif which == "eur":
                    v = prices.get("eur_foil" if foil else "eur")
                elif which == "tix":
                    v = prices.get("tix")
                else:
                    v = None
                return f"{float(v):.2f}" if v not in (None, "", "null") else ""
            except Exception:
                return ""
        def _p(ag, key):
            v = (((ag.get("scry") or {}).get("prices") or {}).get(key))
            try:
                return f"{float(v):.2f}" if v not in (None, "", "null") else ""
            except Exception:
                return ""
        # canonical field -> value function
        FIELD_FNS = OrderedDict([
            ("Quantity",             lambda ag: ag["count"]),
            ("Name",                 lambda ag: ag["name"]),
            ("Finish",               lambda ag: "Foil" if ag["foil"] else "Normal"),
            ("Condition",            lambda ag: "NM"),
            ("Date Added",           lambda ag: time.strftime("%Y-%m-%d", time.localtime(ag["first_ts"])) if ag["first_ts"] else ""),
            ("Language",             lambda ag: (_get(ag, "lang") or "en").upper()),
            ("Purchase Price",       lambda ag: ""),
            ("Tags",                 lambda ag: ""),
            ("Edition Name",         lambda ag: _get(ag, "set_name", "")),
            ("Edition Code",         lambda ag: ag["setc"]),
            ("Multiverse Id",        lambda ag: _first(_get(ag, "multiverse_ids"))),
            ("Scryfall ID",          lambda ag: _get(ag, "id", "")),
            ("MTGO ID",              lambda ag: _get(ag, "mtgo_id", "")),
            ("Collector Number",     lambda ag: ag["cn"]),
            ("Mana Value",           lambda ag: _get(ag, "cmc", "")),
            ("Colors",               lambda ag: "".join(_get(ag, "colors", []))),
            ("Identities",           lambda ag: "".join(_get(ag, "color_identity", []))),
            ("Mana cost",            lambda ag: _get(ag, "mana_cost", "")),
            ("Types",                lambda ag: _split_typeline(_get(ag, "type_line", "")).get("Types","")),
            ("Sub-types",            lambda ag: _split_typeline(_get(ag, "type_line", "")).get("Sub-types","")),
            ("Super-types",          lambda ag: _split_typeline(_get(ag, "type_line", "")).get("Super-types","")),
            ("Rarity",               lambda ag: str(_get(ag, "rarity", "")).title()),
            ("Price (Card Kingdom)",      lambda ag: ""),                # Scryfall doesn't provide CK
            ("Price (TCG Player)",        lambda ag: _p(ag, "usd")),     # Scryfall USD
            ("Price (Star City Games)",   lambda ag: ""),                # not provided
            ("Price (Card Hoarder)",      lambda ag: _p(ag, "tix")),     # Scryfall TIX
            ("Price (Card Market)",       lambda ag: _p(ag, "eur")),     # Scryfall EUR
            ("Scryfall Oracle ID",   lambda ag: _get(ag, "oracle_id", "")),
        ])

        # Parse 'fields' from query; default to a sensible set if none
        raw_fields = []
        # allow repeated ?fields=... or a single comma-separated value
        raw_fields += a.getlist("fields")
        if not raw_fields and a.get("fields"):
            raw_fields = [a.get("fields")]
        user_fields = []
        for chunk in raw_fields:
            for f in (chunk or "").split(","):
                f = f.strip()
                if not f: 
                    continue
                f = _aliases.get(f, f)  # normalize synonyms
                if f in FIELD_FNS and f not in user_fields:
                    user_fields.append(f)

        if not user_fields:
            # Archidekt-ish default
            user_fields = [
                "Quantity","Name","Finish","Condition","Date Added","Language",
                "Edition Name","Edition Code","Collector Number",
                "Scryfall ID","Scryfall Oracle ID",
                "Mana Value","Identities","Mana cost","Types","Sub-types","Super-types","Rarity"
            ]

        # Always ensure Quantity + Name are present (first/second), even if user unchecked by mistake
        for must in ["Quantity", "Name"]:
            if must not in user_fields:
                user_fields.insert(0 if must == "Quantity" else 1, must)

        # Build CSV
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(user_fields)
        for key in sorted(aggs.keys()):
            ag = aggs[key]
            row = [FIELD_FNS[h](ag) for h in user_fields]
            w.writerow(row)

        data = buf.getvalue().encode("utf-8")
        return Response(
            data,
            mimetype="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{fn_base}.csv"'}
        )

    # ----- Plain text (Moxfield/Archidekt paste) -----
    lines = []
    for (name, setc, cn, foil) in sorted(aggs.keys()):
        # rebuild tail exactly like before
        tail = f" ({setc}) {cn}" if setc or cn else ""
        lines.append(f"{aggs[(name,setc,cn,foil)]['count']} {name}{tail}")
    txt = "\n".join(lines) + "\n"
    return Response(
        txt.encode("utf-8"),
        mimetype="text/plain",
        headers={"Content-Disposition": f'attachment; filename="{fn_base}.txt"'}
    )

@app.get("/exports/<path:fname>")
def serve_export(fname):
    # Optional: add simple allowlist to avoid path traversal
    if not re.match(r"^[\w.\-]+\.txt$", fname):
        return jsonify({"ok": False, "error": "invalid filename"}), 400
    return send_from_directory(EXPORT_DIR, fname, mimetype="text/plain", as_attachment=True)

@app.route('/api/snapshot', methods=['POST'])
def api_snapshot():
    global scanned_card, snapshot_seq, preview_card
    action=(request.get_json(silent=True) or {}).get('action','').lower()
    if action=='snap':
        ok=capture_scanned_card_from_live()
        return jsonify({"ok":ok,"error":None if ok else "no live crop"})
    elif action=='clear':
        with scan_lock:
            scanned_card = None
            preview_card = None
            snapshot_seq += 1
            global PROC_PAUSED, PROC_PAUSE_SEQ
            PROC_PAUSED = False
            PROC_PAUSE_SEQ = None

        # Immediately clear the cached /card frame so nothing stale is shown
        try:
            STREAM_CACHE['card'] = _jpeg_bytes(np.zeros((8,8,3), np.uint8), 70)
        except Exception:
            STREAM_CACHE['card'] = b""

        STREAM_STATE['paused'] = False
        STREAM_STATE['paused_reason'] = ''

        _clear_ocr()
        return jsonify({"ok": True})


    return jsonify({"ok":False,"error":"invalid action"}), 400

@app.get('/api/scrysearch')
def api_scrysearch():
    name = (request.args.get('name') or '').strip()
    number = (request.args.get('number') or '').strip()
    set_hint = (request.args.get('set') or '').strip().lower()
    # Build a Scryfall search query from any combination of inputs (name, set, number)
    parts = []
    if name:
        parts.append(f'!"{name}"')
    cn_norm = _normalize_cn_for_search(number)
    if cn_norm:
        parts.append(f"cn:{cn_norm}")
    if set_hint:
        parts.append(f"set:{set_hint}")
    if not parts:
        return jsonify({"ok": True, "results": []})
    q = " ".join(parts)
    data = _scryfall_request("https://api.scryfall.com/cards/search",
                             {"q": q, "unique": "prints", "order": "released", "dir": "desc"})
    results = []
    for c in (data or {}).get("data", [])[:12]:
        img = (c.get("image_uris") or {}).get("small")
        if not img and c.get("card_faces"):
            img = ((c["card_faces"][0].get("image_uris") or {}).get("small"))
        results.append({
            "id": c.get("id"),
            "name": c.get("name"),
            "set": c.get("set"),
            "set_name": c.get("set_name"),
            "collector_number": c.get("collector_number"),
            "image": img,
        })
    return jsonify({"ok": True, "results": results})

def _settings_load():
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except FileNotFoundError:
        return {}
    except Exception:
        return {}

def _settings_save(data: dict):
    os.makedirs(os.path.dirname(os.path.abspath(SETTINGS_PATH)), exist_ok=True)
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)

# ---- Settings helpers (single source of truth) ----
# (removed duplicate definition of _settings_load)

# (removed duplicate definition of _settings_save)

RESTART_KEYS = {
    "CAMERA_DEVICE", "REQ_WIDTH", "REQ_HEIGHT", "REQ_FPS",
    "FOURCC_PRIMARY", "FOURCC_FALLBACK", "HOST", "PORT",
}

_FAST_CFG_KEYS = {
    "OCR_FAST_PATH", "FAST_OCR_MODE", "OCR_LAZY_SET_HINT",
    "OCR_TITLE_SINGLE_PASS", "OCR_TITLE_MAX_W",
    "OCR_NUMBER_FASTPATH", "OCR_SKIP_NUMBER", "OCR_USE_TESSERACT",
}

def _refresh_fast_cfg_from_globals():
    cfg = globals().get("_FAST_CFG")
    if not isinstance(cfg, dict):
        return
    try:
        cfg.update({
            "FAST_PATH": bool(globals().get("OCR_FAST_PATH", True) or globals().get("FAST_OCR_MODE", False)),
            "LAZY_SH":   bool(globals().get("OCR_LAZY_SET_HINT", False)),
            "SINGLE":    bool(globals().get("OCR_TITLE_SINGLE_PASS", True)),
            "MAXW":      int(globals().get("OCR_TITLE_MAX_W", 640)),
            "FAST_NUM":  bool(globals().get("OCR_NUMBER_FASTPATH", True)),
            "SKIP_NUM":  bool(globals().get("OCR_SKIP_NUMBER", False)),
            "USE_TESSERACT": bool(globals().get("OCR_USE_TESSERACT", True) and TESS_AVAILABLE),
            "USE_PADDLE": bool(globals().get("OCR_USE_TESSERACT", True) and TESS_AVAILABLE),
        })
    except Exception:
        pass

def _apply_settings_runtime(changes: dict):
    """Best-effort: apply saved settings to live globals without restart."""
    if not changes:
        return

    manual_quad = None
    manual_enabled = None

    for k, v in changes.items():
        if k in RESTART_KEYS:
            continue
        if k not in globals():
            continue
        try:
            globals()[k] = v
        except Exception:
            pass
        if k == "MANUAL_CROP_QUAD":
            manual_quad = v
        elif k == "MANUAL_CROP_ENABLED":
            manual_enabled = v

    if manual_quad is not None or manual_enabled is not None:
        try:
            _set_manual_crop(
                manual_quad if manual_quad is not None else manual_crop_quad,
                manual_enabled if manual_enabled is not None else manual_crop_enabled,
            )
        except Exception:
            pass

    if _FAST_CFG_KEYS & set(changes.keys()):
        _refresh_fast_cfg_from_globals()

    if ("CARD_THICKNESS_MM" in changes) or ("REMEASURE_EVERY" in changes):
        _update_stack_config(
            thickness=changes.get("CARD_THICKNESS_MM", stack_state.get("thickness")),
            remeasure_every=changes.get("REMEASURE_EVERY", stack_state.get("remeasure_every")),
        )

def _effective_settings():
    # current runtime values (imported from config.py at startup)
    def L(v):  # ensure JSON-friendly for tuples
        return list(v) if isinstance(v, (list, tuple)) else v
    return {
        # Camera
        "CAMERA_DEVICE": CAMERA_DEVICE, "REQ_WIDTH": REQ_WIDTH, "REQ_HEIGHT": REQ_HEIGHT,
        "REQ_FPS": REQ_FPS, "FOURCC_PRIMARY": FOURCC_PRIMARY, "FOURCC_FALLBACK": FOURCC_FALLBACK,
        # Processing / Canvas
        "PROC_MAX_WIDTH": PROC_MAX_WIDTH, "PROC_DOWNSCALE_MAX_W": PROC_DOWNSCALE_MAX_W,
        "CARD_W": CARD_W, "CARD_H": CARD_H,
        "MIN_CARD_AREA_RATIO": MIN_CARD_AREA_RATIO, "MAX_CARD_AREA_RATIO": MAX_CARD_AREA_RATIO,
        "BORDER_MARGIN_PCT": BORDER_MARGIN_PCT, "CARD_ASPECT": CARD_ASPECT, "ASPECT_TOL": ASPECT_TOL,
        "CONFIRM_FRAMES": CONFIRM_FRAMES, "STALE_FRAMES": STALE_FRAMES, "MAX_ASSOC_DIST": MAX_ASSOC_DIST,
        "MAX_CARDS": MAX_CARDS, "DETECT_EVERY_N_FRAMES": DETECT_EVERY_N_FRAMES,
        "RECTANGULARITY_MIN": RECTANGULARITY_MIN, "DETECT_QUAD_PAD_PCT": DETECT_QUAD_PAD_PCT,
        "MANUAL_CROP_ENABLED": manual_crop_enabled,
        "MANUAL_CROP_QUAD": L(manual_crop_quad.tolist()),
        # Capture timing
        "AUTO_CAPTURE_WAIT_S": AUTO_CAPTURE_WAIT_S,
        "AUTOSCAN_OCR_TIMEOUT": AUTOSCAN_OCR_TIMEOUT,
        # OCR & Debug
        "OCR_BACKEND": OCR_BACKEND, "OCR_ONLY_ON_SNAPSHOT": OCR_ONLY_ON_SNAPSHOT,
        "DEBUG_OCR": DEBUG_OCR, "OCR_DEBUG_BOXES": OCR_DEBUG_BOXES, "SHOW_ROI_OVERLAY": SHOW_ROI_OVERLAY,
        "MIN_TITLE_LETTERS": MIN_TITLE_LETTERS, "TEXT_PRESENCE_MIN": TEXT_PRESENCE_MIN,
        "TITLE_ALLOW_TESS_FALLBACK": TITLE_ALLOW_TESS_FALLBACK, "USE_TESS_FOR_TITLES": USE_TESS_FOR_TITLES,
        "OCR_ENABLE_BLACKHAT": OCR_ENABLE_BLACKHAT,
        "FAST_OCR_MODE": _fast_mode(),
        "LIVE_OCR_ONLY_WHEN_STEADY": LIVE_OCR_ONLY_WHEN_STEADY,
        "LIVE_OCR_MIN_INTERVAL": LIVE_OCR_MIN_INTERVAL,
        "USE_OPENCL": USE_OPENCL,
        # Foil
        "FOIL_MIN_SCORE": FOIL_MIN_SCORE, "FOIL_ON_TH": FOIL_ON_TH,
        "FOIL_OFF_TH": FOIL_OFF_TH, "FOIL_DETECT": FOIL_DETECT, "FOIL_EVERY_N_FRAMES": FOIL_EVERY_N_FRAMES,
        # Match
        "MATCH_ENABLE": MATCH_ENABLE, "MATCH_W_HASH": MATCH_W_HASH, "MATCH_W_HIST": MATCH_W_HIST,
        "MATCH_W_ORB": MATCH_W_ORB, "MATCH_TH": MATCH_TH, "NAME_OK_TH": NAME_OK_TH,
        "MATCH_REQUIRE_ORB": MATCH_REQUIRE_ORB, "MATCH_ORB_FAIL_THRESHOLD": MATCH_ORB_FAIL_THRESHOLD,
        "ALWAYS_SCAN_OK": ALWAYS_SCAN_OK, "MATCH_USE_ART": MATCH_USE_ART, "MATCH_ORB_FEATURES": MATCH_ORB_FEATURES,
        "MATCH_FAST_ACCEPT_DELTA": MATCH_FAST_ACCEPT_DELTA, "MATCH_FAST_REJECT_DELTA": MATCH_FAST_REJECT_DELTA,
        "HASH_STABILITY_BITS": HASH_STABILITY_BITS,
        # Prices
        "SCRYFALL_TIMEOUT": SCRYFALL_TIMEOUT, "FX_URL": FX_URL, "FX_TTL_SEC": FX_TTL_SEC,
        # Moonraker / Klipper
        "MOONRAKER_URL": MOONRAKER_URL, "HTTP_POST_URL": HTTP_POST_URL,
        # Stacks / Sorter
        "CARD_THICKNESS_MM": CARD_THICKNESS_MM,
        "REMEASURE_EVERY": REMEASURE_EVERY,
        # Debug & Paths
        "DEBUG_LEVEL": DEBUG_LEVEL,
        "SC_IMG_CACHE_DIR": SC_IMG_CACHE_DIR,
        # Persistence
        "HISTORY_DIR": HISTORY_DIR, "HISTORY_IMG_DIR": HISTORY_IMG_DIR,
        "HISTORY_JSON_EXT": HISTORY_JSON_EXT, "JPEG_QUALITY_SNAP": JPEG_QUALITY_SNAP,
        "JPEG_QUALITY_CMP": JPEG_QUALITY_CMP, "JPEG_QUALITY_THUMB": JPEG_QUALITY_THUMB,
        "HISTORY_API_DEFAULT_LIMIT": HISTORY_API_DEFAULT_LIMIT,
        "HISTORY_API_MAX_LIMIT": HISTORY_API_MAX_LIMIT,
        
        # AI / YOLOv5
        "AI_ENABLED": AI_ENABLED, "AI_USE_FOR_CARDS": AI_USE_FOR_CARDS, "AI_USE_FOR_ROIS": AI_USE_FOR_ROIS,
        "AI_ROIS_SNAPSHOT_ONLY": bool(globals().get("AI_ROIS_SNAPSHOT_ONLY", False)),
        "AI_ONLY_MODE": AI_ONLY_MODE, "AI_MODEL_PATH": AI_MODEL_PATH, "YOLOV5_DIR": YOLOV5_DIR,
        "AI_IMG_SIZE": AI_IMG_SIZE, "AI_CONF_THRES": AI_CONF_THRES, "AI_IOU_THRES": AI_IOU_THRES,
        "AI_MAX_DETS": AI_MAX_DETS, "AI_CLASS_NAMES": AI_CLASS_NAMES,
# Server
        "HOST": HOST, "PORT": PORT,
        # Tracking / motion / appearance
        "TRACK_ALPHA": TRACK_ALPHA, "TRACK_BETA": TRACK_BETA, "TRACK_DEADBAND_PX": TRACK_DEADBAND_PX,
        "LOCK_IOU_THRESH": LOCK_IOU_THRESH, "ACQUIRE_FRAMES": ACQUIRE_FRAMES,
        "DROP_MISS_FRAMES": DROP_MISS_FRAMES, "PREDICT_HOLD": PREDICT_HOLD,
        "STEADY_SPEED_PX": STEADY_SPEED_PX, "WARP_EXPAND_PCT": WARP_EXPAND_PCT,
        "WARP_CROP_PCT": WARP_CROP_PCT, "DETECT_MAX_FPS": DETECT_MAX_FPS,
        "APPEARANCE_ALIGN_ENABLE": APPEARANCE_ALIGN_ENABLE,
        "APPEARANCE_AB_STRENGTH": APPEARANCE_AB_STRENGTH,
        "APPEARANCE_SAT_STRENGTH": APPEARANCE_SAT_STRENGTH,
        "APPEARANCE_GAMMA_CLAMP": L(APPEARANCE_GAMMA_CLAMP),
        "TONE_ALIGN_ENABLE": TONE_ALIGN_ENABLE,
        "TRACK_BOX_BLEND": TRACK_BOX_BLEND,
    }

@app.get("/api/settings")
def api_settings_get():
    # Prefer the saved file; fall back to effective if missing.
    saved = _settings_load() or {}
    eff   = _effective_settings()
    # Merge defaults (eff) under saved so form is pre-filled nicely.
    # Also return the path and effective values for debugging (UI ignores extras).
    merged = {**eff, **saved}
    merged["__path"] = os.path.abspath(SETTINGS_PATH)
    merged["__effective"] = eff
    return jsonify(merged)

@app.post("/api/settings")
def api_settings_post():
    incoming = request.get_json(force=True, silent=True) or {}

    # normalize booleans
    for k, v in list(incoming.items()):
        if isinstance(v, str) and v.lower() in ("true","false","1","0","on","off"):
            incoming[k] = v.lower() in ("true","1","on")

    cur = _settings_load() or {}
    cur.update(incoming)
    _settings_save(cur)

    _apply_settings_runtime(incoming)

    if any(k in incoming for k in ("MATCH_USE_ART","CARD_W","CARD_H")):
        _scry_img_feats.clear()

    needs_restart = any(k in RESTART_KEYS for k in incoming.keys())
    if any(k in incoming for k in ("REMEASURE_EVERY","CARD_THICKNESS_MM")):
        _apply_printer_settings_with_retry(reason="settings_save")
    return jsonify({"ok": True, "saved": incoming, "restart_required": needs_restart})

@app.post("/api/manual_crop")
def api_manual_crop():
    data = request.get_json(force=True, silent=True) or {}
    quad = data.get("quad") or data.get("corners")
    enabled = data.get("enabled")
    new_quad, enabled_state = _set_manual_crop(quad, enabled)
    cur = _settings_load() or {}
    cur["MANUAL_CROP_QUAD"] = new_quad.tolist()
    cur["MANUAL_CROP_ENABLED"] = bool(enabled_state)
    _settings_save(cur)
    return jsonify({"ok": True, "quad": new_quad.tolist(), "enabled": enabled_state})

@app.post("/api/restart")
def api_restart():
    # return immediately, then exit the process after a tiny delay
    def _bye():
        time.sleep(0.4)
        # Prefer graceful exit if you have one; falling back to os._exit
        os._exit(3)
    threading.Thread(target=_bye, daemon=True).start()
    return jsonify({"ok": True, "restarting": True})
@app.post('/api/scan/<int:sid>/edit')
def api_scan_edit(sid):
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    number_raw = (data.get("number") or "").strip()
    set_hint = (data.get("set") or "").strip().lower()
    scry_id = (data.get("scry_id") or "").strip()
    with history_lock:
        target_ref = next((e for e in scan_history if e["id"] == sid), None)
        if not target_ref:
            return jsonify({"ok": False, "error": "not found"}), 404
        # Work on a copy to avoid holding the lock during network/compute
        target = dict(target_ref)
        snap_bytes = target_ref.get("snap_jpg")

    if name:
        target["name"] = name
    if number_raw:
        target["number_raw"] = number_raw
        target["number"] = _parse_collector_for_display(number_raw)
    if set_hint:
        target["set_hint"] = set_hint
    choice = None
    if scry_id:
        choice = _scryfall_request(f"https://api.scryfall.com/cards/{scry_id}") or None
    if choice is None:
        choice = _scryfall_lookup(
            target.get("name",""),
            target.get("number_raw",""),
            set_hint=set_hint or target.get("set_hint","")
        )
        choice = _scry_fix_mismatch(
            choice,
            target.get("name",""),
            target.get("number_raw",""),
            set_hint or target.get("set_hint","")
        )
    if choice:
        target["scry"] = choice
        if not name and choice.get("name"):
            target["name"] = choice.get("name")
        if not number_raw and choice.get("collector_number"):
            try:
                cn_raw = str(choice.get("collector_number") or "").strip()
                if cn_raw:
                    target["number_raw"] = cn_raw
                    target["number"] = _parse_collector_for_display(cn_raw)
            except Exception:
                pass
        if not set_hint and choice.get("set"):
            target["set_hint"] = str(choice.get("set") or "").lower()

    snap = _decode_jpg(snap_bytes)
    score, ok, details = 0.0, False, None
    if choice is None:
        choice = target.get("scry")
    if snap is not None and choice:
        scry_img = _fetch_scry_image(choice)
        if scry_img is not None:
            score, ok, details = compare_snapshot_to_scryfall(snap, scry_img, return_details=True, scry_card=choice)
            if details:
                vis = _render_compare_visual(details)
                target["cmp_stats"] = {k:v for k,v in details.items() if k != "orb_dbg"}
                target["cmp_jpg"] = _jpeg_bytes(vis, JPEG_QUALITY_CMP) if vis is not None else None
    target["match_score"] = round(float(score), 3) if details else target.get("match_score", 0.0)
    target["match_ok"] = bool(ok) if details else target.get("match_ok", False)
    global current_loaded_scan_id
    ts_now = time.time()
    if current_loaded_scan_id == sid:
        with ocr_lock:
            ocr_state["name"] = target.get("name","")
            ocr_state["number"] = target.get("number","")
            ocr_state["number_raw"] = target.get("number_raw","")
            ocr_state["set_hint"] = target.get("set_hint","")
            ocr_state["match_score"] = target["match_score"]
            ocr_state["match_ok"] = target["match_ok"]
            ocr_state["updated_at"] = ts_now
    with cardinfo_lock:
        cardinfo_state["scry"] = target.get("scry") or choice
        cardinfo_state["fx"] = {"rate": _get_usd_to_cad()}
        cardinfo_state["last_updated"] = ts_now
        if details:
            _publish_compare_visual(details)
    with history_lock:
        target_ref = next((e for e in scan_history if e["id"] == sid), None)
        if not target_ref:
            return jsonify({"ok": False, "error": "not found"}), 404
        target_ref.update(target)
        _persist_entry_to_disk(target_ref)
    return jsonify({"ok": True, "id": sid, "match_score": target["match_score"], "match_ok": target["match_ok"]})

def _bytes_to_data_url(blob, mime="image/jpeg"):
    if not blob:
        return None
    encoded = base64.b64encode(blob).decode("ascii")
    return f"data:{mime};base64,{encoded}"

def _summarize_entry_for_modal(entry):
    scry = entry.get("scry") or {}
    return {
        "id": entry.get("id"),
        "name": entry.get("name",""),
        "number": entry.get("number",""),
        "set_hint": entry.get("set_hint",""),
        "status": entry.get("status",""),
        "match_score": entry.get("match_score",0.0),
        "match_ok": entry.get("match_ok"),
        "flagged": entry.get("flagged", False),
        "review_reasons": entry.get("review_reasons", []),
        "review_level": entry.get("review_level"),
        "auto_name_fix": entry.get("auto_name_fix", False),
        "scry_name": scry.get("name",""),
        "scry_set": scry.get("set",""),
        "scry_cn": scry.get("collector_number",""),
        "cmp_data_url": _bytes_to_data_url(entry.get("cmp_jpg")),
        "thumb_url": f"/api/scan/{entry.get('id')}/thumb" if entry.get("id") else None,
    }

def _cleanup_reprocess_cache(now=None):
    now = now or time.time()
    stale = []
    for token, meta in list(_reprocess_cache.items()):
        if now - meta.get("ts", 0) > REPROCESS_CACHE_TTL:
            stale.append(token)
    for token in stale:
        _reprocess_cache.pop(token, None)

def _store_reprocess_result(sid, payload):
    token = uuid.uuid4().hex
    now = time.time()
    with _reprocess_cache_lock:
        _cleanup_reprocess_cache(now)
        _reprocess_cache[token] = {"sid": sid, "ts": now, "payload": payload}
    return token

def _pop_reprocess_result(token):
    with _reprocess_cache_lock:
        meta = _reprocess_cache.pop(token, None)
    return meta

def _serialize_reprocess_candidate(payload, sid):
    scry = payload.get("scry") or {}
    ocr = payload.get("ocr") or {}
    return {
        "id": sid,
        "name": ocr.get("name") or ocr.get("name_raw") or "",
        "number": ocr.get("number") or "",
        "number_raw": ocr.get("number_raw") or "",
        "set_hint": ocr.get("set_hint") or "",
        "match_score": payload.get("match_score", 0.0),
        "match_ok": payload.get("match_ok"),
        "flagged": payload.get("flagged", False),
        "status": payload.get("status"),
        "review_reasons": payload.get("review_reasons", []),
        "review_level": payload.get("review_level"),
        "auto_name_fix": payload.get("auto_name_fix", False),
        "inputs_present": payload.get("inputs_present"),
        "scry_name": scry.get("name",""),
        "scry_set": scry.get("set",""),
        "scry_cn": scry.get("collector_number",""),
        "scry_image": (scry.get("image_uris") or {}).get("normal") or (scry.get("card_faces") or [{}])[0].get("image_uris", {}).get("normal") if scry else None,
        "cmp_data_url": _bytes_to_data_url(payload.get("cmp_jpg")),
    }

def _apply_reprocess_candidate(target, candidate):
    ocr = candidate.get("ocr") or {}
    target["name"] = ocr.get("name") or ocr.get("name_raw") or target.get("name","")
    target["name_conf"] = float(ocr.get("name_conf", target.get("name_conf", 0.0)))
    target["number_raw"] = ocr.get("number_raw") or ocr.get("number") or ""
    target["number"] = _parse_collector_for_display(target["number_raw"])
    target["number_conf"] = round(float(ocr.get("number_conf", target.get("number_conf", 0.0))), 1) if target["number"] else 0.0
    target["set_hint"] = (ocr.get("set_hint") or "").strip().lower()
    target["foil"] = bool(ocr.get("foil", target.get("foil", False)))
    target["foil_score"] = float(ocr.get("foil_score", target.get("foil_score", 0.0)))
    target["scry"] = candidate.get("scry")
    target["match_score"] = round(float(candidate.get("match_score", 0.0)), 3)
    target["match_ok"] = bool(candidate.get("match_ok")) if candidate.get("match_ok") is not None else None
    status = (candidate.get("status") or ("pass" if target["match_ok"] else "fail")).lower()
    target["status"] = status
    target["flagged"] = bool(candidate.get("flagged") or (status != "pass"))
    target["review_reasons"] = list(candidate.get("review_reasons") or [])
    target["review_level"] = candidate.get("review_level")
    target["auto_name_fix"] = bool(candidate.get("auto_name_fix", target.get("auto_name_fix", False)))
    if candidate.get("inputs_present"):
        target["inputs_present"] = dict(candidate.get("inputs_present"))
    # Re-sanitize compare stats in case a previous version left numpy arrays or debug blobs in place.
    target["cmp_stats"] = _cmp_stats_sanitized(candidate.get("cmp_stats") or {})
    target["cmp_jpg"] = candidate.get("cmp_jpg")

def _run_reprocess_on_entry(entry, progress_cb=None):
    perf_enabled = bool(REPROCESS_TIMER_DEBUG)
    perf_marks = []
    perf_start = perf_last = time.perf_counter()

    def _perf_mark(label):
        nonlocal perf_last
        if not perf_enabled:
            return
        now = time.perf_counter()
        perf_marks.append((label, now - perf_last))
        perf_last = now

    def _perf_log(status):
        if not perf_enabled:
            return
        total = time.perf_counter() - perf_start
        detail = ", ".join(f"{label}={dur:.3f}s" for label, dur in perf_marks if dur is not None)
        msg = f"id={entry.get('id')} status={status} total={total:.3f}s"
        if detail:
            msg += f" :: {detail}"
        _dbg("PERF REPROCESS", msg)

    perf_status = "ok"
    try:
        snap = _decode_jpg(entry.get("snap_jpg"))
        _perf_mark("decode_snapshot")
        if snap is None:
            if progress_cb:
                progress_cb("no_snapshot", 1.0)
            perf_status = "no_snapshot"
            return None
        # Optional downscale for faster OCR
        try:
            maxw = int(globals().get("REPROCESS_MAX_WIDTH", 0) or 0)
            if maxw > 0:
                import cv2
                h, w = snap.shape[:2]
                if w > maxw:
                    scale = maxw / float(w)
                    snap = cv2.resize(snap, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
        except Exception:
            pass
        _perf_mark("resize")
        # Optional AI ROI refresh
        if not bool(globals().get("REPROCESS_SKIP_AI_ROIS", False)):
            try:
                with _preserve_ai_rois():
                    _update_ai_rois(snap)
            except Exception:
                pass
        _perf_mark("roi_update")
        try:
            if progress_cb:
                progress_cb("ocr", 0.15)
            # Reprocess: force fast OCR and temporarily disable heavy salvage for speed.
            _fcfg = globals().get("_FAST_CFG", {})
            _save_tess = bool(_fcfg.get("USE_TESSERACT", _fcfg.get("USE_PADDLE", True)))
            _save_foil = globals().get("FOIL_DETECT", True)
            if bool(globals().get("REPROCESS_SKIP_FOIL_DETECT", True)):
                globals()["FOIL_DETECT"] = False
            try:
                with _force_fast_ocr_scope(True):
                    prev_from_snapshot = getattr(_OCR_CONTEXT, "from_snapshot", False)
                    _OCR_CONTEXT.from_snapshot = True
                    try:
                        ocr = ocr_from_card_upright(snap.copy())
                    finally:
                        _OCR_CONTEXT.from_snapshot = prev_from_snapshot
            finally:
                globals()["FOIL_DETECT"] = _save_foil
                _fcfg["USE_TESSERACT"] = _save_tess
                _fcfg["USE_PADDLE"] = _save_tess
        except Exception as exc:
            _dbg("REPROCESS", f"OCR failed: {exc}")
            ocr = {"name": "", "name_raw": "", "number": "", "number_raw": "", "set_hint": ""}
        _perf_mark("ocr")
        name = (ocr.get("name") or ocr.get("name_raw") or "").strip()
        number_raw = (ocr.get("number_raw") or ocr.get("number") or "").strip()
        set_hint = (ocr.get("set_hint") or "").strip().lower()
        choice = None
        if progress_cb:
            progress_cb("ocr", 0.35)
        if name or number_raw:
            try:
                _old_timeout = globals().get("SCRYFALL_TIMEOUT", SCRYFALL_TIMEOUT)
                globals()["SCRYFALL_TIMEOUT"] = float(globals().get("FAST_REPROCESS_SCRY_TIMEOUT", 30.0))
                choice = _scryfall_lookup(name, number_raw, set_hint=set_hint)
                if choice is None and set_hint:
                    choice = _scryfall_lookup(name, number_raw, set_hint="")
            except Exception as exc:
                _dbg("REPROCESS", f"Scryfall lookup failed: {exc}")
                choice = None
            finally:
                globals()["SCRYFALL_TIMEOUT"] = _old_timeout
            _perf_mark("scry_lookup")
        if choice is not None:
            scry_img = _fetch_scry_image(choice)
            _perf_mark("scry_image")
        else:
            scry_img = None
        if progress_cb:
            progress_cb("scry", 0.6)
        match_score, match_ok, cmp_details = 0.0, False, None
        if scry_img is not None:
            try:
                match_score, match_ok, cmp_details = compare_snapshot_to_scryfall(
                    snap, scry_img, return_details=True, scry_card=choice
                )
            except Exception as exc:
                _dbg("REPROCESS", f"Compare failed: {exc}")
                match_score, match_ok, cmp_details = 0.0, False, None
            _perf_mark("compare")
        if progress_cb:
            progress_cb("compare", 0.85)
        cmp_jpg = None
        cmp_stats = {}
        if cmp_details:
            try:
                vis = _render_compare_visual(cmp_details)
                if vis is not None:
                    cmp_jpg = _jpeg_bytes(vis, JPEG_QUALITY_CMP)
                # Ensure compare stats are JSON-safe (strip numpy arrays, debug images, etc.)
                cmp_stats = _cmp_stats_sanitized(cmp_details)
            except Exception:
                cmp_jpg = None
                cmp_stats = {}
            _perf_mark("render_compare")

        decision = _derive_review_outcome(
            ocr,
            choice,
            match_score,
            match_ok,
            snapshot_ok=(snap is not None and getattr(snap, "size", 0) > 0),
            scry_img_ok=(scry_img is not None and getattr(scry_img, "size", 0) > 0),
            match_mode=(choice or {}).get("_match_mode"),
            allow_auto_name_fix=True,
        )
        ocr = decision["ocr"]
        _perf_mark("review")

        if progress_cb:
            progress_cb("done", 1.0)
        return {
            "ocr": ocr,
            "scry": choice,
            "match_score": round(float(match_score or 0.0), 3),
            "match_ok": bool(match_ok) if match_ok is not None else None,
            "flagged": decision["flagged"],
            "status": decision["status"],
            "review_level": decision.get("review_level"),
            "review_reasons": decision["review_reasons"],
            "auto_name_fix": decision.get("auto_name_fix", False),
            "inputs_present": decision.get("inputs_present"),
            "cmp_stats": cmp_stats,
            "cmp_jpg": cmp_jpg,
            "cmp_details": cmp_details,
        }
    except Exception:
        perf_status = "error"
        raise
    finally:
        _perf_log(perf_status)

def _cleanup_reprocess_jobs(now=None):
    now = now or time.time()
    with _reprocess_jobs_lock:
        stale = [jid for jid, job in _reprocess_jobs.items() if (now - job.get("created", now)) > REPROCESS_JOB_TTL]
        for jid in stale:
            _reprocess_jobs.pop(jid, None)

def _reprocess_job_update(job_id, **updates):
    with _reprocess_jobs_lock:
        job = _reprocess_jobs.get(job_id)
        if not job:
            return
        job.update(updates)

def _reprocess_job_record_progress(job_id, phase, progress=None, current_id=None, card_progress=None):
    with _reprocess_jobs_lock:
        job = _reprocess_jobs.get(job_id)
        if not job:
            return
        job["current_phase"] = phase
        if progress is not None:
            prog_clamped = max(0.0, min(1.0, float(progress)))
            job["current_progress"] = prog_clamped
            job["progress"] = prog_clamped
        if current_id is not None:
            job["current_card"] = current_id
        if card_progress is not None:
            job["current_card_progress"] = max(0.0, min(1.0, float(card_progress)))

def _start_single_reprocess_job(sid):
    job_id = uuid.uuid4().hex
    job = {
        "id": job_id,
        "kind": "single",
        "sid": sid,
        "status": "running",
        "created": time.time(),
        "current_phase": "queued",
        "current_progress": 0.0,
        "result": None,
        "error": None,
    }
    with _reprocess_jobs_lock:
        _cleanup_reprocess_jobs(job["created"])
        _reprocess_jobs[job_id] = job

    def _worker():
        with history_lock:
            target_ref = next((e for e in scan_history if e["id"] == sid), None)
            target = dict(target_ref) if target_ref else None
        if not target:
            _reprocess_job_update(job_id, status="error", error="not found")
            return
        old_summary = _summarize_entry_for_modal(target)
        try:
            def _cb(phase, frac):
                _reprocess_job_record_progress(job_id, phase, frac, current_id=sid, card_progress=frac)
            result = _run_reprocess_on_entry(target, progress_cb=_cb)
            if result is None:
                raise RuntimeError("no snapshot available")
            token = _store_reprocess_result(sid, result)
            payload = {
                "token": token,
                "old": old_summary,
                "candidate": _serialize_reprocess_candidate(result, sid),
            }
            _reprocess_job_update(job_id, status="done", result=payload, current_progress=1.0, current_phase="done")
        except Exception as exc:
            _reprocess_job_update(job_id, status="error", error=str(exc))
    threading.Thread(target=_worker, daemon=True).start()
    return job_id

def _start_batch_reprocess_job(entries):
    job_id = uuid.uuid4().hex
    total = len(entries)
    job = {
        "id": job_id,
        "kind": "batch",
        "status": "running",
        "created": time.time(),
        "total": total,
        "done": 0,
        "progress": 0.0,
        "current_card": None,
        "current_phase": "queued",
        "current_card_progress": 0.0,
        "items": [],
        "error": None,
    }
    with _reprocess_jobs_lock:
        _cleanup_reprocess_jobs(job["created"])
        _reprocess_jobs[job_id] = job

    def _worker():
        try:
            if total == 0:
                _reprocess_job_update(job_id, status="done", progress=1.0)
                return
            results = []
            for idx, entry in enumerate(entries):
                rid = entry.get("id")
                old_summary = _summarize_entry_for_modal(entry)
                _reprocess_job_record_progress(
                    job_id, "start_card", progress=(idx/float(total) if total else 0.0),
                    current_id=rid, card_progress=0.0
                )
                def _cb(phase, frac):
                    with _reprocess_jobs_lock:
                        job_ref = _reprocess_jobs.get(job_id)
                        done_cards = job_ref.get("done", 0) if job_ref else 0
                    overall = ((done_cards + frac) / float(total)) if total else 1.0
                    _reprocess_job_record_progress(job_id, phase, overall, current_id=rid, card_progress=frac)
                result = _run_reprocess_on_entry(entry, progress_cb=_cb)
                if result:
                    token = _store_reprocess_result(rid, result)
                    results.append({
                        "id": rid,
                        "token": token,
                        "old": old_summary,
                        "candidate": _serialize_reprocess_candidate(result, rid),
                    })
                with _reprocess_jobs_lock:
                    job_ref = _reprocess_jobs.get(job_id)
                    if job_ref:
                        job_ref["done"] = idx + 1
                        job_ref["progress"] = (idx + 1) / float(total)
                        job_ref["current_card_progress"] = 0.0
                        job_ref["current_card"] = None
            _reprocess_job_update(job_id, status="done", items=results, progress=1.0, current_phase="done")
        except Exception as exc:
            _reprocess_job_update(job_id, status="error", error=str(exc))
    threading.Thread(target=_worker, daemon=True).start()
    return job_id

@app.post('/api/scan/<int:sid>/reprocess')
def api_scan_reprocess(sid):
    mode = (request.args.get("mode") or "").lower()
    if mode in ("async", "1", "true"):
        target = next((dict(e) for e in scan_history if e["id"] == sid), None)
        if not target:
            return jsonify({"ok": False, "error": "not found"}), 404
        job_id = _start_single_reprocess_job(sid)
        return jsonify({"ok": True, "job": job_id})
    target = next((e for e in scan_history if e["id"] == sid), None)
    if not target:
        return jsonify({"ok": False, "error": "not found"}), 404
    result = _run_reprocess_on_entry(target)
    if result is None:
        return jsonify({"ok": False, "error": "no snapshot available"}), 400
    token = _store_reprocess_result(sid, result)
    return jsonify({
        "ok": True,
        "token": token,
        "old": _summarize_entry_for_modal(target),
        "candidate": _serialize_reprocess_candidate(result, sid),
    })

@app.post('/api/scan/<int:sid>/reprocess/apply')
def api_scan_reprocess_apply(sid):
    data = request.get_json(silent=True) or {}
    token = (data.get("token") or "").strip()
    if not token:
        return jsonify({"ok": False, "error": "token required"}), 400
    if _scan_in_progress():
        return jsonify({"ok": False, "error": "busy processing a live scan"}), 409
    meta = _pop_reprocess_result(token)
    if not meta or meta.get("sid") != sid:
        return jsonify({"ok": False, "error": "token expired"}), 400
    candidate = meta.get("payload")
    with history_lock:
        target = next((e for e in scan_history if e["id"] == sid), None)
        if not target:
            return jsonify({"ok": False, "error": "not found"}), 404
        _apply_reprocess_candidate(target, candidate)
        _persist_entry_to_disk(target)

    if current_loaded_scan_id == sid:
        ts_now = time.time()
        with ocr_lock:
            ocr = candidate.get("ocr") or {}
            ocr_state["name"] = target.get("name","")
            ocr_state["name_raw"] = ocr.get("name_raw") or target.get("name","")
            ocr_state["name_conf"] = float(ocr.get("name_conf", target.get("name_conf", 0.0)))
            ocr_state["number"] = target.get("number","")
            ocr_state["number_raw"] = target.get("number_raw","")
            ocr_state["set_hint"] = target.get("set_hint","")
            ocr_state["match_score"] = target.get("match_score",0.0)
            ocr_state["match_ok"] = target.get("match_ok")
            ocr_state["flagged"] = target.get("flagged", False)
            ocr_state["updated_at"] = ts_now
        with cardinfo_lock:
            cardinfo_state["scry"] = target.get("scry")
            cardinfo_state["fx"] = {"rate": _get_usd_to_cad()}
            cardinfo_state["last_updated"] = ts_now
        if candidate.get("cmp_details"):
            _publish_compare_visual(candidate["cmp_details"])
        elif candidate.get("cmp_jpg"):
            jpg = candidate.get("cmp_jpg")
            _update_compare_cache(
                _decode_jpg(jpg) if jpg else None,
                dict(candidate.get("cmp_stats") or {})
            )
    return jsonify({"ok": True, "id": sid})

@app.post('/api/reprocess/flagged')
def api_reprocess_flagged():
    mode = (request.args.get("mode") or "async").lower()
    limit = int(request.args.get("limit") or 0)
    with history_lock:
        snapshots = [dict(e) for e in scan_history if (e.get("flagged") or (str(e.get("status","")).lower() == "fail"))]
    if limit > 0:
        snapshots = snapshots[:limit]
    if mode in ("async", "1", "true"):
        job_id = _start_batch_reprocess_job(snapshots)
        return jsonify({"ok": True, "job": job_id, "total": len(snapshots)})
    results = []
    for entry in snapshots:
        rid = entry.get("id")
        candidate = _run_reprocess_on_entry(entry)
        if candidate is None:
            continue
        token = _store_reprocess_result(rid, candidate)
        results.append({
            "id": rid,
            "token": token,
            "old": _summarize_entry_for_modal(entry),
            "candidate": _serialize_reprocess_candidate(candidate, rid),
        })
    return jsonify({"ok": True, "total": len(snapshots), "items": results})

@app.get('/api/reprocess/job/<job_id>')
def api_reprocess_job(job_id):
    _cleanup_reprocess_jobs()
    with _reprocess_jobs_lock:
        job = _reprocess_jobs.get(job_id)
        if not job:
            return jsonify({"ok": False, "error": "job not found"}), 404
        payload = {
            "ok": True,
            "id": job_id,
            "kind": job.get("kind"),
            "status": job.get("status"),
            "current_phase": job.get("current_phase", "queued"),
            "current_progress": job.get("current_progress", 0.0),
            "error": job.get("error"),
        }
        if job.get("kind") == "single":
            if job.get("status") == "done":
                payload["result"] = job.get("result")
        elif job.get("kind") == "batch":
            payload.update({
                "total": job.get("total", 0),
                "done": job.get("done", 0),
                "progress": job.get("progress", 0.0),
                "current_card": job.get("current_card"),
                "current_card_progress": job.get("current_card_progress", 0.0),
            })
            if job.get("status") == "done":
                payload["items"] = job.get("items", [])
        return jsonify(payload)

# ===== Shared filter helpers =====
def _parse_since(s):
    if not s: return None
    m = re.match(r'^\s*(\d+)\s*([smhd])\s*$', s.lower() or "")
    if not m: return None
    n, u = int(m.group(1)), m.group(2)
    mul = {'s':1, 'm':60, 'h':3600, 'd':86400}[u]
    return time.time() - n*mul

def _parse_scan_filters(a):
    status   = (a.get("status") or "all").lower()
    score_min = max(0.0, min(1.0, float(a.get("score_min") or 0)))
    score_max = max(0.0, min(1.0, float(a.get("score_max") or 1)))
    ts_from   = float(a.get("ts_from")) if a.get("ts_from") else (_parse_since(a.get("since")) or 0)
    ts_to     = float(a.get("ts_to")) if a.get("ts_to") else 1e15
    q         = (a.get("q") or "").strip().lower()
    set_code  = (a.get("set") or "").strip().lower()
    foil_q    = a.get("foil")
    return {
        "status": status, "score_min": score_min, "score_max": score_max,
        "ts_from": ts_from, "ts_to": ts_to, "q": q, "set": set_code, "foil_q": foil_q
    }

def _apply_scan_filters(rows, F):
    def _status_ok(e):
        st = (e.get("status","") or "").lower()
        if not st:
            st = "pass" if e.get("match_ok") else "fail"
        if F["status"] == "all":     return True
        if F["status"] == "pass_no_review": return (st == "pass") and not bool(e.get("flagged"))
        if F["status"] == "pass":    return st == "pass"
        if F["status"] == "fail":    return st == "fail"
        if F["status"] in ("flagged", "review"): return (st == "review") or bool(e.get("flagged"))
        return True

    def _parse_bool(v): return str(v).lower() in ("1","true","yes","y","on")

    out = []
    for e in rows:
        if not _status_ok(e): continue
        if not (F["ts_from"] <= float(e.get("ts", 0)) <= F["ts_to"]): continue
        sc = float(e.get("match_score", 0.0))
        if sc < F["score_min"] or sc > F["score_max"]: continue

        c = e.get("scry") or {}
        if F["q"]:
            ql = F["q"].lower()
            hay = " ".join([
                e.get("name",""),
                c.get("name","") or "",
                (c.get("set","") or ""),
                str(c.get("collector_number") or e.get("number") or "")
            ]).lower()
            if ql not in hay: 
                continue

        if F["set"] and F["set"] != ((c.get("set","") or "").lower()): continue
        if F["foil_q"] is not None:
            want = _parse_bool(F["foil_q"])
            if bool(e.get("foil", False)) != want: continue
        out.append(e)
    return out

def _parse_limit_param(val):
    if val is None:
        return None
    sval = str(val).strip().lower()
    if not sval:
        return None
    if sval in ("all", "*", "max", "infinite", "inf", "unlimited"):
        return 0
    try:
        return int(val)
    except Exception:
        return None

def _resolve_history_limit(requested, total_rows):
    total = max(0, int(total_rows or 0))
    if total == 0:
        return 0
    limit = HISTORY_API_DEFAULT_LIMIT if requested is None else int(requested)
    if limit <= 0:
        limit = total
    max_lim = int(HISTORY_API_MAX_LIMIT or 0)
    if max_lim > 0:
        limit = min(limit, max_lim)
    return max(1, min(limit, total))

@app.route('/api/scans')
def api_scans():
    a = request.args
    sort_by   = (a.get("sort_by") or "ts").lower()
    sort_dir  = (a.get("sort_dir") or "desc").lower()
    limit_req = _parse_limit_param(a.get("limit"))
    offset    = max(0, int(a.get("offset") or 0))

    F = _parse_scan_filters(a)
    with history_lock:
        rows = list(scan_history)
    out = _apply_scan_filters(rows, F)

    def _key(e):
        if sort_by == "score": return float(e.get("match_score", 0.0))
        if sort_by == "name":  return (e.get("name","").lower(), e.get("ts",0))
        if sort_by == "set":   return ((e.get("scry") or {}).get("set",""), e.get("number",""), e.get("ts",0))
        return float(e.get("ts",0))
    out.sort(key=_key, reverse=(sort_dir!="asc"))

    total = len(out)
    pass_count = sum(1 for e in out if str(e.get("status","")).lower() == "pass")
    fail_count = sum(1 for e in out if str(e.get("status","")).lower() == "fail")
    review_count = sum(1 for e in out if str(e.get("status","")).lower() == "review")
    flagged_count = sum(1 for e in out if bool(e.get("flagged")))
    limit = _resolve_history_limit(limit_req, total)
    if offset >= total:
        out = []
    else:
        out = out[offset:offset+limit] if limit else []
    items = [{
        "id": e["id"], "ts": e["ts"], "name": e.get("name",""), "number": e.get("number",""),
        "status": e.get("status","fail"), "match_score": e.get("match_score",0.0),
        "match_ok": e.get("match_ok", False), "flagged": e.get("flagged", False),
        "foil": bool(e.get("foil", False)),
        "scry_name": (e.get("scry") or {}).get("name",""),
        "scry_set":  (e.get("scry") or {}).get("set",""),
        "scry_cn":   (e.get("scry") or {}).get("collector_number",""),
        "set_hint": e.get("set_hint",""),
        "number_raw": e.get("number_raw",""),
        "thumb_ok": bool(e.get("thumb") or e.get("snap_jpg")),
        "review_reasons": e.get("review_reasons", []),
        "review_level": e.get("review_level", "info"),
    } for e in out]
    return jsonify({
        "items": items,
        "total": total,
        "count": total,
        "offset": offset,
        "limit": limit,
        "returned": len(items),
        "summary": {
            "total": total,
            "pass": pass_count,
            "fail": fail_count,
            "review": review_count,
            "flagged": flagged_count,
        },
    })

@app.route('/api/scan/<int:sid>/thumb')
def api_scan_thumb(sid):
    with history_lock:
        for e in scan_history:
            if e["id"] == sid:
                return Response(e.get("thumb") or e.get("snap_jpg") or b"", mimetype='image/jpeg')
    return jsonify({"ok": False, "error": "not found"}), 404

@app.route('/api/scan/<int:sid>/load', methods=['POST'])
def api_scan_load(sid):
    global scanned_card, last_scan_ts, snapshot_seq, current_loaded_scan_id, _last_cmp_img, _last_cmp_stats, preview_card
   
    if _scan_in_progress():
        return jsonify({"ok": False, "error": "busy processing a live scan"}), 409

    with history_lock:
        target_ref = next((e for e in scan_history if e["id"] == sid), None)
        target = dict(target_ref) if target_ref else None
    if not target:
        return jsonify({"ok": False, "error": "not found"}), 404
    snap = _decode_jpg(target.get("snap_jpg"))
    if snap is None:
        return jsonify({"ok": False, "error": "no snapshot bytes"}), 400
    ts_now = time.time()
    with scan_lock:
        preview_card = snap.copy()
    # Make sure the cached /card frame reflects the loaded history item even if the stream is paused
    try:
        STREAM_CACHE['card'] = _jpeg_bytes(preview_card, int(STREAM_STATE.get('quality', 80)))
        STREAM_STATE['paused'] = False
        STREAM_STATE['paused_reason'] = ''
    except Exception:
        pass
    _clear_ocr(preserve_seq=True)
    with ocr_lock:
        ocr_state["name"]       = target.get("name", "")
        ocr_state["name_raw"]   = target.get("name", "")
        ocr_state["name_conf"]  = float(target.get("name_conf", 0.0))
        ocr_state["number"]     = target.get("number", "")
        ocr_state["number_raw"] = target.get("number_raw", "")
        ocr_state["set_hint"]   = target.get("set_hint", "")
        ocr_state["foil"]       = bool(target.get("foil", False))
        ocr_state["match_score"]= float(target.get("match_score", 0.0))
        ocr_state["match_ok"]   = bool(target.get("match_ok", False))
        ocr_state["flagged"]    = bool(target.get("flagged", False))
        ocr_state["status"]     = target.get("status", "")
        ocr_state["updated_at"] = ts_now
    with cardinfo_lock:
        cardinfo_state.clear()
        cardinfo_state["scry"] = target.get("scry")
        cardinfo_state["fx"]   = {"rate": _get_usd_to_cad()}
        cardinfo_state["last_updated"] = ts_now
    cmp_img, cmp_stats = _repair_cmp_visual(target)
    if cmp_img is None:
        cmp_img = _decode_jpg(target.get("cmp_jpg"))
        cmp_stats = cmp_stats or dict(target.get("cmp_stats") or {})
    _update_compare_cache(cmp_img, dict(cmp_stats or {}))
    current_loaded_scan_id = sid
    return jsonify({"ok": True, "loaded_id": sid})

@app.route('/api/scan/<int:sid>/status', methods=['POST'])
def api_scan_status(sid):
    data = request.get_json(silent=True) or {}
    status = (data.get("status") or "all").lower()
    if status == "flagged":
        status = "review"
    if status not in ("pass", "fail", "review"):
        return jsonify({"ok": False, "error": "status must be 'pass', 'fail', or 'review'"}), 400
    with history_lock:
        target = next((e for e in scan_history if e["id"] == sid), None)
        if not target:
            return jsonify({"ok": False, "error": "not found"}), 404
        target["status"] = status
        target["flagged"] = (status != "pass")
        _persist_entry_to_disk(target)
    return jsonify({"ok": True, "id": sid, "status": status})

@app.route('/api/scan/<int:sid>/delete', methods=['POST'])
def api_scan_delete(sid):
    """Delete a scan from memory + disk, and clear viewer state if it was loaded."""
    global current_loaded_scan_id, preview_card, _last_cmp_img, _last_cmp_stats

    # Remove from in-memory history and delete files on disk
    with history_lock:
        idx = next((i for i, e in enumerate(scan_history) if e["id"] == sid), None)
        if idx is None:
            return jsonify({"ok": False, "error": "not found"}), 404
        _ = scan_history.pop(idx)
        _delete_entry_from_disk(sid)

    # If that scan is currently loaded in the viewer, clear UI state
    if current_loaded_scan_id == sid:
        with scan_lock:
            preview_card = None  # don't touch a live capture; just clear preview
        _clear_ocr(preserve_seq=True)
        _update_compare_cache(None, {})
        current_loaded_scan_id = None
        # also clear the cached /card frame so nothing stale is shown
        try:
            STREAM_CACHE['card'] = b""
        except Exception:
            pass

    return jsonify({"ok": True, "deleted_id": sid, "count": len(scan_history)})


# ---------------------------
# Compare stats (debug/UX)
# ---------------------------
@app.get("/api/compare/stats")
def api_compare_stats():
    with compare_lock:
        stats = dict(_last_cmp_stats)
    return jsonify({"ok": True, "stats": stats})

@app.post("/api/badlist/add")
def api_bad_add():
    data = request.get_json(silent=True) or {}
    item = (data.get("item") or "").strip()
    if not item:
        return jsonify({"ok": False, "error": "missing item"}), 400
    with _badlist_lock:
        if item not in bad_cards:
            bad_cards.append(item)
    return jsonify({"ok": True, "count": len(bad_cards)})

@app.post("/api/badlist/remove")
def api_bad_remove():
    data = request.get_json(silent=True) or {}
    item = (data.get("item") or "").strip()
    if not item:
        return jsonify({"ok": False, "error": "missing item"}), 400
    with _badlist_lock:
        try:
            while item in bad_cards:
                bad_cards.remove(item)
        except ValueError:
            pass
    return jsonify({"ok": True, "count": len(bad_cards)})

@app.post("/api/badlist/clear")
def api_bad_clear():
    with _badlist_lock:
        bad_cards.clear()
    return jsonify({"ok": True, "count": 0})

# ---------------------------
# Set-hint OCR (helper)
# ---------------------------
def _read_set_hint(roi_bgr: np.ndarray) -> str:
    """
    Try to read a Scryfall set code printed in the collector band.
    Returns a lowercased code if it matches known set codes, else "".
    """
    try:
        if roi_bgr is None or getattr(roi_bgr, "size", 0) == 0:
            return ""
        seen = {}

        def _add_candidates(raw: str, weight: float = 1.0):
            if not raw:
                return
            txt = (raw or "").upper()
            # First, prefer 3-letter all-cap tokens (better D/O disambiguation)
            tokens = re.findall(r"[A-Z]{3}", txt) + re.findall(r"[A-Z0-9]{2,5}", txt)
            for tok in tokens:
                norm = _normalize_set_code_token(tok)
                if not norm:
                    continue
                if norm in SCRYFALL_SET_CODES:
                    return norm
                prev = seen.get(norm, 0.0)
                seen[norm] = max(prev, float(weight))
            return None

        for sub in _iter_set_hint_rois(roi_bgr):
            if sub is None or getattr(sub, "size", 0) == 0:
                continue

            # Binary (general OCR)
            bin_img = _prep_roi_for_ocr(sub)
            txt, _, _ = _read_text_general(bin_img)
            direct = _add_candidates(txt, 1.1)
            if direct:
                return direct

            # Extra allowlist pass on the same binary image to bias toward set codes
            try:
                bin_bgr = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)
            except Exception:
                bin_bgr = None
            if bin_bgr is not None:
                try:
                    t_allow, conf_allow = _tess_text_from_bgr(
                        bin_bgr,
                        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                        det=False,
                    )
                except Exception:
                    t_allow, conf_allow = "", 0.0
                direct = _add_candidates(t_allow, 1.2 + max(0.0, float(conf_allow)) * 0.01)
                if direct:
                    return direct

            # Extra OCR pass on grayscale (upper-case bias handled in parsing)
            try:
                g = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
                g = cv2.resize(g, None, fx=2.1, fy=2.1, interpolation=cv2.INTER_LINEAR)
                txt, _, _ = _read_text_general(g)
                direct = _add_candidates(txt, 1.15)
                if direct:
                    return direct
            except Exception:
                pass

            code = _fast_read_set_hint(sub)
            if code:
                if code in SCRYFALL_SET_CODES:
                    return code
                seen[code] = max(seen.get(code, 0.0), 1.25)

        def _code_score(c: str):
            exact = 1 if len(c) == 3 else 0
            length_bias = -abs(len(c) - 3)
            digits = sum(ch.isdigit() for ch in c)
            weight = seen.get(c, 0.0)
            return (weight, exact, length_bias, -digits)

        for code in sorted(seen.keys(), key=_code_score, reverse=True):
            if code in SCRYFALL_SET_CODES:
                return code
            close = _closest_set_code(code)
            if close:
                return close
    except Exception:
        pass
    return ""

# ---------------------------
# Scryfall catalogs / DFC map
# ---------------------------
def _index_dfc(card_obj: dict):
    """Populate MDFC face-name → combined-name mapping from a card object."""
    try:
        if not card_obj:
            return
        combined = card_obj.get("name") or ""
        faces = card_obj.get("card_faces") or []
        if " // " not in combined or not faces:
            return
        for face in faces:
            fname = (face.get("name") or "").strip()
            if not fname:
                continue
            DFC_FACE_TO_COMBINED[fname.lower()] = combined
            DFC_FACE_TO_COMBINED_NORM[_norm_key(fname)] = combined
    except Exception:
        pass

def _load_scryfall_catalogs_bg():
    """Warm card names + set codes for better fuzzy/fast matching."""
    global SCRYFALL_CARD_NAMES, SCRYFALL_CARD_NAMES_LOWER, SCRYFALL_SET_CODES
    try:
        # Card names
        r = _HTTP.get("https://api.scryfall.com/catalog/card-names", timeout=SCRYFALL_TIMEOUT)
        if r.status_code == 200:
            j = r.json()
            names = j.get("data", []) or []
            # Deduplicate while preserving order
            seen = set()
            dedup = []
            for n in names:
                nl = n.lower()
                if nl in seen:
                    continue
                seen.add(nl); dedup.append(n)
            SCRYFALL_CARD_NAMES = dedup
            SCRYFALL_CARD_NAMES_LOWER = set(n.lower() for n in dedup)
            _dbg("SCRYFALL DATA", f"Loaded {len(SCRYFALL_CARD_NAMES)} card names.")
        # Set codes
        r2 = _HTTP.get("https://api.scryfall.com/sets", timeout=SCRYFALL_TIMEOUT)
        if r2.status_code == 200:
            j2 = r2.json()
            codes = { (s.get("code") or "").lower() for s in (j2.get("data") or []) if s.get("code") }
            SCRYFALL_SET_CODES = set(c for c in codes if 2 <= len(c) <= 5)
            _dbg("SCRYFALL DATA", f"Loaded {len(SCRYFALL_SET_CODES)} set codes.")
    except Exception as e:
        _dbg("SCRYFALL DATA ERROR", f"load failed: {e}")


# --- Ensure Scryfall warm-up runs only once ---
_scry_warmup_once = {"started": False}

def _start_scry_warmup_once():
    global _scry_warmup_once
    if not _scry_warmup_once.get("started", False):
        _scry_warmup_once["started"] = True
        try:
            threading.Thread(target=_load_scryfall_catalogs_bg, daemon=True).start()
        except Exception as _e:
            _dbg("SCRYFALL DATA", f"Warm-up not started: {_e}")

_reprocess_cache = {}
_reprocess_cache_lock = threading.Lock()
REPROCESS_CACHE_TTL = 300.0
_reprocess_jobs = {}
_reprocess_jobs_lock = threading.Lock()
REPROCESS_JOB_TTL = 900.0
# Ensure any Scryfall choice we resolve helps future DFC lookups
# (Monkey-patch the resolver return path to index faces.)
_orig__scryfall_lookup = _scryfall_lookup
def _scryfall_lookup(name, number_raw, set_hint=""):
    _t0 = time.perf_counter()
    choice = _orig__scryfall_lookup(name, number_raw, set_hint=set_hint)
    try:
        if bool(globals().get("PERF_TIMING_DEBUG", True)):
            _dbg("PERF SCRY", f"lookup={time.perf_counter() - _t0:.3f}s name='{(name or '')[:22]}' set='{set_hint}' num='{number_raw}'")
    except Exception:
        pass
    try:
        _index_dfc(choice)
    except Exception:
        pass
    return choice

# ---------------------------
# Startup / shutdown
# ---------------------------
def _start_workers():
    _ensure_history_dirs()
    try:
        os.makedirs(EXPORT_DIR, exist_ok=True)
    except Exception as e:
        _dbg("EXPORT ERROR", f"mkdirs failed: {e}")
    global scan_history
    try:
        scan_history = _load_history_from_disk()
        _dbg("HISTORY", f"Loaded {len(scan_history)} scans.")
    except Exception as e:
        _dbg("HISTORY ERROR", f"load failed: {e}")
    try:
        _cur = _settings_load()
    except Exception:
        _cur = {}
    # Warm catalogs in the background
    _start_scry_warmup_once()

    # Fire up all workers
    threading.Thread(target=detect_worker,   daemon=True).start()
    threading.Thread(target=video_thread,     daemon=True).start()
    threading.Thread(target=ocr_worker,       daemon=True).start()
    threading.Thread(target=cardinfo_worker,  daemon=True).start()
    threading.Thread(target=ws_thread,        daemon=True).start()
    threading.Thread(target=autoscan_manager, daemon=True).start()
    _apply_printer_settings_with_retry(reason="startup")



@app.get("/api/version")
def api_version():
    # Return the most recent update info we have
    info = dict(UPDATE_INFO)
    info["current"] = APP_VERSION
    return jsonify(info)





# ---------------------------
# Printer Controls (Moonraker/Klipper)

# Compatibility wrapper: normalize _send_gcode() to (ok, err)
def _send_gcode_result(script: str):
    """
    Calls the app's legacy _send_gcode() (which doesn't return anything)
    and converts it to (ok, err). If unavailable, posts directly to HTTP_POST_URL.
    """
    try:
        # prefer app's original sender
        if "_send_gcode" in globals() and callable(_send_gcode):
            try:
                _send_gcode(script)  # legacy returns None
                return True, None
            except Exception as e:
                return False, str(e)
        # fallback: direct HTTP
        url = (HTTP_POST_URL if "HTTP_POST_URL" in globals() else "").strip()
        if not url:
            return False, "HTTP_POST_URL not configured"
        to = globals().get("HTTP_TIMEOUT", 5)
        r = _HTTP.post(url, json={"script": script}, timeout=to)
        ok = (r.status_code == 200)
        return (ok, None if ok else f"HTTP {r.status_code}")
    except Exception as e:
        return False, str(e)
# ---------------------------

def _apply_printer_settings_with_retry(reason="startup", attempts=3, delay=2.0):
    """
    Push card thickness and remeasure cadence down to Klipper. Retries a few
    times so it still sticks if Moonraker isn't ready the moment we start.
    """
    def _worker():
        try:
            cur = _settings_load() or {}
        except Exception:
            cur = {}
        try:
            remeasure = int(cur.get("REMEASURE_EVERY", globals().get("REMEASURE_EVERY", 0)) or 0)
        except Exception:
            remeasure = int(globals().get("REMEASURE_EVERY", 0) or 0)
        try:
            thickness = float(cur.get("CARD_THICKNESS_MM", globals().get("CARD_THICKNESS_MM", 0.305)) or 0.305)
        except Exception:
            thickness = float(globals().get("CARD_THICKNESS_MM", 0.305) or 0.305)

        _update_stack_config(thickness=thickness, remeasure_every=remeasure)

        cmds = [
            f"SET_CARD_THICKNESS T={thickness:.3f}",
            f"SET_REMEASURE_EVERY N={max(0, remeasure)}",
        ]
        last_err = None
        for _i in range(max(1, int(attempts or 1))):
            all_ok = True
            for c in cmds:
                ok, err = _send_gcode_result(c)
                if not ok:
                    all_ok = False
                    last_err = err
                    break
            if all_ok:
                _dbg("PRINTER SETUP", f"Applied printer settings ({reason}) thickness={thickness:.3f} remeasure={remeasure}")
                return
            time.sleep(delay or 0.5)
        if last_err:
            _dbg("PRINTER SETUP", f"Failed to push printer settings ({reason}): {last_err}")
    threading.Thread(target=_worker, daemon=True).start()

def _moonraker_http_base():
    """Translate ws(s)://host/websocket -> http(s)://host"""
    try:
        url = MOONRAKER_URL or ""
    except NameError:
        url = ""
    if not url:
        return None
    u = url.replace("ws://","http://").replace("wss://","https://")
    if "/websocket" in u:
        u = u.split("/websocket",1)[0]
    return u.rstrip("/")

def _mr_get(path):
    base = _moonraker_http_base()
    if not base:
        return None, "Moonraker URL not configured"
    try:
        r = _HTTP.get(base + path, timeout=globals().get('HTTP_TIMEOUT', 5))
        return r, None
    except Exception as e:
        return None, str(e)

def _mr_post(path, payload=None):
    base = _moonraker_http_base()
    if not base:
        return None, "Moonraker URL not configured"
    try:
        if payload is None:
            payload = {}
        r = _HTTP.post(base + path, json=payload, timeout=globals().get('HTTP_TIMEOUT', 5))
        return r, None
    except Exception as e:
        return None, str(e)

def _list_macros_from_config():
    """Query Moonraker's configfile object and extract [gcode_macro *] section names."""
    try:
        r, err = _mr_get("/printer/objects/query?configfile")
        if err or r is None or r.status_code != 200:
            raise RuntimeError(err or f"HTTP {getattr(r,'status_code',None)}")
        j = r.json() or {}
        cfg = ((j.get("result") or {}).get("status") or {}).get("configfile") or {}
        conf = cfg.get("config") or {}
        names = []
        for sec in conf.keys():
            if isinstance(sec, str) and sec.lower().startswith("gcode_macro "):
                nm = sec.split(" ",1)[1].strip()
                if nm and nm.upper() != "M112":  # skip reserved
                    names.append(nm)
        names.sort()
        return names
    except Exception as e:
        _dbg("MACROS", f"Falling back to static list: {e}")
        # Fallback: a reasonable fixed list based on your posted config
        return [
            "SORTER_HELP","SORT_ONE","SORT_10","SET_TIMEOUT_S","CANCEL_SCAN_WAIT",
            "SORTER_HOME","SET_START_HEIGHT_GUIDED","SET_START_HEIGHT_OK","SET_CARD_THICKNESS",
            "SET_REMEASURE_EVERY","REMEASURE_STARTS",
            "INIT_STACKS","SET_START_FROM_CURRENT","VAC_ON","VAC_OFF","VAC_ZERO",
            "MOVE_TO_SCAN","PLACE_FINISHED","PLACE_REJECT","SCAN_OK","SCAN_FAIL",
            "SORTER_CYCLE_INTERACTIVE","RUN_SORTER_INTERACTIVE","SHOW_STACKS","SET_SCAN_PARK",
        ]

@app.get("/api/printer/macros")
def api_printer_macros():
    try:
        names = _list_macros_from_config()
        return jsonify({"macros": names})
    except Exception as e:
        return jsonify({"macros": [], "error": str(e)}), 500

@app.post("/api/printer/gcode")
def api_printer_gcode():
    j = request.get_json(force=True, silent=True) or {}
    script = (j.get("script") or "").strip()
    if not script:
        return jsonify({"ok": False, "error": "Missing 'script'"}), 400
    ok, err = _send_gcode_result(script)
    return jsonify({"ok": ok, "error": err})

@app.post("/api/printer/macro")
def api_printer_macro():
    j = request.get_json(force=True, silent=True) or {}
    name = (j.get("name") or "").strip()
    params = j.get("params")
    args = j.get("args")
    if not name:
        return jsonify({"ok": False, "error": "Missing 'name'"}), 400
    # Build command line
    line = name
    if isinstance(params, dict) and params:
        parts = []
        for k, v in params.items():
            if v is None: continue
            parts.append(f"{k}={v}")
        if parts:
            line += " " + " ".join(parts)
    elif isinstance(args, str) and args.strip():
        line += " " + args.strip()
    ok, err = _send_gcode_result(line)
    return jsonify({"ok": ok, "cmd": line, "error": err})

@app.post("/api/printer/home")
def api_printer_home():
    j = request.get_json(force=True, silent=True) or {}
    axes = (j.get("axes") or "XYZ").upper()
    safe = bool(j.get("safe", True))
    # sanitize axes
    axes = "".join([a for a in axes if a in "XYZ"])
    if not axes:
        axes = "XYZ"
    cmd = "G28 " + " ".join(list(axes))
    ok, err = _send_gcode_result(cmd)
    if ok and safe:
        _send_gcode("_SAFE_RAISE")
    return jsonify({"ok": ok, "error": err, "cmd": cmd})

@app.post("/api/printer/move")
def api_printer_move():
    j = request.get_json(force=True, silent=True) or {}
    x = j.get("x"); y = j.get("y"); z = j.get("z")
    f = j.get("f") or j.get("feedrate") or 6000
    absolute = bool(j.get("absolute", False))
    axes = []
    if isinstance(x, (int, float)): axes.append(f"X{float(x):.3f}")
    if isinstance(y, (int, float)): axes.append(f"Y{float(y):.3f}")
    if isinstance(z, (int, float)): axes.append(f"Z{float(z):.3f}")
    if not axes:
        return jsonify({"ok": False, "error": "No movement specified"}), 400
    cmds = []
    if not absolute:
        cmds.append("G91")
    cmds.append("G1 " + " ".join(axes) + f" F{int(f)}")
    if not absolute:
        cmds.append("G90")
    ok_all = True; last_err = None
    for c in cmds:
        ok, err = _send_gcode_result(c)
        if not ok:
            ok_all = False; last_err = err; break
    return jsonify({"ok": ok_all, "error": last_err, "cmds": cmds})

@app.post("/api/printer/estop")
def api_printer_estop():
    # Try Moonraker endpoint if present; always send M112 as well.
    _mr_post("/printer/emergency_stop", {})
    ok, err = _send_gcode_result("M112")
    return jsonify({"ok": ok, "error": err})


# ---------------------------
# Graceful shutdown (fixed)
# ---------------------------
_shutdown_started = False
def _graceful_shutdown(*_a):
    global _shutdown_started
    if _shutdown_started:
        return
    _shutdown_started = True
    print("\nShutting down…")
    try:
        shutdown_evt.set()
    except Exception:
        pass
    try:
        ws = ws_ref.get("ws")
        if ws is not None:
            try: ws.close()
            except Exception: pass
    except Exception:
        pass
    def _stop_http():
        try:
            srv = _httpd_ref.get("srv")
            if srv is not None:
                srv.shutdown()
        except Exception as e:
            _dbg("WEB SERVER ERROR", f"shutdown() failed: {e}")
    threading.Thread(target=_stop_http, daemon=True).start()

# ---------------------------
# Startup / shutdown
# ---------------------------
def run_server(host: str, port: int):
    _start_workers()
    httpd = make_server(host, port, app, threaded=True, request_handler=WSGIRequestHandler)
    _httpd_ref["srv"] = httpd
    _dbg("WEB SERVER", f"Serving on http://{host}:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        # normal Ctrl+C path
        pass
    finally:
        try:
            if not globals().get("_shutdown_started", False):
                _graceful_shutdown()
        except Exception:
            pass
        try:
            httpd.server_close()
        except Exception:
            pass
        time.sleep(0.2)


# --- start Scryfall warm-up on import ---
try:
    _start_scry_warmup_once()
except Exception as _e:
    _dbg("SCRYFALL DATA", f"Warm-up not started: {_e}")
# --- end warm-up ---
# === BEGIN FAST OCR PATCH (drop-in / removable) ===
try:
    _FAST_CFG = {
        "FAST_PATH": bool(globals().get("OCR_FAST_PATH", True) or globals().get("FAST_OCR_MODE", False)),
        "LAZY_SH":   bool(globals().get("OCR_LAZY_SET_HINT", False)),
        "SINGLE":    bool(globals().get("OCR_TITLE_SINGLE_PASS", True)),
        "USE_TESSERACT": bool(globals().get("OCR_USE_TESSERACT", True) and TESS_AVAILABLE),
        "USE_PADDLE":  bool(globals().get("OCR_USE_TESSERACT", True) and TESS_AVAILABLE),  # legacy flag name; mirrors USE_TESSERACT for compatibility
        "MAXW":      int(globals().get("OCR_TITLE_MAX_W", 640)),
        "FAST_NUM":  bool(globals().get("OCR_NUMBER_FASTPATH", True)),
        "SKIP_NUM": bool(globals().get("OCR_SKIP_NUMBER", False)),
    }
except Exception:
    _FAST_CFG = {"FAST_PATH": True, "LAZY_SH": True, "SINGLE": True, "USE_TESSERACT": bool(TESS_AVAILABLE), "USE_PADDLE": bool(TESS_AVAILABLE), "MAXW": 640, "FAST_NUM": True, "SKIP_NUM": False}

_OCR_CONTEXT = threading.local()

def _use_tesseract_backend() -> bool:
    cfg = globals().get("_FAST_CFG", {}) or {}
    return bool(cfg.get("USE_TESSERACT", cfg.get("USE_PADDLE", True))) and bool(TESS_AVAILABLE)

def _ocr_fast_only():
    return bool(getattr(_OCR_CONTEXT, "fast_only", False))

@contextmanager
def _force_fast_ocr_scope(enable=False):
    if not enable:
        yield
        return
    prev = getattr(_OCR_CONTEXT, "fast_only", False)
    _OCR_CONTEXT.fast_only = True
    try:
        yield
    finally:
        _OCR_CONTEXT.fast_only = prev

_AI_NAME_LOGS = {"count": 0}
_AI_NUM_LOGS = {"count": 0}
_AI_SET_LOGS = {"count": 0}
_ROI_SAVE_COUNTS = {}

def _tess_text_from_bgr(bgr, maxw=None, allowlist=None, deadline=None, det=False):
    """
    Tesseract-backed OCR wrapper (former Paddle entrypoint).
    det flag kept for signature compatibility but ignored.
    """
    if not _use_tesseract_backend():
        return "", 0.0
    try:
        import cv2 as _cv2, time as _time, numpy as _np
        if deadline is not None and _time.perf_counter() >= deadline:
            return "", 0.0
        if bgr is None or getattr(bgr, "size", 0) == 0:
            return "", 0.0
        if not isinstance(bgr, _np.ndarray):
            bgr = _np.asarray(bgr)
        if bgr.dtype != _np.uint8:
            bgr = _np.clip(bgr, 0, 255).astype(_np.uint8)
        h, w = bgr.shape[:2]
        if getattr(bgr, "ndim", 0) == 2:
            bgr = _cv2.cvtColor(bgr, _cv2.COLOR_GRAY2BGR)
        if maxw and w > maxw:
            scale = maxw / float(w)
            bgr = _cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=_cv2.INTER_AREA)
            h, w = bgr.shape[:2]
        if deadline is not None and _time.perf_counter() >= deadline:
            return "", 0.0
        # Tesseract expects RGB
        rgb = _cv2.cvtColor(bgr, _cv2.COLOR_BGR2RGB)
        config = "--psm 7 --oem 3"
        if allowlist:
            config += f" -c tessedit_char_whitelist={allowlist}"
        data = pytesseract.image_to_data(rgb, output_type=_TessOutput.DICT, config=config)
        texts, confs = [], []
        for txt, conf_str in zip(data.get("text", []), data.get("conf", [])):
            txt = (txt or "").strip()
            if not txt:
                continue
            try:
                conf_val = float(conf_str)
            except Exception:
                conf_val = -1.0
            if conf_val < 0:
                continue
            texts.append(txt)
            confs.append(conf_val / 100.0)
        if not texts:
            return "", 0.0
        line = " ".join(texts).strip()
        max_conf = max(confs) if confs else 0.0
        conf = max(0.0, min(100.0, 100.0 * float(max_conf)))
        return line, conf
    except Exception:
        return "", 0.0

# Legacy alias for callers still using the old name.
_paddle_text_from_bgr = _tess_text_from_bgr

def _paddle_salvage_title(img, foil=False):
    """Try broader OCR passes when the fast path fails to read a title."""
    if img is None or getattr(img, "size", 0) == 0 or not _use_tesseract_backend():
        return "", 0.0
    rois = []
    try:
        if _ai_enabled():
            for key in ("name", "set_name"):
                roi = _ai_crop(img, key)
                if roi is not None and getattr(roi, "size", 0) > 0:
                    rois.append(roi)
    except Exception:
        pass
    for _roi_fn in (_crop_title_roi_top, _crop_title_roi_alt):
        try:
            roi = _roi_fn(img)
            if roi is not None and getattr(roi, "size", 0) > 0:
                rois.append(roi)
        except Exception:
            continue
    try:
        wide_top = _roi_rel(img, [0.02, 0.01, 0.98, 0.18])
        if wide_top is not None and getattr(wide_top, "size", 0) > 0:
            rois.append(wide_top)
    except Exception:
        pass
    best_txt, best_conf = "", 0.0
    for roi in rois:
        if roi is None or getattr(roi, "size", 0) == 0:
            continue
        src = _enhance_for_foil(roi) if foil else roi
        txt, conf = _tess_text_from_bgr(src, det=False)
        txt = _clean_title_text(txt)
        if txt and conf > best_conf:
            best_txt, best_conf = txt, conf
    return best_txt, best_conf

def _quick_title_from_roi(roi_bgr, foil=False):
    if roi_bgr is None or getattr(roi_bgr, "size", 0) == 0:
        return "", 0.0, "none"
    try:
        import cv2 as _cv2, numpy as _np
        h, w = roi_bgr.shape[:2]
        maxw = max(120, int(_FAST_CFG.get("MAXW", 640)))
        if w > maxw:
            scale = maxw / float(w)
            roi_bgr = _cv2.resize(roi_bgr, (int(w*scale), int(h*scale)), interpolation=_cv2.INTER_AREA)
        src = _enhance_for_foil(roi_bgr) if foil else roi_bgr
        if _use_tesseract_backend():
            t, c = _tess_text_from_bgr(src)
            if t:
                return _clean_title_text(t), c, "Tesseract"
        return "", 0.0, "none"
    except Exception:
        return "", 0.0, "none"

try:
    __SLOW_OCR_IMPL = _orig_ocr_from_card_upright
except NameError:
    __SLOW_OCR_IMPL = None

def _fast_ocr_from_card_upright(img):
    """Simplified OCR: use Tesseract on AI-detected regions without fast/slow paths."""
    _perf_dbg = bool(globals().get("PERF_TIMING_DEBUG", True))
    _t_start = time.perf_counter()
    ai_dt = name_dt = num_dt = set_dt = 0.0
    try:
        if _ai_enabled():
            snapshot_only = bool(globals().get("AI_ROIS_SNAPSHOT_ONLY", False))
            from_snapshot = bool(getattr(_OCR_CONTEXT, "from_snapshot", False))
            _t_ai = time.perf_counter()
            if (not snapshot_only) or from_snapshot:
                _update_ai_rois(img)
            ai_dt = time.perf_counter() - _t_ai
    except Exception:
        pass

    foil = False
    foil_score = 0.0
    try:
        if globals().get("FOIL_DETECT", True):
            foil, foil_score = _detect_foil_card(img)
    except Exception:
        foil = False
        foil_score = 0.0

    # --- Name from AI ROI (fallback to top/alt crop) ---
    name_txt, name_conf = "", 0.0
    name_perf = [] if bool(globals().get("PERF_TIMING_DEBUG", True)) else None
    _t_name = time.perf_counter()
    try:
        name_rois = []
        # Primary: AI name box
        if _ai_enabled():
            r = _ai_crop_exact(img, "name")
            if r is not None and getattr(r, "size", 0) > 0:
                name_rois.append(("ai-name", r))
                _save_roi_debug("ai-name", r)
        # Secondary: top band of AI card box (helps when name box is cropped tight)
        card_roi = _ai_crop(img, "card") if (_ai_enabled() and not _AI_ROI_STRICT) else None
        if card_roi is not None and getattr(card_roi, "size", 0) > 0:
            pad = float(globals().get("AI_CARD_NAME_PAD", 0.04))
            try:
                h, w = card_roi.shape[:2]
                y1 = min(h, int((0.22 + pad) * h))
                top_band = card_roi[:y1, :]
            except Exception:
                top_band = _slice_frac(card_roi, 0.00, 0.22)
            if top_band is not None and getattr(top_band, "size", 0) > 0:
                name_rois.append(("ai-card-top", top_band))
        if not _AI_ROI_STRICT:
            # Fallbacks: legacy top/alt crops
            r_top = _crop_title_roi_top(img)
            if r_top is not None and getattr(r_top, "size", 0) > 0:
                name_rois.append(("top", r_top))
            r_alt = _crop_title_roi_alt(img)
            if r_alt is not None and getattr(r_alt, "size", 0) > 0:
                name_rois.append(("alt", r_alt))

        best_txt, best_conf = "", 0.0
        maxw_val = max(900, int(globals().get("OCR_TITLE_MAX_W", 640)))
        early_exit = bool(globals().get("OCR_TITLE_EARLY_EXIT", True))
        early_conf = float(globals().get("OCR_TITLE_EARLY_CONF", 88.0))

        def _iter_name_variants(roi):
            """Yield a few enhanced versions of the ROI to improve Paddle recognition."""
            if roi is None or getattr(roi, "size", 0) == 0:
                return
            yield "raw", roi
            try:
                g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if getattr(roi, "ndim", 0) == 3 else roi
                g = cv2.resize(g, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
                g = cv2.GaussianBlur(g, (3, 3), 0)
                th = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 23, 9)
                th_inv = 255 - th
                th_bgr = cv2.cvtColor(th_inv, cv2.COLOR_GRAY2BGR)
                yield "thresh", th_bgr
            except Exception:
                pass
            if foil:
                try:
                    enh = _enhance_for_foil(roi)
                    if enh is not None and getattr(enh, "size", 0) > 0:
                        yield "foil", enh
                except Exception:
                    pass
            try:
                prep = _prep_roi_for_ocr(roi)
                if prep is not None and getattr(prep, "size", 0) > 0:
                    yield "prep", prep
            except Exception:
                pass

        def _read_name_variant(src):
            if src is None or getattr(src, "size", 0) == 0:
                return "", 0.0
            t, c = _tess_text_from_bgr(src, maxw=maxw_val, det=False)
            cleaned = _clean_title_text(t)
            return cleaned, c

        for lbl, roi in name_rois:
            if roi is None or getattr(roi, "size", 0) == 0:
                continue
            roi_for_ocr = _pad_roi_for_ocr(roi, pad_px=6)
            for variant, src in _iter_name_variants(roi_for_ocr):
                _t_var = time.perf_counter()
                txt, conf = _read_name_variant(src)
                if name_perf is not None:
                    try:
                        name_perf.append((lbl, variant, time.perf_counter() - _t_var, bool(txt)))
                    except Exception:
                        pass
                if txt and conf >= best_conf:
                    best_txt, best_conf = _maybe_correct_ocr_name(txt, conf), conf
                     # Early exit once we have a strong hit from an AI-guided ROI
                if best_conf >= early_conf and early_exit:
                    break
            if best_conf >= early_conf and early_exit:
                break
        if (not best_txt) and (not _AI_ROI_STRICT):
            try:
                fallback_rois = []
                try:
                    r_top = _crop_title_roi_top(img)
                    if r_top is not None and getattr(r_top, "size", 0) > 0:
                        fallback_rois.append(("fallback-top", r_top))
                except Exception:
                    pass
                try:
                    r_alt = _crop_title_roi_alt(img)
                    if r_alt is not None and getattr(r_alt, "size", 0) > 0:
                        fallback_rois.append(("fallback-alt", r_alt))
                except Exception:
                    pass
                for lbl, roi in fallback_rois:
                    if roi is None or getattr(roi, "size", 0) == 0:
                        continue
                    roi_for_ocr = _pad_roi_for_ocr(roi, pad_px=6)
                    for variant, src in _iter_name_variants(roi_for_ocr):
                        _t_var = time.perf_counter()
                        txt, conf = _read_name_variant(src)
                        if name_perf is not None:
                            try:
                                name_perf.append((lbl, variant, time.perf_counter() - _t_var, bool(txt)))
                            except Exception:
                                pass
                        if txt and conf >= best_conf:
                            best_txt, best_conf = _maybe_correct_ocr_name(txt, conf), conf
                        if best_conf >= early_conf and early_exit:
                            break
                    if best_conf >= early_conf and early_exit:
                        break
                if (not best_txt):
                    salvage_txt, salvage_conf = _paddle_salvage_title(img, foil=foil)
                    salvage_txt = _clean_title_text(salvage_txt)
                    if salvage_txt:
                        best_txt, best_conf = _maybe_correct_ocr_name(salvage_txt, salvage_conf), salvage_conf
            except Exception:
                pass
        if best_txt:
            name_txt, name_conf = best_txt, best_conf
            try:
                if name_conf < 70.0:
                    lex = _score_against_lexicon(name_txt)
                    name_conf = max(name_conf, 100.0 * float(lex))
            except Exception:
                pass
    except Exception:
        pass
    name_dt = time.perf_counter() - _t_name
    if name_perf is not None:
        try:
            top = sorted(name_perf, key=lambda t: t[2], reverse=True)[:5]
            summary = ", ".join(f"{lbl}:{var} {dur:.3f}s{'*' if hit else ''}" for lbl, var, dur, hit in top)
            if best_conf >= early_conf and early_exit:
                summary += " (early-exit)"
            _dbg("PERF OCR NAME", summary)
        except Exception:
            pass

    # --- Collector number from AI ROIs ---
    number_tok, number_conf = "", 0.0
    skip_number = bool(globals().get("OCR_SKIP_NUMBER", False))
    _t_num = time.perf_counter()
    if not skip_number:
        try:
            rois = []
            if _ai_enabled():
                ai_card = _ai_crop_relaxed(img, "card", pad_x=_AI_CARD_PAD_RIGHT, pad_y=_AI_CARD_PAD_Y)
                if ai_card is not None and getattr(ai_card, "size", 0) > 0:
                    try:
                        br = _slice_frac(ai_card, 0.70, 0.99, 0.58, 0.99)
                        if br is not None and getattr(br, "size", 0) > 0:
                            rois.append(("ai-card-bottom-right", br))
                        band = _slice_frac(ai_card, 0.68, 1.0)
                        if band is not None and getattr(band, "size", 0) > 0:
                            rois.append(("ai-card-band", band))
                            try:
                                h, w = band.shape[:2]
                                right = band[:, int(0.55 * w):]
                                if right is not None and getattr(right, "size", 0) > 0:
                                    rois.append(("ai-card-band-right", right))
                                tight = band[:, int(0.72 * w):]
                                if tight is not None and getattr(tight, "size", 0) > 0:
                                    rois.append(("ai-card-band-tight", tight))
                            except Exception:
                                pass
                    except Exception:
                        pass
                rois.extend(list(_ai_card_number_rois(img)))
                for lbl, roi in rois:
                    _save_roi_debug(lbl, roi)
            if (not rois) and (not _AI_ROI_STRICT):
                card_roi = _card_crop_or_full(img)
                if card_roi is not None and getattr(card_roi, "size", 0) > 0:
                    band_roi = _slice_frac(card_roi, 0.68, 1.0)
                    if band_roi is not None and getattr(band_roi, "size", 0) > 0:
                        rois.append(("card-band", band_roi))
                        try:
                            h, w = band_roi.shape[:2]
                            right = band_roi[:, int(0.55 * w):]
                            if right is not None and getattr(right, "size", 0) > 0:
                                rois.append(("card-band-right", right))
                            tight = band_roi[:, int(0.72 * w):]
                            if tight is not None and getattr(tight, "size", 0) > 0:
                                rois.append(("card-band-tight", tight))
                        except Exception:
                            pass
            try:
                number_deadline = time.perf_counter() + max(0.35, float(globals().get("OCR_NUM_FAST_CAP_S", 3.5)))
            except Exception:
                number_deadline = None

            for label, roi in rois:
                if roi is None or getattr(roi, "size", 0) == 0:
                    continue
                tok, conf = _fast_read_number(roi, source_tag=label, deadline=number_deadline)
                if conf > number_conf:
                    number_tok, number_conf = tok, conf
                if number_tok and str(number_tok).isdigit() and number_conf >= max(_FAST_NUM_CONF_EXIT, 72.0):
                    break

            if not number_tok:
                try:
                    tok, conf = _read_collector_number(img, foil=foil)
                    if conf > number_conf:
                        number_tok, number_conf = tok, conf
                except Exception:
                    pass
        except Exception:
            pass

    number_raw = str(number_tok or "")
    number_disp = _parse_collector_for_display(number_raw)
    num_dt = time.perf_counter() - _t_num
    if not number_disp:
        number_conf = 0.0
        # Slow but more thorough number OCR fallback when the fast path missed
        if not skip_number:
            try:
                tok_slow, conf_slow = _read_collector_number(img, foil=foil)
                if tok_slow:
                    number_raw = tok_slow
                    number_disp = _parse_collector_for_display(number_raw)
                    number_conf = max(number_conf, float(conf_slow or 0.0)) if number_disp else 0.0
            except Exception:
                pass

    # track set_hint even if we fall back later
    set_hint = ""

    # --- Fallback to slower OCR if we still have nothing meaningful ---
    try:
        # If AI-based OCR failed to produce a name or number,
        # fall back to the original slower implementation regardless of ROI strictness.
        if (not name_txt) and (not number_disp) and (__SLOW_OCR_IMPL is not None):
            slow = __SLOW_OCR_IMPL(img)
            if isinstance(slow, dict):
                if (not name_txt) and slow.get("name"):
                    name_txt = slow.get("name") or slow.get("name_raw") or ""
                    name_conf = float(slow.get("name_conf") or 0.0)
                if (not number_disp) and slow.get("number"):
                    number_disp = slow.get("number") or ""
                    number_raw = slow.get("number_raw") or number_disp
                    number_conf = float(slow.get("number_conf") or 0.0)
                if not set_hint:
                    set_hint = (slow.get("set_hint") or "").strip().lower()
                if slow.get("foil") is not None:
                    foil = bool(slow.get("foil"))
                if slow.get("foil_score") is not None:
                    foil_score = float(slow.get("foil_score") or 0.0)
    except Exception:
        pass

    # --- Set hint from AI set_name / bottom band (tight ROI) ---
    _t_set = time.perf_counter()
    try:
        roi = _ai_set_code_roi(img) if _ai_enabled() else None
        if (roi is None or getattr(roi, "size", 0) == 0) and _ai_enabled():
            # Fallback: raw AI 'set_name' box if band ROI is unavailable.
            roi = _ai_crop_exact(img, "set_name")
        if roi is not None and getattr(roi, "size", 0) > 0:
            _save_roi_debug("ai-set", roi)
            roi_for_ocr = _pad_roi_for_ocr(roi, pad_px=4)
            try:
                if DEBUG_OCR:
                    _dbg("OCR SET ROI", f"shape={getattr(roi_for_ocr,'shape',None)} tess_backend={_use_tesseract_backend()}")
            except Exception:
                pass
            txt_raw, conf_raw = _tess_text_from_bgr(
                roi_for_ocr,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            )
            txt_raw = (txt_raw or "").strip().upper()
            if txt_raw:
                try:
                    code = _normalize_set_code_token(txt_raw)
                    if code and code not in SCRYFALL_SET_CODES:
                        code = _closest_set_code(code) or code
                except Exception:
                    code = txt_raw
                if code and 2 <= len(code) <= 5:
                    set_hint = code.lower()
                    if DEBUG_OCR and _AI_SET_LOGS["count"] < 4:
                        _AI_SET_LOGS["count"] += 1
                        try:
                            _dbg(
                                "OCR SET ROI",
                                f"set_hint='{set_hint}' raw='{txt_raw}' shape={roi.shape[1]}x{roi.shape[0]}",
                            )
                        except Exception:
                            pass
    except Exception:
        pass
    if not set_hint:
        try:
            card_roi = _card_crop_or_full(img)
            band = _slice_frac(card_roi, 0.68, 1.0) if card_roi is not None else None
            set_hint = (_read_set_hint(band) or "").strip().lower() if band is not None else ""
        except Exception:
            pass
    if not set_hint:
        try:
            set_name_txt = _read_set_name_roi(img)
            set_hint = (_set_code_from_set_name(set_name_txt) or "").strip().lower()
        except Exception:
            pass
    set_dt = time.perf_counter() - _t_set

    # Final consolidation: resolve set hint using all fallbacks (band, name, icon) if still weak
    try:
        if (not set_hint) or (set_hint and set_hint not in SCRYFALL_SET_CODES):
            resolved = _resolve_set_hint(img, initial_hint=set_hint, name_hint=name_txt)
            if resolved:
                set_hint = resolved
    except Exception:
        pass

    if _perf_dbg:
        try:
            total = time.perf_counter() - _t_start
            _dbg("PERF OCR DETAIL", f"ai={ai_dt:.3f}s name={name_dt:.3f}s num={num_dt:.3f}s set={set_dt:.3f}s total={total:.3f}s snapshot={bool(getattr(_OCR_CONTEXT,'from_snapshot',False))}")
        except Exception:
            pass

    return {
        "name": name_txt or "",
        "name_raw": name_txt or "",
        "name_conf": float(name_conf or 0.0),
        "number": number_disp,
        "number_raw": number_raw,
        "number_conf": float(number_conf or 0.0) if number_disp else 0.0,
        "set_hint": set_hint or "",
        "foil": bool(foil),
        "foil_score": float(foil_score or 0.0),
        "provider": "Tesseract",
    }

# Patch both names so any call hits fast path
ocr_from_card_upright = _fast_ocr_from_card_upright
_orig_ocr_from_card_upright = _fast_ocr_from_card_upright

try:
    _dbg("PERF TUNING", f"FAST OCR path={'on' if _FAST_CFG.get('FAST_PATH', True) else 'off'}; lazy_set_hint={'on' if _FAST_CFG.get('LAZY_SH', True) else 'off'}; tesseract_backend={'on' if _use_tesseract_backend() else 'off'}")
except Exception:
    pass

# Re-arm timers now that we've patched OCR functions
try:
    if '_enable_perf_debug' in globals():
        _enable_perf_debug()
except Exception:
    pass
# === END FAST OCR PATCH ===


if __name__ == "__main__":
    HOST = str(globals().get("APP_HOST", os.environ.get("APP_HOST", "0.0.0.0")))
    PORT = int(globals().get("APP_PORT", os.environ.get("APP_PORT", 5000)))
    try:
        signal.signal(signal.SIGINT,  lambda *_: threading.Thread(target=_graceful_shutdown, daemon=True).start())
        signal.signal(signal.SIGTERM, lambda *_: threading.Thread(target=_graceful_shutdown, daemon=True).start())
    except Exception:
        pass
    run_server(HOST, PORT)
