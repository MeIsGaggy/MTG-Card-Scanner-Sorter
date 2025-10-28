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


def _dbg(tag, msg, level=1):
    try:
        print(f"[{tag}] {msg}")
        _BOOT_LOG.append((level, _t.time(), str(tag), str(msg)))
    except Exception:
        pass

def _dbg2(tag, msg):
    _dbg(tag, msg, level=2)
    
try:
    import os, json, re, threading, time, signal, io, base64, shutil, glob
    from collections import OrderedDict, deque
    from werkzeug.serving import make_server, WSGIRequestHandler
    import pytesseract
    import cv2, numpy as np, requests
    from flask import Flask, Response, jsonify, request, send_file, render_template, send_from_directory
    from rapidocr_onnxruntime import RapidOCR
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
try:
    RAPIDOCR_AVAILABLE = True
    _dbg("RAPID OCR", "RapidOCR is available")
except Exception as e:
    RapidOCR, RAPIDOCR_AVAILABLE = None, False
    _dbg("RAPID OCR", "Failed to import RapidOCR: {e}")
try:
    FUZZ_AVAILABLE = True
    _dbg("RAPID FUZZ", "RapidFuzz is available")
except Exception:
    FUZZ_AVAILABLE = False
    process = fuzz = None
    _dbg("RAPID FUZZ", "RapidFuzz not available; fuzzy matching disabled.")

CORES = os.cpu_count() or 4
os.environ.setdefault("OMP_NUM_THREADS", str(CORES - 2))
os.environ.setdefault("ORT_NUM_THREADS", str(CORES - 2))
os.environ.setdefault("OMP_WAIT_POLICY", "ACTIVE")
try:
    cv2.setNumThreads(CORES); cv2.setUseOptimized(True); os.nice(-5)
except Exception:
    pass

PRIMARY_PROVIDER_LABEL = "RapidOCR" if "rapidocr" in (OCR_BACKEND or "").lower() else "Tesseract"

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
PROC_DOWNSCALE_MAX_W    = globals().get("PROC_DOWNSCALE_MAX_W", PROC_MAX_WIDTH)  # still honor your config
SC_IMG_CACHE_DIR        = globals().get("SC_IMG_CACHE_DIR", "./.scry_img_cache") # disk cache for card images
USE_OPENCL              = globals().get("USE_OPENCL", True)

# PERF: new runtime tuning knobs (can be overridden in config.py)
LIVE_OCR_ONLY_WHEN_STEADY = globals().get("LIVE_OCR_ONLY_WHEN_STEADY", True)  # only OCR live frames when steady
LIVE_OCR_MIN_INTERVAL     = globals().get("LIVE_OCR_MIN_INTERVAL", 0.35)      # seconds between live OCR passes
FOIL_EVERY_N_FRAMES       = globals().get("FOIL_EVERY_N_FRAMES", 30)           # compute foil detection every N frames

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
# =========================
#MARK: DEBUG
# =========================
# ---- UPGRADE LOGGER & REPLAY BOOT LINES ----
def _dbg(tag, msg, level=1):
    # real logger that also prints depending on DEBUG_LEVEL
    try:
        _push_log(tag, msg, level)
    finally:
        if (level == 1 and DEBUG_LEVEL > 0) or (level == 2 and DEBUG_LEVEL > 1):
            print(f"[{tag}] {msg}")

def _dbg2(tag, msg):  # convenience
    _dbg(tag, msg, level=2)

# replay anything captured before the real logger existed
for level, ts, tag, msg in _BOOT_LOG:
    try:
        _push_log(tag, msg, level)
    except Exception:
        pass
_BOOT_LOG.clear()


if DEBUG_SAVE_ROI:
    try:
        os.makedirs(DEBUG_DIR, exist_ok=True)
    except Exception:
        pass

# =========================
#MARK: GLOBALS
# =========================
app = Flask(__name__, static_folder='static', template_folder='templates')

APP_VERSION = get_current_version()
UPDATE_INFO = {'current': APP_VERSION, 'latest': APP_VERSION, 'is_update': False, 'html_url': ''}
    
def ui_js_fallback():
    p = os.path.join(APP_DIR, "static", "js", "ui.js")
    try:
        return send_file(p, mimetype="application/javascript")
    except Exception:
        return Response("/* ui.js missing */", mimetype="application/javascript")

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

shutdown_evt = threading.Event()

# Cache for frozen stream frames (encoded JPG bytes)
STREAM_CACHE = {'live': None, 'card': None}

cap, output_frame, current_card_crop, scanned_card = None, None, None, None
last_scan_ts = 0.0

tracks = OrderedDict()

ws_ref = {"ws": None, "connected": False}
printer_state = {
    "awaiting": False, "job_id": None, "last_ready_line": "", "last_decision": None,
    "ws_connected": False, "last_error": None,
    "scan_captured": False, "ready_since": 0.0, "ack_received_at": 0.0,
    "messages_seen": 0, "ack_ok_jobs": set()
}

# Give UI something sane on first /api/state call
scanner_state = {
    "locked": False,
    "locked_frames": 0,
    "locked_area": 0.0,
    "steady": False,
    "foil": False,
    "foil_score": 0.0,
}
ocr_state = {"provider": PRIMARY_PROVIDER_LABEL}
cardinfo_state = {}

_rapidocr_engine = None
_HTTP = requests.Session()
_HTTP.headers.update({"User-Agent": "mtg-scanner/1.1"})
_scry_cache, _fx_cache = {}, {"rate": None, "exp": 0}
_scry_img_cache = {}

SCRYFALL_CARD_NAMES = []
SCRYFALL_CARD_NAMES_LOWER = set()
SCRYFALL_SET_CODES = set()

APPEARANCE_ALIGN_ENABLE   = globals().get("APPEARANCE_ALIGN_ENABLE", True)
APPEARANCE_AB_STRENGTH    = globals().get("APPEARANCE_AB_STRENGTH", 0.55)  # 0..1, chroma transfer strength
APPEARANCE_SAT_STRENGTH   = globals().get("APPEARANCE_SAT_STRENGTH", 0.65) # 0..1, HSV S scaling strength
APPEARANCE_GAMMA_CLAMP    = globals().get("APPEARANCE_GAMMA_CLAMP", (0.65, 1.6))

# Where to save exported decklists (on the device running this app)
EXPORT_DIR = globals().get("EXPORT_DIR", "./exports")
os.makedirs(EXPORT_DIR, exist_ok=True)
# Bad list (in-memory ring + files on disk)
try:
    os.makedirs(BAD_DIR, exist_ok=True)
except Exception:
    pass
bad_cards = deque(maxlen=500)

# ======= Review / History (persistent + in-memory) =======
scan_history = []          # newest-first
scan_id_seq = 0
current_loaded_scan_id = None
history_lock = threading.Lock()
# --- post-snapshot pause control (processing pause separate from streaming pause)
PROC_PAUSED = False
PROC_PAUSE_SEQ = None  # snapshot_seq this pause belongs to

STREAM_STATE = {
    "enabled": bool(globals().get("STREAM_ENABLED", True)),          # on/off
    "fps":     int(globals().get("STREAM_FPS", 15)),                 # 1–60
    "quality": int(globals().get("STREAM_JPEG_QUALITY", 68)),        # 30–95

    'paused': False,
    'paused_reason': ''
}

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

def _scry_cache_key(card_obj, scry_bgr=None, use_art=False, tgt=None, roi_art=None):
    # stable base: scryfall id or image url or in-memory pointer
    base = (card_obj or {}).get("id") or (((card_obj or {}).get("image_uris") or {}).get("normal"))
    if not base and scry_bgr is not None:
        base = id(scry_bgr)
    return (base, bool(use_art), tuple(tgt or ()), tuple(map(float, roi_art or ROI_ART)))


def _get_cached_reference_feats(card_obj, scry_bgr, use_art=False, tgt=(640,900)):
    key = _scry_cache_key(card_obj, scry_bgr, use_art=use_art, tgt=tgt, roi_art=ROI_ART)
    feats = _lru_get(_scry_img_feats, key)
    if feats is not None:
        # ensure compatibility in case code changes
        if feats.get("use_art") == bool(use_art) and feats.get("tgt") == tuple(tgt):
            return feats

    if scry_bgr is None or scry_bgr.size == 0:
        return None

    B0 = _center_crop(scry_bgr, 0.98)
    if use_art:  # << use the parameter, not the global
        B0 = _crop_art(B0)

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
    num = _roi_rel(img, ROI_NUMBER_WIDE)
    try:
        top_bin = _prep_roi_for_ocr(top)
        alt_bin = _prep_roi_for_ocr(alt)
    except Exception:
        return 0.0
    s = 0.0
    s += 1.0 if _has_text_quick(top_bin) else 0.0
    s += 0.6 if _has_text_quick(alt_bin) else 0.0
    # digits band — super cheap ink check
    s += 0.5 if _ink_present_quick(num, thresh=0.01) else 0.0
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
    Build counts keyed by (name, set, cn). only_status: 'pass' or 'all'
    """
    counts = {}
    for e in list(scan_history):
        if only_status == "pass" and (e.get("status") or "fail") != "pass":
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


def save_scan_entry(snap_img, ocr, scry, cmp_details, match_score, match_ok, flagged, status=None):
    """Create a review record and persist it."""
    global scan_id_seq, scan_history
    sid = int(time.time()*1000) + (scan_id_seq % 1000)
    scan_id_seq += 1

    cmp_jpg = None
    if cmp_details:
        vis = _render_compare_visual(cmp_details)
        if vis is not None:
            cmp_jpg = _jpeg_bytes(vis, JPEG_QUALITY_CMP)

    # --- Build review reasons & optional name auto-fix ---
    review_reasons = []
    match_mode = (scry or {}).get("_match_mode")
    auto_name_fix = False

    # If Scryfall was found by CN+SET (or CN only), note that title was unreliable
    if match_mode in ("cn_set_only", "cn_only"):
        review_reasons.append("Card name not reliably read — identified by set + collector number.")

    # Visual match failed → reason
    try:
        ms = float(match_score)
    except Exception:
        ms = 0.0
    if match_ok is False:
        try:
            th = float(MATCH_TH)
        except Exception:
            th = 0.0
        review_reasons.append(f"Scryfall candidate did not visually match (score {ms:.2f} below threshold {th:.2f}).")

    # No Scryfall at all → reason
    if not scry:
        review_reasons.append("No Scryfall result; manual selection required.")

    # Your requested behavior:
    # If we matched by cn_set_only and the visual match passed, fix the saved name to Scryfall's.
    if (match_mode == "cn_set_only") and bool(match_ok) and scry and scry.get("name"):
        ocr = dict(ocr or {})
        ocr["name"] = scry.get("name") or ocr.get("name") or ""
        auto_name_fix = True
        flagged = True  # still flag for review as requested
        review_reasons.append("Name auto-corrected from OCR to Scryfall (cn_set_only + visual match passed).")

    entry = {
        "id": sid,
        "ts": time.time(),
        "name": (ocr or {}).get("name") or (ocr or {}).get("name_raw") or "",
        "name_conf": float((ocr or {}).get("name_conf", 0.0)),
        "number": (ocr or {}).get("number") or "",
        "number_raw": (ocr or {}).get("number_raw") or "",
        "set_hint": (ocr or {}).get("set_hint") or "",
        "foil": bool((ocr or {}).get("foil", False)),
        "match_score": round(float(ms), 3),
        "match_ok": bool(match_ok) if match_ok is not None else None,
        "flagged": bool(flagged),
        "status": status or ("pass" if match_ok else "fail"),
        "scry": scry,
        "cmp_stats": {k: v for k, v in (cmp_details or {}).items() if k != "orb_dbg"},
        "snap_jpg": _jpeg_bytes(snap_img, JPEG_QUALITY_SNAP),
        "thumb": _make_thumb(snap_img, 120),
        "cmp_jpg": cmp_jpg,

        # NEW: rich review context
        "review_reasons": review_reasons,
        "auto_name_fix": bool(auto_name_fix),
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
    global _rapidocr_engine
    if _rapidocr_engine is None and RAPIDOCR_AVAILABLE:
        try:
            _rapidocr_engine = RapidOCR()
        except Exception:
            _rapidocr_engine = None
    return _rapidocr_engine

def _get_price(ag, key):
    """Return a price string from ag['scry']['prices'][key] (e.g., 'usd', 'usd_foil', 'eur', 'tix')."""
    try:
        prices = _get(ag, "prices") or {}
        v = prices.get(key)
        if v in (None, "", "null"):
            return ""
        # normalize to plain string; Archidekt accepts raw decimal text
        return f"{float(v):.2f}"
    except Exception:
        return ""


def order_points(pts):
    rect = np.zeros((4,2), dtype='float32')
    s = pts.sum(axis=1)
    rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
    d = np.diff(pts, axis=1)
    rect[1], rect[3] = pts[np.argmin(d)], pts[np.argmax(d)]
    return rect

def warp_card(frame, quad):
    M = cv2.getPerspectiveTransform(
        quad.astype('float32'),
        np.array([[0,0],[CARD_W-1,0],[CARD_W-1,CARD_H-1],[0,CARD_H-1]], dtype='float32')
    )
    warped = cv2.warpPerspective(frame, M, (CARD_W, CARD_H))
    px = int(CARD_W * 0.01); py = int(CARD_H * 0.01)
    warped = warped[py:CARD_H-py, px:CARD_W-px]
    warped = cv2.resize(warped, (CARD_W, CARD_H), interpolation=cv2.INTER_AREA)
    warped = _auto_orient_upright(warped)
    return warped

def _roi_rel_pad(img, roi, pad_x=0.02, pad_y=0.00):
    h, w = img.shape[:2]
    x0,y0,x1,y1 = roi
    x0 = max(0.0, x0 - pad_x); x1 = min(1.0, x1 + pad_x)
    y0 = max(0.0, y0 - pad_y); y1 = min(1.0, y1 + pad_y)
    X0, Y0, X1, Y1 = int(x0*w), int(y0*h), int(x1*w), int(y1*h)
    if X1<=X0 or Y1<=Y0: return np.zeros((1,1,3), dtype=np.uint8)
    return img[Y0:Y1, X0:X1]

# use a hair of horizontal padding on the title bands
_crop_title_roi_top = lambda img: _roi_rel_pad(img, ROI_TITLE_TOP, pad_x=0.02, pad_y=0.0)
_crop_title_roi_alt = lambda img: _roi_rel_pad(img, ROI_TITLE_ALT, pad_x=0.02, pad_y=0.0)

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

def _detect_foil_card(card_bgr: np.ndarray):
    rois = [
        _roi_rel(card_bgr, ROI_TITLE_TOP),
        _roi_rel(card_bgr, ROI_TITLE_ALT),
        _roi_rel(card_bgr, ROI_NUMBER_WIDE),
    ]
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

def _read_text_general(bin_img):
    inverted = 255 - bin_img
    if "rapidocr" in OCR_BACKEND and RAPIDOCR_AVAILABLE and (eng := _get_rapid_engine()):
        try:
            res = eng(cv2.cvtColor(inverted, cv2.COLOR_GRAY2BGR))[0]
            if res:
                res.sort(key=lambda r: r[0][0][0])
                txt = " ".join(str(r[1]) for r in res).strip()
                if txt:
                    return txt, max(float(r[2]) for r in res) * 100.0, "RapidOCR"
        except Exception:
            pass
    if "tesseract" in OCR_BACKEND and pytesseract:
        try:
            cfg = "--oem 3 --psm 6"
            txt = pytesseract.image_to_string(inverted, config=cfg, lang="eng").strip()
            if txt:
                return txt, 30.0, "Pytesseract"
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

    eng = _get_rapid_engine() if "rapidocr" in OCR_BACKEND and RAPIDOCR_AVAILABLE else None

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

    # ---------- RapidOCR helpers ----------
    def add_rapid(bgr, tag):
        if eng is None:
            return
        try:
            res = (eng(bgr)[0] or [])
            if not res:
                return

            # top-line join (group boxes near the top)
            items = [((r[0][0][0], 0.5*(r[0][0][1]+r[0][2][1])), r[1], float(r[2])*100.0) for r in res]
            ys = [p[1] for (p, _, _) in items]
            y0 = min(ys) if ys else 0.0
            top_band_h = 0.12 * bgr.shape[0]
            top_line = sorted([it for it in items if (it[0][1] - y0) <= top_band_h], key=lambda it: it[0][0])
            if top_line:
                joined = " ".join(t for _, t, _ in top_line).strip()
                bestc = max(c for _, _, c in top_line)
                maybe_add(joined, bestc, f"RapidOCR-topline/{tag}", relax=True)

            # all boxes individually + “join all” as a last resort
            for _, txt, score in items:
                maybe_add(txt, score, f"RapidOCR/{tag}")
            joined_all = " ".join(r[1] for r in res).strip()
            if joined_all:
                bestc = max(float(r[2]) for r in res) * 100.0
                maybe_add(joined_all, bestc, f"RapidOCR-joined/{tag}")
        except Exception as e:
            _dbg("OCR ERROR", f"{tag} Rapid error: {e}")

    # ---------- Tesseract helpers ----------
    def add_tess(bin_img, tag, psm=6):
        if not USE_TESS_FOR_TITLES or pytesseract is None or "tesseract" not in OCR_BACKEND:
            return
        try:
            cfg = f'--oem 3 --psm {psm} -c load_system_dawg=0 -c load_freq_dawg=0'
            txt = pytesseract.image_to_string(255 - bin_img, config=cfg, lang="eng").strip()
            if txt:
                maybe_add(txt, 42.0, f"Tesseract/{tag}")
        except Exception:
            pass

    # ---------- Build preprocess variants ----------
    roi_color = roi_src_bgr
    if foil:
        roi_color = _enhance_for_foil(roi_color)

    # color → Rapid
    add_rapid(_prep_roi_for_rapid_bgr(roi_color), "color")
    if foil:
        add_rapid(_prep_roi_for_rapid_bgr(_enhance_for_foil(roi_src_bgr)), "foil-color")

    # binary → Rapid & Tess
    bin_img = _prep_roi_for_ocr(roi_src_bgr)
    add_rapid(cv2.cvtColor(255 - bin_img, cv2.COLOR_GRAY2BGR), "inv-bin")
    add_tess(bin_img, "bin-psm6", psm=6)
    add_tess(bin_img, "bin-psm7", psm=7)

    # blackhat (great on shiny / low ink)
    if globals().get('OCR_ENABLE_BLACKHAT', False):
        g = cv2.cvtColor(roi_src_bgr, cv2.COLOR_BGR2GRAY)
        bh = _blackhat_bin(g)
        add_rapid(cv2.cvtColor(bh, cv2.COLOR_GRAY2BGR), "blackhat")
        add_tess(bh, "blackhat-psm7", psm=7)

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
    t = _normalize_num_chars(txt)
    for m in re.finditer(r'(?<!\d)([0-9\s|IlOo]{1,5})\s*/\s*([0-9\s|IlOo]{1,5})(?!\d)', t):
        L = re.sub(r'\D', '', m.group(1))[:4]
        R = re.sub(r'\D', '', m.group(2))[:4]
        if L and R:
            yield f"{int(L)}/{int(R)}"
    for m in re.finditer(r'(?<!\d)([0-9\s|IlOo]{2,8})(?!\d)', t):
        digits = re.sub(r'\D', '', m.group(1))
        if 2 <= len(digits) <= 4:
            yield str(int(digits))
    singles = re.findall(r'\d', t)
    if 2 <= len(singles) <= 4:
        yield str(int(''.join(singles)))

_TESS_NUM_CFG = (
    '--oem 1 --psm 7 '
    '-c tessedit_char_whitelist="0123456789/ " '
    '-c classify_bln_numeric_mode=1 '
    '-c load_system_dawg=0 -c load_freq_dawg=0'
)

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
        if len(t) == 4 and t.endswith("0") and t[:-1] in best:
            del best[t]
    def prio(k: str):
        if "/" in k:
            return (3, 0, 10**6)
        digits = int(re.sub(r"\D", "", k) or 0)
        closeness = -abs(len(k) - 3)
        return (2, closeness, digits)
    for tok in sorted(best.keys(), key=prio, reverse=True):
        yield tok, best[tok]

def _current_settings_effective():
    """
    Snapshot of the *actual* values in use (env > settings.json > defaults).
    Returned as plain key/value pairs for the Settings UI.
    """
    try:
        from . import config as cfg
    except ImportError:
        import config as cfg
        keys = [
            # Camera
            "CAMERA_DEVICE","REQ_WIDTH","REQ_HEIGHT","REQ_FPS","FOURCC_PRIMARY","FOURCC_FALLBACK",
            # Processing / Canvas
            "PROC_MAX_WIDTH","CARD_W","CARD_H","MIN_CARD_AREA_RATIO","MAX_CARD_AREA_RATIO",
            "BORDER_MARGIN_PCT","CARD_ASPECT","ASPECT_TOL","CONFIRM_FRAMES","STALE_FRAMES",
            "MAX_ASSOC_DIST","MAX_CARDS",
            # Autoscan / Steady
            "STEADY_MIN_FRAMES","AUTO_CAPTURE_WAIT_S","AUTOSCAN_OCR_TIMEOUT",
            "AUTOSCAN_SCRY_TIMEOUT","AUTOSCAN_IMG_TIMEOUT",
            # Detection cadence/robustness
            "DETECT_EVERY_N_FRAMES","RECTANGULARITY_MIN",
            # OCR & Debug
            "OCR_BACKEND","OCR_ONLY_ON_SNAPSHOT","DEBUG_OCR","OCR_DEBUG_BOXES","SHOW_ROI_OVERLAY",
            "MIN_TITLE_LETTERS","TEXT_PRESENCE_MIN","TITLE_ALLOW_TESS_FALLBACK",
            # Foil
            "FOIL_DETECT","FOIL_MIN_SCORE","FOIL_ON_TH","FOIL_OFF_TH",
            # Match
            "MATCH_ENABLE","MATCH_W_HASH","MATCH_W_HIST","MATCH_W_ORB","MATCH_TH","NAME_OK_TH",
            "ALWAYS_SCAN_OK","MATCH_USE_ART",
            # Scryfall / Prices
            "SCRYFALL_TIMEOUT","FX_URL","FX_TTL_SEC",
            # Moonraker
            "MOONRAKER_URL","HTTP_POST_URL",
            # ROIs
            "ROI_TITLE_TOP","ROI_TITLE_ALT","ROI_NUMBER_MAIN","ROI_NUMBER_WIDE",
            "ROI_NUMBER_TALL","ROI_NUMBER_NARROW","ROI_ART",
            # Paths / Server
            "DEBUG_LEVEL","DEBUG_SAVE_ROI","DEBUG_DIR","BAD_DIR",
            "HISTORY_DIR","HISTORY_IMG_DIR","HISTORY_JSON_EXT",
            "JPEG_QUALITY_SNAP","JPEG_QUALITY_CMP","JPEG_QUALITY_THUMB",
            "HOST","PORT",
        ]
        out = {}
        for k in keys:
            if hasattr(cfg, k):
                v = getattr(cfg, k)
                if isinstance(v, tuple): v = list(v)
                out[k] = v
        return out


def _read_collector_number(img, foil: bool = False):
    rois = [ROI_NUMBER_NARROW, ROI_NUMBER_MAIN, ROI_NUMBER_TALL, ROI_NUMBER_WIDE]
    best_roi = None
    for roi in rois:
        crop = _roi_rel(img, roi)
        if crop is None or crop.size == 0:
            continue
        src = _enhance_for_foil(crop) if foil else crop
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        if not _ink_present_quick(gray):
            continue
        g = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        g = cv2.GaussianBlur(g, (3, 3), 0)
        bin_img = cv2.adaptiveThreshold(
            g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
            17 if foil else 23, 3
        )
        if pytesseract is not None and "tesseract" in OCR_BACKEND:
            try:
                raw = pytesseract.image_to_string(bin_img, config=_TESS_NUM_CFG, lang="eng") or ""
                for tok, conf in _num_tokens_from_text(raw):
                    return tok, conf
            except Exception:
                pass
        if best_roi is None and _ink_present_quick(bin_img, thresh=0.02):
            best_roi = crop
        if foil:
            bh = _blackhat_bin(gray)
            if pytesseract is not None and "tesseract" in OCR_BACKEND:
                try:
                    raw2 = pytesseract.image_to_string(bh, config=_TESS_NUM_CFG, lang="eng") or ""
                    for tok, conf in _num_tokens_from_text(raw2):
                        return tok, max(conf, 39.0)
                except Exception:
                    pass
    if best_roi is not None and RAPIDOCR_AVAILABLE and "rapidocr" in OCR_BACKEND:
        try:
            eng = _get_rapid_engine()
            rbgr = _prep_roi_for_rapid_bgr(_enhance_for_foil(best_roi) if foil else best_roi)
            res = (eng(rbgr)[0] or [])
            raw = " ".join(r[1] for r in res)
            for tok, conf in _num_tokens_from_text(raw):
                return tok, max(conf, 39.0)
        except Exception:
            pass
    return "", 0.0

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

def _largest_external_contour(bin_img):
    cnts, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(cnts, key=cv2.contourArea) if cnts else None

def _rect_to_box(rect):
    return order_points(cv2.boxPoints(rect).astype(np.float32))

def _illum_norm_gray(bgr_small: np.ndarray) -> np.ndarray:
    g  = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2GRAY)
    bg = cv2.GaussianBlur(g, (31, 31), 0)
    bg = np.maximum(bg, 1)

    # Already 0..255 when inputs are uint8 — do NOT multiply again.
    norm = cv2.divide(g, bg, scale=255)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(norm)

def _shadow_robust_mask(bgr_small: np.ndarray) -> np.ndarray:
    """
    Keep the card, reject white-sheet background + soft shadows.
    Pipeline:
      - illumination-normalized gray
      - adaptive inverse threshold (card ~ white)
      - add dilated Canny edges (stabilize borders)
      - drop connected components that touch the image border
      - close/open + trim 1% frame to avoid edge locks
    """
    g = _illum_norm_gray(bgr_small)

    # Card is darker than paper => after norm, invert binary so card = 255
    th = cv2.adaptiveThreshold(
        g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 2
    )

    edges = cv2.Canny(g, 36, 110)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), 1)

    mask = cv2.max(th, edges)

    # Remove any blob that touches the image border
    h, w = mask.shape[:2]
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros_like(mask)
    for i in range(1, num):
        x, y, ww, hh, area = stats[i]
        touches = (x <= 0) or (y <= 0) or (x + ww >= w - 1) or (y + hh >= h - 1)
        if touches:
            continue
        if area < 0.004 * (w * h):  # drop tiny flecks
            continue
        keep[labels == i] = 255
    mask = keep

    k1 = _K1 if _K1 is not None else cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    k2 = _K2 if _K2 is not None else cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k1, 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k2, 1)

    pad = max(2, int(0.01 * min(h, w)))
    mask[:pad, :] = 0; mask[-pad:, :] = 0; mask[:, :pad] = 0; mask[:, -pad:] = 0
    return mask

def _rectangularity(cnt, rect) -> float:
    """Contour area divided by its minAreaRect box area."""
    area = cv2.contourArea(cnt)
    (w, h) = rect[1]
    box_area = max(1.0, float(w) * float(h))
    return float(area) / box_area

def detect_cards(frame):
    # --- crop to current ROI (pixels) ---
    with _detect_roi_lock:
        roi_norm = list(DETECT_ROI)
    rx0, ry0, rx1, ry1 = _abs_roi_from_norm(frame, roi_norm)
    sub = frame[ry0:ry1, rx0:rx1]
    if sub is None or sub.size == 0:
        return []

    # --- downscale within ROI for speed ---
    maxw = PROC_DOWNSCALE_MAX_W
    scale = maxw / sub.shape[1] if sub.shape[1] > maxw else 1.0
    small = cv2.resize(sub, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    mask = _shadow_robust_mask(small)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return []

    Hs, Ws = small.shape[:2]
    img_area = float(Hs * Ws)
    edge_margin = int(BORDER_MARGIN_PCT * max(Hs, Ws))

    best = None
    best_score = -1.0

    aspect_target = CARD_ASPECT
    aspect_tol    = ASPECT_TOL
    min_area      = MIN_CARD_AREA_RATIO
    max_area      = MAX_CARD_AREA_RATIO
    rect_min      = RECTANGULARITY_MIN

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
        if not (aspect_target - aspect_tol <= aspect <= aspect_target + aspect_tol):
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

        # reject if touching ROI edges (in ROI-space)
        x = box[:, 0]; y = box[:, 1]
        if (x.min() <= edge_margin or x.max() >= (Ws - edge_margin) or
            y.min() <= edge_margin or y.max() >= (Hs - edge_margin)):
            continue

        aspect = min(rect[1][0], rect[1][1]) / max(rect[1][0], rect[1][1])
        aspect_score = 1.0 - min(1.0, abs(aspect - aspect_target) / (aspect_target + 1e-6))
        score = 0.55 * area_norm + 0.30 * rectness + 0.15 * aspect_score
        if score > best_score:
            best_score = score
            best = rect

    if best is None:
        return []

    # lock to exact aspect, unscale, and offset back into full-frame coords
    best = _fix_rect_aspect(best, target_aspect=aspect_target)
    quad_small = order_points(cv2.boxPoints(best).astype(np.float32))
    quad_full = (quad_small / scale).astype(np.float32)
    quad_full[:, 0] += rx0
    quad_full[:, 1] += ry0

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

def _enhance_for_foil(roi_bgr: np.ndarray) -> np.ndarray:
    if roi_bgr is None or roi_bgr.size == 0:
        return np.zeros((1,1,3), dtype=np.uint8)
    lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8,8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    bgr = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = (s.astype(np.float32) * 0.6).clip(0,255).astype(np.uint8)
    hsv2 = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR)

def ocr_from_card_upright(img):
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
    set_hint = _read_set_hint(_roi_rel(img, ROI_NUMBER_WIDE)) or ""

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

def _scryfall_request(url, params=None, tries=2):
    if _scry_offline_now():
        return None
    params = params or {}
    for i in range(tries):
        try:
            r = _HTTP.get(url, params=params, timeout=SCRYFALL_TIMEOUT)
            if r.status_code == 200:
                return r.json()
        except Exception as e:
            if i+1 == tries:
                _dbg("SCRYFALL ERROR", f"Request failed after {tries} tries: {e}")
            _SCRY_OFFLINE["until"] = time.time() + 60.0  # 1 minute backoff
        time.sleep(0.15)
    return None

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

def _hist_sim(bgrA, bgrB):
    hsvA = cv2.cvtColor(bgrA, cv2.COLOR_BGR2HSV)
    hsvB = cv2.cvtColor(bgrB, cv2.COLOR_BGR2HSV)
    def hist(hsv):
        h = cv2.calcHist([hsv],[0],None,[32],[0,180]); s = cv2.calcHist([hsv],[1],None,[32],[0,256])
        cv2.normalize(h, h); cv2.normalize(s, s); return h, s
    hA,sA = hist(hsvA); hB,sB = hist(hsvB)
    cH = cv2.compareHist(hA, hB, cv2.HISTCMP_CORREL)
    cS = cv2.compareHist(sA, sB, cv2.HISTCMP_CORREL)
    return float((cH + 1.0) * 0.5 * 0.6 + (cS + 1.0) * 0.5 * 0.4)

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
        A0 = _crop_art(A0); B0 = _crop_art(B0)

    ref_feats = _get_cached_reference_feats(scry_card, scry_bgr, use_art=MATCH_USE_ART, tgt=tgt) or {}
    A_orb = cv2.resize(_normalize_illum(A0), tgt, interpolation=cv2.INTER_AREA)
    A_hist = cv2.resize(A0, tgt, interpolation=cv2.INTER_AREA)

    # Hash similarity (with cached reference hashes if available)
    ah = _ahash64(A_orb)
    dh = _dhash64(A_orb)
    ah2 = ref_feats.get("ah2"); dh2 = ref_feats.get("dh2")
    if ah2 is None or dh2 is None:
        B_orb_tmp = cv2.resize(_normalize_illum(B0), tgt, interpolation=cv2.INTER_AREA)
        ah2 = _ahash64(B_orb_tmp); dh2 = _dhash64(B_orb_tmp)
    sim_hash = 1.0 - min(_hamming64(ah, ah2), _hamming64(dh, dh2)) / 64.0

    # Histogram similarity (HSV H+S)
    hsvA = cv2.cvtColor(A_hist, cv2.COLOR_BGR2HSV)
    hA = cv2.calcHist([hsvA],[0],None,[32],[0,180]); sA = cv2.calcHist([hsvA],[1],None,[32],[0,256])
    cv2.normalize(hA, hA); cv2.normalize(sA, sA)
    hB = ref_feats.get("hB"); sB = ref_feats.get("sB")
    if hB is None or sB is None:
        hsvB = cv2.cvtColor(cv2.resize(B0, tgt, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV)
        hB = cv2.calcHist([hsvB],[0],None,[32],[0,180]); sB = cv2.calcHist([hsvB],[1],None,[32],[0,256])
        cv2.normalize(hB, hB); cv2.normalize(sB, sB)
    cH = cv2.compareHist(hA, hB, cv2.HISTCMP_CORREL)
    cS = cv2.compareHist(sA, sB, cv2.HISTCMP_CORREL)
    sim_hist = float((cH + 1.0) * 0.5 * 0.6 + (cS + 1.0) * 0.5 * 0.4)

    fast_accept = (sim_hash >= 0.90 and sim_hist >= 0.85)
    fast_reject = (sim_hash <= 0.40 and sim_hist <= 0.50)

    if (fast_accept or fast_reject) and not return_details:
        score = MATCH_W_HASH * sim_hash + MATCH_W_HIST * sim_hist
        ok = fast_accept
        return float(score), ok

    # 3) ORB (use cached reference keypoints/descriptors if available)
    sim_orb, good, orb_dbg = 0.0, [], None
    ref_kits_ready = ref_feats.get("kB") is not None and ref_feats.get("dB") is not None
    if ref_kits_ready:
        sim_orb, good, orb_dbg = _orb_sim_v2_cached(A_orb, ref_feats)
    else:
        # fallback: compute fresh
        B_orb = cv2.resize(_normalize_illum(B0), tgt, interpolation=cv2.INTER_AREA)
        sim_orb, good, orb_dbg = _orb_sim_v2(A_orb, B_orb)

    # Full weighted score
    score = MATCH_W_HASH * sim_hash + MATCH_W_HIST * sim_hist + MATCH_W_ORB * sim_orb
    ok = bool(score >= MATCH_TH)

    if not return_details:
        return float(score), ok

    details = {
        "sim_hash": float(sim_hash),
        "sim_hist": float(sim_hist),
        "sim_orb":  float(sim_orb),
        "score":    float(score),
        "ok":       ok,
        "weights": {"hash": MATCH_W_HASH, "hist": MATCH_W_HIST, "orb": MATCH_W_ORB},
        "orb_dbg": {"A": A_orb, "B": ref_feats.get("B_orb") if ref_feats.get("B_orb") is not None else None, "orb_data": orb_dbg}
    }
    return float(score), ok, details


def _crop_art(img):
    if img is None or img.size == 0:
        return img
    h, w = img.shape[:2]
    x0, y0, x1, y1 = ROI_ART
    x0, y0, x1, y1 = int(x0*w), int(y0*h), int(x1*w), int(y1*h)
    if x1-x0 < 40 or y1-y0 < 40:
        return img
    return img[y0:y1, x0:x1]

def _render_compare_visual(details):
    try:
        A = details["orb_dbg"]["A"]
        B = details["orb_dbg"]["B"]
        orb_data = details["orb_dbg"]["orb_data"]
        if A is None or B is None:
            return None
        if orb_data is not None:
            kA, kB, good, mask = orb_data
            matchesMask = mask.ravel().tolist() if mask is not None else None
            flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            vis = cv2.drawMatches(A, kA, B, kB, good, None,
                                  matchesMask=matchesMask, flags=flags)
        else:
            vis = cv2.hconcat([A, B])
        t = f"HASH {details['sim_hash']:.2f} • HIST {details['sim_hist']:.2f} • ORB {details['sim_orb']:.2f}  ⇒  SCORE {details['score']:.3f}  ({'MATCH' if details['ok'] else 'REVIEW'})"
        pad = 36
        canvas = np.zeros((vis.shape[0]+pad, vis.shape[1], 3), dtype=np.uint8)
        canvas[:,:] = (18,18,24)
        canvas[pad:pad+vis.shape[0], :vis.shape[1]] = vis
        cv2.putText(canvas, t, (10,24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230,230,230), 1, cv2.LINE_AA)
        return canvas
    except Exception:
        return None

def _publish_compare_visual(details):
    global _last_cmp_img, _last_cmp_stats
    img = _render_compare_visual(details)
    with compare_lock:
        _last_cmp_img = img
        _last_cmp_stats = {k: v for k, v in details.items() if k != "orb_dbg"}

# =========================
#MARK: THREADS / WORKERS
# =========================
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
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
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
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        except Exception:
            pass
    return cap if cap.isOpened() else None

def video_thread():
    global cap, output_frame, current_card_crop, scanner_state
    cap = _open_camera()
    if not cap:
        _dbg("CAMERA ERROR", f"Could not open camera {CAMERA_DEVICE}")
        return
    _dbg("CAMERA INFO", f"{int(cap.get(3))}x{int(cap.get(4))} @ {cap.get(5):.1f}fps")
    scanner_state.update({'locked': False, 'locked_frames': 0, 'locked_area': 0, 'steady': False})
    frame_i = 0

    # Initialize the local ROI version we've seen so far
    with _detect_roi_lock:
        roi_ver_seen = _detect_roi_version

    try:
        while not shutdown_evt.is_set():
            with _detect_roi_lock:
                rv = _detect_roi_version

            if rv != roi_ver_seen:
                with video_lock:
                    tracks.clear()
                    scanner_state.update({'locked': False, 'locked_frames': 0, 'steady': False})
                roi_ver_seen = rv

            if PROC_PAUSED:
                time.sleep(0.03)
                continue

            ok, frame = cap.read()
            if not ok:
                time.sleep(0.02)
                continue

            dets = []
            if frame_i % DETECT_EVERY_N_FRAMES == 0:
                try:
                    dets = detect_cards(frame)
                    scanner_state.pop('detect_error', None)
                except Exception as e:
                    dets = []
                    scanner_state['detect_error'] = str(e)
            frame_i += 1

            with video_lock:
                for t in tracks.values():
                    t['updated'] = False

                for d in dets:
                    d['centroid'] = d['box'].mean(axis=0)
                    best_id, best_dist = None, MAX_ASSOC_DIST
                    for tid, t in tracks.items():
                        dist = np.hypot(*(d['centroid'] - t['centroid']))
                        if dist < best_dist:
                            best_dist, best_id = dist, tid
                    if best_id is not None:
                        t = tracks[best_id]
                        t['box'] = (t['box'] * 0.85 + d['box'] * 0.15).astype(np.float32)
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
                            'updated': True
                        }

                for tid in [tid for tid, t in list(tracks.items()) if not t['updated']]:
                    t = tracks[tid]
                    t['missed'] = t.get('missed', 0) + 1
                    if t['missed'] > STALE_FRAMES:
                        del tracks[tid]

                out = frame.copy()
                locked = next((t for t in tracks.values() if t.get('seen', 0) >= CONFIRM_FRAMES), None)
                if locked:
                    current_card_crop = warp_card(frame, locked['box'])
                    steady = locked.get('seen', 0) >= STEADY_MIN_FRAMES
                    scanner_state.update({
                        'locked': True,
                        'locked_frames': locked['seen'],
                        'locked_area': locked['area'],
                        'steady': steady
                    })
                    # PERF: throttle foil detection to reduce per-frame cost
                    if FOIL_DETECT and (frame_i % max(1, int(FOIL_EVERY_N_FRAMES)) == 0):
                        try:
                            isf, score = _detect_foil_card(current_card_crop)
                            prev = float(scanner_state.get("foil_score", 0.0))
                            sm = 0.7 * prev + 0.3 * score if prev else score
                            was = bool(scanner_state.get("foil", False))
                            now = (sm >= FOIL_ON_TH) or (was and sm >= FOIL_OFF_TH)
                            scanner_state.update({"foil": now, "foil_score": round(sm, 2)})
                        except Exception:
                            pass
                    elif not FOIL_DETECT:
                        scanner_state.pop("foil", None)
                        scanner_state.pop("foil_score", None)
                    cv2.polylines(out, [locked['box'].astype(int)], True, (0, 255, 0), 2)
                else:
                    current_card_crop = None
                    scanner_state.update({'locked': False, 'locked_frames': 0, 'steady': False})

                output_frame = out
    finally:
        if cap:
            cap.release()
            print("Camera released.")

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
        try:
            res = ocr_from_card_upright(snap)
            res["name"] = _maybe_correct_ocr_name(res["name_raw"], res["name_conf"])
            if DEBUG_OCR:
                if res["name"] != res["name_raw"]:
                    _dbg("OCR INFO", f"'{res['name_raw']}' -> '{res['name']}'")
                _dbg("OCR INFO", f"name='{res['name']}'(c:{res['name_conf']:.1f}) num='{res['number']}'(c:{res['number_conf']:.1f}) set='{res.get('set_hint','')}' foil={res.get('foil')} ({res.get('foil_score',0):.2f})")
            with ocr_lock:
                ocr_state.update(res)
                ocr_state["updated_at"] = time.time()
                ocr_state["seq"] = local_seq
        except Exception as e:
            with ocr_lock:
                ocr_state["last_error"] = str(e)
        finally:
            # PERF: mark last run for live OCR throttling
            if not is_snapshot:
                last_live_ocr_ts = time.time()
        
        try:
            global PROC_PAUSED, PROC_PAUSE_SEQ
            snap_seq = None
            with ocr_lock:
                snap_seq = ocr_state.get("seq")
            if PROC_PAUSED and (PROC_PAUSE_SEQ is not None) and (snap_seq == PROC_PAUSE_SEQ):
                PROC_PAUSED = False
                PROC_PAUSE_SEQ = None
        except Exception:
            pass

def _clear_ocr():
    with ocr_lock:
        ocr_state.clear()
        ocr_state.update({
            "name": "", "name_raw": "", "name_conf": 0.0,
            "number": "", "number_raw": "", "number_conf": 0.0,
            "set_hint": "", "updated_at": time.time(),
            "provider": PRIMARY_PROVIDER_LABEL,
            "foil": False, "foil_score": 0.0,
            "match_score": 0.0, "match_ok": None, "flagged": False,
            "seq": None,
        })
    with cardinfo_lock:
        cardinfo_state.clear()

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

def cardinfo_worker():
    last_key = None
    while not shutdown_evt.is_set():
        time.sleep(0.2)

        # Snapshot OCR fields
        with ocr_lock:
            name       = (ocr_state.get("name") or "").strip()
            number_raw = (ocr_state.get("number_raw") or "").strip()
            set_hint   = (ocr_state.get("set_hint") or "").strip().lower()
            updated_at = ocr_state.get("updated_at", 0)
            seq        = ocr_state.get("seq")

        # Only work once per OCR update; require at least name or number
        if not (name or number_raw) or updated_at == cardinfo_state.get("last_updated", 0):
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
                    roi_img = _roi_rel(scanned_card, ROI_NUMBER_WIDE)
            if roi_img is None:
                with video_lock:
                    if current_card_crop is not None:
                        roi_img = _roi_rel(current_card_crop, ROI_NUMBER_WIDE)
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

        usd_to_cad = _get_usd_to_cad()

        # Warm image cache (ignore failures)
        try:
            _ = _fetch_scry_image(choice)
        except Exception:
            pass

        # Publish
        with cardinfo_lock:
            cardinfo_state["scry"] = choice
            cardinfo_state["fx"] = {"rate": usd_to_cad}
            cardinfo_state["last_updated"] = updated_at

# =========================
#MARK: MOONRAKER WS + AUTOSCAN
# =========================

def _send_gcode(script, timeout=None):
    # Use a longer timeout for long-running moves/homing/macros; Moonraker blocks until completion.
    try:
        s = (script or "").strip().upper()
        long = s.startswith(("G28", "RUN_SORTER_INTERACTIVE", "SORTER_HOME", "SORTER_CYCLE_INTERACTIVE", "MOVE_TO_SCAN", "PICKUP_CARD"))
        to = timeout if timeout is not None else (120 if long else 30)
        _HTTP.post(HTTP_POST_URL, json={"script": script}, timeout=to)
    except Exception as e:
        _dbg("WEBSOCKET ERROR", f"gcode post failed: {e}")

def _send_ack_ok(job): _send_gcode(f"M118 ACK_OK job={job}")
def _send_scan_ok(job): _send_gcode("SCAN_OK")
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
            if "READY_TO_SCAN" in line:
                job = _parse_ready_line(line)
                if job is None: return
                with printer_lock:
                    printer_state["last_ready_line"] = line
                    printer_state["awaiting"] = True
                    printer_state["job_id"] = job
                    seen = printer_state["ack_ok_jobs"]
                    if job not in seen:
                        seen.add(job)
                        _send_ack_ok(job)
                        _dbg("WEBSOCKET", f"READY_TO_SCAN -> ACK_OK (job={job})")

                # Auto-pause streams while the printer requests a scan
                STREAM_STATE['paused'] = True
                STREAM_STATE['paused_reason'] = f'ready_to_scan job={job}'
            elif "ACK_SCAN_OK" in line:
                _dbg("WEBSOCKET", f"Printer ACKed SCAN_OK {line}")
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

        # 2) Wait until the card is steady
        while not shutdown_evt.is_set():
            with video_lock:
                steady = bool(scanner_state.get("steady", False))
            if steady:
                break
            time.sleep(0.02)

        # 3) Small grace period then capture a snapshot
        time.sleep(AUTO_CAPTURE_WAIT_S)
        ok = capture_scanned_card_from_live()
        with scan_lock:
            cur_seq = snapshot_seq
        
        if not ok:
            _dbg("AUTO-SNAPSHOT WARNING", f"Auto-snap failed for job number:{job} (no card?)")

        # 4) Wait briefly for OCR results tied to THIS snapshot seq
        deadline = time.time() + AUTOSCAN_OCR_TIMEOUT
        nm = ""; cn_raw = ""; shint = ""; ocr_updated_at = 0.0
        got_ocr = False
        while time.time() < deadline:
            with ocr_lock:
                seq_ok  = (ocr_state.get("seq") == cur_seq)
                nm      = (ocr_state.get("name") or "").strip()
                cn_raw  = (ocr_state.get("number_raw") or "").strip()
                shint   = (ocr_state.get("set_hint") or "").strip().lower()
                ocr_updated_at = float(ocr_state.get("updated_at", 0))
                got_ocr = seq_ok and bool(nm or cn_raw)
            if got_ocr:
                break
            time.sleep(0.02)
        if not got_ocr:
            _dbg("OCR WARNING", f"OCR timeout ({AUTOSCAN_OCR_TIMEOUT:.1f}s) for job={job} (seq={cur_seq})")

        # 5) Prefer the cardinfo_worker result for this OCR update
        choice = None
        with cardinfo_lock:
            if cardinfo_state.get("last_updated") == ocr_updated_at:
                choice = cardinfo_state.get("scry")

        # 6) Single, de-duplicated fallback lookup (no repeats)
        if choice is None and (nm or cn_raw):
            try:
                # First try with OCR’s set hint (if any)…
                choice = _scryfall_lookup_once(nm, cn_raw, shint, cur_seq)
                # …then fall back to no set constraint
                if choice is None and shint:
                    choice = _scryfall_lookup_once(nm, cn_raw, "", cur_seq)
            except Exception:
                choice = None

        # 7) Grab snapshot & Scryfall image (cache is warmed by cardinfo_worker when possible)
        with scan_lock:
            snap_img = scanned_card.copy() if scanned_card is not None else None
        scry_img = _fetch_scry_image(choice) if choice is not None else None

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



        visual_ready = (choice is not None and snap_img is not None and scry_img is not None)
        flagged = (not visual_ready) or (not match_ok)

        if not visual_ready:
            _dbg("COMPARE", f"FLAGGED (no Scryfall image or snapshot) job={job}")
        elif not match_ok:
            _dbg("COMPARE", f"FLAGGED job={job} score={match_score:.3f} (visual threshold {MATCH_TH})")
        else:
            _dbg("COMPARE", f"VISUAL MATCH job={job} score={match_score:.3f} (>= {MATCH_TH})")

        # 9) Publish quick summary to OCR state for UI
        with ocr_lock:
            ocr_state["match_score"] = round(float(match_score), 3)
            ocr_state["match_ok"] = bool(match_ok)
            ocr_state["flagged"] = bool(flagged)

        # 10) Persist review entry & resume processing/stream
        try:
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
            save_scan_entry(
                snap_img if snap_img is not None else np.zeros((10,10,3), np.uint8),
                ocr_snapshot,
                choice,
                cmp_details,
                match_score,
                match_ok,
                flagged
            )

            # Resume processing for this snapshot
            global PROC_PAUSED, PROC_PAUSE_SEQ
            PROC_PAUSED = False
            PROC_PAUSE_SEQ = None
            STREAM_STATE['paused'] = False
            STREAM_STATE['paused_reason'] = ''

        except Exception as e:
            _dbg("HISTORY ERROR", f"save_scan_entry failed: {e}")

        # 11) Notify printer
        if ALWAYS_SCAN_OK or not flagged:
            _send_scan_ok(job or 0)
            _dbg("WEBSOCKET", f"Sent SCAN_OK (job={job})")
            STREAM_STATE['paused'] = False
            STREAM_STATE['paused_reason'] = ''
            with printer_lock:
                printer_state["last_decision"] = "SCAN_OK (flagged)" if flagged else "SCAN_OK"
        else:
            _send_scan_fail(job or 0)
            _dbg("WEBSOCKET", f"Sent SCAN_FAIL (job={job})")
            with printer_lock:
                printer_state["last_decision"] = "SCAN_FAIL"

        # 12) Clear pending flag for this job
        with printer_lock:
            printer_state["awaiting"] = False

# =========================
#MARK: STREAM / UI
# =========================
def _draw_ocr_debug_boxes(img):
    h,w = img.shape[:2]
    def draw(roi,c,th=2):
        x0=int(roi[0]*w); y0=int(roi[1]*h); x1=int(roi[2]*w); y1=int(roi[3]*h)
        cv2.rectangle(img,(x0,y0),(x1,y1),c,th)
    draw(ROI_TITLE_TOP,(0,255,0))
    draw(ROI_TITLE_ALT,(0,200,255))
    draw(ROI_NUMBER_MAIN,(220,120,255))
    draw(ROI_NUMBER_WIDE,(160,80,255))
    draw(ROI_NUMBER_TALL,(200,60,255))
    draw(ROI_NUMBER_NARROW,(255,80,120))

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
            with scan_lock:
                if preview_card is not None:
                    img = preview_card.copy()
                elif scanned_card is not None:
                    img = scanned_card.copy()
            if img is None:
                with video_lock:
                    img = current_card_crop.copy() if current_card_crop is not None else None

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
    STREAM_STATE['fps'] = max(1, min(60, int(STREAM_STATE.get('fps', 25))))
    STREAM_STATE['quality'] = max(40, min(95, int(STREAM_STATE.get('quality', 80))))
    return jsonify({"ok": True, **STREAM_STATE})

@app.route('/compare.jpg')
def compare_jpg():
    with compare_lock:
        img = _last_cmp_img
    if img is None:
        empty = np.zeros((1,1,3), dtype=np.uint8)
        ok, enc = cv2.imencode('.png', empty)
        return Response(enc.tobytes(), mimetype='image/png')
    ok, enc = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY_CMP])
    return Response(enc.tobytes(), mimetype='image/jpeg')

# ---------------------------
# UI file routes
# ---------------------------

@app.get("/api/logs")
def api_logs():
    """
    Poll console logs.
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
        out.append({
            "id":   int(e["id"]),
            "ts":   float(e["ts"]),
            "tag":  e["tag"],
            "msg":  e["msg"],
            "lvl":  int(e.get("lvl", 1)),   # <— add this
            "line": f"[{e['tag']}] {e['msg']}",
        })
    return jsonify({"ok": True, "items": out, "next": int(nxt)})


@app.post("/api/logs/clear")
def api_logs_clear():
    with log_lock:
        LOG_RING.clear()
    return jsonify({"ok": True})

def load_settings_file():
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except FileNotFoundError:
        return {}
    except Exception:
        return {}

def save_settings_file(data: dict):
    os.makedirs(os.path.dirname(os.path.abspath(SETTINGS_PATH)), exist_ok=True)
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)

def parse_roi_csv(s: str):
    try:
        nums = [float(x.strip()) for x in str(s).split(",")]
        if len(nums) == 4: return nums
    except Exception:
        pass
    return None

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
        "match": {
            "ok": ocr_snap.get("match_ok"),
            "score": ocr_snap.get("match_score"),
        },
    })
    state["bad_count"] = len(bad_cards)
    state["loaded_scan_id"] = current_loaded_scan_id
    return jsonify(state)

# =========================
#MARK: DETECTION ROI (UI <-> server)
# =========================
# Normalized ROI in [0,1] coords: [x0, y0, x1, y1]
DETECT_ROI = [0.0, 0.0, 1.0, 1.0]
_detect_roi_lock = threading.Lock()
_detect_roi_version = 0  # bump whenever the ROI changes

def _abs_roi_from_norm(frame, roi):
    """Convert normalized ROI to absolute pixel bounds (clamped)."""
    h, w = frame.shape[:2]
    x0 = max(0, min(w, int(roi[0] * w)))
    y0 = max(0, min(h, int(roi[1] * h)))
    x1 = max(0, min(w, int(roi[2] * w)))
    y1 = max(0, min(h, int(roi[3] * h)))
    if x1 <= x0 or y1 <= y0:
        return (0, 0, w, h)
    return (x0, y0, x1, y1)

@app.get("/api/detect_roi")
def api_get_detect_roi():
    with _detect_roi_lock:
        return jsonify({"ok": True, "roi": list(DETECT_ROI)})

@app.post("/api/detect_roi")
def api_set_detect_roi():
    data = request.get_json(silent=True) or {}
    roi = data.get("roi")
    if not isinstance(roi, (list, tuple)) or len(roi) != 4:
        return jsonify({"ok": False, "error": "roi must be [x0,y0,x1,y1]"}), 400
    try:
        x0, y0, x1, y1 = [float(x) for x in roi]
    except Exception:
        return jsonify({"ok": False, "error": "roi must be float-like values"}), 400

    # normalize + clamp + sort
    x0, x1 = sorted((max(0.0, min(1.0, x0)), max(0.0, min(1.0, x1))))
    y0, y1 = sorted((max(0.0, min(1.0, y0)), max(0.0, min(1.0, y1))))
    if (x1 - x0) < 0.05 or (y1 - y0) < 0.05:  # avoid degenerate tiny boxes
        return jsonify({"ok": False, "error": "roi too small"}), 400

    with _detect_roi_lock:
        DETECT_ROI[:] = [x0, y0, x1, y1]
        global _detect_roi_version
        _detect_roi_version += 1 
    try:
        cur = _settings_load()
    except Exception:
        cur = {}
    try:
        if not isinstance(cur, dict):
            cur = {}
        cur["DETECT_ROI"] = [x0, y0, x1, y1]
        try:
            _settings_save(cur)
        except Exception:
            # fallback to legacy helper if present
            save_settings_file(cur)
    except Exception as e:
        _dbg("SETTINGS ERROR", f"save DETECT_ROI failed: {e}")

    return jsonify({"ok": True, "roi": list(DETECT_ROI)})


@app.route('/card')
def card(): return Response(_stream_frames('card'), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.after_request
def _no_cache(resp):
    p = request.path
    if p.startswith('/api/scan/') and p.endswith('/thumb'):
        resp.headers['Cache-Control'] = 'public, max-age=31536000, immutable'
    else:
        resp.headers['Cache-Control'] = 'no-store'
    return resp

@app.route('/api/carddata')
def api_carddata():
    with cardinfo_lock: payload = dict(cardinfo_state)
    return jsonify(payload)

@app.route('/api/badlist')
def api_bad():
    return jsonify({"items": list(bad_cards), "count": len(bad_cards)})

# --- Photometric normalization (WB + tone match to reference) ---

TONE_ALIGN_ENABLE = globals().get("TONE_ALIGN_ENABLE", False)

def _gray_world_wb(bgr: np.ndarray) -> np.ndarray:
    """
    Simple gray-world white balance. Fast and robust for cheap LED color casts.
    """
    if bgr is None or bgr.size == 0:
        return bgr
    img = bgr.astype(np.float32)
    b, g, r = cv2.split(img)
    eps = 1e-6
    mean_b, mean_g, mean_r = b.mean() + eps, g.mean() + eps, r.mean() + eps
    gray = (mean_b + mean_g + mean_r) / 3.0
    kb, kg, kr = gray/mean_b, gray/mean_g, gray/mean_r
    b *= kb; g *= kg; r *= kr
    out = cv2.merge((b, g, r))
    return np.clip(out, 0, 255).astype(np.uint8)

def _build_lut_from_cdfs(cdf_src: np.ndarray, cdf_ref: np.ndarray) -> np.ndarray:
    """
    Given two 256-length CDFs (0..1), build a uint8 LUT that maps src->ref.
    """
    lut = np.zeros(256, dtype=np.uint8)
    j = 0
    for i in range(256):
        while j < 255 and cdf_ref[j] < cdf_src[i]:
            j += 1
        lut[i] = j
    return lut

def _u8_cdf(u8: np.ndarray) -> np.ndarray:
    hist = cv2.calcHist([u8],[0],None,[256],[0,256]).ravel().astype(np.float64)
    cdf  = hist.cumsum()
    if cdf[-1] <= 0:  # avoid div-by-zero
        return np.linspace(0, 1, 256, dtype=np.float64)
    return cdf / cdf[-1]

def _match_l_to_reference(src_bgr: np.ndarray, ref_bgr: np.ndarray) -> np.ndarray:
    """
    Histogram-match the L channel of src to ref in LAB space.
    Keeps A/B as-is to avoid weird hue shifts.
    """
    if src_bgr is None or ref_bgr is None or src_bgr.size == 0 or ref_bgr.size == 0:
        return src_bgr

    src_lab = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2LAB)
    ref_lab = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2LAB)
    Ls, As, Bs = cv2.split(src_lab)
    Lr, _,  _  = cv2.split(ref_lab)

    # Guard degenerate cases; fall back to linear percentile matching
    std_s = float(Ls.std()); std_r = float(Lr.std())
    if std_s < 1.5 or std_r < 1.5:
        # percentile-based affine (robust to speculars)
        s_lo, s_hi = np.percentile(Ls, [3, 97])
        r_lo, r_hi = np.percentile(Lr, [3, 97])
        scale = (r_hi - r_lo) / max(1.0, (s_hi - s_lo))
        offset = r_lo - scale * s_lo
        Lm = np.clip(Ls.astype(np.float32) * scale + offset, 0, 255).astype(np.uint8)
    else:
        cdf_s = _u8_cdf(Ls)
        cdf_r = _u8_cdf(Lr)
        lut = _build_lut_from_cdfs(cdf_s, cdf_r)
        Lm = cv2.LUT(Ls, lut)

    out_lab = cv2.merge((Lm, As, Bs))
    return cv2.cvtColor(out_lab, cv2.COLOR_LAB2BGR)

def _tone_align_to_reference(src_bgr: np.ndarray, ref_bgr: np.ndarray) -> np.ndarray:
    """
    Full photometric align: gray-world WB then L-channel histogram match.
    """
    if src_bgr is None or ref_bgr is None:
        return src_bgr
    src = _gray_world_wb(src_bgr)
    ref = _gray_world_wb(ref_bgr)
    return _match_l_to_reference(src, ref)

def _estimate_whitepoint(bgr: np.ndarray):
    """Robust white estimate: bright & low-sat cluster; fallback to gray-world."""
    if bgr is None or bgr.size == 0: return (1.0, 1.0, 1.0)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mask = ((s < 35) & (v > 200))
    if int(mask.sum()) < 600:  # fallback: top 0.5% brightest, lenient sat
        thr = np.percentile(v, 99.5)
        mask = (v >= thr) & (s < 60)
    region = bgr[mask] if np.any(mask) else bgr.reshape(-1,3)
    wp = region.astype(np.float32).mean(axis=0)  # B,G,R
    return tuple(np.clip(wp, 1.0, 255.0))

def _whitepoint_align(src_bgr: np.ndarray, ref_bgr: np.ndarray) -> np.ndarray:
    """Von Kries/diagonal gains to map src white → ref white."""
    if src_bgr is None or ref_bgr is None: return src_bgr
    swb = np.array(_estimate_whitepoint(src_bgr), dtype=np.float32)
    rwb = np.array(_estimate_whitepoint(ref_bgr), dtype=np.float32)
    gains = (rwb / np.maximum(swb, 1e-3)).reshape(1,1,3)
    out = src_bgr.astype(np.float32) * gains
    return np.clip(out, 0, 255).astype(np.uint8)

def _percentile_gamma_match_l(src_bgr: np.ndarray, ref_bgr: np.ndarray) -> np.ndarray:
    """Match L using two percentiles + a gentle gamma to align the midtone."""
    if src_bgr is None or ref_bgr is None: return src_bgr
    src_lab = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2LAB); Ls, As, Bs = cv2.split(src_lab)
    ref_lab = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2LAB); Lr, _, _  = cv2.split(ref_lab)

    s_lo, s_md, s_hi = np.percentile(Ls, [5, 50, 95]).astype(np.float32)
    r_lo, r_md, r_hi = np.percentile(Lr, [5, 50, 95]).astype(np.float32)
    # normalize to 0..1 within robust range
    n = np.clip((Ls.astype(np.float32) - s_lo) / max(1.0, (s_hi - s_lo)), 0, 1)
    s_md_n = np.clip((s_md - s_lo) / max(1.0, (s_hi - s_lo)), 1e-3, 1-1e-3)
    r_md_n = np.clip((r_md - r_lo) / max(1.0, (r_hi - r_lo)), 1e-3, 1-1e-3)
    # gamma so that median maps to median
    g = float(np.log(r_md_n) / np.log(s_md_n))
    g = float(np.clip(g, APPEARANCE_GAMMA_CLAMP[0], APPEARANCE_GAMMA_CLAMP[1]))
    n = np.power(n, g)
    Lm = (n * (r_hi - r_lo) + r_lo).astype(np.uint8)
    out_lab = cv2.merge((Lm, As, Bs))
    return cv2.cvtColor(out_lab, cv2.COLOR_LAB2BGR)

def _reinhard_ab_transfer(src_bgr: np.ndarray, ref_bgr: np.ndarray, strength=0.6) -> np.ndarray:
    """Soft chroma transfer on A/B (means+stds), blended by strength."""
    if src_bgr is None or ref_bgr is None: return src_bgr
    s_lab = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2LAB); Ls, As, Bs = cv2.split(s_lab)
    r_lab = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2LAB); _, Ar, Br = cv2.split(r_lab)

    def _match(ch_s, ch_r):
        s = ch_s.astype(np.float32); r = ch_r.astype(np.float32)
        ms, ss = float(s.mean()), float(s.std() + 1e-6)
        mr, sr = float(r.mean()), float(r.std() + 1e-6)
        out = (s - ms) * (sr / ss) + mr
        return np.clip(out, 0, 255).astype(np.uint8)

    A_t = _match(As, Ar); B_t = _match(Bs, Br)
    # Blend so we don't overdrive hues
    A_o = cv2.addWeighted(As, 1.0 - strength, A_t, strength, 0)
    B_o = cv2.addWeighted(Bs, 1.0 - strength, B_t, strength, 0)
    return cv2.cvtColor(cv2.merge((Ls, A_o, B_o)), cv2.COLOR_LAB2BGR)

def _match_saturation_to_ref(src_bgr: np.ndarray, ref_bgr: np.ndarray, strength=0.65) -> np.ndarray:
    """Scale HSV S to bring medians closer; blended by strength."""
    if src_bgr is None or ref_bgr is None: return src_bgr
    shsv = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    rhsv = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    s_src = np.percentile(shsv[:,:,1], 50)
    s_ref = np.percentile(rhsv[:,:,1], 50)
    if s_src <= 1e-3: return src_bgr
    k = float(np.clip(s_ref / s_src, 0.6, 1.6))
    S_new = np.clip(shsv[:,:,1] * (1.0 + strength*(k-1.0)), 0, 255)
    shsv[:,:,1] = S_new
    return cv2.cvtColor(shsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def _appearance_match_to_reference(src_bgr: np.ndarray, ref_bgr: np.ndarray) -> np.ndarray:
    """Full appearance alignment: WP → tone → chroma(A/B) → saturation."""
    if src_bgr is None or ref_bgr is None: return src_bgr
    x = _whitepoint_align(src_bgr, ref_bgr)
    x = _percentile_gamma_match_l(x, ref_bgr)
    x = _reinhard_ab_transfer(x, ref_bgr, strength=APPEARANCE_AB_STRENGTH)
    x = _match_saturation_to_ref(x, ref_bgr, strength=APPEARANCE_SAT_STRENGTH)
    return x

def capture_scanned_card_from_live():
    global scanned_card, last_scan_ts, snapshot_seq, preview_card
    global PROC_PAUSED, PROC_PAUSE_SEQ

    # Take the current detected crop, if any
    with video_lock:
        snap = current_card_crop.copy() if current_card_crop is not None else None
    if snap is None:
        return False

    # Install the new snapshot
    with scan_lock:
        preview_card = None            # ensure preview doesn't mask the new snap
        scanned_card = snap.copy()
        last_scan_ts = time.time()
        snapshot_seq += 1
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
    STREAM_STATE['paused'] = True
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
      status: 'pass' | 'all'                     (default 'pass')
      filename: optional, else auto timestamp
    Returns JSON with file path and a /exports/... URL.
    """
    data = request.get_json(silent=True) or {}
    fmt = (data.get("fmt") or "archidekt").lower()
    status = (data.get("status") or "all").lower()
    if status not in ("pass", "all"): status = "pass"

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
    if not name:
        return jsonify({"ok": True, "results": []})
    q = f'!"{name}"'
    cn_norm = _normalize_cn_for_search(number)
    if cn_norm: q += f" cn:{cn_norm}"
    if set_hint: q += f" set:{set_hint}"
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

def _parse_roi_csv(s: str):
    try:
        vals = [float(x.strip()) for x in str(s).split(",")]
        return vals if len(vals) == 4 else None
    except Exception:
        return None

# ---- Settings helpers (single source of truth) ----
# (removed duplicate definition of _settings_load)

# (removed duplicate definition of _settings_save)

def _effective_settings():
    # current runtime values (imported from config.py at startup)
    def L(v):  # ensure JSON-friendly for tuples
        return list(v) if isinstance(v, (list, tuple)) else v
    return {
        # Camera
        "CAMERA_DEVICE": CAMERA_DEVICE, "REQ_WIDTH": REQ_WIDTH, "REQ_HEIGHT": REQ_HEIGHT,
        "REQ_FPS": REQ_FPS, "FOURCC_PRIMARY": FOURCC_PRIMARY, "FOURCC_FALLBACK": FOURCC_FALLBACK,
        # Processing / Canvas
        "PROC_MAX_WIDTH": PROC_MAX_WIDTH, "CARD_W": CARD_W, "CARD_H": CARD_H,
        "MIN_CARD_AREA_RATIO": MIN_CARD_AREA_RATIO, "MAX_CARD_AREA_RATIO": MAX_CARD_AREA_RATIO,
        "BORDER_MARGIN_PCT": BORDER_MARGIN_PCT, "CARD_ASPECT": CARD_ASPECT, "ASPECT_TOL": ASPECT_TOL,
        "CONFIRM_FRAMES": CONFIRM_FRAMES, "STALE_FRAMES": STALE_FRAMES, "MAX_ASSOC_DIST": MAX_ASSOC_DIST,
        "MAX_CARDS": MAX_CARDS, "DETECT_EVERY_N_FRAMES": DETECT_EVERY_N_FRAMES,
        "RECTANGULARITY_MIN": RECTANGULARITY_MIN,
        # Autoscan / Steady
        "STEADY_MIN_FRAMES": STEADY_MIN_FRAMES, "AUTO_CAPTURE_WAIT_S": AUTO_CAPTURE_WAIT_S,
        "AUTOSCAN_OCR_TIMEOUT": AUTOSCAN_OCR_TIMEOUT, "AUTOSCAN_SCRY_TIMEOUT": AUTOSCAN_SCRY_TIMEOUT,
        "AUTOSCAN_IMG_TIMEOUT": AUTOSCAN_IMG_TIMEOUT,
        # OCR & Debug
        "OCR_BACKEND": OCR_BACKEND, "OCR_ONLY_ON_SNAPSHOT": OCR_ONLY_ON_SNAPSHOT,
        "DEBUG_OCR": DEBUG_OCR, "OCR_DEBUG_BOXES": OCR_DEBUG_BOXES, "SHOW_ROI_OVERLAY": SHOW_ROI_OVERLAY,
        "MIN_TITLE_LETTERS": MIN_TITLE_LETTERS, "TEXT_PRESENCE_MIN": TEXT_PRESENCE_MIN,
        "TITLE_ALLOW_TESS_FALLBACK": TITLE_ALLOW_TESS_FALLBACK,
        # Foil
        "FOIL_MIN_SCORE": FOIL_MIN_SCORE, "FOIL_ON_TH": FOIL_ON_TH,
        "FOIL_OFF_TH": FOIL_OFF_TH, "FOIL_DETECT": FOIL_DETECT,
        # Match
        "MATCH_ENABLE": MATCH_ENABLE, "MATCH_W_HASH": MATCH_W_HASH, "MATCH_W_HIST": MATCH_W_HIST,
        "MATCH_W_ORB": MATCH_W_ORB, "MATCH_TH": MATCH_TH, "NAME_OK_TH": NAME_OK_TH,
        "ALWAYS_SCAN_OK": ALWAYS_SCAN_OK, "MATCH_USE_ART": MATCH_USE_ART,
        # Prices
        "SCRYFALL_TIMEOUT": SCRYFALL_TIMEOUT, "FX_URL": FX_URL, "FX_TTL_SEC": FX_TTL_SEC,
        # Moonraker / Klipper
        "MOONRAKER_URL": MOONRAKER_URL, "HTTP_POST_URL": HTTP_POST_URL,
        # ROIs
        "ROI_TITLE_TOP": L(ROI_TITLE_TOP), "ROI_TITLE_ALT": L(ROI_TITLE_ALT),
        "ROI_NUMBER_MAIN": L(ROI_NUMBER_MAIN), "ROI_NUMBER_WIDE": L(ROI_NUMBER_WIDE),
        "ROI_NUMBER_TALL": L(ROI_NUMBER_TALL), "ROI_NUMBER_NARROW": L(ROI_NUMBER_NARROW),
        "ROI_ART": L(ROI_ART),
        # Debug & Paths
        "DEBUG_LEVEL": DEBUG_LEVEL, "DEBUG_SAVE_ROI": DEBUG_SAVE_ROI,
        "DEBUG_DIR": DEBUG_DIR, "BAD_DIR": BAD_DIR,
        # Persistence
        "HISTORY_DIR": HISTORY_DIR, "HISTORY_IMG_DIR": HISTORY_IMG_DIR,
        "HISTORY_JSON_EXT": HISTORY_JSON_EXT, "JPEG_QUALITY_SNAP": JPEG_QUALITY_SNAP,
        "JPEG_QUALITY_CMP": JPEG_QUALITY_CMP, "JPEG_QUALITY_THUMB": JPEG_QUALITY_THUMB,
        # Server
        "HOST": HOST, "PORT": PORT,
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

    # normalize booleans; also let ROI accept "x,y,x,y"
    for k, v in list(incoming.items()):
        if isinstance(v, str) and v.lower() in ("true","false","1","0","on","off"):
            incoming[k] = v.lower() in ("true","1","on")
        if k.startswith("ROI_") and isinstance(v, str):
            try:
                vals = [float(x.strip()) for x in v.split(",")]
                if len(vals) == 4:
                    incoming[k] = vals
            except Exception:
                pass

    cur = _settings_load() or {}
    cur.update(incoming)
    _settings_save(cur)

    if any(k in incoming for k in ("MATCH_USE_ART","ROI_ART","CARD_W","CARD_H")):
        _scry_img_feats.clear()

    restart_keys = {
        "CAMERA_DEVICE","REQ_WIDTH","REQ_HEIGHT","REQ_FPS",
        "FOURCC_PRIMARY","FOURCC_FALLBACK","HOST","PORT"
    }
    needs_restart = any(k in restart_keys for k in incoming.keys())
    return jsonify({"ok": True, "saved": incoming, "restart_required": needs_restart})

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
    target = next((e for e in scan_history if e["id"] == sid), None)
    if not target:
        return jsonify({"ok": False, "error": "not found"}), 404
    if name:        target["name"] = name
    if number_raw:  target["number_raw"] = number_raw; target["number"] = _parse_collector_for_display(number_raw)
    if set_hint:    target["set_hint"] = set_hint
    choice = None
    if scry_id:
        choice = _scryfall_request(f"https://api.scryfall.com/cards/{scry_id}") or None
    if choice is None:
        choice = _scryfall_lookup(target.get("name",""), target.get("number_raw",""), set_hint=set_hint or target.get("set_hint",""))
    target["scry"] = choice
    snap = _decode_jpg(target.get("snap_jpg"))
    score, ok, details = 0.0, False, None
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
    if current_loaded_scan_id == sid:
        ts_now = time.time()
        with ocr_lock:
            ocr_state["name"] = target.get("name","")
            ocr_state["number"] = target.get("number","")
            ocr_state["number_raw"] = target.get("number_raw","")
            ocr_state["set_hint"] = target.get("set_hint","")
            ocr_state["match_score"] = target["match_score"]
            ocr_state["match_ok"] = target["match_ok"]
            ocr_state["updated_at"] = ts_now
        with cardinfo_lock:
            cardinfo_state["scry"] = choice
            cardinfo_state["fx"] = {"rate": _get_usd_to_cad()}
            cardinfo_state["last_updated"] = ts_now
        if details:
            _publish_compare_visual(details)
    with history_lock:
        _persist_entry_to_disk(target)
    return jsonify({"ok": True, "id": sid, "match_score": target["match_score"], "match_ok": target["match_ok"]})

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
        if F["status"] == "all":     return True
        if F["status"] == "pass":    return (st == "pass") or bool(e.get("match_ok"))
        if F["status"] == "fail":    return (st == "fail") or bool(e.get("flagged"))
        if F["status"] == "flagged": return bool(e.get("flagged"))
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

@app.route('/api/scans')
def api_scans():
    a = request.args
    sort_by   = (a.get("sort_by") or "ts").lower()
    sort_dir  = (a.get("sort_dir") or "desc").lower()
    limit     = max(1, min(500, int(a.get("limit") or 200)))
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
    out = out[offset:offset+limit]
    items = [{
        "id": e["id"], "ts": e["ts"], "name": e.get("name",""), "number": e.get("number",""),
        "status": e.get("status","fail"), "match_score": e.get("match_score",0.0),
        "match_ok": e.get("match_ok", False), "flagged": e.get("flagged", False),
        "foil": bool(e.get("foil", False)),
        "scry_name": (e.get("scry") or {}).get("name",""),
        "scry_set":  (e.get("scry") or {}).get("set",""),
        "scry_cn":   (e.get("scry") or {}).get("collector_number",""),
        "review_reasons": e.get("review_reasons", []),
    } for e in out]
    return jsonify({"items": items, "total": total, "offset": offset, "limit": limit})

@app.route('/api/scan/<int:sid>/thumb')
def api_scan_thumb(sid):
    for e in scan_history:
        if e["id"] == sid:
            return Response(e.get("thumb") or e.get("snap_jpg") or b"", mimetype='image/jpeg')
    return jsonify({"ok": False, "error": "not found"}), 404

@app.route('/api/scan/<int:sid>/load', methods=['POST'])
def api_scan_load(sid):
    global scanned_card, last_scan_ts, snapshot_seq, current_loaded_scan_id, _last_cmp_img, _last_cmp_stats, preview_card
   
    target = next((e for e in scan_history if e["id"] == sid), None)
    if not target:
        return jsonify({"ok": False, "error": "not found"}), 404
    snap = _decode_jpg(target.get("snap_jpg"))
    if snap is None:
        return jsonify({"ok": False, "error": "no snapshot bytes"}), 400
    ts_now = time.time()
    with scan_lock:
        preview_card = snap.copy()
    _clear_ocr()
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
        ocr_state["updated_at"] = ts_now
    with cardinfo_lock:
        cardinfo_state.clear()
        cardinfo_state["scry"] = target.get("scry")
        cardinfo_state["fx"]   = {"rate": _get_usd_to_cad()}
        cardinfo_state["last_updated"] = ts_now
    jpg = target.get("cmp_jpg")
    with compare_lock:
        _last_cmp_img   = _decode_jpg(jpg) if jpg else None
        _last_cmp_stats = dict(target.get("cmp_stats") or {})
    current_loaded_scan_id = sid
    return jsonify({"ok": True, "loaded_id": sid})

@app.route('/api/scan/<int:sid>/status', methods=['POST'])
def api_scan_status(sid):
    data = request.get_json(silent=True) or {}
    status = (data.get("status") or "all").lower()
    if status not in ("pass", "fail"):
        return jsonify({"ok": False, "error": "status must be 'pass' or 'fail"}), 400
    target = next((e for e in scan_history if e["id"] == sid), None)
    if not target:
        return jsonify({"ok": False, "error": "not found"}), 404
    target["status"] = status
    target["flagged"] = (status == "fail")
    with history_lock:
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
        _clear_ocr()
        with compare_lock:
            _last_cmp_img = None
            _last_cmp_stats = {}
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

# ---------------------------
# Bad list persistence (optional)
# ---------------------------
_badlist_lock = threading.Lock()

def _badlist_path_for(item: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", (item or ""))[:96]
    return os.path.join(BAD_DIR, f"{safe}.txt")

def _badlist_persist_add(item: str):
    try:
        path = _badlist_path_for(item)
        with open(path, "w", encoding="utf-8") as f:
            f.write(item)
    except Exception as e:
        _dbg("HISTORY ERROR", f"add persist failed: {e}")

def _badlist_persist_remove(item: str):
    try:
        path = _badlist_path_for(item)
        if os.path.exists(path):
            os.remove(path)
    except Exception as e:
        _dbg("HISTORY ERROR", f"rm persist failed: {e}")

def _badlist_load_from_disk():
    items = []
    try:
        for p in glob.glob(os.path.join(BAD_DIR, "*.txt")):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    s = f.read().strip()
                    if s:
                        items.append(s)
            except Exception:
                pass
    except Exception as e:
        _dbg("HISTORY ERROR", f"load failed: {e}")
    return items

@app.post("/api/badlist/add")
def api_bad_add():
    data = request.get_json(silent=True) or {}
    item = (data.get("item") or "").strip()
    if not item:
        return jsonify({"ok": False, "error": "missing item"}), 400
    with _badlist_lock:
        if item not in bad_cards:
            bad_cards.append(item)
            _badlist_persist_add(item)
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
            _badlist_persist_remove(item)
        except ValueError:
            pass
    return jsonify({"ok": True, "count": len(bad_cards)})

@app.post("/api/badlist/clear")
def api_bad_clear():
    with _badlist_lock:
        bad_cards.clear()
        try:
            for p in glob.glob(os.path.join(BAD_DIR, "*.txt")):
                os.remove(p)
        except Exception as e:
            _dbg("HISTORY ERROR", f"clear failed: {e}")
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
        if roi_bgr is None or roi_bgr.size == 0:
            return ""
        # Build several candidates from different preprocessings
        candidates = []

        # Binary (general OCR)
        bin_img = _prep_roi_for_ocr(roi_bgr)
        txt, _, _ = _read_text_general(bin_img)
        if txt:
            candidates.append(txt)

        # Color RapidOCR
        if RAPIDOCR_AVAILABLE and "rapidocr" in OCR_BACKEND:
            rbgr = _prep_roi_for_rapid_bgr(roi_bgr)
            try:
                eng = _get_rapid_engine()
                res = (eng(rbgr)[0] or [])
                if res:
                    candidates.append(" ".join(r[1] for r in res))
            except Exception:
                pass

        # Heuristic extraction: 2–5 uppercase letters/digits
        for raw in candidates:
            for tok in re.findall(r"[A-Z0-9]{2,5}", (raw or "").upper()):
                code = tok.lower()
                if code in SCRYFALL_SET_CODES:
                    return code
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

# Ensure any Scryfall choice we resolve helps future DFC lookups
# (Monkey-patch the resolver return path to index faces.)
_orig__scryfall_lookup = _scryfall_lookup
def _scryfall_lookup(name, number_raw, set_hint=""):
    choice = _orig__scryfall_lookup(name, number_raw, set_hint=set_hint)
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
        for itm in _badlist_load_from_disk():
            bad_cards.append(itm)
    except Exception as e:
        _dbg("HISTORY ERROR", f"load failed: {e}")
    try:
        _cur = _settings_load()
    except Exception:
        _cur = {}
    try:
        _roi = (_cur or {}).get("DETECT_ROI")
        if isinstance(_roi, (list, tuple)) and len(_roi) == 4:
            x0, y0, x1, y1 = [float(x) for x in _roi]
            x0, x1 = sorted((max(0.0, min(1.0, x0)), max(0.0, min(1.0, x1))))
            y0, y1 = sorted((max(0.0, min(1.0, y0)), max(0.0, min(1.0, y1))))
            with _detect_roi_lock:
                DETECT_ROI[:] = [x0, y0, x1, y1]
    except Exception as e:
        _dbg("SETTINGS ERROR", f"apply DETECT_ROI failed: {e}")

    # Warm catalogs in the background
    threading.Thread(target=_load_scryfall_catalogs_bg, daemon=True).start()

    # Fire up all workers
    threading.Thread(target=video_thread,     daemon=True).start()
    threading.Thread(target=ocr_worker,       daemon=True).start()
    threading.Thread(target=cardinfo_worker,  daemon=True).start()
    threading.Thread(target=ws_thread,        daemon=True).start()
    threading.Thread(target=autoscan_manager, daemon=True).start()



# --- Legacy static fallbacks (in case of reverse proxies or path quirks) ---
@app.get("/ui.js")
def _legacy_ui_js():
    return send_from_directory(app.static_folder, "js/ui.js")

@app.get("/app.css")
def _legacy_app_css():
    return send_from_directory(app.static_folder, "css/app.css")

# Quick debug endpoint to verify paths on device
@app.get("/_debug/static")
def _debug_static():
    return {
        "static_folder": app.static_folder,
        "template_folder": app.template_folder,
        "ui_js_exists": os.path.exists(os.path.join(app.static_folder, "js", "ui.js")),
        "css_exists": os.path.exists(os.path.join(app.static_folder, "css", "app.css")),
    }
@app.context_processor
def inject_version():
    return {"APP_VERSION": APP_VERSION}

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

if __name__ == "__main__":
    HOST = str(globals().get("APP_HOST", os.environ.get("APP_HOST", "0.0.0.0")))
    PORT = int(globals().get("APP_PORT", os.environ.get("APP_PORT", 5000)))
    try:
        signal.signal(signal.SIGINT,  lambda *_: threading.Thread(target=_graceful_shutdown, daemon=True).start())
        signal.signal(signal.SIGTERM, lambda *_: threading.Thread(target=_graceful_shutdown, daemon=True).start())
    except Exception:
        pass
    run_server(HOST, PORT)