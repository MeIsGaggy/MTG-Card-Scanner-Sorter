# Settings Reference (MTG Scanner)

How values flow: environment variables override `settings.json`, which overrides built‑in defaults in `config.py`. The UI reads from `/api/settings` and writes back to `settings.json`; most changes take effect immediately unless noted. Camera device/size/FPS and server bind (`HOST`/`PORT`) typically need a restart.

## Camera
- `CAMERA_DEVICE` — Video device path (e.g., `/dev/video0`).
- `REQ_WIDTH` / `REQ_HEIGHT` — Requested capture resolution; restart required.
- `REQ_FPS` — Requested capture frame rate; restart required.
- `FOURCC_PRIMARY` / `FOURCC_FALLBACK` — Preferred/backup pixel format codes.

## Processing & Canvas
- `PROC_MAX_WIDTH` / `PROC_DOWNSCALE_MAX_W` — Max width for processing and pre-detection downscale.
- `CARD_W` / `CARD_H` — Canvas size used for warping the card crop.
- `MIN_CARD_AREA_RATIO` / `MAX_CARD_AREA_RATIO` — Allowed card area as fraction of frame.
- `BORDER_MARGIN_PCT` — Padding around detected card to keep margins.
- `CARD_ASPECT` / `ASPECT_TOL` — Target aspect ratio and tolerance.
- `RECTANGULARITY_MIN` — Minimum rectangle score for a valid quad.
- `CARD_DETECT_MIN_SCORE` — Minimum score to treat a contour as a card.
- `CARD_RING_OUTSIDE_MIN_FRAC` — Required fraction of border that must live outside the card.
- `DETECT_QUAD_PAD_PCT` — Extra padding applied to detection quads before warp.
- `CARD_BG_BLUR` — Blur kernel size for background smoothing.
- `CARD_CLAHE_CLIP` / `CARD_CLAHE_TILE` — CLAHE parameters for card normalization.
- `CARD_BLACKHAT_KERNEL` — Kernel size for blackhat preprocessing on card masks.
- `CARD_CHANNEL_THRESH_C` — Constant added during channel thresholding.
- `CARD_MASK_MIN_COMPONENT` — Minimum connected component size for card masks.
- `CARD_MASK_BORDER_TRIM_PCT` — Border trim on masks to avoid edge artifacts.
- `CARD_EDGE_MIN_STRENGTH` / `CARD_EDGE_MIN_CONTRAST` / `CARD_EDGE_RING_FRAC` — Edge strength/contrast/ring guards for borders.
- `CARD_EDGE_SNAP_FRAC` — Snap factor when aligning detected edges.
- `CARD_EDGE_CONTRAST_WEIGHT` / `CARD_EDGE_CONTRAST_STEP` — Weight and step size when searching for edge contrast.
- `CARD_CROP_LIVE` — If false, keep the full frame in the live feed instead of the cropped card.
- `QUAD_STICKY_PX` — Pixel hysteresis applied to quad movement.
- `QUAD_SMOOTH_MIN_ALPHA` / `QUAD_SMOOTH_MAX_ALPHA` — Bounds for quad smoothing strength.
- `MANUAL_CROP_ENABLED` / `MANUAL_CROP_QUAD` — Enable and set manual quad points (UI drag handles).

## Tracking, Steady, Autoscan
- `CONFIRM_FRAMES` / `STALE_FRAMES` — Frames to confirm a card and to drop stale tracks.
- `MAX_ASSOC_DIST` — Max pixel distance to associate detections to tracks.
- `MAX_CARDS` — Limit of concurrent card tracks.
- `STEADY_MIN_FRAMES` / `STEADY_RELAX_FRAMES` — Frames required for steady state and relaxation.
- `STEADY_PROMOTE_S` — Time in steady before autoscan promotion.
- `AUTO_CAPTURE_WAIT_S` — Delay after READY_TO_SCAN before the snapshot (lets the stream settle).
- `AUTOSCAN_OCR_TIMEOUT` — Timeout for OCR after the snapshot.
- `DETECT_EVERY_N_FRAMES` / `DETECT_MAX_FPS` — Detection cadence caps.
- `FOIL_EVERY_N_FRAMES` — Interval for foil detection passes.
- `STEADY_SPEED_PX` — Speed threshold (px/frame) for steady detection.
- `LOCK_IOU_THRESH` — IoU threshold to lock a track to a detection.
- `ACQUIRE_FRAMES` / `DROP_MISS_FRAMES` / `PREDICT_HOLD` — Frames to acquire, drop, and hold predictions for a track.
- `TRACK_ALPHA` / `TRACK_BETA` / `TRACK_DEADBAND_PX` — Tracker smoothing and deadband weights.
- `TRACK_BOX_BLEND` — Blend factor when merging track boxes with new detections.
- `WARP_EXPAND_PCT` / `WARP_CROP_PCT` — Expand/crop percentages around warped cards.
- `MANUAL_SCAN_BYPASS_STEADY` — Allow manual scan without steady check.
- `FAST_OCR_MODE` — Favor speed for autoscan OCR.

## Capture & Orientation
- `SNAPSHOT_CARD_SCALE` — Upsample factor applied to saved/ocr snapshots.
- `AUTO_ORIENT` — Enable automatic card orientation.

## OCR & Debug
- `OCR_BACKEND` — Comma list of OCR backends to try.
- `OCR_ONLY_ON_SNAPSHOT` — If true, OCR only on captured snapshot (not live).
- `DEBUG_OCR` — Emit OCR debug logs.
- `OCR_DEBUG_BOXES` — Draw OCR boxes in the UI.
- `SHOW_ROI_OVERLAY` — Draw ROI overlays.
- `MIN_TITLE_LETTERS` — Minimum title length to accept.
- `TEXT_PRESENCE_MIN` — Text presence threshold for detecting text regions.
- `TITLE_ALLOW_TESS_FALLBACK` — Allow Tesseract fallback when primary fails.
- `USE_TESS_FOR_TITLES` — Force Tesseract for titles.
- `OCR_ENABLE_BLACKHAT` — Enable blackhat preprocessing on OCR crops.
- `LIVE_OCR_ONLY_WHEN_STEADY` / `LIVE_OCR_MIN_INTERVAL` — Gate live OCR to steady state and rate limit it.
- `OCR_SKIP_NUMBER` / `OCR_FAST_NUMBER_CONF_EXIT` — Skip collector number OCR or early-exit by confidence.
- `CARD_STREAM_LIVE` / `LIVE_CROP_WHILE_PAUSED` — Control live card crop display and whether the crop updates while paused.

## OCR Fast Path & Reprocess
- `OCR_FAST_PATH` / `OCR_LAZY_SET_HINT` — Enable fast OCR path and reuse set hints.
- `OCR_TITLE_SINGLE_PASS` — Single-pass title OCR (skip second attempt).
- `OCR_USE_EASYOCR` — Use EasyOCR shortcut flow.
- `OCR_TITLE_MAX_W` — Max width for title crop in fast path.
- `OCR_NUMBER_FASTPATH` — Enable fast-path number OCR.
- `REPROCESS_FORCE_FAST_OCR` — Force fast OCR during reprocess.
- `REPROCESS_MAX_WIDTH` — Downscale snapshot width before re-OCR (0 = off).
- `REPROCESS_SKIP_AI_ROIS` — Skip AI ROI refresh during reprocess.
- `REPROCESS_SKIP_FOIL_DETECT` — Skip foil detect during reprocess.

## Foil Detection
- `FOIL_MIN_SCORE` — Minimum foil score to consider foil.
- `FOIL_ON_TH` / `FOIL_OFF_TH` — Hysteresis thresholds to toggle foil state.
- `FOIL_DETECT` — Enable foil detection.

## Match / Comparison
- `MATCH_ENABLE` — Enable visual matching.
- `MATCH_W_HASH` / `MATCH_W_HIST` / `MATCH_W_ORB` — Weights for hash, histogram, and ORB scores.
- `MATCH_TH` — Overall match pass threshold.
- `NAME_OK_TH` — Name similarity threshold.
- `ALWAYS_SCAN_OK` — Force scan OK even if match fails.
- `MATCH_USE_ART` — Include art region in comparison.
- `MATCH_REQUIRE_ORB` — Require ORB match to pass.
- `MATCH_ORB_FAIL_THRESHOLD` — Threshold to flag ORB failure.
- `MATCH_ORB_FEATURES` — Number of ORB features to extract.
- `MATCH_FAST_ACCEPT_DELTA` / `MATCH_FAST_REJECT_DELTA` — Early accept/reject margins.
- `HASH_STABILITY_BITS` — Bits that must change before re-OCR triggers.

## Stream / UI Feeds
- `STREAM_ENABLED` — Enable MJPEG streaming.
- `STREAM_FPS` — Stream frame rate; defaults to camera FPS.
- `STREAM_JPEG_QUALITY` — JPEG quality for stream frames.
- `STREAM_WARP_DOWNSCALE` — Downscale factor before warping for stream only.
- `STREAM_CARD_SCALE` — Scale for card crop in UI stream.
- `SNAPSHOT_CARD_SCALE` — Upsample factor for saved/ocr snapshots.
- `CARD_STREAM_LIVE` / `LIVE_CROP_WHILE_PAUSED` — Send live card crops and keep updating while paused.
- `CARD_CROP_LIVE` — Show cropped card in the live feed instead of the full frame.

## Moonraker / Klipper
- `MOONRAKER_URL` — WebSocket URL to Moonraker.
- `HTTP_POST_URL` — HTTP endpoint for G-code script posting.
- `REQUIRE_PRINTER_ACK` — Wait for printer ACK before proceeding.

## Stacks / Sorter
- `CARD_THICKNESS_MM` — Card thickness estimate pushed to Klipper.
- `REMEASURE_EVERY` — Cards between auto re-measure; 0 disables mid-run probe.

## AI / YOLOv5
- `AI_ENABLED` — Master AI toggle.
- `AI_USE_FOR_CARDS` / `AI_USE_FOR_ROIS` — Use AI to find card box / ROIs.
- `AI_ONLY_MODE` — Use AI ROIs only (ignore static ROIs).
- `AI_MODEL_PATH` — Path to YOLO model.
- `YOLOV5_DIR` — Local YOLOv5 repo path.
- `AI_IMG_SIZE` — Inference image size.
- `AI_CONF_THRES` / `AI_IOU_THRES` — Confidence and IoU thresholds.
- `AI_MAX_DETS` — Maximum detections per inference.
- `AI_CLASS_NAMES` — Class labels list (order-sensitive).
- `AI_BOX_THICKNESS` / `AI_LEGEND_FONT_SCALE` / `AI_LEGEND_FONT_THICKNESS` / `AI_LEGEND_LINE_THICKNESS` / `AI_LEGEND_PAD` — Overlay settings for AI debugging legends.
- `AI_CARD_MIN_CONF`, `AI_SETNAME_MIN_CONF`, `AI_SETNAME_MIN_Y`, `AI_SETNAME_MAX_Y`, `AI_SETNAME_MIN_WIDTH`, `AI_SETNAME_MIN_HEIGHT`, `AI_CARD_MIN_AREA`, `AI_CARD_MAX_TOP` — Heuristics to filter detections by confidence, position, and size.
- `AI_NAME_PAD_X` / `AI_NAME_PAD_Y` — Padding applied to name ROI.
- `AI_TOPLINE_FRAC_MIN` / `AI_TOPLINE_MULT` — Heuristics for merging text lines.
- `AI_CARD_PAD_LEFT` / `AI_CARD_PAD_RIGHT` / `AI_CARD_PAD_Y` — Extra padding around detected cards before ROI extraction.
- `SET_ICON_FALLBACK_ENABLE` / `SET_NAME_FALLBACK_ENABLE` — Fall back to set icon/name heuristics when OCR fails.

## Export & Caches
- `EXPORT_DIR` — Export output directory.
- `SC_IMG_CACHE_DIR` — Scryfall image cache directory.
- `SCRY_IMG_LRU_MAX` / `SCRY_FEATS_LRU_MAX` — In-memory LRU sizes for images/features.
- `SCRY_SETS_CACHE_PATH` / `SCRY_SETS_CACHE_TTL_S` — Cache path and TTL for Scryfall set data.

## Scryfall & Prices
- `SCRYFALL_TIMEOUT` — HTTP timeout for Scryfall calls.
- `FX_URL` — Exchange rate API URL.
- `FX_TTL_SEC` — Cache TTL for FX data.

## Appearance / Photometric
- `APPEARANCE_ALIGN_ENABLE` — Enable appearance alignment.
- `APPEARANCE_AB_STRENGTH` / `APPEARANCE_SAT_STRENGTH` — Chroma and saturation match strengths.
- `APPEARANCE_GAMMA_CLAMP` — Allowed gamma clamp range.
- `TONE_ALIGN_ENABLE` — Enable tone alignment.

## Persistence & Server
- `HISTORY_DIR` / `HISTORY_IMG_DIR` — History storage paths.
- `HISTORY_JSON_EXT` — Extension for history JSON files.
- `JPEG_QUALITY_SNAP` / `JPEG_QUALITY_CMP` / `JPEG_QUALITY_THUMB` — JPEG qualities.
- `HISTORY_API_DEFAULT_LIMIT` / `HISTORY_API_MAX_LIMIT` — API pagination limits.
- `HOST` / `PORT` — Server bind address/port; restart required.

## Debug / Paths / Performance
- `DEBUG_LEVEL` — Verbosity level.
- `DEBUG_SAVE_ROI` — Save ROI crops.
- `DEBUG_DIR` — Directory for debug artifacts.
- `BAD_DIR` — Directory for bad-list images.
- `USE_OPENCL` — Allow OpenCL acceleration when available.
