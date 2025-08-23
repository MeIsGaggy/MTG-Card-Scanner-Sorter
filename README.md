# MTG Card Scanner

A fast, Raspberry Pi–friendly web app for scanning Magic: The Gathering cards. It uses a USB camera, performs live detection/OCR on-card regions, looks up details via Scryfall, and gives you an interactive UI to review, compare, and export results.

---
## ⚠️ Disclaimers

* **Processing Speed:** This app is not designed to be the fasted available for scanning TCG cards. It was made as a personal project to make minimal modifications to a 3D printer running Klipper to have it sort through my thousands of Magic cards. Apps like Manabox are much faster, however they require manually holding or mounting the phone and running cards under. Even with a 3D printed holder, the reliablity is lackluster at times. This app tried to maintain performance on a Raspberry Pi while trying to keep reliablity as high as possible.
* **AI Assistance:** I am a hobbist, not a professional programmer. I combined my 8+ years of hobby/personal programming with ChatGPT to make something that would have either not been possible given my current skillset or would have taken so long I would have lost interest. If you are not a fan of AI assisted development, please look elsewhere.
* **Reliability:** I've done my best to get this to be as stable and reliable as possible so far. Based on my current testing it will correctly read the card over 90% of the time. Given the odd special foil or layout it may have a hard time. That's why I have a comaparison that runs after the OCR to compare the card scanned to the one it thinks it is. If it's below the set tollerance, it will flag it for manual review. 

---

## ✨ Features

* **One‑command start** via CLI: `mtg-scanner`
* **Live preview** with ROI (region-of-interest) overlay and steady‑frame auto‑capture
* **Rapid OCR** (RapidOCR ONNX) with optional **Tesseract fallback**
* **Card name matching** (configurable hashing/hist/ORB weights)
* **Scryfall**-powered lookups and price panel (with optional FX rate for CAD)
* **History & exports**: snapshots, thumbnails, and CSV/JSON exports
* **Update check** (optional) against your GitHub Releases on startup
* Designed to run well on **Raspberry Pi** (uses OpenCV headless)


---

## 📦 Installation

### Requirements

* Python **3.9+** (3.11 tested)
* A USB/UVC camera available as `/dev/video*`
* On Raspberry Pi OS / Debian, recommended system packages:

  ```bash
  sudo apt update
  sudo apt install -y tesseract-ocr libatlas-base-dev libjpeg-dev zlib1g-dev
  ```

### Download repository
```bash
gh repo clone MeIsGaggy/MTG-Card-Scanner-Sorter
cd MTG-Card-Scanner-Sorter/
```

### Create a virtualenv and install

From your project root (the folder containing `pyproject.toml`):

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -e .        # editable install during development
# If OpenCV wheels complain, use headless:
pip uninstall -y opencv-python && pip install opencv-python-headless
```
---

## 🔔 Updates

If there is an update available, it will be logged in the apps console. Run the following command to install the update

```bash
pip install --upgrade --force-reinstall \
  "git+https://github.com/MeIsGaggy/MTG-Card-Scanner-Sorter@[VERSION CODE]#egg=mtg_scanner"
```
Where it states [VERSION CODE] you need to paste the vX.X.X from the latest release
---

## 🚀 Quick start (Exports are optional)

```bash
export SETTINGS_PATH="./settings.json"          # optional (defaults to ./settings.json)
export MTG_SCANNER_REPO="<user>/<repo>"        # optional: enables the GitHub update check
mtg-scanner
```

Open a browser to `http://<raspberrypi>:5000` (or whatever host/port you configured).

> settings.json can be created within the app by opening the App Settings then clicking on Save.
> Repo does not need to be set. This is for following my updates, or can be changed to follow your own if you fork

---

## ⚙️ Configuration

Settings are read in this order **(env → `settings.json` → internal defaults)**. You can point the app to a custom config location via `SETTINGS_PATH`.

Create a `settings.json` next to where you run `mtg-scanner` (or copy `src/mtg_scanner/settings.example.json`):

```json
{
  "HOST": "0.0.0.0",
  "PORT": 5000,
  "CAMERA_DEVICE": "/dev/video0",
  "REQ_WIDTH": 1920,
  "REQ_HEIGHT": 1080,
  "REQ_FPS": 30,
  "OCR_BACKEND": "rapidocr,tesseract",
  "OCR_ONLY_ON_SNAPSHOT": true,
  "SHOW_ROI_OVERLAY": true,
  "SCRYFALL_TIMEOUT": 2.5,
  "FX_URL": "https://api.exchangerate.host/latest?base=USD&symbols=CAD",
  "FX_TTL_SEC": 43200,
  "DEBUG_LEVEL": 1
}
```

### Common keys (quick reference)

| Key                                         |                Default | What it does                                                                |
| ------------------------------------------- | ---------------------: | --------------------------------------------------------------------------- |
| `HOST`                                      |            `"0.0.0.0"` | Bind address for the web server. Use `127.0.0.1` to restrict to local only. |
| `PORT`                                      |                 `5000` | Port for the web UI.                                                        |
| `CAMERA_DEVICE`                             |        `"/dev/video0"` | Which camera device to open.                                                |
| `REQ_WIDTH` / `REQ_HEIGHT` / `REQ_FPS`      |         `1920/1080/30` | Camera request settings (driver may choose the closest mode).               |
| `OCR_BACKEND`                               | `"rapidocr,tesseract"` | Backends to try (RapidOCR first, fall back to Tesseract).                   |
| `OCR_ONLY_ON_SNAPSHOT`                      |                 `true` | If `true`, full OCR runs on snapshots; live OCR is throttled for speed.     |
| `SHOW_ROI_OVERLAY`                          |                 `true` | Draws ROI boxes on the live preview.                                        |
| `MATCH_ENABLE`                              |                    `1` | Enables name/image matching.                                                |
| `MATCH_W_HASH`/`MATCH_W_HIST`/`MATCH_W_ORB` |           see defaults | Relative weights when comparing frames to the reference.                    |
| `SCRYFALL_TIMEOUT`                          |                  `2.5` | Seconds to wait for Scryfall before skipping.                               |
| `FX_URL` / `FX_TTL_SEC`                     |           see defaults | FX endpoint and TTL for USD→CAD price conversion.                           |
| `HISTORY_DIR`                               |          `"./history"` | Where to persist JSON/thumbnail artifacts.                                  |
| `JPEG_QUALITY_SNAP`                         |                   `82` | JPEG quality for stored snapshots.                                          |

> Many more tuning options exist (detection thresholds, foil detection, ROI presets, debug switches). The defaults are sane—tune only if you need finer control.

---

## 🖥️ Using the App

The UI is split into panels:

* **Live** — camera feed with ROI overlay. Use the **Snapshot** action to lock a frame for OCR and matching. Live OCR is throttled unless you disable `OCR_ONLY_ON_SNAPSHOT`.
* **Detected Card** — the perspective‑corrected card image and parsed text.
* **Comparison** — quick visual comparison stats and matching scores.
* **Connection & Processing** — camera/processing status and log console. Use **Clear Logs** if needed.
* **Card Info & Prices** — Scryfall details, set symbol, and prices (optionally FX‑converted).

Additional actions:

* **History / Scans** — browse past captures, load a scan, or delete it.
* **Export** — export current/selected results to CSV/JSON (see the Export modal).
* **Settings** — open the settings modal to live‑tweak common options (also persisted via `settings.json`).

---

## 🔄 Update check on startup (optional)

If you set `MTG_SCANNER_REPO` to your GitHub repo slug (e.g. `"MeIsGaggy/MTG-Card-Scanner-Sorter"`), the app will query **latest GitHub Release** on boot and print a one‑line status (and can be wired to the in‑app console). Tag new releases as `vX.Y.Z` and publish them on GitHub to enable this flow.

---

## 🧰 CLI and environment

* Start the app: `mtg-scanner`
* Override config file location: `export SETTINGS_PATH=/path/to/settings.json`
* Enable update check: `export MTG_SCANNER_REPO=<user>/<repo>`

---

## 🧪 API (advanced)

These are some of the commonly used endpoints that power the UI. Most users don’t need them directly, but they’re handy for automation.

* `GET /` — main UI
* `GET /api/state` — app state & stats
* `POST /api/snapshot` — take a snapshot
* `GET /api/scans` — list saved scans
* `POST /api/scan/<id>/load` — load a scan by id
* `GET /api/scan/<id>/thumb` — thumbnail
* `POST /api/scan/<id>/delete` — delete a scan
* `GET|POST /api/settings` — get/update settings
* `POST /api/logs/clear` — clear in‑app logs
* `GET /api/compare/stats` — comparison metrics
* `POST /api/badlist/add|remove|clear` — manage excluded results

> Endpoints may evolve; use your browser devtools (Network tab) to discover current calls from `ui.js`.

---

## 🛠️ Run as a service (Linux / Raspberry Pi)

Create `/etc/systemd/system/mtg-scanner.service`:

```ini
[Unit]
Description=MTG Scanner
After=network-online.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/MTG_SCANNER_PROD
Environment=SETTINGS_PATH=/home/pi/MTG_SCANNER_PROD/settings.json
Environment=MTG_SCANNER_REPO=MeIsGaggy/MTG-Card-Scanner-Sorter
ExecStart=/home/pi/MTG_SCANNER_PROD/.venv/bin/mtg-scanner
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Then enable:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now mtg-scanner
```

---

## 🧩 Customizing & extending

* **UI tweaks**: edit `templates/index.html`, `static/css/app.css`, and `static/js/ui.js`. No restart is needed for minor CSS/JS in editable installs; otherwise restart the service.
* **Detection/OCR tuning**: adjust thresholds in `settings.json` (e.g., `RECTANGULARITY_MIN`, `TEXT_PRESENCE_MIN`, `MATCH_*` weights). Keep a backup of a good configuration.
* **Alternate currencies**: point `FX_URL` to another base/target pair or disable FX conversion in the UI.
* **Printer / Moonraker hooks**: `MOONRAKER_URL` and `HTTP_POST_URL` exist for optional integrations—leave them as defaults if unused.



## 📄 License

See **LICENSE** in this repository for the license. 

No commercial or for-profit use allowed. Personal projects only.
