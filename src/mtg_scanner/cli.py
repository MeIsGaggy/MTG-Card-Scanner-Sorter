import os
from .update import get_current_version, check_for_update_async

def main():
    from .config import HOST, PORT
    repo = os.environ.get("MTG_SCANNER_REPO")  # e.g. "yourname/mtg-scanner"
    if repo:
        check_for_update_async(repo, get_current_version())

    # Prefer the project's own run_server if present
    try:
        from .app import run_server as _run
        _run(HOST, PORT)
        return
    except Exception:
        pass

    # Fallback to Flask dev server
    try:
        from .app import app
    except Exception as e:
        raise SystemExit(f"Failed to import app: {e}")
    app.run(host=HOST, port=PORT, threaded=True)
