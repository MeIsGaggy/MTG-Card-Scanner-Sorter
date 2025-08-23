import threading, requests, os, re
from importlib.metadata import version as _pkg_version, PackageNotFoundError

def _norm(v: str):
    v = re.sub(r"^[vV]", "", str(v or ""))
    parts = [int(p) for p in re.findall(r"\d+", v)]
    return parts or [0]

def get_current_version() -> str:
    try:
        return _pkg_version("mtg-scanner")
    except PackageNotFoundError:
        try:
            # fallback to package attribute (editable installs)
            from . import __version__
            return __version__
        except Exception:
            return "0.0.0"

def check_for_update(repo: str, current_version: str):
    """
    Query GitHub Releases for `repo` (e.g. 'yourname/mtg-scanner').
    Returns dict with current, latest, is_update, tag, html_url.
    """
    if not repo:
        return {"current": current_version, "latest": current_version, "is_update": False, "tag": "", "html_url": ""}
    url = f"https://api.github.com/repos/{repo}/releases/latest"
    r = requests.get(url, timeout=4)
    r.raise_for_status()
    data = r.json() if r.headers.get("content-type","").startswith("application/json") else {}
    tag = data.get("tag_name") or data.get("name") or ""
    latest = re.sub(r"^[vV]", "", str(tag))
    cur = re.sub(r"^[vV]", "", str(current_version or ""))
    return {
        "current": cur,
        "latest": latest,
        "is_update": _norm(latest) > _norm(cur),
        "tag": tag,
        "html_url": data.get("html_url") or data.get("url") or "",
    }

def check_for_update_async(repo: str, current_version: str, on_result=None):
    def worker():
        try:
            res = check_for_update(repo, current_version)
            if on_result:
                on_result(res)
            else:
                print(f"[UPDATE] current={res['current']} latest={res['latest']} available={res['is_update']} url={res.get('html_url') or ''}")
        except Exception as e:
            print(f"[UPDATE] failed: {e}")
    t = threading.Thread(target=worker, daemon=True)
    t.start()
