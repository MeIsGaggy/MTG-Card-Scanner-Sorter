(function () {
  // ----- DOM -----
  const sc = document.getElementById("scanner");
  const p1 = document.getElementById("printer1");
  const ocrEl = document.getElementById("ocr");
  const ci = document.getElementById("cardinfo");
  const badcount = document.getElementById("badcount");
  const badbox = document.getElementById("badbox");
  const revList = document.getElementById("revList");
  const revCount = document.getElementById("revCount");
  const btnPass = document.getElementById("btnMarkPass");
  const btnFail = document.getElementById("btnMarkFail");
  const btnDelete = document.getElementById("btnDelete");

  // live / streams
  const btnStream = document.getElementById("btnStream");
  const liveWrap  = document.getElementById("liveWrap");
  const liveImg   = document.getElementById("live");
  const cardImg   = document.getElementById("card");
  const cmpImg    = document.getElementById("cmpimg");
  const manualSvg = document.getElementById("manualCropSvg");
  const manualPoly = document.getElementById("manualCropPoly");
  const manualHandles = Array.from(document.querySelectorAll(".manual-handle"));
  const manualSaved = document.getElementById("manualSaved");
  const btnManualReset = document.getElementById("btnManualReset");

  // modal
  const btnEdit   = document.getElementById("btnEdit");
  const btnReprocess = document.getElementById("btnReprocess");
  const btnReprocessBatch = document.getElementById("btnReprocessBatch");
  const modal     = document.getElementById("editModal");
  const btnClose  = document.getElementById("editClose");
  const eiName    = document.getElementById("eiName");
  const eiSet     = document.getElementById("eiSet");
  const eiNum     = document.getElementById("eiNum");
  const eiSearch  = document.getElementById("eiSearch");
  const eiApply   = document.getElementById("eiApply");
  const eiResults = document.getElementById("eiResults");
  const reModal      = document.getElementById("reprocessModal");
  const reprocessClose = document.getElementById("reprocessClose");
  const reprocessApply = document.getElementById("reprocessApply");
  const reprocessReject= document.getElementById("reprocessReject");
  const reprocessStatus= document.getElementById("reprocessStatus");
  const reprocessContent = document.getElementById("reprocessContent");
  const reprocessProgressBar = document.getElementById("reprocessProgressBar");
  const reOldName    = document.getElementById("reOldName");
  const reOldSet     = document.getElementById("reOldSet");
  const reOldNumber  = document.getElementById("reOldNumber");
  const reOldScore   = document.getElementById("reOldScore");
  const reOldFlag    = document.getElementById("reOldFlag");
  const reOldCmp     = document.getElementById("reOldCmp");
  const reOldReasons = document.getElementById("reOldReasons");
  const reNewName    = document.getElementById("reNewName");
  const reNewSet     = document.getElementById("reNewSet");
  const reNewNumber  = document.getElementById("reNewNumber");
  const reNewScore   = document.getElementById("reNewScore");
  const reNewFlag    = document.getElementById("reNewFlag");
  const reNewCmp     = document.getElementById("reNewCmp");
  const reNewReasons = document.getElementById("reNewReasons");
  const reNewScry    = document.getElementById("reNewScry");
  const batchModal   = document.getElementById("reprocessBatchModal");
  const batchClose   = document.getElementById("reprocessBatchClose");
  const batchDone    = document.getElementById("reprocessBatchDone");
  const batchStatus  = document.getElementById("reprocessBatchStatus");
  const batchList    = document.getElementById("reprocessBatchList");
  const batchOverallBar = document.getElementById("reprocessBatchOverallBar");
  const btnQuarantine = document.getElementById("btnQuarantine");
  const quarantineModal = document.getElementById("quarantine-modal");
  const quarantineClose = document.getElementById("quarantine-close");
  const quarantineList = document.getElementById("quarantine-list");
  const qId = document.getElementById("q-id");
  const qName = document.getElementById("q-name");
  const qSet = document.getElementById("q-set");
  const qNumber = document.getElementById("q-number");
  const qStatus = document.getElementById("q-status");
  const qMatch = document.getElementById("q-match");
  const qFlagged = document.getElementById("q-flagged");
  const qReasons = document.getElementById("q-reasons");
  const qStatusText = document.getElementById("q-status-text");
  const qAutoFix = document.getElementById("q-autofix");
  const qSave = document.getElementById("q-save");
  const qDelete = document.getElementById("q-delete");
  const btnBackup = document.getElementById("btn-backup");
  const backupModal = document.getElementById("backup-modal");
  const backupClose = document.getElementById("backup-close");
  const backupExport = document.getElementById("backup-export");
  const backupExportStatus = document.getElementById("backup-export-status");
  const backupCompact = document.getElementById("backup-compact");
  const backupImport = document.getElementById("backup-import");
  const backupImportStatus = document.getElementById("backup-import-status");
  const backupFile = document.getElementById("backup-file");
  const backupMode = document.getElementById("backup-mode");
  const backupReassign = document.getElementById("backup-reassign");
  const consoleBox   = document.getElementById("consoleBox");
  const btnLogPause  = document.getElementById("btnLogPause");
  const btnLogClear  = document.getElementById("btnLogClear");

  let currentLoadedId = null;
  let lastScansSig = "";
  let lastState = null;
  let chosenScryId = null;
  let streamOn = true;
  let logAfter = 0;
  let logPaused = false;
  let historyQueryQS = "";  // list view filters
let historyModalQueryQS = "";  // modal-only filters
let filterContext = 'list';
  let modalOpen = false;
  let reprocessToken = null;
  let reprocessTargetId = null;
  let reprocessBatchItems = [];
  let reprocessBatchJobId = null;
  let reprocessProgressStop = null;
  let batchOverallProgressStop = null;
  const SCAN_META_LIMIT = 2000;
  const scanMetaCache = new Map();
  let quarantineItems = [];
  let activeQuarantineId = null;
  let activeEditId = null;
  const manualDefaultQuad = [[0.18, 0.10], [0.82, 0.10], [0.82, 0.92], [0.18, 0.92]];
  let manualQuad = manualDefaultQuad.map(p => p.slice());
  let manualEnabled = true;
  let manualDragIdx = null;
  let manualSaveTimer = null;
  let lastCardinfoSeq = 0;
  let lastCardinfoEtag = "";
  let cardinfoPending = false;
  let lastCmpSeq = 0;
  // ------- Card magnifier (uses /card stream, not /live) -------
  let lens = null;
  let lensZoom = 2.0;
  const LENS_MIN = 1.25;
  const LENS_MAX = 6.0;
  const LENS_SIZE = 200;
  function getDisplayedBox(imgEl) {
    if (!imgEl) return null;
    const rect = imgEl.getBoundingClientRect();
    const iw = Number(imgEl.naturalWidth || rect.width || 0);
    const ih = Number(imgEl.naturalHeight || rect.height || 0);
    if (!rect.width || !rect.height || !iw || !ih) return rect;
    const imgRatio = iw / ih;
    const boxRatio = rect.width / rect.height;
    let w, h, offsetX, offsetY;
    if (boxRatio > imgRatio) {
      h = rect.height;
      w = h * imgRatio;
      offsetX = (rect.width - w) / 2;
      offsetY = 0;
    } else {
      w = rect.width;
      h = w / imgRatio;
      offsetX = 0;
      offsetY = (rect.height - h) / 2;
    }
    return {
      left: rect.left + offsetX,
      top: rect.top + offsetY,
      width: w,
      height: h,
    };
  }
  function ensureLens() {
    if (lens) return lens;
    lens = document.createElement("div");
    lens.id = "cardMagnifier";
    Object.assign(lens.style, {
      position: "fixed",
      width: `${LENS_SIZE}px`,
      height: `${LENS_SIZE}px`,
      border: "2px solid #4a9cff",
      borderRadius: "50%",
      boxShadow: "0 6px 20px rgba(0,0,0,0.35)",
      overflow: "hidden",
      pointerEvents: "none",
      zIndex: 9999,
      backgroundRepeat: "no-repeat",
      backgroundSize: "cover",
      display: "none",
    });
    document.body.appendChild(lens);
    return lens;
  }
  function updateLensBackground(url) {
    ensureLens();
    lens.style.backgroundImage = `url('${url}')`;
  }
  function positionLens(e, imgRect) {
    if (!lens || !imgRect) return;
    const withinX = e.clientX >= imgRect.left && e.clientX <= imgRect.left + imgRect.width;
    const withinY = e.clientY >= imgRect.top && e.clientY <= imgRect.top + imgRect.height;
    if (!(withinX && withinY)) {
      lens.style.display = "none";
      return;
    }
    lens.style.display = "block";
    const x = e.clientX;
    const y = e.clientY;
    lens.style.left = `${x - LENS_SIZE/2}px`;
    lens.style.top  = `${y - LENS_SIZE/2}px`;
    const relX = (e.clientX - imgRect.left) / imgRect.width;
    const relY = (e.clientY - imgRect.top) / imgRect.height;
    const bgX = -(relX * imgRect.width * lensZoom - LENS_SIZE/2);
    const bgY = -(relY * imgRect.height * lensZoom - LENS_SIZE/2);
    lens.style.backgroundSize = `${imgRect.width * lensZoom}px ${imgRect.height * lensZoom}px`;
    lens.style.backgroundPosition = `${bgX}px ${bgY}px`;
  }
  function attachMagnifier(imgEl, defaultSrc) {
    if (!imgEl) return;
    let rect = null;
    imgEl.addEventListener("mouseenter", () => {
      rect = getDisplayedBox(imgEl);
      const src = imgEl.getAttribute("src") || defaultSrc || "/card";
      updateLensBackground(src);
      ensureLens().style.display = "block";
    });
    imgEl.addEventListener("mousemove", (e) => {
      if (!lens || !rect) return;
      rect = getDisplayedBox(imgEl) || rect;
      positionLens(e, rect);
    });
    imgEl.addEventListener("mouseleave", () => {
      rect = null;
      if (lens) lens.style.display = "none";
    });
    imgEl.addEventListener("wheel", (e) => {
      if (!lens || lens.style.display === "none") return;
      e.preventDefault();
      const delta = e.deltaY > 0 ? -0.2 : 0.2;
      lensZoom = Math.min(LENS_MAX, Math.max(LENS_MIN, lensZoom + delta));
      if (!rect) rect = imgEl.getBoundingClientRect();
      positionLens(e, rect);
    }, { passive: false });
  }

  function rememberScanMeta(items) {
    if (!Array.isArray(items)) return;
    if (scanMetaCache.size > SCAN_META_LIMIT) {
      scanMetaCache.clear();
    }
    items.forEach(it => {
      if (!it || typeof it.id === "undefined") return;
      const prev = scanMetaCache.get(it.id) || {};
      scanMetaCache.set(it.id, Object.assign({}, prev, it));
    });
  }
  function makeThumbPlaceholder() {
    const div = document.createElement("div");
    div.className = "thumb thumb-missing";
    div.textContent = "No Image";
    div.setAttribute("aria-label", "No snapshot available");
    return div;
  }
  function attachThumbFallbacks(scope) {
    if (!scope) return;
    scope.querySelectorAll("img.thumb").forEach(img => {
      img.addEventListener("error", () => {
        const ph = makeThumbPlaceholder();
        img.replaceWith(ph);
      }, { once: true });
    });
  }
  // Bind magnifier once the card image exists
  attachMagnifier(cardImg, "/card");
  attachMagnifier(cmpImg, "/compare.jpg");

  // Card Info collapse
  const ciPanel = document.getElementById("cardinfo-panel");
  const ciToggle = document.getElementById("cardinfo-toggle");
  const ciRestore = document.getElementById("cardinfo-restore");
  ciToggle?.addEventListener("click", () => {
    const collapsed = document.body.classList.toggle("ci-collapsed");
    ciToggle.textContent = collapsed ? "Expand" : "Collapse";
    ciToggle.setAttribute("aria-expanded", collapsed ? "false" : "true");
    if (ciRestore) {
      ciRestore.setAttribute("aria-expanded", collapsed ? "false" : "true");
    }
  });
  ciRestore?.addEventListener("click", () => {
    const nowCollapsed = document.body.classList.contains("ci-collapsed");
    if (nowCollapsed) {
      document.body.classList.remove("ci-collapsed");
      ciToggle && (ciToggle.textContent = "Collapse", ciToggle.setAttribute("aria-expanded","true"));
      ciRestore.setAttribute("aria-expanded", "true");
    }
  });

  // ----- Manual crop overlay -----
  function normalizeManualQuad(raw) {
    if (!Array.isArray(raw) || raw.length !== 4) return manualDefaultQuad.map(p => p.slice());
    return raw.map(pt => {
      const x = Math.min(1, Math.max(0, Number(pt?.[0] ?? 0)));
      const y = Math.min(1, Math.max(0, Number(pt?.[1] ?? 0)));
      return [x, y];
    });
  }
  function getLiveImageBox() {
    if (!liveWrap) return null;
    const rect = liveWrap.getBoundingClientRect();
    const iw = Number(liveImg?.naturalWidth || rect.width || 0);
    const ih = Number(liveImg?.naturalHeight || rect.height || 0);
    if (!rect.width || !rect.height || !iw || !ih) return null;
    const imgRatio = iw / ih;
    const boxRatio = rect.width / rect.height;
    let w, h, offsetX, offsetY;
    if (boxRatio > imgRatio) {
      h = rect.height;
      w = h * imgRatio;
      offsetX = (rect.width - w) / 2;
      offsetY = 0;
    } else {
      w = rect.width;
      h = w / imgRatio;
      offsetX = 0;
      offsetY = (rect.height - h) / 2;
    }
    // convert to page coords for pointer math
    const left = rect.left + offsetX;
    const top = rect.top + offsetY;
    return { left, top, width: w, height: h, rect };
  }
  function drawManualQuad() {
    if (!manualSvg || !manualPoly) return;
    const box = getLiveImageBox();
    if (!box) return;
    const pts = manualQuad.map(p => {
      const px = (((box.left - box.rect.left) + p[0] * box.width) / box.rect.width) * 100;
      const py = (((box.top - box.rect.top) + p[1] * box.height) / box.rect.height) * 100;
      return `${px.toFixed(2)},${py.toFixed(2)}`;
    }).join(" ");
    manualPoly.setAttribute("points", pts);
    manualHandles.forEach((h, idx) => {
      const p = manualQuad[idx] || [0, 0];
      const px = (((box.left - box.rect.left) + p[0] * box.width) / box.rect.width) * 100;
      const py = (((box.top - box.rect.top) + p[1] * box.height) / box.rect.height) * 100;
      h.setAttribute("cx", px.toFixed(2));
      h.setAttribute("cy", py.toFixed(2));
    });
  }
  function setManualEnabled(on) {
    manualEnabled = !!on;
    if (manualSvg) manualSvg.style.display = manualEnabled ? "block" : "none";
    const hint = document.querySelector(".manual-hint");
    if (hint) hint.style.display = manualEnabled ? "flex" : "none";
  }
  function saveManualCrop(debounce = true) {
    if (debounce) {
      clearTimeout(manualSaveTimer);
      manualSaveTimer = setTimeout(() => saveManualCrop(false), 200);
      return;
    }
    if (!manualEnabled) {
      setManualEnabled(true);
    }
    if (manualSaved) {
      manualSaved.classList.remove("error");
      manualSaved.textContent = "Saving…";
    }
    fetch("/api/manual_crop", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ quad: manualQuad, enabled: true })
    }).then(r => r.json())
      .then(() => {
        if (manualSaved) manualSaved.textContent = "Saved";
      })
      .catch(() => {
        if (manualSaved) {
          manualSaved.textContent = "Save failed";
          manualSaved.classList.add("error");
        }
      });
  }
  async function loadManualCrop() {
    if (!manualSvg) return;
    try {
      const res = await fetch("/api/settings?ts=" + Date.now());
      const data = await res.json();
      manualQuad = normalizeManualQuad(data?.MANUAL_CROP_QUAD ?? manualDefaultQuad);
      manualEnabled = data?.MANUAL_CROP_ENABLED !== false;
    } catch (e) {
      manualQuad = manualDefaultQuad.map(p => p.slice());
      manualEnabled = true;
    }
    setManualEnabled(manualEnabled);
    drawManualQuad();
    if (manualSaved) manualSaved.textContent = "Saved";
  }
  if (manualSvg && manualHandles.length) {
    manualHandles.forEach((h, idx) => {
      h.addEventListener("pointerdown", (ev) => {
        manualDragIdx = idx;
        try { h.setPointerCapture(ev.pointerId); } catch (e) {}
        ev.preventDefault();
      });
      h.addEventListener("pointermove", (ev) => {
        if (manualDragIdx !== idx) return;
        const box = getLiveImageBox();
        if (!box) return;
        const x = Math.min(1, Math.max(0, (ev.clientX - box.left) / box.width));
        const y = Math.min(1, Math.max(0, (ev.clientY - box.top) / box.height));
        manualQuad[idx] = [x, y];
        drawManualQuad();
        saveManualCrop(true);
      });
      ["pointerup", "pointercancel", "pointerleave"].forEach(evt => {
        h.addEventListener(evt, () => {
          if (manualDragIdx === idx) {
            manualDragIdx = null;
            saveManualCrop(false);
          }
        });
      });
    });
  }
  btnManualReset?.addEventListener("click", () => {
    manualQuad = manualDefaultQuad.map(p => p.slice());
    drawManualQuad();
    saveManualCrop(false);
  });
  window.addEventListener("resize", drawManualQuad);
  liveImg?.addEventListener("load", () => setTimeout(drawManualQuad, 60));

  function esc(val) {
    return String(val ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  // ----- Console log helpers -----
  function formatLogTime(ts) {
    if (!ts && ts !== 0) return "";
    try {
      const d = new Date(ts * 1000);
      return d.toLocaleTimeString("en-US", { hour12: false });
    } catch {
      return "";
    }
  }
  function renderLogLine(item) {
    const row = document.createElement("div");
    row.className = "logline";
    const t = document.createElement("span");
    t.className = "t";
    t.textContent = formatLogTime(item.ts || Date.now()/1000);
    const tag = document.createElement("span");
    const sev = (item.sev || "info").toLowerCase();
    tag.className = `tag tag-${sev}`;
    tag.textContent = item.tag || sev.toUpperCase();
    const msg = document.createElement("span");
    msg.className = "g";
    msg.textContent = item.msg || item.line || "";
    row.append(t, tag, msg);
    return row;
  }
  function appendLogs(items) {
    if (!consoleBox || !Array.isArray(items) || !items.length) return;
    const atBottom = (consoleBox.scrollTop + consoleBox.clientHeight + 20) >= consoleBox.scrollHeight;
    items.forEach(item => consoleBox.appendChild(renderLogLine(item)));
    while (consoleBox.childNodes.length > 600) {
      consoleBox.removeChild(consoleBox.firstChild);
    }
    if (atBottom) consoleBox.scrollTop = consoleBox.scrollHeight;
  }
  async function pollLogs() {
    if (!consoleBox) return;
    if (logPaused) { setTimeout(pollLogs, 1200); return; }
    try {
      const res = await fetch(`/api/logs?after=${logAfter}`);
      if (res.ok) {
        const data = await res.json();
        if (data && Array.isArray(data.items)) {
          appendLogs(data.items);
          if (typeof data.next === "number") logAfter = data.next;
        }
      }
    } catch (err) {
      console.error("log poll failed", err);
    }
    setTimeout(pollLogs, 1200);
  }
  if (consoleBox) pollLogs();
  btnLogPause?.addEventListener("click", () => {
    logPaused = !logPaused;
    btnLogPause.textContent = logPaused ? "Resume" : "Pause";
    if (!logPaused) pollLogs();
  });
  btnLogClear?.addEventListener("click", async () => {
    try { await fetch('/api/logs/clear', { method: 'POST' }); } catch {}
    if (consoleBox) consoleBox.innerHTML = "";
    logAfter = 0;
  });

  function openManualEditById(id) {
    if (!id) return false;
    const meta = scanMetaCache.get(id);
    if (meta) {
      openEditModal(Object.assign({ id }, meta));
      return true;
    }
    return false;
  }
  function setProgressBar(el, fraction) {
    if (!el) return;
    const pct = Math.max(0, Math.min(1, Number(fraction || 0))) * 100;
    el.style.width = pct.toFixed(1) + "%";
  }
  function startProgressRamp(el, max=0.9, step=0.01, speed=120) {
    if (!el) return null;
    let pct = 0.05;
    setProgressBar(el, pct);
    const timer = setInterval(() => {
      pct = Math.min(max, pct + step + Math.random() * step * 0.5);
      setProgressBar(el, pct);
      if (pct >= max) clearInterval(timer);
    }, speed);
    return () => clearInterval(timer);
  }
  function stopIndeterminateBar(stopFn, el, fill=0) {
    if (typeof stopFn === "function") {
      try { stopFn(); } catch {}
    }
    if (el) setProgressBar(el, fill);
    return null;
  }
  async function fetchWithFallback(urls, options) {
    let lastErr;
    for (const url of urls) {
      try {
        const res = await fetch(url, options);
        if (!res.ok) {
          lastErr = new Error(`HTTP ${res.status}`);
          continue;
        }
        return await res.json();
      } catch (err) {
        lastErr = err;
      }
    }
    throw lastErr || new Error("Request failed");
  }

function pill(cls, txt, title) {
  return '<span class="pill ' + cls + '"' +
         (title ? ' title="' + String(title).replace(/"/g,'&quot;') + '"' : '') +
         '>' + txt + '</span>';
}
function updatePills(el, pills) {
  if (!el) return;
  el.innerHTML = (pills || []).map(p => pill(p.cls, p.txt, p.title)).join("");
}

  function money(val, cur) {
    cur = cur || "USD";
    if (val === null || val === undefined || Number.isNaN(val)) return "—";
    try { return new Intl.NumberFormat("en-US", { style: "currency", currency: cur }).format(val); }
    catch { return cur + " " + String(val); }
  }

  function refreshCompare(token) {
    const cmp = document.getElementById("cmpimg");
    if (!cmp) return;
    const seq = (typeof token === "number" && !Number.isNaN(token)) ? token : Date.now();
    lastCmpSeq = seq;
    cmp.src = "/compare.jpg?seq=" + seq;
  }
 
  
  // ---------- Stream pause/freeze ----------
  function freezeImage(img) {
    if (!img) return;
    if (!img.complete || img.naturalWidth === 0 || img.naturalHeight === 0) return;
    const c = document.createElement('canvas');
    c.width = img.naturalWidth; c.height = img.naturalHeight;
    const ctx = c.getContext('2d'); try { ctx.drawImage(img, 0, 0); } catch { return; }
    if (!img.dataset.prevSrc) img.dataset.prevSrc = img.src;
    img.src = c.toDataURL('image/jpeg', 0.92);
  }
  async function setStreamServerState(enabled) {
    try {
      await fetch('/api/stream', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ enabled })});
    } catch {}
  }
  btnStream?.addEventListener('click', async ()=>{
    if (streamOn) {
      freezeImage(liveImg); freezeImage(cardImg);
      await setStreamServerState(false);
      btnStream.textContent = 'Resume'; streamOn = false;
    } else {
      await setStreamServerState(true);
      const bust = Date.now();
      liveImg.src = '/live?ts='+bust;
      cardImg.src = '/card?ts='+bust;
      btnStream.textContent = 'Pause'; streamOn = true;
    }
  });

  // ---------- Scryfall card info ----------
  async function updateCardInfo(s, attempt = 0) {
    if (!ci || !s || !s.name) { if (ci) ci.innerHTML = ""; return; }
    const updatedAt = Number(s.updated_at || 0);
    if (updatedAt && updatedAt <= lastCardinfoSeq && attempt === 0) return;
    if (cardinfoPending) return;

    const headers = {};
    if (lastCardinfoEtag) headers["If-None-Match"] = lastCardinfoEtag;

    cardinfoPending = true;
    try {
      const r = await fetch("/api/carddata", {
        cache: "no-cache",
        headers
      });
      const et = r.headers.get("ETag");
      const etTs = (() => {
        const m = (et || "").match(/card-(\d+)/);
        return m ? Number(m[1]) / 1000 : null;
      })();
      if (r.status === 304) {
        if (et) lastCardinfoEtag = et;
        if (etTs !== null && !Number.isNaN(etTs)) {
          lastCardinfoSeq = Math.max(lastCardinfoSeq, etTs);
        }
        return;
      }
      if (!r.ok) return;
      if (et) lastCardinfoEtag = et;

      const d = await r.json();
      if (!d || typeof d.last_updated !== "number") { if (ci) ci.innerHTML = ""; return; }

      const lu = Number(d.last_updated || updatedAt || 0);
      if (lu <= lastCardinfoSeq && attempt === 0) return;
      lastCardinfoSeq = lu;

      // Retry briefly if cardinfo backend is a little behind OCR
      if (lu < updatedAt && attempt < 3) { setTimeout(() => updateCardInfo(s, attempt + 1), 250); return; }

      if (!d.scry) { if (attempt < 3) setTimeout(() => updateCardInfo(s, attempt + 1), 250); return; }
      const c = d.scry || {};
      const p = c.prices || {};
      const fx = (d.fx && d.fx.rate) || 1.3;

      const usd = { n: p.usd ? Number(p.usd) : null, f: p.usd_foil ? Number(p.usd_foil) : null, e: p.usd_etched ? Number(p.usd_etched) : null };
      const cad = { n: usd.n != null ? usd.n * fx : null, f: usd.f != null ? usd.f * fx : null, e: usd.e != null ? usd.e * fx : null };

      const image =
        (c.image_uris && c.image_uris.normal) ||
        (Array.isArray(c.card_faces) && c.card_faces[0] && c.card_faces[0].image_uris && c.card_faces[0].image_uris.normal) ||
        null;

      const oracle = String(c.oracle_text || "").replace(/\n/g, "<br>");
      const flavor = String(c.flavor_text || "").replace(/\n/g, "<br>");
      const pt = (c.power || c.toughness) ? String(c.power || "") + "/" + String(c.toughness || "") : (c.loyalty ? String(c.loyalty) + " loyalty" : "");
      const setIcon = c.set ? "https://svgs.scryfall.io/sets/" + c.set + ".svg" : null;
      const tag = (txt) => '<span class="tag">' + txt + "</span>";
      const kw = Array.isArray(c.keywords) ? c.keywords : [];

      const priceBlock = (
        (usd && (usd.n != null || usd.f != null || usd.e != null))
          ? (
            '<div class="prices prices-below">' +
              '<div class="phead">Prices</div>' +
              '<div class="prow">' +
                (usd.n != null ? tag("Normal: " + money(usd.n) + " / " + money(cad.n, "CAD")) : "") +
                (usd.f != null ? tag("Foil: "   + money(usd.f) + " / " + money(cad.f, "CAD")) : "") +
                (usd.e != null ? tag("Etched: " + money(usd.e) + " / " + money(cad.e, "CAD")) : "") +
              "</div>" +
            "</div>"
          )
          : ''
      );

      ci.innerHTML =
        '<div class="cardbox">' +
          '<div class="left">' +
            (image ? ('<img class="art" src="' + image + '" alt="Card">') : "") +
            priceBlock +
          "</div>" +
          '<div class="right">' +
            '<div class="title"><div class="name">' + (c.name || "") + "</div>" + (c.mana_cost ? ('<div class="mana">' + c.mana_cost + "</div>") : "") + "</div>" +
            '<div class="subline">' +
              tag(c.type_line || "") +
              (pt ? tag(pt) : "") +
              tag(String(c.rarity || "").toUpperCase()) +
              tag(c.set_name || "") +
              (c.collector_number ? tag("#" + c.collector_number) : "") +
              (c.released_at ? tag(c.released_at) : "") +
              (setIcon ? ('<img class="seticon" src="' + setIcon + '" alt="' + (c.set || "") + '">') : "") +
            "</div>" +
            '<div class="textbox">' +
              (oracle ? ('<div class="oracle">' + oracle + "</div>") : "") +
              (flavor ? ('<div class="flavor">“' + flavor + "”</div>") : "") +
              (kw.length ? ('<div class="keywords">' + kw.map(tag).join(" ") + "</div>") : "") +
            "</div>" +
          "</div>" +  
        "</div>";   
    } catch (e) {
      console.error("updateCardInfo failed", e);
    } finally {
      cardinfoPending = false;
    }
  }

  // ---------- Bad list ----------
  function fetchBad() {
    fetch("/api/badlist?ts=" + Date.now())
      .then(r => (r.ok ? r.json() : null))
      .then(d => {
        if (!d) return;
        if (badcount) badcount.textContent = d.count || 0;
        if (Array.isArray(d.items) && badbox) {
          const lines = d.items.slice(0, 12).map(x =>
            new Date((x.ts || 0) * 1000).toLocaleTimeString() +
            " • job=" + (x.job != null ? x.job : "?") +
            " • " + (x.ocr_name || "?") +
            " • score=" + Number(x.match_score || 0).toFixed(2) +
            " • conf=" + Number(x.name_conf || 0).toFixed(1) +
            (x.scry_name ? (" • scry=" + x.scry_name + " #" + (x.scry_cn || "")) : "")
          );
          badbox.textContent = lines.join("\n");
        }
      })
      .catch(() => {});
  }

  // -- modal open/close
const $ = s => document.querySelector(s);
const fmodal = $('#filter-modal');
const backdrop = $('#filter-backdrop');

function openFilter() {
  openBackdrop();
  fmodal.classList.remove('hidden');
}
function closeFilter() {
  fmodal.classList.add('hidden');
  hideBackdropIfNone();
}
// set filterContext list on btn-open-filter
$('#btn-open-filter')?.addEventListener('click', () => { filterContext='list'; applyFilterFieldsFromQS(historyQueryQS); openFilter(); });
$('#filter-close')?.addEventListener('click', closeFilter);
backdrop?.addEventListener('click', ()=>{
  closeFilter();
  closeExport?.();
  closeEditModal?.();
  closeBackupModal?.();
  closeHistoryModal();
  document.getElementById("history-options-modal")?.classList.add("hidden");
  closeQuarantineModal();
});
document.addEventListener('keydown', (e)=>{
  if(e.key==='Escape'){
    closeFilter();
    closeExport?.();
    closeEditModal?.();
    closeBackupModal?.();
    closeHistoryModal();
    document.getElementById("history-options-modal")?.classList.add("hidden");
    closeQuarantineModal();
  }
});



// -- build query from modal inputs
function buildHistoryQuery() {
  const qs = new URLSearchParams();
  const s = id => document.getElementById(id);

  const status  = s('hf-status').value;
  const scoreMin= s('hf-scoremin').value;
  const since   = s('hf-since').value;
  const q       = s('hf-q').value;
  const setc    = s('hf-set').value;
  const foil    = s('hf-foil').value;
  const sortBy  = s('hf-sortby').value;
  const sortDir = s('hf-sortdir').value;

  if (status)  qs.set('status', status);
  if (scoreMin)qs.set('score_min', scoreMin);
  if (since)   qs.set('since', since);
  if (q)       qs.set('q', q);
  if (setc)    qs.set('set', setc);
  if (foil)    qs.set('foil', foil);
  qs.set('sort_by',  sortBy);
  qs.set('sort_dir', sortDir);
  qs.set('limit', 'all');
  return qs.toString();
}

// -- history refresh (reuse your renderer)
async function refreshHistoryList() {
  historyQueryQS = buildHistoryQuery();
  await fetchScans(); // reuse the normal fetch, but with current filters
}

// buildHistoryQuery() stays as-is

// REPLACE refreshHistoryList with:
async function refreshHistoryList() {
  historyQueryQS = buildHistoryQuery();
  await fetchScans(); // reuse the normal fetch, but with current filters
}

// REPLACE fetchScans() with:
async function fetchScans() {
  if (modalOpen) return;
  if (!revList && !revCount) return;
  const qs = historyQueryQS ? historyQueryQS + '&' : '';
  const res = await fetch('/api/scans?' + qs + 'ts=' + Date.now());
  const d = await res.json();

  // include the filter in the signature so dedupe works correctly
  const sig = (qs || '') + '|' + JSON.stringify((d.items || []).map(it => [it.id, it.status, it.match_score, it.scry_set, it.scry_cn]));
  if (sig === lastScansSig) return;
  lastScansSig = sig;

  renderScans(d); // <- this was "renderHistoryItems" before (typo)
}

// In the filter modal handlers:
$('#hf-apply')?.addEventListener('click', () => { 
  closeFilter(); 
  if (filterContext === 'history') { historyModalQueryQS = buildHistoryQuery(); try { refreshHistoryGrid(); } catch(e) {} } else { historyQueryQS = buildHistoryQuery(); fetchScans(); }
});

$('#hf-reset')?.addEventListener('click', () => {
  ['hf-status','hf-scoremin','hf-since','hf-q','hf-set','hf-foil','hf-sortby','hf-sortdir'].forEach(id=>{
    const el = document.getElementById(id); if (!el) return;
    if (el.tagName === 'SELECT') el.selectedIndex = 0; else el.value = '';
  });
  document.getElementById('hf-status').value = 'all';
  document.getElementById('hf-sortby').value = 'ts';
  document.getElementById('hf-sortdir').value = 'desc';
  if (filterContext === 'history') { historyModalQueryQS = ""; try { refreshHistoryGrid(); } catch(e) {} } else { historyQueryQS = ""; fetchScans(); }
});

// (optional) if you add a "ManaBox CSV" button:
$('#btn-export-manabox')?.addEventListener('click', () => exportWith('manabox'));

// and slightly tweak exportWith to reuse the current filters:
function exportWith(fmt) {
  const url = new URL('/api/scans/export/download', window.location.origin);
  const params = new URLSearchParams(historyQueryQS || buildHistoryQuery());
  if (fmt === 'txt') params.set('status','pass'); // keep your pass-only default for deck paste
  params.set('fmt', fmt);
  url.search = params.toString();
  window.location.href = url.toString();
}

$('#btn-export-txt')?.addEventListener('click', ()=>exportWith('txt'));

// Load history once on page load
refreshHistoryList();

// ===== Settings Modal =====
// Call openSettings() from your existing settings button.

const SETTINGS_SCHEMA = [
  {
    title: "Camera",
    items: [
      {k:"CAMERA_DEVICE", label:"Camera device"},
      {k:"REQ_WIDTH", label:"Requested width", type:"int"},
      {k:"REQ_HEIGHT", label:"Requested height", type:"int"},
      {k:"REQ_FPS", label:"FPS", type:"int"},
      {k:"FOURCC_PRIMARY", label:"Primary FOURCC"},
      {k:"FOURCC_FALLBACK", label:"Fallback FOURCC"},
    ]
  },
  {
    title: "Processing / Canvas",
    items: [
      {k:"PROC_MAX_WIDTH", label:"Processing downscale max width", type:"int"},
      {k:"PROC_DOWNSCALE_MAX_W", label:"Detection downscale max width", type:"int"},
      {k:"CARD_W", label:"Card canvas width", type:"int"},
      {k:"CARD_H", label:"Card canvas height", type:"int"},
      {k:"MIN_CARD_AREA_RATIO", label:"Min card area ratio", type:"float"},
      {k:"MAX_CARD_AREA_RATIO", label:"Max card area ratio", type:"float"},
      {k:"BORDER_MARGIN_PCT", label:"Border margin (0..1)", type:"float"},
      {k:"CARD_ASPECT", label:"Card aspect (min/max)", type:"float"},
      {k:"ASPECT_TOL", label:"Aspect tolerance", type:"float"},
      {k:"CONFIRM_FRAMES", label:"Confirm frames", type:"int"},
      {k:"STALE_FRAMES", label:"Stale frames", type:"int"},
      {k:"MAX_ASSOC_DIST", label:"Track assoc. distance", type:"int"},
      {k:"MAX_CARDS", label:"Max cards", type:"int"},
      {k:"DETECT_EVERY_N_FRAMES", label:"Detect cadence (N frames)", type:"int"},
      {k:"RECTANGULARITY_MIN", label:"Rectangularity min (0..1)", type:"float"},
      {k:"DETECT_QUAD_PAD_PCT", label:"Quad pad pct (0..0.1)", type:"float"},
    ]
  },
  {
    title: "Capture",
    items: [
      {k:"AUTO_CAPTURE_WAIT_S", label:"Capture grace delay (s)", type:"float"},
      {k:"AUTOSCAN_OCR_TIMEOUT", label:"OCR timeout (s)", type:"float"},
    ]
  },
  {
    title: "OCR & Debug",
    items: [
      {k:"OCR_ONLY_ON_SNAPSHOT", label:"OCR only on snapshot", type:"bool"},
      {k:"DEBUG_OCR", label:"Debug OCR", type:"bool"},
      {k:"OCR_DEBUG_BOXES", label:"Draw OCR boxes", type:"bool"},
      {k:"SHOW_ROI_OVERLAY", label:"Show ROI overlay", type:"bool"},
      {k:"MIN_TITLE_LETTERS", label:"Min title letters", type:"int"},
      {k:"TEXT_PRESENCE_MIN", label:"Text presence min", type:"float"},
      {k:"TITLE_ALLOW_TESS_FALLBACK", label:"Allow tess fallback", type:"bool"},
      {k:"USE_TESS_FOR_TITLES", label:"Use Tesseract for titles", type:"bool"},
      {k:"OCR_ENABLE_BLACKHAT", label:"Enable Blackhat preproc", type:"bool"},
    ]
  },
  {
    title: "Performance / Throttling",
    items: [
      {k:"FAST_OCR_MODE", label:"Fast OCR mode", type:"bool"},
      {k:"LIVE_OCR_ONLY_WHEN_STEADY", label:"Live OCR only when steady", type:"bool"},
      {k:"LIVE_OCR_MIN_INTERVAL", label:"Live OCR min interval (s)", type:"float"},
      {k:"FOIL_EVERY_N_FRAMES", label:"Foil detect every N frames", type:"int"},
      {k:"USE_OPENCL", label:"Use OpenCL (if available)", type:"bool"},
      {k:"SC_IMG_CACHE_DIR", label:"Scry image cache dir"},
    ]
  },
  {
    title: "Foil Detection",
    items: [
      {k:"FOIL_MIN_SCORE", label:"Foil min score", type:"float"},
      {k:"FOIL_ON_TH", label:"Foil ON threshold", type:"float"},
      {k:"FOIL_OFF_TH", label:"Foil OFF threshold", type:"float"},
      {k:"FOIL_DETECT", label:"Enable foil detect", type:"bool"},
    ]
  },
  {
    title: "Match / Comparison",
    items: [
      {k:"MATCH_ENABLE", label:"Enable visual match", type:"bool"},
      {k:"MATCH_W_HASH", label:"Weight: hash", type:"float"},
      {k:"MATCH_W_HIST", label:"Weight: hist", type:"float"},
      {k:"MATCH_W_ORB",  label:"Weight: ORB",  type:"float"},
      {k:"MATCH_TH",     label:"Match pass threshold", type:"float"},
      {k:"NAME_OK_TH",   label:"Name OK threshold", type:"float"},
      {k:"ALWAYS_SCAN_OK", label:"Mark scan OK even if match fails", type:"bool"},
      {k:"MATCH_USE_ART",label:"Use art region for compare", type:"bool"},
    ]
  },
  {
    title: "Advanced Match (Perf)",
    items: [
      {k:"MATCH_ORB_FEATURES", label:"ORB features", type:"int"},
      {k:"MATCH_FAST_ACCEPT_DELTA", label:"Fast accept delta", type:"float"},
      {k:"MATCH_FAST_REJECT_DELTA", label:"Fast reject delta", type:"float"},
      {k:"HASH_STABILITY_BITS", label:"Min changed bits to re-OCR", type:"int"},
    ]
  },
  
  {
    title: "AI / YOLOv5",
    items: [
      {k:"AI_ENABLED", label:"Enable AI", type:"bool"},
      {k:"AI_ONLY_MODE", label:"AI ROIs only (ignore static ROIs)", type:"bool"},
      {k:"AI_USE_FOR_CARDS", label:"Use AI to find the card box", type:"bool"},
      {k:"AI_USE_FOR_ROIS", label:"Use AI to find title/set/etc ROIs", type:"bool"},
      {k:"AI_MODEL_PATH", label:"Model path (best.pt)"},
      {k:"YOLOV5_DIR", label:"Local yolov5 repo dir"},
      {k:"AI_IMG_SIZE", label:"AI image size", type:"int"},
      {k:"AI_CONF_THRES", label:"AI conf threshold", type:"float"},
      {k:"AI_IOU_THRES", label:"AI IoU threshold", type:"float"},
      {k:"AI_MAX_DETS", label:"AI max detections", type:"int"},
    ]
  },
  {
    title: "Scryfall & Prices",
    items: [
      {k:"SCRYFALL_TIMEOUT", label:"HTTP timeout (s)", type:"float"},
      {k:"FX_URL", label:"FX URL"},
      {k:"FX_TTL_SEC", label:"FX TTL (s)", type:"int"},
    ]
  },
  {
    title: "Moonraker / Klipper",
    items: [
      {k:"MOONRAKER_URL", label:"WebSocket URL"},
      {k:"HTTP_POST_URL", label:"G-code HTTP URL"},
    ]
  },
  {
    title: "Stacks / Sorter",
    items: [
      {k:"CARD_THICKNESS_MM", label:"Card thickness (mm)", type:"float"},
      {k:"REMEASURE_EVERY", label:"Remeasure every N cards (0 = off)", type:"int"},
    ]
  },
  {
    title: "Debug & Paths",
    items: [
      {k:"DEBUG_LEVEL", label:"Debug level (1-4)", type:"int"},
    ]
  },
  {
    title: "Persistence & Server",
    items: [
      {k:"HISTORY_DIR", label:"History dir"},
      {k:"HISTORY_IMG_DIR", label:"History images dir"},
      {k:"HISTORY_JSON_EXT", label:"History JSON ext"},
      {k:"JPEG_QUALITY_SNAP", label:"JPEG quality (snap)", type:"int"},
      {k:"JPEG_QUALITY_CMP",  label:"JPEG quality (compare)", type:"int"},
      {k:"JPEG_QUALITY_THUMB",label:"JPEG quality (thumb)", type:"int"},
      {k:"HOST", label:"Bind host"},
      {k:"PORT", label:"Port", type:"int"},
    ]
  },

  {
    title: "Tracking & Warp",
    items: [
      {k:"TRACK_ALPHA", label:"Track alpha (stiffness)", type:"float"},
      {k:"TRACK_BETA", label:"Track beta (velocity)", type:"float"},
      {k:"TRACK_DEADBAND_PX", label:"Deadband (px)", type:"float"},
      {k:"LOCK_IOU_THRESH", label:"Lock IoU threshold", type:"float"},
      {k:"ACQUIRE_FRAMES", label:"Frames to lock", type:"int"},
      {k:"DROP_MISS_FRAMES", label:"Misses to drop", type:"int"},
      {k:"PREDICT_HOLD", label:"Predict-only frames", type:"int"},
      {k:"STEADY_SPEED_PX", label:"Steady speed (px)", type:"float"},
      {k:"WARP_EXPAND_PCT", label:"Warp expand %", type:"float"},
      {k:"WARP_CROP_PCT", label:"Warp crop %", type:"float"},
      {k:"DETECT_MAX_FPS", label:"Detect max fps", type:"int"},
    ]
  },


  {
    title: "Appearance / Photometric",
    items: [
      {k:"APPEARANCE_ALIGN_ENABLE", label:"Enable appearance alignment", type:"bool"},
      {k:"APPEARANCE_AB_STRENGTH", label:"Chroma transfer strength", type:"float"},
      {k:"APPEARANCE_SAT_STRENGTH", label:"Saturation match strength", type:"float"},
      {k:"APPEARANCE_GAMMA_CLAMP", label:"Gamma clamp (min,max)"},
      {k:"TONE_ALIGN_ENABLE", label:"Tone alignment", type:"bool"},
    ]
  },

];

const SETTINGS_BASE_KEYS = new Set(SETTINGS_SCHEMA.flatMap((sec) => sec.items.map((it) => it.k)));
let currentSettingsSchema = SETTINGS_SCHEMA;

function inferSettingType(val) {
  if (typeof val === "boolean") return "bool";
  if (typeof val === "number") return Number.isInteger(val) ? "int" : "float";
  if (Array.isArray(val) || (val && typeof val === "object")) return "json";
  return "text";
}

function humanizeSettingKey(key) {
  return (key || "")
    .split("_")
    .map((part) => (part ? (part[0].toUpperCase() + part.slice(1).toLowerCase()) : ""))
    .join(" ")
    .trim() || key;
}

function buildSettingsSchema(data) {
  const extras = [];
  const known = new Set(SETTINGS_BASE_KEYS);
  Object.keys(data || {}).sort().forEach((k) => {
    if (!k || k.startsWith("__")) return;
    if (known.has(k)) return;
    if (k.toUpperCase() !== k) return;
    extras.push({ k, label: humanizeSettingKey(k), type: inferSettingType(data[k]) });
  });
  if (!extras.length) return SETTINGS_SCHEMA;
  const base = SETTINGS_SCHEMA.map((sec) => ({ title: sec.title, items: sec.items.slice() }));
  base.push({ title: "Other Settings", items: extras });
  return base;
}

// ===== Settings Modal (fixed, long scrollable form) =====

// Wire the “open settings” button
document.getElementById('btn-open-settings')?.addEventListener('click', openSettings);

// Cache DOM nodes with the IDs that actually exist in index.html
const settingsBackdrop = document.getElementById('settings-backdrop');
const settingsModal    = document.getElementById('settings-modal');
const settingsForm     = document.getElementById('settings-form');

function openSettings() {
  modalOpen = true;                                  
  settingsBackdrop?.classList.remove('hidden');
  settingsModal?.classList.remove('hidden');
  renderSettings(true);
}
function closeSettings() {
  settingsBackdrop?.classList.add('hidden');
  settingsModal?.classList.add('hidden');
  modalOpen = false;                                  
}


// Close on ✕ and on backdrop click
document.getElementById('settings-close')?.addEventListener('click', closeSettings);
settingsBackdrop?.addEventListener('click', (e) => {
  if (e.target === settingsBackdrop) closeSettings();
});

// Save + reset buttons
document.getElementById('settings-save')?.addEventListener('click', (e) => {
  e.preventDefault();
  saveSettings();
});
document.getElementById('settings-reset')?.addEventListener('click', (e) => {
  e.preventDefault();
  renderSettings(true); // re-fetch current/effective values
});

// Build the entire long scrollable form (no tabs)
async function renderSettings(forceFresh = false) {
  if (!settingsForm) return;
  settingsForm.innerHTML = '<div class="mm-field">Loading…</div>';

  let cur = {};
  try {
    const res = await fetch('/api/settings' + (forceFresh ? ('?t=' + Date.now()) : ''));
    cur = await res.json();
  } catch (err) {
    settingsForm.innerHTML = '<div class="mm-field">Failed to load settings (click Reset to retry)</div>';
    return;
  }
  // cur already contains the merged {**effective, **saved}; helper keys (__effective, __path) are ignored by the schema.

  const schema = buildSettingsSchema(cur);
  currentSettingsSchema = schema;

  const frag = document.createDocumentFragment();

  schema.forEach((sec) => {
    const section = document.createElement('section');
    section.className = 'mm-section';

    const h = document.createElement('h3');
    h.textContent = sec.title;
    // make sticky by default; remove if you don't want it:
    h.className = 'sticky';
    section.appendChild(h);

    const grid = document.createElement('div');
    grid.className = 'mm-grid';
    sec.items.forEach((it) => grid.appendChild(fieldEl(it, cur[it.k])));
    section.appendChild(grid);

    frag.appendChild(section);
  });

  settingsForm.innerHTML = '';
  settingsForm.appendChild(frag);
}


// Render one field
function fieldEl(it, val) {
  const wrap = document.createElement('div');
  wrap.className = 'mm-field';
  const id = 's__' + it.k;
  const type = it.type || inferSettingType(val);

  const label = document.createElement('label');
  label.htmlFor = id;
  label.textContent = it.label;
  wrap.appendChild(label);

  if (type === 'bool') {
    const sel = document.createElement('select');
    sel.id = id;
    sel.className = 'mm-input';
    sel.dataset.type = 'bool';
    ['true', 'false'].forEach((v) => {
      const o = document.createElement('option');
      o.value = v;
      o.textContent = v;
      const valStr = String(val).toLowerCase();
      if ((val === true && v === 'true') || (val === false && v === 'false') || valStr === v) {
        o.selected = true;
      }
      sel.appendChild(o);
    });
    wrap.appendChild(sel);
    return wrap;
  }

  if (type === 'json') {
    const ta = document.createElement('textarea');
    ta.id = id;
    ta.className = 'mm-input';
    ta.dataset.type = 'json';
    try {
      ta.value = JSON.stringify(val, null, 2);
    } catch (e) {
      ta.value = String(val ?? '');
    }
    ta.rows = Math.min(6, Math.max(2, ta.value.split('\n').length));
    wrap.appendChild(ta);
    return wrap;
  }

  const inp = document.createElement('input');
  inp.id = id;
  inp.type = 'text';
  inp.value = val ?? '';
  inp.className = 'mm-input';
  inp.dataset.type = type;
  if (type === 'int') inp.inputMode = 'numeric';
  if (type === 'float') inp.inputMode = 'decimal';
  wrap.appendChild(inp);
  return wrap;
}

// Collect all values across the whole long form
function collect() {
  const out = {};
  const errors = [];
  const schema = currentSettingsSchema || SETTINGS_SCHEMA;
  schema.forEach((sec) => {
    sec.items.forEach((it) => {
      const id = 's__' + it.k;
      const el = document.getElementById(id);
      if (!el) return;
      if (el.classList) el.classList.remove('mm-error');
      const type = it.type || el.dataset?.type || 'text';
      const raw = (el.value ?? '').trim();
      if (type === 'bool') {
        out[it.k] = (el.value === 'true');
        return;
      }
      if (type === 'int' || type === 'float') {
        const num = (type === 'int') ? parseInt(raw || '0', 10) : parseFloat(raw || '0');
        if (Number.isNaN(num)) {
          errors.push(`${it.k}: invalid number`);
          el.classList?.add('mm-error');
          return;
        }
        out[it.k] = num;
        return;
      }
      if (type === 'json') {
        if (!raw) { out[it.k] = null; return; }
        try {
          out[it.k] = JSON.parse(raw);
        } catch (err) {
          errors.push(`${it.k}: ${err.message}`);
          el.classList?.add('mm-error');
        }
        return;
      }
      out[it.k] = el.value;
    });
  });
  if (errors.length) {
    throw new Error(errors.join('\n'));
  }
  return out;
}

async function saveSettings() {
  let payload;
  try {
    payload = collect();
  } catch (err) {
    alert('Settings not saved: ' + err.message);
    return;
  }
  const res = await fetch('/api/settings', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  let j = {};
  try {
    j = await res.json();
  } catch (e) {}
  if (!res.ok || !j.ok) {
    alert('Save failed');
    return;
  }
  alert(
    'Settings saved.' +
      (j.restart_required ? ' (Restart recommended for some changes.)' : '')
  );
}


document.getElementById("exportBtn")?.addEventListener("click", () => {
  const qs = buildHistoryQuery();
  // force pass-only for deck exports if you prefer:
  const u = new URLSearchParams(qs);
  u.set('status','pass');
  window.location.href = '/api/scans/export/download?' + u.toString();
});
  // ---------- Review list ----------
  function renderScans(scans) {
    if (!revList || !revCount) return;
    const items = Array.isArray(scans.items) ? scans.items : [];
    rememberScanMeta(items);

    revList.innerHTML = items.map(function (it) {
      const st = String(it.status || "fail").toLowerCase();
      const cls = (st === "pass" ? "ok" : (st === "review" ? "warn" : "bad"));
      const score = Number(it.match_score || 0);
      const numStr = (it.number ? (` • #${esc(it.number)}`) : "");
      const thumbHtml = it.thumb_ok
        ? `<img class="thumb" loading="lazy" src="/api/scan/${it.id}/thumb" alt="scan ${it.id}">`
        : '<div class="thumb thumb-missing" aria-label="No snapshot">No Image</div>';
      const setStr = it.scry_set
        ? `<span>${esc(String(it.scry_set).toUpperCase())} ${esc(it.scry_cn || "")}</span>`
        : (it.set_hint ? `<span>${esc(String(it.set_hint).toUpperCase())}</span>` : "");
      const warn = (it.flagged && Array.isArray(it.review_reasons) && it.review_reasons.length)
        ? `<span class="pill warn" title="${it.review_reasons.join(' • ').replace(/"/g,'&quot;')}">Review Req.</span>`
        : "";
      return `
        <div class="revitem" data-id="${it.id}" data-status="${esc(it.status || "fail")}">
          ${thumbHtml}
          <div style="flex:1;min-width:0;display:flex;flex-direction:column;gap:4px;">
            <div style="display:flex;align-items:center;gap:8px;justify-content:space-between;">
              <div style="min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;"><b>${esc(it.name || "(unnamed)")}</b>${numStr}</div>
              <span class="pill ${cls}">VISUAL - ${String((it.status || "FAIL")).toUpperCase()}</span>
            </div>
            <div class="meta">
              <span>Visual Score: ${(score * 100).toFixed(2)}%</span>
              ${setStr}
              ${warn}
            </div>
            <div class="revitem-actions">
              <button type="button" class="btn btn-mini btn-ghost rev-edit" data-id="${it.id}">Manual Edit</button>
            </div>
          </div>
        </div>
      `;
    }).join("");

    const summary = scans?.summary || null;
    const total = Number(
      (summary && summary.total) ??
      (scans && (scans.total ?? scans.count)) ??
      items.length ??
      0
    );
    const shown = items.length;
    const passCount = summary?.pass ?? items.filter(it => (it.status || "").toLowerCase() === "pass").length;
    const failCount = summary?.fail ?? items.filter(it => (it.status || "").toLowerCase() === "fail").length;
    const reviewCount = summary?.review ?? items.filter(it => (it.status || "").toLowerCase() === "review").length;
    let label = `${total.toLocaleString()} cards`;
    if (total > shown && shown > 0) {
      label += ` (showing ${shown.toLocaleString()})`;
    }
    const breakdown = [
      typeof passCount === "number" ? `Pass ${passCount.toLocaleString()}` : null,
      typeof failCount === "number" ? `Fail ${failCount.toLocaleString()}` : null,
      typeof reviewCount === "number" ? `Review ${reviewCount.toLocaleString()}` : null,
    ].filter(Boolean);
    if (breakdown.length) {
      label += ` • ${breakdown.join(" | ")}`;
    }
    revCount.textContent = label;
    attachThumbFallbacks(revList);
    Array.from(revList.querySelectorAll(".revitem")).forEach(function (el) {
      el.addEventListener("click", async function () {
        const id = Number(el.getAttribute("data-id"));
        try {
          const resp = await fetch(`/api/scan/${id}/load`, { method: "POST" });
          const res = await resp.json().catch(() => ({}));
          if (!resp.ok || !res?.ok) {
            alert(res?.error || "Unable to load this scan right now (scanner busy?).");
            return;
          }
          currentLoadedId = id;
          if (btnPass) btnPass.disabled = false;
          if (btnFail) btnFail.disabled = false;
          if (btnEdit) btnEdit.disabled = false;
          if (btnDelete) btnDelete.disabled = false;
          refreshCompare(Date.now());
          fetchState();
        } catch (err) {
          console.error("History load failed", err);
        }
      });
    });
    revList.querySelectorAll(".rev-edit").forEach(btn => {
      btn.addEventListener("click", (ev) => {
        ev.stopPropagation();
        const id = Number(btn.getAttribute("data-id"));
        if (!openManualEditById(id)) {
          currentLoadedId = id;
          openEditModal({ id });
        }
      });
    });
  }
  async function fetchScans() {
    if (!revList && !revCount) return;
    const qs = historyQueryQS ? historyQueryQS + '&' : '';
    const res = await fetch('/api/scans?' + qs + 'ts=' + Date.now());
    const d = await res.json();

    // include the filter in the signature so dedupe works correctly
    const sig = (qs || '') + '|' + JSON.stringify((d.items || []).map(it => [it.id, it.status, it.match_score, it.scry_set, it.scry_cn]));
    if (sig === lastScansSig) return;
    lastScansSig = sig;

    renderScans(d); // <- this was "renderHistoryItems" before (typo)
  }
  // ---------- State / pills ----------
  function fetchState() {
    if (modalOpen) return;
    fetch("/api/state?ts=" + Date.now())
      .then(r => r.json())
      .then(function (s) {
        lastState = s;

        const scannerPills = [];
        scannerPills.push({ txt: (s.snapshot_present ? ("Snapshot " + (s.snapshot_age_sec || 0) + "s ago") : "No Snapshot"), cls: (s.snapshot_present ? "ok" : "warn") });
        const blurVal = (typeof s.steady_blur === "number") ? s.steady_blur.toFixed(1) : null;
        const motionVal = (typeof s.steady_motion === "number") ? s.steady_motion.toFixed(3) : null;
        const steadyTitle = (blurVal !== null && motionVal !== null) ? ("blur " + blurVal + " • motion " + motionVal) : "";
        scannerPills.push({ txt: (s.steady ? "STEADY" : "UNSTEADY"), cls: (s.steady ? "ok" : "warn"), title: steadyTitle });
        updatePills(sc, scannerPills);

        const p1p = [{ txt: (s.ws_connected ? "WS connected" : "WS offline"), cls: (s.ws_connected ? "ok" : "bad") }];
        if (s.awaiting) p1p.push({ txt: "Most recent scan - Job #" + s.job_id, cls: "info" });
        if (s.last_decision) p1p.push({ txt: s.last_decision, cls: "ok" });
        const stacks = s.stacks || {};
        const rem = stacks.remaining_cards || [];
        const start = stacks.start_cards || [];
        const heights = stacks.current_height || [];
        [0, 1].forEach((i) => {
          const rc = rem[i];
          const startTxt = (typeof start[i] === "number") ? ` of ${start[i]}` : "";
          const hTxt = (typeof heights[i] === "number") ? ` (${heights[i].toFixed(2)}mm)` : "";
          const txt = (typeof rc === "number")
            ? `Stack ${i + 1}: ${rc}${startTxt} cards${hTxt}`
            : `Stack ${i + 1}: —${startTxt} ${hTxt}`.trim();
          p1p.push({ txt, cls: (typeof rc === "number" ? "ok" : "warn") });
        });
        updatePills(p1, p1p);

        const ocrPills = [];
        const foilNow = (typeof s.ocr_foil !== "undefined") ? s.ocr_foil : s.foil;
        const foilSc  = (typeof s.ocr_foil_score !== "undefined") ? s.ocr_foil_score : s.foil_score;
        if (s.snapshot_present && typeof foilNow !== "undefined") {
          const fs = (typeof foilSc === "number" ? foilSc.toFixed(2) : "—");
          ocrPills.push({ txt: (foilNow ? ("Foil • " + fs) : "Non-foil"), cls: (foilNow ? "ok" : "warn") });
        }
        if (s.name) { ocrPills.push({ txt: "Name: " + s.name, cls: "ok" }, { txt: "conf " + Math.round(s.name_conf || 0), cls: "ok" }); }
        else { ocrPills.push({ txt: "Name: —", cls: "warn" }); }
        if (s.number) {
          const numConf = Math.max(0, Math.round(s.number_conf || 0));
          const confCls = (numConf >= 65) ? "ok" : (numConf >= 30 ? "warn" : "bad");
          const numCls = (numConf >= 20) ? "ok" : "warn";
          ocrPills.push({ txt: "No: " + s.number, cls: numCls });
          ocrPills.push({ txt: "conf " + numConf, cls: confCls });
        }
        if (s.set_hint) ocrPills.push({ txt: "Set: " + String(s.set_hint).toUpperCase(), cls: "ok" });
        if (typeof s.match_ok !== "undefined" && s.match_ok !== null) {
          const ms = Number(s.match_score || 0).toFixed(2);
          ocrPills.push({ txt: (s.match_ok ? "MATCH " : "REVIEW ") + ms, cls: (s.match_ok ? "ok" : "bad") });
        }
        if (s.flagged) ocrPills.push({ txt: "FLAGGED", cls: "bad" });
        if (s.flagged && Array.isArray(s.review_reasons) && s.review_reasons.length) {
          const why = s.review_reasons.join(" • ");
          ocrPills.push({ txt: "Why?", cls: "warn", title: why });
        }

        if (s.last_error) ocrPills.push({ txt: "OCR err: " + s.last_error, cls: "bad" });
        updatePills(ocrEl, ocrPills);

        const cmpSeq = Number(s.cmp_seq || 0);
        if (cmpSeq && cmpSeq !== lastCmpSeq) refreshCompare(cmpSeq);

        currentLoadedId = s.loaded_scan_id || currentLoadedId;
        if (btnPass) btnPass.disabled = !currentLoadedId;
        if (btnFail) btnFail.disabled = !currentLoadedId;
        if (btnEdit) btnEdit.disabled = !currentLoadedId;
        if (btnReprocess) btnReprocess.disabled = !currentLoadedId;
        if (btnDelete) btnDelete.disabled = !currentLoadedId;
        updateCardInfo(s);
        fetchBad();
      })
      .catch(e => console.error("fetchState error", e));
  }

  // ---------- Snapshot buttons (if added later) ----------
  const btnSnap = document.getElementById("btnSnap");
  const btnClear = document.getElementById("btnClear");
  function snap(action) {
    fetch("/api/snapshot", { method:"POST", headers:{ "Content-Type":"application/json" }, body: JSON.stringify({ action }) })
      .catch(console.error).then(() => setTimeout(() => { fetchState(); fetchScans(); }, 200));
  }
  if (btnSnap)  btnSnap.onclick  = () => snap("snap");
  if (btnClear) btnClear.onclick = () => snap("clear");

  if (btnPass) btnPass.onclick = function () {
    if (!currentLoadedId) return;
    fetch("/api/scan/" + currentLoadedId + "/status", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ status: "pass" })
    }).then(fetchScans);
  };
  if (btnFail) btnFail.onclick = function () {
    if (!currentLoadedId) return;
    fetch("/api/scan/" + currentLoadedId + "/status", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ status: "fail" })
    }).then(fetchScans);
  };
  if (btnDelete) btnDelete.onclick = function () {
    if (!currentLoadedId) return;
    if (!confirm("Delete this scan from history? This removes the entry and images from disk.")) return;
    fetch("/api/scan/" + currentLoadedId + "/delete", { method: "POST" })
      .then(r => r.json())
      .then(res => {
        if (res && res.ok) {
          currentLoadedId = null;
          fetchScans();
          fetchState();
          refreshCompare(Date.now());
        }
      }).catch(() => {});
  };

  // ---------- Edit modal ----------
  function openEditModal(prefillMeta) {
    if (!modal) return;
    const meta = prefillMeta || lastState || {};
    const targetId = (prefillMeta && prefillMeta.id) || currentLoadedId;
    if (!targetId) return;
    activeEditId = targetId;
    const name = meta.name || meta.name_raw || meta.scry_name || "";
    const set  = meta.set_hint || meta.scry_set || "";
    const num  = meta.number_raw || meta.number || meta.scry_cn || "";
    eiName.value = name;
    eiSet.value = set;
    eiNum.value = num;
    eiResults.innerHTML = ""; chosenScryId = null;
    // ensure it's visible even if it has the 'hidden' utility class
    modal.classList.remove('hidden');
    // show shared backdrop (same one used by filter/export)
    openBackdrop();
    modal.classList.add("show"); modal.setAttribute("aria-hidden","false");
  }
  function closeEditModal() {
    modal.classList.remove('show');
    modal.setAttribute('aria-hidden','true');
    modal.classList.add('hidden');                   // <-- important
    activeEditId = null;
    chosenScryId = null;
    hideBackdropIfNone();
  }
  btnEdit?.addEventListener('click', (e)=>{ e.preventDefault(); if (currentLoadedId) openEditModal(); });

  function setReprocessStatus(msg) {
    if (reprocessStatus) reprocessStatus.textContent = msg;
  }
  function renderReasons(el, reasons) {
    if (!el) return;
    const list = Array.isArray(reasons) ? reasons : [];
    if (!list.length) {
      el.innerHTML = '<li class="muted">None</li>';
    } else {
      el.innerHTML = list.map(r => `<li>${esc(r)}</li>`).join("");
    }
  }

  function attachReprocessMagnifiers() {
    attachMagnifier(reOldCmp, reOldCmp?.getAttribute("src") || "");
    attachMagnifier(reNewCmp, reNewCmp?.getAttribute("src") || "");
    document.querySelectorAll("#reprocessBatchList img").forEach(img => {
      attachMagnifier(img, img.getAttribute("src") || "");
    });
  }
  function fillReprocessColumns(oldData, newData) {
    const oldScore = typeof oldData?.match_score === "number" ? (Number(oldData.match_score)*100).toFixed(2) + "%" : "—";
    const newScore = typeof newData?.match_score === "number" ? (Number(newData.match_score)*100).toFixed(2) + "%" : "—";
    if (reOldName) reOldName.textContent = oldData?.name || "(unnamed)";
    if (reOldSet) reOldSet.textContent = (oldData?.set_hint || "").toUpperCase() || "—";
    if (reOldNumber) reOldNumber.textContent = oldData?.number || "—";
    if (reOldScore) reOldScore.textContent = oldScore;
    if (reOldFlag) reOldFlag.textContent = oldData?.flagged ? "Yes" : "No";
    if (reOldCmp) {
      if (oldData?.cmp_data_url) {
        reOldCmp.src = oldData.cmp_data_url;
        reOldCmp.classList.remove("hidden");
      } else {
        reOldCmp.classList.add("hidden");
        reOldCmp.removeAttribute("src");
      }
    }
    renderReasons(reOldReasons, oldData?.review_reasons);

    if (reNewName) {
      reNewName.value = newData?.name || "";
      reNewName.setAttribute("placeholder", "(unnamed)");
    }
    if (reNewSet) {
      reNewSet.value = (newData?.set_hint || "").toUpperCase();
      reNewSet.setAttribute("placeholder", "Set code");
    }
    if (reNewNumber) {
      reNewNumber.value = newData?.number || "";
      reNewNumber.setAttribute("placeholder", "Collector #");
    }
    if (reNewScore) reNewScore.textContent = newScore;
    if (reNewFlag) reNewFlag.textContent = newData?.flagged ? "Yes" : "No";
    if (reNewCmp) {
      if (newData?.cmp_data_url) {
        reNewCmp.src = newData.cmp_data_url;
        reNewCmp.classList.remove("hidden");
      } else {
        reNewCmp.classList.add("hidden");
        reNewCmp.removeAttribute("src");
      }
    }
    if (reNewScry) {
      const setText = newData?.scry_set ? String(newData.scry_set).toUpperCase() : "";
      const cn = newData?.scry_cn || "";
      let scryTxt = newData?.scry_name || "";
      if (setText || cn) {
        scryTxt = `${setText} ${cn}`.trim() + (scryTxt ? ` • ${scryTxt}` : "");
      }
      reNewScry.textContent = scryTxt || "—";
    }
    renderReasons(reNewReasons, newData?.review_reasons);
  }
  async function openReprocessModalForScan(sid) {
    if (!reModal || !sid) return;
    reprocessTargetId = sid;
    reprocessToken = null;
    reprocessApply?.setAttribute("disabled","disabled");
    setReprocessStatus("Starting reprocess…");
    setProgressBar(reprocessProgressBar, 0);
    reprocessProgressStop = startProgressRamp(reprocessProgressBar, 0.85, 0.008, 100);
    reprocessContent?.classList.add("hidden");
    reModal.classList.remove("hidden");
    openBackdrop();
    reModal.classList.add("show");
    try {
      const data = await fetchWithFallback(
        [`/api/scan/${sid}/reprocess?mode=sync`, `/api/scan/${sid}/reprocess`],
        {method:'POST'}
      );
      if (!data || data.ok === false) {
        throw new Error((data && data.error) || "Unable to reprocess");
      }
      reprocessToken = data.token;
      fillReprocessColumns(data.old || {}, data.candidate || {});
      attachReprocessMagnifiers();
      setReprocessStatus("Review the new detection and accept to overwrite.");
      reprocessContent?.classList.remove("hidden");
      reprocessApply?.removeAttribute("disabled");
      setProgressBar(reprocessProgressBar, 1);
    } catch (err) {
      setReprocessStatus("Reprocess failed: " + err.message);
      setProgressBar(reprocessProgressBar, 0);
    } finally {
      reprocessProgressStop = stopIndeterminateBar(reprocessProgressStop, null);
    }
  }
  function closeReprocessModal() {
    if (!reModal) return;
    reModal.classList.remove("show");
    setTimeout(()=>{
      reModal.classList.add("hidden");
      hideBackdropIfNone();
    }, 50);
    reprocessToken = null;
    reprocessTargetId = null;
    reprocessApply?.setAttribute("disabled","disabled");
    reprocessContent?.classList.add("hidden");
    setReprocessStatus("Preparing…");
    reprocessProgressStop = stopIndeterminateBar(reprocessProgressStop, reprocessProgressBar, 0);
  }
  btnReprocess?.addEventListener("click", (e) => {
    e.preventDefault();
    if (currentLoadedId) {
      openReprocessModalForScan(currentLoadedId);
    }
  });
  reprocessClose?.addEventListener("click", closeReprocessModal);
  reprocessReject?.addEventListener("click", closeReprocessModal);
  reModal?.addEventListener("click", (e) => {
    if (e.target === reModal || e.target.classList.contains("backdrop")) {
      closeReprocessModal();
    }
  });
  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && reModal?.classList.contains("show")) {
      closeReprocessModal();
    }
  });
  reprocessApply?.addEventListener("click", async () => {
    if (!reprocessToken || !reprocessTargetId) return;
    reprocessApply.setAttribute("disabled","disabled");
    setReprocessStatus("Applying new result…");
    try {
      const override = {};
      if (reNewName && reNewName.value.trim()) override.name = reNewName.value.trim();
      if (reNewSet && reNewSet.value.trim()) override.set_hint = reNewSet.value.trim();
      if (reNewNumber && reNewNumber.value.trim()) override.number = reNewNumber.value.trim();
      const res = await fetch(`/api/scan/${reprocessTargetId}/reprocess/apply`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ token: reprocessToken, override })
      });
      const data = await res.json();
      if (!data || !data.ok) {
        throw new Error((data && data.error) || "Unable to apply");
      }
      closeReprocessModal();
      fetchState();
      fetchScans();
      try { refreshHistoryGrid(); } catch {}
    } catch (err) {
      setReprocessStatus("Apply failed: " + err.message);
      reprocessApply?.removeAttribute("disabled");
    }
  });

  function setBatchStatus(msg) {
    if (batchStatus) batchStatus.textContent = msg;
  }
  function formatBatchCounts(flagged, failed) {
    const parts = [];
    if (typeof flagged === "number") parts.push(`${flagged} flagged`);
    if (typeof failed === "number") parts.push(`${failed} failed`);
    return parts.join(", ");
  }
  function closeBatchModal() {
    if (!batchModal) return;
    if (reprocessBatchJobId) {
      fetch(`/api/reprocess/job/${reprocessBatchJobId}`, {method:'DELETE'}).catch(()=>{});
      reprocessBatchJobId = null;
    }
    batchModal.classList.remove("show");
    setTimeout(()=>batchModal.classList.add("hidden"), 50);
    reprocessBatchItems = [];
    if (batchList) {
      batchList.innerHTML = "";
      batchList.classList.add("hidden");
    }
    setBatchStatus("Preparing…");
    batchOverallProgressStop = stopIndeterminateBar(batchOverallProgressStop, batchOverallBar, 0);
    hideBackdropIfNone();
  }
  function renderBatchList() {
    if (!batchList) return;
    if (!reprocessBatchItems.length) {
      batchList.innerHTML = "";
      batchList.classList.add("hidden");
      setBatchStatus("No review items need updates.");
      return;
    }
    batchList.classList.remove("hidden");
    const html = reprocessBatchItems.map(item => {
      const oldName = esc(item.old?.name || "(unnamed)");
      const newName = esc(item.candidate?.name || "(unnamed)");
      const oldSet = esc((item.old?.set_hint || "").toUpperCase()) || "—";
      const newSet = esc((item.candidate?.set_hint || "").toUpperCase()) || "—";
      const oldNumber = esc(item.old?.number || "—");
      const newNumber = esc(item.candidate?.number || "—");
      const oldScore = typeof item.old?.match_score === "number" ? (Number(item.old.match_score)*100).toFixed(1)+"%" : "—";
      const newScore = typeof item.candidate?.match_score === "number" ? (Number(item.candidate.match_score)*100).toFixed(1)+"%" : "—";
      const oldReasons = (item.old?.review_reasons || []).map(r => `<li>${esc(r)}</li>`).join("") || '<li class="muted">None</li>';
      const newReasons = (item.candidate?.review_reasons || []).map(r => `<li>${esc(r)}</li>`).join("") || '<li class="muted">None</li>';
      const scryTxt = (() => {
        const setTxt = item.candidate?.scry_set ? String(item.candidate.scry_set).toUpperCase() : "";
        const cn = item.candidate?.scry_cn || "";
        let nm = item.candidate?.scry_name || "";
        if (setTxt || cn) {
          nm = `${setTxt} ${cn}`.trim() + (nm ? ` • ${nm}` : "");
        }
        return nm || "—";
      })();
      const scanImg = item.old?.thumb_url || item.old?.cmp_data_url || "";
      const scryImg = item.candidate?.scry_image || item.candidate?.cmp_data_url || "";
      const imageHtml = (scanImg || scryImg) ? `
        <div class="reproc-batch-images">
          <div class="img-slot">
            ${scanImg ? `<span>Scan</span><img src="${esc(scanImg)}" alt="Scan #${item.id}">` : ""}
          </div>
          <div class="img-slot">
            ${scryImg ? `<span>Scryfall</span><img src="${esc(scryImg)}" alt="Scryfall match for #${item.id}">` : ""}
          </div>
        </div>
      ` : "";
      return `
        <div class="reproc-batch-item" data-token="${item.token}">
          <h4>#${item.id} ${oldName}</h4>
          ${imageHtml}
          <div class="reproc-batch-columns">
            <div>
              <div class="reproc-field"><span>Current</span><strong>${oldName}</strong></div>
              <div class="reproc-field"><span>Set</span><strong>${oldSet}</strong></div>
              <div class="reproc-field"><span>Number</span><strong>${oldNumber}</strong></div>
              <div class="reproc-field"><span>Match</span><strong>${oldScore}</strong></div>
              <ul class="reproc-reasons">${oldReasons}</ul>
            </div>
            <div>
              <div class="reproc-field"><span>New OCR</span><input class="mm-input reproc-edit" data-token="${item.token}" data-edit="name" type="text" value="${newName}" aria-label="New name"></div>
              <div class="reproc-field sub"><span>Suggested</span><em>${esc(scryTxt)}</em></div>
              <div class="reproc-field"><span>Set</span><input class="mm-input reproc-edit" data-token="${item.token}" data-edit="set_hint" type="text" value="${newSet}" aria-label="New set code"></div>
              <div class="reproc-field"><span>Number</span><input class="mm-input reproc-edit" data-token="${item.token}" data-edit="number" type="text" value="${newNumber}" aria-label="New collector number"></div>
              <div class="reproc-field"><span>Match</span><strong>${newScore}</strong></div>
              <ul class="reproc-reasons">${newReasons}</ul>
            </div>
          </div>
          <div class="reproc-batch-actions">
            <button type="button" class="btn btn-ghost" data-rebatch-action="reject" data-token="${item.token}">Reject</button>
            <button type="button" class="btn btn-primary" data-rebatch-action="accept" data-token="${item.token}">Accept</button>
          </div>
        </div>
      `;
    }).join("");
    batchList.innerHTML = html;
    attachReprocessMagnifiers();
  }
  async function openBatchModal() {
    if (!batchModal) return;
    openBackdrop();
    batchModal.classList.remove("hidden");
    batchModal.classList.add("show");
    setBatchStatus("Reprocessing review items…");
    setProgressBar(batchOverallBar, 0);
    batchOverallProgressStop = null;
    batchList?.classList.add("hidden");
    reprocessBatchItems = [];
    try {
      let data = null;
      try {
        const res = await fetch('/api/reprocess/flagged?mode=async', {method:'POST'});
        data = await res.json();
      } catch (_err) {
        data = null;
      }
      if (!data || data.ok === false || (!data.job && !Array.isArray(data?.items))) {
        batchOverallProgressStop = startProgressRamp(batchOverallBar, 0.85, 0.0065, 160);
        data = await fetchWithFallback(
          ['/api/reprocess/flagged?mode=sync', '/api/reprocess/flagged'],
          {method:'POST'}
        );
      }
      if (!data || !data.ok) {
        throw new Error((data && data.error) || "Unable to start batch");
      }
      const flaggedCt = (typeof data.flagged === "number") ? data.flagged : null;
      const failedCt  = (typeof data.failed === "number") ? data.failed : null;
      const countsTxt = formatBatchCounts(flaggedCt, failedCt);
      const totalQueued = typeof data.total === "number" ? data.total : (Array.isArray(data?.items) ? data.items.length : 0);
      if (!totalQueued) {
        setBatchStatus("No flagged or failed cards to reprocess.");
        setProgressBar(batchOverallBar, 0);
        return;
      }
      if (countsTxt) {
        setBatchStatus(`Found ${totalQueued} cards (${countsTxt}). Starting…`);
      } else {
        setBatchStatus(`Found ${totalQueued} cards to reprocess. Starting…`);
      }
      if (data.job) {
        const total = data.total || 0;
        const jobData = await pollReprocessJob(data.job, (status)=>{
          const totalCards = status?.total || total || 1;
          const done = Math.min(totalCards, status?.done || 0);
          const overall = totalCards ? (done / totalCards) : 1;
          setProgressBar(batchOverallBar, overall);
          const phase = status?.current_phase ? String(status.current_phase).replace(/_/g," ") : "";
          const currentLabel = status?.current_card ? `Card #${status.current_card}` : "";
          const countsLabel = formatBatchCounts(
            typeof status?.flagged === "number" ? status.flagged : flaggedCt,
            typeof status?.failed === "number" ? status.failed : failedCt
          );
          const base = `Processing ${done}/${totalCards}`;
          const withCounts = countsLabel ? `${base} (${countsLabel})` : base;
          setBatchStatus(`${withCounts} ${currentLabel} ${phase}`.trim());
        }, 250);
        reprocessBatchItems = Array.isArray(jobData?.items) ? jobData.items : [];
      } else {
        // synchronous fallback - use ramp to hint activity
        batchOverallProgressStop = startProgressRamp(batchOverallBar, 0.85, 0.0065, 160);
        reprocessBatchItems = Array.isArray(data.items) ? data.items : [];
      }
      reprocessBatchJobId = data.job || null;
      if (!reprocessBatchItems.length) {
        setBatchStatus("No review items require reprocessing.");
      } else {
        const counts = formatBatchCounts(flaggedCt, failedCt);
        const reviewLabel = `Review ${reprocessBatchItems.length} updated card${reprocessBatchItems.length === 1 ? "" : "s"}.`;
        setBatchStatus(counts ? `${reviewLabel} (${counts})` : reviewLabel);
      }
      renderBatchList();
      setProgressBar(batchOverallBar, 1);
    } catch (err) {
      setBatchStatus("Batch reprocess failed: " + err.message);
      setProgressBar(batchOverallBar, 0);
    } finally {
      batchOverallProgressStop = stopIndeterminateBar(batchOverallProgressStop, null);
      reprocessBatchJobId = null;
    }
  }
  btnReprocessBatch?.addEventListener("click", (e)=>{
    e.preventDefault();
    openBatchModal();
  });
  batchClose?.addEventListener("click", closeBatchModal);
  batchDone?.addEventListener("click", closeBatchModal);
  batchModal?.addEventListener("click", (e)=>{
    if (e.target === batchModal || e.target.classList.contains("backdrop")) closeBatchModal();
  });
  window.addEventListener("keydown", (e)=>{
    if (e.key === "Escape" && batchModal?.classList.contains("show")) closeBatchModal();
  });
  batchList?.addEventListener("click", async (e)=>{
    const btn = e.target.closest("[data-rebatch-action]");
    if (!btn) return;
    const action = btn.getAttribute("data-rebatch-action");
    const token = btn.getAttribute("data-token");
    if (!token) return;
    if (action === "reject") {
      reprocessBatchItems = reprocessBatchItems.filter(it => it.token !== token);
      renderBatchList();
      return;
    }
    if (action === "accept") {
      btn.disabled = true;
      try {
        const item = reprocessBatchItems.find(it => it.token === token);
        if (!item) throw new Error("Token expired");
        const cardEl = btn.closest("[data-token]");
        const override = {};
        const nameInput = cardEl?.querySelector('[data-edit="name"]');
        const setInput = cardEl?.querySelector('[data-edit="set_hint"]');
        const numInput = cardEl?.querySelector('[data-edit="number"]');
        if (nameInput && nameInput.value.trim()) override.name = nameInput.value.trim();
        if (setInput && setInput.value.trim()) override.set_hint = setInput.value.trim();
        if (numInput && numInput.value.trim()) override.number = numInput.value.trim();
        const res = await fetch(`/api/scan/${item.id}/reprocess/apply`, {
          method: 'POST',
          headers: {'Content-Type':'application/json'},
          body: JSON.stringify({token, override})
        });
        const data = await res.json();
        if (!data || !data.ok) throw new Error((data && data.error) || "Apply failed");
        reprocessBatchItems = reprocessBatchItems.filter(it => it.token !== token);
        renderBatchList();
        fetchScans();
        fetchState();
        try { refreshHistoryGrid(); } catch {}
      } catch (err) {
        btn.disabled = false;
        setBatchStatus("Apply failed: " + err.message);
      }
    }
  });

  // ---- Quarantine Recovery ----
  function setQuarantineStatus(msg) {
    if (qStatusText) qStatusText.textContent = msg || "";
  }
  function fillQuarantineForm(item) {
    activeQuarantineId = item?.id || null;
    if (!item) {
      if (qId) qId.value = "";
      if (qName) qName.value = "";
      if (qSet) qSet.value = "";
      if (qNumber) qNumber.value = "";
      if (qStatus) qStatus.value = "pass";
      if (qMatch) qMatch.value = "";
      if (qFlagged) qFlagged.value = "false";
      if (qReasons) qReasons.value = "";
      setQuarantineStatus("");
      return;
    }
    if (qId) qId.value = item.id ?? "";
    if (qName) qName.value = item.meta?.name || "";
    if (qSet) qSet.value = (item.meta?.set_hint || "").toUpperCase();
    if (qNumber) qNumber.value = item.meta?.number || "";
    if (qStatus) qStatus.value = (item.meta?.status || "pass");
    if (qMatch) qMatch.value = (typeof item.meta?.match_score === "number") ? item.meta.match_score : "";
    if (qFlagged) qFlagged.value = item.meta?.flagged ? "true" : "false";
    if (qReasons) qReasons.value = Array.isArray(item.meta?.review_reasons) ? item.meta.review_reasons.join("\n") : "";
    setQuarantineStatus(item.parse_ok ? "Parsed ok. Edit or auto-fix to restore." : "Parse failed. Try auto-fix or edit fields then save.");
  }
  function renderQuarantineList() {
    if (!quarantineList) return;
    if (!quarantineItems.length) {
      quarantineList.innerHTML = '<div class="quarantine-item"><strong>No quarantined files.</strong></div>';
      fillQuarantineForm(null);
      return;
    }
    const html = quarantineItems.map(item => {
      const cls = item.id === activeQuarantineId ? "quarantine-item active" : "quarantine-item";
      const name = esc(item.meta?.name || "(unnamed)");
      const setTxt = esc((item.meta?.set_hint || "").toUpperCase());
      const num = esc(item.meta?.number || "");
      const status = esc(item.meta?.status || "");
      return `<div class="${cls}" data-qid="${item.id}">
        <h4>#${item.id} ${name}</h4>
        <div class="meta">${setTxt} ${num}</div>
        <div class="meta">Reason: ${esc(item.reason || "unknown")} • Status: ${status || "?"}</div>
      </div>`;
    }).join("");
    quarantineList.innerHTML = html;
  }
  async function loadQuarantineList() {
    try {
      const res = await fetch('/api/history/quarantine');
      const data = await res.json();
      if (!data || !data.ok) throw new Error((data && data.error) || "Unable to load quarantine");
      quarantineItems = Array.isArray(data.items) ? data.items : [];
      if (quarantineItems.length) {
        activeQuarantineId = quarantineItems[0].id;
        renderQuarantineList();
        fillQuarantineForm(quarantineItems[0]);
      } else {
        renderQuarantineList();
        fillQuarantineForm(null);
      }
    } catch (err) {
      setQuarantineStatus("Load failed: " + err.message);
    }
  }
  function openQuarantineModal() {
    if (!quarantineModal) return;
    openBackdrop();
    quarantineModal.classList.remove("hidden");
    quarantineModal.setAttribute("aria-hidden","false");
    loadQuarantineList();
  }
  function closeQuarantineModal() {
    if (!quarantineModal) return;
    quarantineModal.classList.add("hidden");
    quarantineModal.setAttribute("aria-hidden","true");
    activeQuarantineId = null;
    setQuarantineStatus("");
    hideBackdropIfNone();
  }
  quarantineList?.addEventListener("click", (e)=>{
    const card = e.target.closest("[data-qid]");
    if (!card) return;
    const id = Number(card.getAttribute("data-qid"));
    const item = quarantineItems.find(it => Number(it.id) === id);
    if (item) {
      activeQuarantineId = id;
      renderQuarantineList();
      fillQuarantineForm(item);
    }
  });
  btnQuarantine?.addEventListener("click", (e)=>{ e.preventDefault(); openQuarantineModal(); });
  quarantineClose?.addEventListener("click", closeQuarantineModal);
  qAutoFix?.addEventListener("click", async ()=>{
    if (!activeQuarantineId) return;
    setQuarantineStatus("Auto-fixing…");
    try {
      const res = await fetch(`/api/history/quarantine/${activeQuarantineId}/autofix`, {method:'POST'});
      const data = await res.json();
      if (!data || !data.ok) throw new Error((data && data.error) || "Auto-fix failed");
      quarantineItems = quarantineItems.filter(it => it.id !== activeQuarantineId);
      renderQuarantineList();
      fillQuarantineForm(quarantineItems[0] || null);
      fetchScans();
      fetchState();
      try { refreshHistoryGrid(); } catch {}
      setQuarantineStatus("Restored successfully.");
    } catch (err) {
      setQuarantineStatus("Auto-fix failed: " + err.message);
    }
  });
  qSave?.addEventListener("click", async ()=>{
    if (!activeQuarantineId) return;
    setQuarantineStatus("Saving…");
    const body = {
      entry: {
        name: qName?.value || "",
        set_hint: (qSet?.value || "").toLowerCase(),
        number: qNumber?.value || "",
        status: qStatus?.value || "pass",
        match_score: qMatch?.value ? Number(qMatch.value) : null,
        flagged: (qFlagged?.value === "true"),
        review_reasons: qReasons?.value ? qReasons.value.split("\n").map(s=>s.trim()).filter(Boolean) : [],
        match_ok: (qStatus?.value === "pass"),
      }
    };
    try {
      const res = await fetch(`/api/history/quarantine/${activeQuarantineId}/save`, {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify(body)
      });
      const data = await res.json();
      if (!data || !data.ok) throw new Error((data && data.error) || "Save failed");
      quarantineItems = quarantineItems.filter(it => it.id !== activeQuarantineId);
      renderQuarantineList();
      fillQuarantineForm(quarantineItems[0] || null);
      fetchScans();
      fetchState();
      try { refreshHistoryGrid(); } catch {}
      setQuarantineStatus("Restored successfully.");
    } catch (err) {
      setQuarantineStatus("Save failed: " + err.message);
    }
  });
  qDelete?.addEventListener("click", async ()=>{
    if (!activeQuarantineId) return;
    if (!confirm("Delete quarantined files for card #" + activeQuarantineId + "?")) return;
    setQuarantineStatus("Deleting…");
    try {
      const res = await fetch(`/api/history/quarantine/${activeQuarantineId}`, {method:'DELETE'});
      const data = await res.json();
      if (!data || !data.ok) throw new Error((data && data.error) || "Delete failed");
      quarantineItems = quarantineItems.filter(it => it.id !== activeQuarantineId);
      renderQuarantineList();
      fillQuarantineForm(quarantineItems[0] || null);
      setQuarantineStatus("Deleted.");
    } catch (err) {
      setQuarantineStatus("Delete failed: " + err.message);
    }
  });

  // ---- All History (fullscreen grid) ----
  function ensureHistoryModals() {
    if (!document.getElementById('history-modal')) {
      const tpl = document.createElement('div');
      tpl.innerHTML = `
      <div id="history-modal" class="mm-modal hidden history-popup" aria-hidden="true"
        style="position:fixed;top:20px;bottom:20px;left:50%;transform:translate(-50%,0);width:min(1200px,calc(100vw - 40px));max-width:calc(100vw - 40px);height:calc(100vh - 40px);max-height:calc(100vh - 40px);display:flex;flex-direction:column;overflow:hidden;z-index:100;">
        <div class="mm-modal__header" style="position:sticky;top:0;z-index:1;">
          <div class="mm-title">All History</div><button class="btn btn-pill" id="history-filter">Filters…</button>
          <button class="btn btn-pill btn-ghost" id="history-close">Close</button>
        </div>
        <div class="mm-modal__body" style="flex:1 1 auto;min-height:0;overflow:auto;padding:16px;">
          <div id="history-grid" class="history-grid history-grid-inline" aria-label="All history grid"></div>
        </div>
      </div>
      <div id="history-options-modal" class="mm-modal hidden" style="max-width:420px;z-index:120;" aria-hidden="true">
        <div class="mm-modal__header">
          <div class="mm-title">Choose Action</div>
          <button class="btn btn-pill btn-ghost" id="histopt-close">Close</button>
        </div>
        <div class="mm-modal__body">
          <div class="mm-grid" style="grid-template-columns: 1fr;">
            <button class="btn btn-pill btn-primary" id="histopt-pass">Mark Pass</button>
            <button class="btn btn-pill"              id="histopt-fail">Mark Fail</button>
            <button class="btn btn-pill"              id="histopt-edit">Load Data</button>
            <button class="btn btn-pill"              id="histopt-reprocess">Reprocess…</button>
            <button class="btn btn-pill btn-ghost"    id="histopt-delete">Delete</button>
          </div>
        </div>
      </div>`;
      document.body.appendChild(tpl);
    }
  }

  const btnOpenHistory = document.getElementById("btn-open-history");
  const filterBackdrop = document.getElementById("filter-backdrop");

  let histSelectedId = null;
  function openBackdrop() { filterBackdrop?.classList.remove("hidden"); }
  function _anyModalOpen() {
    // Detect any visible modal (legacy .modal or newer .mm-modal)
    const selector = ".mm-modal:not(.hidden):not([aria-hidden=\"true\"]), .modal.show:not(.hidden):not([aria-hidden=\"true\"]), .modal:not(.hidden)[aria-modal=\"true\"]";
    return !!document.querySelector(selector);
  }
  function hideBackdropIfNone() {
    requestAnimationFrame(()=>{ if (!_anyModalOpen()) filterBackdrop?.classList.add("hidden"); });
  }
  function closeHistoryModal() {
    const modal = document.getElementById("history-modal");
    const opts = document.getElementById("history-options-modal");
    modal?.classList.add("hidden");
    opts?.classList.add("hidden");
    hideBackdropIfNone();
  }

  function openBackupModal() {
    if (!backupModal) return;
    openBackdrop();
    backupModal.classList.remove("hidden");
    backupModal.setAttribute("aria-hidden", "false");
    if (backupExportStatus) backupExportStatus.textContent = "";
    if (backupImportStatus) backupImportStatus.textContent = "";
  }
  function closeBackupModal() {
    if (!backupModal) return;
    backupModal.classList.add("hidden");
    backupModal.setAttribute("aria-hidden", "true");
    hideBackdropIfNone();
  }

  async function runBackupExport() {
    if (!backupExportStatus) return;
    backupExportStatus.textContent = "Exporting…";
    try {
      const res = await fetch("/api/history/export/full", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ compact: !!backupCompact?.checked, compact_images: !!backupCompact?.checked })
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok || !data?.ok) {
        throw new Error(data?.error || "Export failed");
      }
      backupExportStatus.textContent = `Export ready (${data.count || 0} items). Downloading…`;
      if (data.url) {
        window.location.href = data.url;
      }
    } catch (err) {
      backupExportStatus.textContent = `Export failed: ${err.message}`;
    }
  }

  async function runBackupImport() {
    if (!backupImportStatus) return;
    const file = backupFile?.files?.[0];
    if (!file) {
      backupImportStatus.textContent = "Choose a .tar.gz file to import.";
      return;
    }
    backupImportStatus.textContent = "Importing…";
    try {
      const fd = new FormData();
      fd.append("file", file);
      fd.append("mode", backupMode?.value || "replace");
      fd.append("reassign_on_conflict", backupReassign?.checked ? "true" : "false");
      const res = await fetch("/api/history/import", { method: "POST", body: fd });
      const data = await res.json().catch(() => ({}));
      if (!res.ok || !data?.ok) {
        throw new Error(data?.error || "Import failed");
      }
      backupImportStatus.textContent = `Imported ${data.imported || 0} items (${data.mode}).`;
      fetchScans();
      try { refreshHistoryGrid(); } catch (e) {}
    } catch (err) {
      backupImportStatus.textContent = `Import failed: ${err.message}`;
    }
  }

  function renderHistoryGrid(items) {
    const grid = document.getElementById("history-grid");
    if (!grid) return;
    grid.classList.add("history-grid-inline");
    Object.assign(grid.style, {
      display: "grid",
      gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))",
      gap: "10px",
    });
    rememberScanMeta(items);
    grid.innerHTML = items.map(function (it) {
      const st = String(it.status || "fail").toLowerCase();
      const cls = (st === "pass" ? "ok" : (st === "review" ? "warn" : "bad"));
      const score = Number(it.match_score || 0);
      const numStr = it.number ? `#${esc(it.number)}` : "";
      const setLine = it.scry_set
        ? `${esc(String(it.scry_set).toUpperCase())} ${esc(it.scry_cn || "")}`
        : (it.set_hint ? `${esc(String(it.set_hint).toUpperCase())}` : "");
      const scryName = it.scry_name ? `→ ${esc(it.scry_name)}` : "";
      const warn = (it.flagged && Array.isArray(it.review_reasons) && it.review_reasons.length)
        ? `<span class="pill warn" title="${(it.review_reasons||[]).join(' • ').replace(/"/g,'&quot;')}">Review Req.</span>`
        : "";
      const thumbHtml = it.thumb_ok
        ? `<img class="thumb" loading="lazy" src="/api/scan/${it.id}/thumb" alt="scan ${it.id}">`
        : '<div class="thumb thumb-missing" aria-label="No snapshot">No Image</div>';
      const metaLine = [
        `${(score*100).toFixed(1)}%`,
        setLine
      ].filter(Boolean).join(" • ");
      return `
        <div class="hcard" data-id="${it.id}">
          ${thumbHtml}
          <div class="row">
            <div class="title">${esc(it.name || "(unnamed)")} ${numStr ? "• " + numStr : ""}</div>
            <span class="pill ${cls}">${String((it.status || "FAIL")).toUpperCase()}</span>
          </div>
          <div class="meta">
            <span>${metaLine}</span>
            ${scryName ? `<span class="meta-scry">${scryName}</span>` : ""}
            ${warn}
          </div>
        </div>
      `;
    }).join("");
    // Card click => options modal
    grid.querySelectorAll(".hcard").forEach(function(card){
      card.addEventListener("click", function(){
        histSelectedId = Number(card.getAttribute("data-id"));
        document.getElementById("history-options-modal")?.classList.remove("hidden");
        openBackdrop();
      });
    });
    attachThumbFallbacks(grid);
  }

  
  function isHistoryModalOpen() {
    const m = document.getElementById("history-modal");
    return !!(m && !m.classList.contains("hidden"));
  }
  function refreshHistoryGrid() {
    if (!isHistoryModalOpen()) return;
    const baseQS = historyModalQueryQS || buildHistoryQuery();
    const qs = baseQS ? baseQS + '&' : '';
    fetch('/api/scans?' + qs + 'ts=' + Date.now())
      .then(r => r.json())
      .then(d => renderHistoryGrid((d && d.items) || []));
  }


  function applyFilterFieldsFromQS(qsStr) {
    try {
      const params = new URLSearchParams(qsStr || '');
      const setIf = (id, val) => { const el = document.getElementById(id); if (el && val !== null) el.value = val; };
      const statusVal = params.get('status');
      setIf('hf-status',  (statusVal === 'flagged' ? 'review' : statusVal) ?? 'all');
      setIf('hf-scoremin',params.get('score_min')?? '');
      setIf('hf-since',   params.get('since')    ?? '');
      setIf('hf-q',       params.get('q')        ?? '');
      setIf('hf-set',     params.get('set')      ?? '');
      setIf('hf-foil',    params.get('foil')     ?? '');
      setIf('hf-sortby',  params.get('sort_by')  ?? 'ts');
      setIf('hf-sortdir', params.get('sort_dir') ?? 'desc');
    } catch(e) {}
  }

function openHistoryModal() {
    ensureHistoryModals();
    const modal = document.getElementById("history-modal");
    const closeBtn = document.getElementById("history-close");
    // Bind close once
    
    const hfBtn = document.getElementById("history-filter");
    if (hfBtn && !hfBtn._bound) {
      hfBtn.addEventListener("click", function(){
        // reuse the existing filter modal
        filterContext = 'history'; applyFilterFieldsFromQS(historyModalQueryQS); openFilter?.();
      });
      hfBtn._bound = true;
    }
if (!closeBtn._boundClose) {
      closeBtn.addEventListener("click", closeHistoryModal);
      closeBtn._boundClose = true;
    }
    openBackdrop();
    modal?.classList.remove("hidden");
    const baseQS = historyModalQueryQS || buildHistoryQuery();
    const qs = baseQS ? baseQS + '&' : '';
    fetch('/api/scans?' + qs + 'ts=' + Date.now())
      .then(r => r.json())
      .then(d => renderHistoryGrid((d && d.items) || []));
  }

  // Options modal handlers (create on demand too)
  function bindHistoryOptionHandlers() {
    const passBtn = document.getElementById("histopt-pass");
    const failBtn = document.getElementById("histopt-fail");
    const reviewBtn = document.getElementById("histopt-review");
    const editBtn = document.getElementById("histopt-edit");
    const reproBtn= document.getElementById("histopt-reprocess");
    const delBtn  = document.getElementById("histopt-delete");
    const optClose= document.getElementById("histopt-close");
    const optModal= document.getElementById("history-options-modal");

    function closeOpt(){ optModal?.classList.add("hidden"); hideBackdropIfNone(); }

    if (!passBtn._bound) {
      passBtn.addEventListener("click", function(){
        if (!histSelectedId) return;
        fetch('/api/scan/'+histSelectedId+'/status', {
          method:'POST', headers:{'Content-Type':'application/json'},
          body: JSON.stringify({status:'pass'})
        }).then(()=>{ closeOpt(); fetchScans(); openHistoryModal(); });
      });
      passBtn._bound = true;
    }
    if (!failBtn._bound) {
      failBtn.addEventListener("click", function(){
        if (!histSelectedId) return;
        fetch('/api/scan/'+histSelectedId+'/status', {
          method:'POST', headers:{'Content-Type':'application/json'},
          body: JSON.stringify({status:'fail'})
        }).then(()=>{ closeOpt(); fetchScans(); openHistoryModal(); });
      });
      failBtn._bound = true;
    }
    if (reviewBtn && !reviewBtn._bound) {
      reviewBtn.addEventListener("click", function(){
        if (!histSelectedId) return;
        fetch('/api/scan/'+histSelectedId+'/status', {
          method:'POST', headers:{'Content-Type':'application/json'},
          body: JSON.stringify({status:'review'})
        }).then(()=>{ closeOpt(); fetchScans(); openHistoryModal(); });
      });
      reviewBtn._bound = true;
    }
    if (!delBtn._bound) {
      delBtn.addEventListener("click", function(){
        if (!histSelectedId) return;
        fetch('/api/scan/'+histSelectedId+'/delete', {method:'POST'})
          .then(()=>{ closeOpt(); fetchScans(); openHistoryModal(); });
      });
      delBtn._bound = true;
    }
    if (!editBtn._bound) {
      editBtn.addEventListener("click", async function(){
        if (!histSelectedId) return;
        if (openManualEditById(histSelectedId)) {
          closeOpt();
          document.getElementById("history-modal")?.classList.add("hidden");
          return;
        }
        try {
          const resp = await fetch(`/api/scan/${histSelectedId}/load`, {method:'POST'});
          const res = await resp.json().catch(() => ({}));
          if (!resp.ok || !res?.ok) {
            alert(res?.error || "Unable to load this scan right now (scanner busy?).");
            return;
          }
          currentLoadedId = histSelectedId;
          document.getElementById("history-modal")?.classList.add("hidden");
          closeOpt();
          fetchState();
          setTimeout(()=>openEditModal(), 80);
        } catch (err) {
          console.error("History load failed", err);
        }
      });
      editBtn._bound = true;
    }
    if (reproBtn && !reproBtn._bound) {
      reproBtn.addEventListener("click", function(){
        if (!histSelectedId) return;
        closeOpt();
        document.getElementById("history-modal")?.classList.add("hidden");
        openReprocessModalForScan(histSelectedId);
      });
      reproBtn._bound = true;
    }
    if (!optClose._bound) {
      optClose.addEventListener("click", closeOpt);
      optClose._bound = true;
    }
  }

  // Hook up the button (works even if button is added later)
  (function attachHistoryButtonObserver(){
    function bind(){
      const btn = document.getElementById("btn-open-history");
      if (!btn || btn._bound) return;
      btn.addEventListener("click", function(){
        ensureHistoryModals();
        bindHistoryOptionHandlers();
        openHistoryModal();
      });
      btn._bound = true;
    }
    // Try now
    bind();
    // And observe future mutations (in case toolbar is dynamically replaced)
    const obs = new MutationObserver(bind);
    obs.observe(document.body, {subtree:true, childList:true});
  })();


  if (btnClose) btnClose.onclick = closeEditModal;
  modal?.addEventListener("click", (e) => { if (e.target === modal || e.target.classList.contains("backdrop")) closeEditModal(); });
  window.addEventListener("keydown", (e) => { if (e.key === "Escape" && modal?.classList.contains("show")) closeEditModal(); });



  if (eiSearch) eiSearch.onclick = function () {
    const qs = new URLSearchParams({ name: eiName.value || "", number: eiNum.value || "", set: eiSet.value || "" });
    fetch("/api/scrysearch?" + qs.toString())
      .then(r => r.json())
      .then(res => {
        const rows = (res.results || []).map(c => (
          '<div class="revitem" data-scryid="'+c.id+'">' +
            (c.image ? '<img class="thumb" loading="lazy" src="'+c.image+'">' : '') +
            '<div style="flex:1;min-width:0;display:flex;flex-direction:column;gap:4px;">' +
              '<div><b>'+c.name+'</b></div>' +
              '<div class="meta"><span>'+String(c.set||"").toUpperCase()+' #'+(c.collector_number||'')+'</span> <span>'+(c.set_name||'')+'</span></div>' +
            '</div>' +
            '<button class="useBtn">Use</button>' +
          '</div>'
        )).join("");
        eiResults.innerHTML = rows || '<div class="pill warn">No results</div>';
        Array.from(eiResults.querySelectorAll(".useBtn")).forEach(btn => {
          btn.addEventListener("click", function(){
            const row = btn.closest(".revitem");
            chosenScryId = row.getAttribute("data-scryid");
            Array.from(eiResults.children).forEach(ch => ch.style.outline = "");
            row.style.outline = "2px solid #4a9cff";
            const meta = row.querySelector(".meta span")?.textContent || "";
            const parts = meta.split(" ");
            if (parts.length >= 1) eiSet.value = (parts[0] || "").toLowerCase();
            if (meta.includes("#")) eiNum.value = meta.split("#").pop();
          });
        });
      });
  };
  if (eiApply) eiApply.onclick = function () {
    const targetId = activeEditId || currentLoadedId;
    if (!targetId) return;
    const payload = { name: eiName.value || "", number: eiNum.value || "", set: (eiSet.value || "").toLowerCase(), scry_id: chosenScryId || null };
    fetch("/api/scan/" + targetId + "/edit", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload)
    })
    .then(r => r.json())
    .then(res => {
      if (res && res.ok) {
        refreshCompare(Date.now());
        fetchState();
        fetchScans();
        activeEditId = null;
        closeEditModal();
      }
    });
  };

  // ===== Export CSV dialog =====
const exportModal = document.getElementById('export-modal');
const exportClose = document.getElementById('export-close');
let exportCardSphereMode = false;
function openExport(){ 
  // build checkboxes each time (so default states reset nicely)
  buildExportCheckboxes();
  openBackdrop();
  exportModal?.classList.remove('hidden'); 
  exportModal?.setAttribute('aria-hidden','false');
}
function closeExport(){ 
  exportModal?.classList.add('hidden'); 
  exportModal?.setAttribute('aria-hidden','true');
  hideBackdropIfNone();
}
exportClose?.addEventListener('click', closeExport);

// Export field sets
const CARDSPHERE_FIELDS = ["Name","Set","Quantity","Foil","Condition","Language"];
const BASE_FIELDS = [
  "Quantity","Name","Foil","Condition","Date Added","Language","Purchase Price","Tags",
  "Edition Name","Edition Code","Multiverse Id","Scryfall ID","MTGO ID",
  "Collector Number","Mana Value","Colors","Identities","Mana cost",
  "Types","Sub-types","Super-types","Rarity",
  "Price (Card Kingdom)","Price (TCG Player)","Price (Star City Games)","Price (Card Hoarder)","Price (Card Market)",
  "Scryfall Oracle ID"
];
const ALL_FIELDS = [...new Set([...BASE_FIELDS, ...CARDSPHERE_FIELDS])];
const DEFAULT_FIELDS = [
  "Quantity","Name","Foil","Condition","Date Added","Language",
  "Edition Name","Edition Code","Collector Number",
  "Scryfall ID","Scryfall Oracle ID",
  "Mana Value","Identities","Mana cost","Types","Sub-types","Super-types","Rarity"
];
let exportFieldOrder = DEFAULT_FIELDS.slice();

// Example row (all fields) to preview the CSV output dynamically.
const EXPORT_EXAMPLE_VALUES = {
  "Quantity": "1",
  "Name": "Kithkin Billyrider",
  "Foil": "false",
  "Condition": "NM",
  "Conditions": "NM",
  "Date Added": "2025-12-07",
  "Language": "English",
  "Languages": "EN",
  "Purchase Price": "0.15",
  "Tags": "bulk",
  "Set": "MOM",
  "Edition Name": "March of the Machine",
  "Edition Code": "MOM",
  "Multiverse Id": "607041",
  "Scryfall ID": "0535b69f-247d-49c9-97e1-d988700578ab",
  "MTGO ID": "109854",
  "Collector Number": "24",
  "Mana Value": "3.0",
  "Colors": "W",
  "Identities": "W",
  "Mana cost": "{2}{W}",
  "Types": "Creature",
  "Sub-types": "Kithkin Knight",
  "Super-types": "None",
  "Rarity": "Common",
  "Price (Card Kingdom)": "0.05",
  "Price (TCG Player)": "0.04",
  "Price (Star City Games)": "0.04",
  "Price (Card Hoarder)": "0.03",
  "Price (Card Market)": "0.12",
  "Scryfall Oracle ID": "142d2063-d692-42a6-b828-42d998914589"
};
const exportExampleBox = document.getElementById('export-example');
const exportExampleText = document.getElementById('export-example-text');

function buildExportCheckboxes(){
  const wrap = document.getElementById('export-field-list');
  if (!wrap) return;
  exportCardSphereMode = false;
  wrap.innerHTML = ALL_FIELDS.map(h =>
    `<label class="mm-check"><input type="checkbox" data-f="${h}" ${DEFAULT_FIELDS.includes(h)?'checked':''}> ${h}</label>`
  ).join('');
  exportFieldOrder = DEFAULT_FIELDS.slice();
  bindExportCheckboxListeners();
  renderExportExample();
}
function setExportFields(fields){
  exportCardSphereMode = fields.length === CARDSPHERE_FIELDS.length && fields.every((f, i) => CARDSPHERE_FIELDS[i] === f);
  exportFieldOrder = fields.slice();
  document.querySelectorAll('#export-field-list input[type=checkbox]').forEach(cb => {
    cb.checked = fields.includes(cb.dataset.f);
  });
  renderExportExample();
}
function selectedFields(){
  const checked = Array.from(document.querySelectorAll('#export-field-list input[type=checkbox]'))
    .filter(cb => cb.checked).map(cb => cb.dataset.f);
  if (exportFieldOrder && exportFieldOrder.length) {
    const ordered = exportFieldOrder.filter(f => checked.includes(f));
    const extras = checked.filter(f => !exportFieldOrder.includes(f));
    return ordered.concat(extras);
  }
  return checked;
}
function bindExportCheckboxListeners(){
  document.querySelectorAll('#export-field-list input[type=checkbox]').forEach(cb => {
    cb.addEventListener('change', renderExportExample);
  });
}
function renderExportExample(){
  if (!exportExampleText) return;
  const fields = selectedFields();
  if (!fields.length) {
    exportExampleText.textContent = 'Select at least one field to preview the CSV row.';
    exportExampleBox?.classList.add('muted');
    return;
  }
  exportExampleBox?.classList.remove('muted');
  const row = fields.map(f => (EXPORT_EXAMPLE_VALUES[f] ?? "")).join(',');
  exportExampleText.textContent = row;
}

window.addEventListener('DOMContentLoaded', () => {
  const chip = document.getElementById('version-chip');
  if (!chip) return;

  fetch('/api/version')
    .then(r => r.json())
    .then(info => {
      const cur = info.current || '0.0.0';
      const latest = info.latest || cur;
      if (info.is_update) {
        chip.style.background = 'rgba(255,185,0,.9)';
        chip.style.color = '#000';
        chip.innerHTML = info.html_url
          ? `<a href="${info.html_url}" target="_blank" rel="noopener">v${cur} → v${latest}</a>`
          : `v${cur} → v${latest}`;
      } else {
        chip.textContent = `v${cur}`;
      }
    })
    .catch(() => { chip.textContent = 'v0.0.0'; });
});


// modal buttons
document.getElementById('export-selectall')?.addEventListener('click', ()=> setExportFields(ALL_FIELDS));
document.getElementById('export-clear')?.addEventListener('click', ()=> setExportFields([]));
document.getElementById('export-defaults')?.addEventListener('click', ()=> setExportFields(DEFAULT_FIELDS));
document.getElementById('export-cardsphere')?.addEventListener('click', ()=> setExportFields(CARDSPHERE_FIELDS));

// open modal from toolbar button
document.getElementById('btn-export-csv')?.addEventListener('click', openExport);

// do the export
document.getElementById('export-go')?.addEventListener('click', ()=>{
  const fields = selectedFields();
  const qs = new URLSearchParams(historyQueryQS || buildHistoryQuery());
  qs.set('fmt','csv');
  if (fields.length) qs.set('fields', fields.join(','));
  if (exportCardSphereMode) qs.set('cardsphere_import','1');
  // NOTE: we export using whatever filters are set; decklist export still forces pass-only
  window.location.href = '/api/scans/export/download?' + qs.toString();
  closeExport();
});

// Backup/restore
btnBackup?.addEventListener('click', openBackupModal);
backupClose?.addEventListener('click', closeBackupModal);
backupExport?.addEventListener('click', runBackupExport);
backupImport?.addEventListener('click', runBackupImport);

  // ---------- boot ----------
  function boot() {
    loadManualCrop();
    fetchState();
    fetchScans();
    setInterval(fetchState, 300);
    setInterval(fetchScans, 900);
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();

  const sh = document.getElementById("showbad");
  if (sh) sh.onclick = function (e) { e.preventDefault(); fetchBad(); };
})();

// ===== Simple macro wiring (override via capture) =====
(function(){
  // Helper to send a macro
  async function sendMacro(name, params) {
    const body = { name };
    if (params) body.params = params;
    const r = await fetch('/api/printer/macro', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body) });
    try { return await r.json(); } catch { return null; }
  }

  // HOME -> SORTER_HOME
  const homeBtn = document.getElementById('btnHomePrep');
  if (homeBtn) {
    homeBtn.addEventListener('click', async (e) => {
      e.stopImmediatePropagation(); e.preventDefault();
      const d = await sendMacro('SORTER_HOME');
      if (d?.ok) log('t', 'SORTER_HOME sent');
      else log('t', 'Home failed: ' + (d?.error || 'unknown'));
    }, true); // capture
  }
  // START -> modal (COUNT only) -> RUN_SORTER_INTERACTIVE
  const startBtn = document.getElementById('btnStartRun');
  const startModal = document.getElementById('start-modal');
  const startClose = document.getElementById('start-close');
  const startCancel= document.getElementById('start-cancel');
  const startGo    = document.getElementById('start-go');
  const startCount = document.getElementById('start-count');

  function openStart() { startModal?.classList.remove('hidden'); }
  function closeStart() { startModal?.classList.add('hidden'); }

  if (startBtn) startBtn.addEventListener('click', (e)=>{ e.stopImmediatePropagation(); e.preventDefault(); openStart(); }, true);
  if (startClose) startClose.addEventListener('click', (e)=>{ e.stopImmediatePropagation(); e.preventDefault(); closeStart(); }, true);
  if (startCancel) startCancel.addEventListener('click', (e)=>{ e.stopImmediatePropagation(); e.preventDefault(); closeStart(); }, true);
  if (startModal) startModal.addEventListener('click', (e)=>{ if (e.target === startModal){ e.stopImmediatePropagation(); e.preventDefault(); closeStart(); } }, true);

  if (startGo) startGo.addEventListener('click', async (e) => {
    e.stopImmediatePropagation(); e.preventDefault();
    const n = Math.max(1, parseInt(startCount?.value || "1", 10));
    const d = await sendMacro('RUN_SORTER_INTERACTIVE', { COUNT: n });
    if (d?.ok) log('t', 'RUN_SORTER_INTERACTIVE COUNT=' + n);
    else log('t', 'Start failed: ' + (d?.error || 'unknown'));
    closeStart();
  }, true);
})();


// ===== Printer Controls modal wiring =====
(function(){
  const btn       = document.getElementById('btn-open-printer');
  const modal     = document.getElementById('printer-modal');
  const close1    = document.getElementById('printer-close');
  const close2    = document.getElementById('printer-close2');
  const backdrop  = document.getElementById('filter-backdrop') || document.querySelector('.backdrop');

  function openPrinter(){
    if (!modal) return;
    modal.classList.remove('hidden');
    if (backdrop && backdrop.classList) backdrop.classList.remove('hidden');
    try { modalOpen = true; } catch {}
    loadMacros();
    restorePrefs();
  }
  function closePrinter(){
    if (!modal) return;
    modal.classList.add('hidden');
    if (backdrop && backdrop.classList) backdrop.classList.add('hidden');
    try { modalOpen = false; } catch {}
  }

  btn && btn.addEventListener('click', (e)=>{ e.preventDefault(); e.stopImmediatePropagation(); openPrinter(); }, true);
  close1 && close1.addEventListener('click', (e)=>{ e.preventDefault(); e.stopImmediatePropagation(); closePrinter(); }, true);
  close2 && close2.addEventListener('click', (e)=>{ e.preventDefault(); e.stopImmediatePropagation(); closePrinter(); }, true);
  if (backdrop && !backdrop.id) { backdrop.addEventListener('click', (e)=>{ closePrinter(); }, true); }
  document.addEventListener('keydown', (e)=>{ if (e.key === 'Escape') closePrinter(); });

  async function loadMacros(){
    try {
      const r = await fetch('/api/printer/macros', {cache:'no-store'});
      if (!r.ok) return;
      const d = await r.json();
      const sel = document.getElementById('macro-list');
      if (!sel) return;
      sel.innerHTML='';
      (d.macros || []).forEach(n => {
        const opt = document.createElement('option');
        opt.value = n; opt.textContent = n; sel.appendChild(opt);
      });
    } catch {}
  }

  function setStepActive(val){
    document.querySelectorAll('#printer-modal .pc-step').forEach(b => {
      if (b.getAttribute('data-step') === String(val)) b.classList.add('active');
      else b.classList.remove('active');
    });
  }
  function parseStep(){
    const active = document.querySelector('#printer-modal .pc-step.active');
    const v = active ? parseFloat(active.getAttribute('data-step')||'1') : 1;
    return isFinite(v) ? Math.max(0.01, v) : 1;
  }
  document.querySelectorAll('#printer-modal .pc-step').forEach(b => {
    b.addEventListener('click', (e)=>{
      e.preventDefault();
      const v = b.getAttribute('data-step');
      setStepActive(v);
      try { localStorage.setItem('pcStep', String(v)); } catch {}
    });
  });

  function setSpeedActive(val){
    document.querySelectorAll('#printer-modal .pc-speed').forEach(b => {
      if (b.getAttribute('data-speed') === String(val)) b.classList.add('active');
      else b.classList.remove('active');
    });
    const inp = document.getElementById('pc-speed-input');
    if (inp) inp.value = String(val);
  }
  function parseSpeed(){
    const active = document.querySelector('#printer-modal .pc-speed.active');
    if (active){
      const v = parseFloat(active.getAttribute('data-speed')||'6000');
      return isFinite(v) ? Math.max(1, v) : 6000;
    }
    const inp = document.getElementById('pc-speed-input');
    const v = inp ? parseFloat(inp.value||'6000') : 6000;
    return isFinite(v) ? Math.max(1, v) : 6000;
  }
  document.querySelectorAll('#printer-modal .pc-speed').forEach(b => {
    b.addEventListener('click', (e)=>{
      e.preventDefault();
      const v = b.getAttribute('data-speed');
      setSpeedActive(v);
      try { localStorage.setItem('pcSpeed', String(v)); } catch {}
    });
  });
  const speedInput = document.getElementById('pc-speed-input');
  if (speedInput){
    speedInput.addEventListener('change', ()=>{
      document.querySelectorAll('#printer-modal .pc-speed').forEach(b=>b.classList.remove('active'));
      try { localStorage.setItem('pcSpeed', String(speedInput.value||'6000')); } catch {}
    });
  }
  function restorePrefs(){
    try { setStepActive(localStorage.getItem('pcStep') || '1'); } catch { setStepActive('1'); }
    try {
      const s = localStorage.getItem('pcSpeed');
      if (s) setSpeedActive(s); else setSpeedActive(6000);
    } catch { setSpeedActive(6000); }
  }

  async function jog(dx,dy,dz){
    try {
      const body = {};
      if (typeof dx === 'number') body.x = dx;
      if (typeof dy === 'number') body.y = dy;
      if (typeof dz === 'number') body.z = dz;
      body.f = parseSpeed();
      const r = await fetch('/api/printer/move', {
        method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)
      });
      const d = await r.json().catch(()=>null);
      if (!d || !d.ok) { try { log('t', 'Jog failed: ' + (d && d.error || 'unknown')); } catch {} }
    } catch {}
  }
  document.querySelectorAll('#printer-modal [data-jog]').forEach(el => {
    el.addEventListener('click', (e)=>{
      e.preventDefault(); e.stopImmediatePropagation();
      const step = parseStep();
      const spec = (el.getAttribute('data-jog')||'').toLowerCase();
      const m = spec.match(/([xyz]):([+-]?\d+(?:\.\d+)?)/);
      if (!m) return;
      const axis = m[1], dir = parseFloat(m[2]) || 0;
      const dx = axis==='x' ? dir*step : undefined;
      const dy = axis==='y' ? dir*step : undefined;
      const dz = axis==='z' ? dir*step : undefined;
      jog(dx,dy,dz);
    });
  });

  async function home(axes){
    try {
      const r = await fetch('/api/printer/home', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({axes, safe:true})});
      const d = await r.json().catch(()=>null);
      if (d && d.ok) { try { log('t', 'Homing ' + axes); } catch {} }
      else { try { log('t', 'Home failed: ' + (d && d.error || 'unknown')); } catch {} }
    } catch {}
  }
  document.getElementById('home-all')?.addEventListener('click', (e)=>{ e.preventDefault(); home('XYZ'); });
  document.getElementById('home-xy')?.addEventListener('click', (e)=>{ e.preventDefault(); home('XY'); });
  document.getElementById('home-x')?.addEventListener('click', (e)=>{ e.preventDefault(); home('X'); });
  document.getElementById('home-y')?.addEventListener('click', (e)=>{ e.preventDefault(); home('Y'); });
  document.getElementById('home-z')?.addEventListener('click', (e)=>{ e.preventDefault(); home('Z'); });

  document.getElementById('macro-run')?.addEventListener('click', async (e)=>{
    e.preventDefault(); e.stopImmediatePropagation();
    const sel = document.getElementById('macro-list');
    const paramsLine = (document.getElementById('macro-params')||{}).value || '';
    const name = sel && sel.value;
    if (!name) return;
    const params = {};
    (paramsLine.trim().split(/\s+/).filter(Boolean)).forEach(tok => {
      const i = tok.indexOf('=');
      if (i>0){
        const k = tok.slice(0,i).trim();
        let v = tok.slice(i+1).trim();
        if (/^-?\d+(?:\.\d+)?$/.test(v)) v = Number(v);
        params[k] = v;
      }
    });
    const body = Object.keys(params).length ? {name, params} : {name};
    try {
      const r = await fetch('/api/printer/macro', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
      const d = await r.json().catch(()=>null);
      if (d && d.ok) { try { log('t', 'Macro sent: ' + (d.cmd || name)); } catch {} }
      else { try { log('t', 'Macro failed: ' + (d && d.error || 'unknown')); } catch {} }
    } catch {}
  });

  document.getElementById('gcode-send')?.addEventListener('click', async (e)=>{
    e.preventDefault(); e.stopImmediatePropagation();
    const line = (document.getElementById('gcode-line')||{}).value || '';
    if (!line.trim()) return;
    try {
      const r = await fetch('/api/printer/gcode', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({script: line})});
      const d = await r.json().catch(()=>null);
      if (d && d.ok) { try { log('t', 'G-code sent: ' + line); } catch {} }
      else { try { log('t', 'G-code failed: ' + (d && d.error || 'unknown')); } catch {} }
    } catch {}
  });

  async function doEstop(){ try { await fetch('/api/printer/estop', {method:'POST'}); try { log('t','EMERGENCY STOP sent'); } catch {} } catch {} }
  document.getElementById('printer-estop')?.addEventListener('click', (e)=>{ e.preventDefault(); e.stopImmediatePropagation(); doEstop(); });
  if (!window.estop) { window.estop = doEstop; }
})();
  async function pollReprocessJob(jobId, onUpdate, intervalMs=600) {
    if (!jobId) throw new Error("job missing");
    while (true) {
      const res = await fetch(`/api/reprocess/job/${jobId}?ts=${Date.now()}`);
      const data = await res.json();
      if (!data || !data.ok) throw new Error((data && data.error) || "Job lookup failed");
      onUpdate?.(data);
      if (data.status === "done") return data;
      if (data.status === "error") throw new Error(data.error || "Job failed");
      await new Promise(r => setTimeout(r, intervalMs));
    }
  }
