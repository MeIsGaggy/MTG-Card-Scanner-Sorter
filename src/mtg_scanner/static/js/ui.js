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

  // ROI overlay controls
  const roiBox  = document.getElementById("roiOverlay");
  const roiReset= document.getElementById("roiReset");
  const roiRead = document.getElementById("roiReadout");

  // modal
  const btnEdit   = document.getElementById("btnEdit");
  const modal     = document.getElementById("editModal");
  const btnClose  = document.getElementById("editClose");
  const eiName    = document.getElementById("eiName");
  const eiSet     = document.getElementById("eiSet");
  const eiNum     = document.getElementById("eiNum");
  const eiSearch  = document.getElementById("eiSearch");
  const eiApply   = document.getElementById("eiApply");
  const eiResults = document.getElementById("eiResults");
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
  let historyQueryQS = "";
  let modalOpen = false;

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

  // ---------- ROI overlay ----------
  // Compute the displayed portion of the <img> when object-fit:contain is used
  function imageRectOnScreen() {
    const wrap = liveWrap.getBoundingClientRect();
    const iw = liveImg.naturalWidth || 1920, ih = liveImg.naturalHeight || 1080;
    const arImg = iw / ih, arWrap = wrap.width / wrap.height;
    let w,h,x,y;
    if (arImg > arWrap) { w = wrap.width; h = w/arImg; x = wrap.left; y = wrap.top + (wrap.height-h)/2; }
    else { h = wrap.height; w = h*arImg; y = wrap.top; x = wrap.left + (wrap.width-w)/2; }
    return {left:x, top:y, width:w, height:h, right:x+w, bottom:y+h};
  }
  function overlayToNorm() {
    const IR = imageRectOnScreen();
    const OR = roiBox.getBoundingClientRect();
    const x0 = (OR.left   - IR.left) / IR.width;
    const y0 = (OR.top    - IR.top ) / IR.height;
    const x1 = (OR.right  - IR.left) / IR.width;
    const y1 = (OR.bottom - IR.top ) / IR.height;
    return [
      Math.max(0,Math.min(1,x0)), Math.max(0,Math.min(1,y0)),
      Math.max(0,Math.min(1,x1)), Math.max(0,Math.min(1,y1))
    ];
  }
  function normToOverlay([x0,y0,x1,y1]) {
    const IR = imageRectOnScreen();
    const wrap = liveWrap.getBoundingClientRect();
    const L = IR.left + IR.width  * x0 - wrap.left;
    const T = IR.top  + IR.height * y0 - wrap.top;
    const W = IR.width  * (x1-x0);
    const H = IR.height * (y1-y0);
    roiBox.style.left = L + "px";
    roiBox.style.top  = T + "px";
    roiBox.style.width  = W + "px";
    roiBox.style.height = H + "px";
  }
  async function pushROIToServer() {
    try {
      const rr = overlayToNorm();
      roiRead.textContent = `ROI ${rr.map(n=>Math.round(n*100)).join('% , ')}%`;
      await fetch('/api/detect_roi', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({ roi: rr })
      });
    } catch {}
  }
  async function initROI() {
    try {
      const r = await fetch('/api/detect_roi'); const d = await r.json();
      const roi = (d && d.roi && d.roi.length===4) ? d.roi : [0,0,1,1];
      normToOverlay(roi);
      roiRead.textContent = `ROI ${roi.map(n=>Math.round(n*100)).join('% , ')}%`;
    } catch {
      normToOverlay([0,0,1,1]);
      roiRead.textContent = 'ROI 0–100%';
    }
  }
  // Drag/resize (mouse & touch)
  let dragMode=null, start=null, startRect=null;
  function startDrag(e){
    e.preventDefault();
    const t = e.target;
    dragMode = (t.classList && t.classList.contains('h')) ? t.classList[1] : 'move';
    const r = roiBox.getBoundingClientRect();
    start = {x:(e.touches?e.touches[0].clientX:e.clientX), y:(e.touches?e.touches[0].clientY:e.clientY)};
    startRect = {x:r.left, y:r.top, w:r.width, h:r.height};
    window.addEventListener('mousemove', onDrag, {passive:false});
    window.addEventListener('touchmove', onDrag, {passive:false});
    window.addEventListener('mouseup', endDrag, {passive:false});
    window.addEventListener('touchend', endDrag, {passive:false});
  }

  
function esc(s){
  return String(s).replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
}
// ui.js
function appendLogLine(e){
  if (!consoleBox) return;
  const nearBottom = (consoleBox.scrollTop + consoleBox.clientHeight) >= (consoleBox.scrollHeight - 8);
  const t = new Date((e.ts || 0)*1000).toLocaleTimeString();
  const lvl = (typeof e.lvl === "number") ? e.lvl : 1;
  const sev = (e.sev) ? String(e.sev) : ({0:'ok',1:'info',2:'info',3:'warn',4:'err'}[lvl] || 'info');
  const html =
    `<span class="t">${esc(t)}</span> ` +
    `<span class="tag tag-${sev}">[${esc(e.tag)}]</span> ${esc(e.msg)}
`;
  consoleBox.insertAdjacentHTML('beforeend', html);
  if (nearBottom) consoleBox.scrollTop = consoleBox.scrollHeight;
}

function fetchLogs(){
  if (modalOpen) return;
  if (logPaused) return;
  fetch(`/api/logs?after=${logAfter}&limit=200&ts=${Date.now()}`)
    .then(r => r.ok ? r.json() : null)
    .then(d => {
      if (!d || !Array.isArray(d.items)) return;
      d.items.forEach(appendLogLine);
      if (typeof d.next === "number") logAfter = Math.max(logAfter, d.next);
    })
    .catch(()=>{});
}
btnLogPause?.addEventListener("click", () => {
  logPaused = !logPaused;
  btnLogPause.textContent = logPaused ? "Resume" : "Pause";
});
btnLogClear?.addEventListener("click", () => {
  if (consoleBox) consoleBox.textContent = "";
  logAfter = 0;   // start over from beginning of ring
  // If you enabled the POST /api/logs/clear route server-side, you can also:
  // fetch('/api/logs/clear', {method:'POST'}).catch(()=>{});
});
  function onDrag(e){
    if(!dragMode) return;
    e.preventDefault();
    const px = (e.touches?e.touches[0].clientX:e.clientX);
    const py = (e.touches?e.touches[0].clientY:e.clientY);
    const dx = px - start.x, dy = py - start.y;
    const IR = imageRectOnScreen();
    const wrap = liveWrap.getBoundingClientRect();
    let x = startRect.x, y = startRect.y, w = startRect.w, h = startRect.h;
    if (dragMode==='move') { x+=dx; y+=dy; }
    if (dragMode==='nw'){ x+=dx; y+=dy; w-=dx; h-=dy; }
    if (dragMode==='ne'){ y+=dy; w+=dx; h-=dy; }
    if (dragMode==='se'){ w+=dx; h+=dy; }
    if (dragMode==='sw'){ x+=dx; w-=dx; h+=dy; }
    // clamp to the visible image rect
    const pad=14;
    x = Math.max(IR.left, Math.min(x, IR.right-pad));
    y = Math.max(IR.top , Math.min(y, IR.bottom-pad));
    w = Math.max(pad, Math.min(w, IR.right - x));
    h = Math.max(pad, Math.min(h, IR.bottom - y));
    roiBox.style.left   = (x - wrap.left) + 'px';
    roiBox.style.top    = (y - wrap.top)  + 'px';
    roiBox.style.width  = w + 'px';
    roiBox.style.height = h + 'px';
  }
  async function endDrag(e){
    if(!dragMode) return;
    dragMode=null;
    window.removeEventListener('mousemove', onDrag);
    window.removeEventListener('touchmove', onDrag);
    window.removeEventListener('mouseup', endDrag);
    window.removeEventListener('touchend', endDrag);
    await pushROIToServer();
  }
  roiBox.addEventListener('mousedown', startDrag, {passive:false});
  roiBox.addEventListener('touchstart', startDrag, {passive:false});
  roiReset?.addEventListener('click', async ()=>{
    normToOverlay([0,0,1,1]);
    roiRead.textContent='ROI 0–100%';
    await fetch('/api/detect_roi',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({roi:[0,0,1,1]})});
  });
  // keep ROI lined up when image loads or window resizes
  liveImg.addEventListener('load', initROI);
  window.addEventListener('resize', initROI);
  document.addEventListener('DOMContentLoaded', initROI);

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
  async function updateCardInfo(s) {
    if (!ci || !s || !s.name) { if (ci) ci.innerHTML = ""; return; }
    try {
      const r = await fetch("/api/carddata?ts=" + Date.now());
      if (!r.ok) return;
      const d = await r.json();
      if (!d || typeof d.last_updated !== "number" || Math.abs((d.last_updated || 0) - (s.updated_at || 0)) > 0.5) return;

      if (!d.scry) { ci.innerHTML = pill("warn", "Card not found: " + (s.name_raw || s.name)); return; }
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

      ci.innerHTML =
        '<div class="cardbox">' +
          '<div class="left">' + (image ? ('<img class="art" src="' + image + '" alt="Card">') : "") + "</div>" +
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
            // ⬇️ Prices lives INSIDE .right now
            (
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
            ) +
          "</div>" +  
        "</div>";   
    } catch {}
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
  backdrop.classList.remove('hidden');
  fmodal.classList.remove('hidden');
}
function closeFilter() {
  backdrop.classList.add('hidden');
  fmodal.classList.add('hidden');
}
$('#btn-open-filter')?.addEventListener('click', openFilter);
$('#filter-close')?.addEventListener('click', closeFilter);
backdrop?.addEventListener('click', ()=>{ closeFilter(); closeExport?.(); closeEditModal?.(); });
document.addEventListener('keydown', (e)=>{ if(e.key==='Escape'){ closeFilter(); closeExport?.(); closeEditModal?.(); }});



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
  qs.set('limit', '200');
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
  historyQueryQS = buildHistoryQuery(); 
  fetchScans(); 
});

$('#hf-reset')?.addEventListener('click', () => {
  ['hf-status','hf-scoremin','hf-since','hf-q','hf-set','hf-foil','hf-sortby','hf-sortdir'].forEach(id=>{
    const el = document.getElementById(id); if (!el) return;
    if (el.tagName === 'SELECT') el.selectedIndex = 0; else el.value = '';
  });
  document.getElementById('hf-status').value = 'all';
  document.getElementById('hf-sortby').value = 'ts';
  document.getElementById('hf-sortdir').value = 'desc';
  historyQueryQS = ""; 
  fetchScans();
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
    ]
  },
  {
    title: "Autoscan / Steady",
    items: [
      {k:"STEADY_MIN_FRAMES", label:"Steady min frames", type:"int"},
      {k:"AUTO_CAPTURE_WAIT_S", label:"Auto capture wait (s)", type:"float"},
      {k:"AUTOSCAN_OCR_TIMEOUT", label:"OCR timeout (s)", type:"float"},
      {k:"AUTOSCAN_SCRY_TIMEOUT", label:"Scryfall timeout (s)", type:"float"},
      {k:"AUTOSCAN_IMG_TIMEOUT", label:"Image timeout (s)", type:"float"},
    ]
  },
  {
    title: "OCR & Debug",
    items: [
      {k:"OCR_BACKEND", label:"OCR backend (comma list)"},
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
    title: "Debug & Paths",
    items: [
      {k:"DEBUG_LEVEL", label:"Debug level", type:"int"},
      {k:"DEBUG_SAVE_ROI", label:"Save ROI crops", type:"bool"},
      {k:"DEBUG_DIR", label:"Debug dir"},
      {k:"BAD_DIR", label:"Bad-list dir"},
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
];

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

const res = await fetch('/api/settings' + (forceFresh ? ('?t=' + Date.now()) : ''));
let cur = await res.json();
// If the backend ever returns {__effective, __path}, we only use the flat key-values for the form.
if (cur && cur.__effective && cur.__path) {
  // cur already contains the merged {**effective, **saved}, so just keep it.
  // (extra keys are ignored by the schema)
}


  const frag = document.createDocumentFragment();

  SETTINGS_SCHEMA.forEach((sec) => {
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

  const label = document.createElement('label');
  label.htmlFor = id;
  label.textContent = it.label;
  wrap.appendChild(label);

  if (it.type === 'bool') {
    const sel = document.createElement('select');
    sel.id = id;
    ['true', 'false'].forEach((v) => {
      const o = document.createElement('option');
      o.value = v;
      o.textContent = v;
      if (String(val).toLowerCase() === v) o.selected = true;
      sel.appendChild(o);
    });
    wrap.appendChild(sel);
    return wrap;
  }

  if (it.type === 'roi') {
    const r = Array.isArray(val) ? val : [0, 0, 1, 1];
    const box = document.createElement('div');
    box.className = 'roi';
    ['x0', 'y0', 'x1', 'y1'].forEach((name, i) => {
      const input = document.createElement('input');
      input.type = 'number';
      input.step = '0.001';
      input.min = '0';
      input.max = '1';
      input.id = `${id}_${name}`;
      input.value = r[i] ?? 0;
      box.appendChild(input);
    });
    wrap.appendChild(box);
    return wrap;
  }

  const inp = document.createElement('input');
  inp.id = id;
  inp.type = 'text';
  inp.value = val ?? '';
  if (it.type === 'int') inp.inputMode = 'numeric';
  if (it.type === 'float') inp.inputMode = 'decimal';
  wrap.appendChild(inp);
  return wrap;
}

// Collect all values across the whole long form
function collect() {
  const out = {};
  SETTINGS_SCHEMA.forEach((sec) => {
    sec.items.forEach((it) => {
      const id = 's__' + it.k;
      if (it.type === 'bool') {
        out[it.k] = (document.getElementById(id).value === 'true');
      } else if (it.type === 'roi') {
        const vals = ['x0', 'y0', 'x1', 'y1'].map((n) =>
          parseFloat(document.getElementById(`${id}_${n}`).value || '0')
        );
        out[it.k] = vals;
      } else {
        const v = document.getElementById(id).value;
        if (it.type === 'int') out[it.k] = parseInt(v || '0', 10);
        else if (it.type === 'float') out[it.k] = parseFloat(v || '0');
        else out[it.k] = v;
      }
    });
  });
  return out;
}

async function saveSettings() {
  const payload = collect();
  const res = await fetch('/api/settings', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  const j = await res.json();
  if (!j.ok) {
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

    revList.innerHTML = items.map(function (it) {
      const cls = (it.status === "pass" ? "ok" : "bad");
      const score = Number(it.match_score || 0);
      const numStr = (it.number ? ("• #" + it.number) : "");
      return (
        '<div class="revitem" data-id="' + it.id + '" data-status="' + (it.status || "fail") + '">' +
          '<img class="thumb" loading="lazy" src="/api/scan/' + it.id + '/thumb">' +
          '<div style="flex:1;min-width:0;display:flex;flex-direction:column;gap:4px;">' +
            '<div style="display:flex;align-items:center;gap:8px;justify-content:space-between;">' +
              '<div style="min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;"><b>' +
                (it.name || "(unnamed)") + "</b> " + numStr +
              "</div>" +
              '<span class="pill ' + cls + '">' + String((it.status || "FAIL")).toUpperCase() + "</span>" +
            "</div>" +
            '<div class="meta"><span>score ' + score.toFixed(2) + "</span>" +
              (it.scry_set ? ('<span>' + String(it.scry_set).toUpperCase() + " " + (it.scry_cn || "") + "</span>") : "") +
              (it.scry_name ? ('<span>→ ' + it.scry_name + "</span>") : "") +
              (it.flagged && Array.isArray(it.review_reasons) && it.review_reasons.length
                ? '<span class="pill warn" title="' + it.review_reasons.join(' • ').replace(/"/g,'&quot;') + '">Review Req.</span>'
                : "") +
            "</div>" +
          "</div>" +
        "</div>"
      );
    }).join("");

    revCount.textContent = (Number(scans.count) || items.length) + " items";
    Array.from(revList.querySelectorAll(".revitem")).forEach(function (el) {
      el.addEventListener("click", function () {
        const id = Number(el.getAttribute("data-id"));
        fetch("/api/scan/" + id + "/load", { method: "POST" })
          .then(r => r.json())
          .then(function (res) {
            if (res && res.ok) {
              currentLoadedId = id;
              if (btnPass) btnPass.disabled = false;
              if (btnFail) btnFail.disabled = false;
              if (btnEdit) btnEdit.disabled = false;
              if (btnDelete) btnDelete.disabled = false;
              const cmp = document.getElementById("cmpimg");
              if (cmp) cmp.src = "/compare.jpg?ts=" + Date.now();
              fetchState();
            }
          });
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

        const scannerPills = [
          { txt: (s.locked ? "Locked" : "Searching"), cls: (s.locked ? "ok" : "warn") }
        ];
        const foilNow = (typeof s.foil !== "undefined") ? s.foil : s.ocr_foil;
        const foilSc  = (typeof s.foil_score !== "undefined") ? s.foil_score : s.ocr_foil_score;
        if (typeof foilNow !== "undefined") {
          const fs = (typeof foilSc === "number" ? foilSc.toFixed(2) : "—");
          scannerPills.push({ txt: (foilNow ? ("Foil • " + fs) : "Non-foil"), cls: (foilNow ? "ok" : "warn") });
        }
        scannerPills.push({ txt: (s.snapshot_present ? ("Snapshot " + (s.snapshot_age_sec || 0) + "s ago") : "No Snapshot"), cls: (s.snapshot_present ? "ok" : "warn") });
        if (s.steady) scannerPills.push({ txt: "STEADY", cls: "ok" });
        updatePills(sc, scannerPills);

        const p1p = [{ txt: (s.ws_connected ? "WS connected" : "WS offline"), cls: (s.ws_connected ? "ok" : "bad") }];
        if (s.awaiting) p1p.push({ txt: "Most recent scan - Job #" + s.job_id, cls: "warn" });
        if (s.last_decision) p1p.push({ txt: s.last_decision, cls: "ok" });
        updatePills(p1, p1p);

        const ocrPills = [];
        if (s.name) { ocrPills.push({ txt: "Name: " + s.name, cls: "ok" }, { txt: "conf " + Math.round(s.name_conf || 0), cls: "ok" }); }
        else { ocrPills.push({ txt: "Name: —", cls: "warn" }); }
        if (s.number) { ocrPills.push({ txt: "No: " + s.number, cls: "ok" }, { txt: "conf " + Math.round(s.number_conf || 0), cls: "ok" }); }
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

        if (s.provider) ocrPills.push({ txt: "Provider: " + s.provider, cls: "ok" });
        if (s.last_error) ocrPills.push({ txt: "OCR err: " + s.last_error, cls: "bad" });
        updatePills(ocrEl, ocrPills);

        const cmp = document.getElementById("cmpimg");
        if (cmp) cmp.src = "/compare.jpg?ts=" + Date.now();

        currentLoadedId = s.loaded_scan_id || currentLoadedId;
        if (btnPass) btnPass.disabled = !currentLoadedId;
        if (btnFail) btnFail.disabled = !currentLoadedId;
        if (btnEdit) btnEdit.disabled = !currentLoadedId;
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
          const cmp = document.getElementById("cmpimg");
          if (cmp) cmp.src = "/compare.jpg?ts=" + Date.now();
        }
      }).catch(() => {});
  };

  // ---------- Edit modal ----------
  function openEditModal() {
    if (!modal) return;
    const name = (lastState && (lastState.name || lastState.name_raw)) || "";
    const set  = (lastState && (lastState.set_hint || "")) || "";
    const num  = (lastState && (lastState.number_raw || lastState.number || "")) || "";
    eiName.value = name; eiSet.value = set; eiNum.value = num;
    eiResults.innerHTML = ""; chosenScryId = null;
    // ensure it's visible even if it has the 'hidden' utility class
    modal.classList.remove('hidden');
    // show shared backdrop (same one used by filter/export)
    document.getElementById('filter-backdrop')?.classList.remove('hidden');
    modal.classList.add("show"); modal.setAttribute("aria-hidden","false");
  }
  function closeEditModal() {
    modal.classList.remove('show');
    modal.setAttribute('aria-hidden','true');
    modal.classList.add('hidden');                   // <-- important
    const f = document.getElementById('filter-modal');
    const e = document.getElementById('export-modal');
    if ((f?.classList.contains('hidden') ?? true) && (e?.classList.contains('hidden') ?? true)) {
      document.getElementById('filter-backdrop')?.classList.add('hidden');
    }
  }
  btnEdit?.addEventListener('click', (e)=>{ e.preventDefault(); if (currentLoadedId) openEditModal(); });

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
    if (!currentLoadedId) return;
    const payload = { name: eiName.value || "", number: eiNum.value || "", set: (eiSet.value || "").toLowerCase(), scry_id: chosenScryId || null };
    fetch("/api/scan/" + currentLoadedId + "/edit", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload)
    })
    .then(r => r.json())
    .then(res => {
      if (res && res.ok) {
        const cmp = document.getElementById("cmpimg");
        if (cmp) cmp.src = "/compare.jpg?ts=" + Date.now();
        fetchState();
        fetchScans();
        closeEditModal();
      }
    });
  };

  // ===== Export CSV dialog =====
const exportModal = document.getElementById('export-modal');
const exportClose = document.getElementById('export-close');
function openExport(){ 
  // build checkboxes each time (so default states reset nicely)
  buildExportCheckboxes();
  document.getElementById('filter-backdrop')?.classList.remove('hidden');
  exportModal?.classList.remove('hidden'); 
  exportModal?.setAttribute('aria-hidden','false');
}
function closeExport(){ 
  exportModal?.classList.add('hidden'); 
  exportModal?.setAttribute('aria-hidden','true');
  document.getElementById('filter-backdrop')?.classList.add('hidden');
}
exportClose?.addEventListener('click', closeExport);

// Archidekt-like field set
const ALL_FIELDS = [
  "Quantity","Name","Finish","Condition","Date Added","Language","Purchase Price","Tags",
  "Edition Name","Edition Code","Multiverse Id","Scryfall ID","MTGO ID",
  "Collector Number","Mana Value","Colors","Identities","Mana cost",
  "Types","Sub-types","Super-types","Rarity",
  "Price (Card Kingdom)","Price (TCG Player)","Price (Star City Games)","Price (Card Hoarder)","Price (Card Market)",
  "Scryfall Oracle ID"
];
const DEFAULT_FIELDS = [
  "Quantity","Name","Finish","Condition","Date Added","Language",
  "Edition Name","Edition Code","Collector Number",
  "Scryfall ID","Scryfall Oracle ID",
  "Mana Value","Identities","Mana cost","Types","Sub-types","Super-types","Rarity"
];

function buildExportCheckboxes(){
  const wrap = document.getElementById('export-field-list');
  if (!wrap) return;
  wrap.innerHTML = ALL_FIELDS.map(h =>
    `<label class="mm-check"><input type="checkbox" data-f="${h}" ${DEFAULT_FIELDS.includes(h)?'checked':''}> ${h}</label>`
  ).join('');
}
function selectedFields(){
  return Array.from(document.querySelectorAll('#export-field-list input[type=checkbox]'))
    .filter(cb => cb.checked).map(cb => cb.dataset.f);
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
document.getElementById('export-selectall')?.addEventListener('click', ()=>{
  document.querySelectorAll('#export-field-list input[type=checkbox]').forEach(cb => cb.checked = true);
});
document.getElementById('export-clear')?.addEventListener('click', ()=>{
  document.querySelectorAll('#export-field-list input[type=checkbox]').forEach(cb => cb.checked = false);
});
document.getElementById('export-defaults')?.addEventListener('click', buildExportCheckboxes);

// open modal from toolbar button
document.getElementById('btn-export-csv')?.addEventListener('click', openExport);

// do the export
document.getElementById('export-go')?.addEventListener('click', ()=>{
  const fields = selectedFields();
  const qs = new URLSearchParams(historyQueryQS || buildHistoryQuery());
  qs.set('fmt','csv');
  if (fields.length) qs.set('fields', fields.join(','));
  // NOTE: we export using whatever filters are set; decklist export still forces pass-only
  window.location.href = '/api/scans/export/download?' + qs.toString();
  closeExport();
});

  // ---------- boot ----------
  function boot() {
    fetchState();
    fetchScans();
    setInterval(fetchState, 800);
    setInterval(fetchScans, 1500);
      setInterval(fetchLogs, 700); 
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();

  const sh = document.getElementById("showbad");
  if (sh) sh.onclick = function (e) { e.preventDefault(); fetchBad(); };
})();


document.getElementById('btnMeasureDeck')?.addEventListener('click', async () => {
  try {
    const script = `_RAISE_TO_TRAVEL
SET_START_HEIGHT_OK`;
    const r = await fetch('/api/printer/gcode', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({script})
    });
    const d = await r.json();
    if (d?.ok) log('t', 'Measuring START height…');
    else log('t', 'Measure failed: ' + (d?.error || 'unknown'));
  } catch {}
});



document.getElementById('btnMeasureDeck')?.addEventListener('click', async () => {
  try {
    const body = { name: 'SET_START_HEIGHT' };
    const r = await fetch('/api/printer/macro', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify(body)
    });
    const d = await r.json();
    if (d?.ok) log('t', 'Measuring START height…');
    else log('t', 'Measure failed: ' + (d?.error || 'unknown'));
  } catch {}
});



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

  // MEASURE -> SET_START_HEIGHT_OK
  const measureBtn = document.getElementById('btnMeasureDeck');
  if (measureBtn) {
    measureBtn.addEventListener('click', async (e) => {
      e.stopImmediatePropagation(); e.preventDefault();
      const d = await sendMacro('SET_START_HEIGHT_OK');
      if (d?.ok) log('t', 'SET_START_HEIGHT_OK sent');
      else log('t', 'Measure failed: ' + (d?.error || 'unknown'));
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

