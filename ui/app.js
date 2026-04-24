/* ── app.js — Self-Pruning NN Dashboard ──────────────────────────── */

// ── Lambda reference data ─────────────────────────────────────────────
const LAMBDA_DATA = {
  "1e-5": { acc: 56, sparsity: 10,  profile: "Near-dense baseline" },
  "1e-4": { acc: 54, sparsity: 30,  profile: "Light pruning" },
  "1e-3": { acc: 51, sparsity: 60,  profile: "Balanced ⭐" },
  "5e-3": { acc: 47, sparsity: 80,  profile: "Aggressive" },
  "1e-2": { acc: 43, sparsity: 92,  profile: "Maximum sparsity" },
};

// ── Colours ───────────────────────────────────────────────────────────
const BLUE   = "#58a6ff";
const GREEN  = "#3fb950";
const ORANGE = "#f0883e";
const PURPLE = "#bc8cff";
const PALETTE = [BLUE, GREEN, ORANGE, PURPLE, "#e05252"];
const BG_DARK  = "#0d1117";
const BG_PANEL = "#161b22";
const MUTED    = "#8b949e";
const TEXT     = "#e6edf3";

// ── Helper: parse lr slider value ─────────────────────────────────────
function lrFromSlider(v) {
  const exp = parseFloat(v);
  const val = Math.pow(10, exp);
  // nice string
  if (Number.isInteger(exp)) return `1e${exp}`;
  return val.toExponential(1).replace(/\.?0+e/, "e");
}

// ── Helper: active lambdas ────────────────────────────────────────────
function getActiveLambdas() {
  return [...document.querySelectorAll(".ms-btn.active")].map(b => b.dataset.val);
}

// ── Build CLI command string ──────────────────────────────────────────
function buildCmd() {
  const lambdas = getActiveLambdas();
  const epochs  = document.getElementById("epochsSlider").value;
  const temp    = parseFloat(document.getElementById("tempSlider").value).toFixed(3);
  const batch   = document.getElementById("batchSlider").value;
  const lr      = lrFromSlider(document.getElementById("lrSlider").value);
  const dev     = document.querySelector(".device-btn.active").dataset.dev;

  let cmd = `python main.py`;
  if (lambdas.length) cmd += ` --lambdas ${lambdas.join(" ")}`;
  cmd += ` --epochs ${epochs}`;
  cmd += ` --temperature ${temp}`;
  cmd += ` --batch_size ${batch}`;
  cmd += ` --lr ${lr}`;
  if (dev === "cuda") cmd += ` # (CUDA device auto-detected)`;
  return cmd;
}

// ── Render prediction cards ───────────────────────────────────────────
function renderPredCards() {
  const lambdas = getActiveLambdas();
  const container = document.getElementById("predCards");
  container.innerHTML = "";

  if (!lambdas.length) {
    container.innerHTML = `<div style="color:var(--muted);font-size:.85rem;padding:16px 0">Select at least one λ value above.</div>`;
    return;
  }

  lambdas.forEach((lam, i) => {
    const d = LAMBDA_DATA[lam] || { acc: "—", sparsity: "—", profile: "—" };
    const card = document.createElement("div");
    card.className = "pred-card";
    card.style.borderColor = PALETTE[i % PALETTE.length] + "44";
    card.innerHTML = `
      <div class="pred-lambda" style="color:${PALETTE[i % PALETTE.length]}">${lam}</div>
      <div class="pred-metric">
        <span class="pred-metric-val pred-acc-val" style="color:${PALETTE[i % PALETTE.length]}">${d.acc}%</span>
        <span class="pred-metric-label">Est. Test Acc</span>
      </div>
      <div class="pred-metric">
        <span class="pred-metric-val pred-sp-val" style="color:${ORANGE}">${d.sparsity}%</span>
        <span class="pred-metric-label">Est. Sparsity</span>
      </div>
    `;
    container.appendChild(card);
  });
}

// ── Update CLI preview ────────────────────────────────────────────────
function updateCmd() {
  document.getElementById("cmdPreview").textContent = buildCmd();
  renderPredCards();
}

// ── Multi-select buttons ──────────────────────────────────────────────
document.querySelectorAll(".ms-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    btn.classList.toggle("active");
    updateCmd();
  });
});

// ── Sliders ───────────────────────────────────────────────────────────
document.getElementById("epochsSlider").addEventListener("input", function () {
  document.getElementById("epochsVal").textContent = this.value;
  updateCmd();
});
document.getElementById("tempSlider").addEventListener("input", function () {
  document.getElementById("tempVal").textContent = parseFloat(this.value).toFixed(3);
  updateCmd();
});
document.getElementById("batchSlider").addEventListener("input", function () {
  document.getElementById("batchVal").textContent = this.value;
  updateCmd();
});
document.getElementById("lrSlider").addEventListener("input", function () {
  document.getElementById("lrVal").textContent = lrFromSlider(this.value);
  updateCmd();
});

// ── Device toggle ─────────────────────────────────────────────────────
document.querySelectorAll(".device-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".device-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    updateCmd();
  });
});

// ── Copy button ───────────────────────────────────────────────────────
document.getElementById("copyCmd").addEventListener("click", function () {
  const txt = document.getElementById("cmdPreview").textContent;
  navigator.clipboard.writeText(txt).then(() => {
    this.textContent = "✓ Copied!";
    this.classList.add("copied");
    setTimeout(() => {
      this.innerHTML = `<svg viewBox="0 0 20 20" fill="currentColor"><path d="M8 3a1 1 0 011-1h2a1 1 0 110 2H9a1 1 0 01-1-1z"/><path d="M6 3a2 2 0 00-2 2v11a2 2 0 002 2h8a2 2 0 002-2V5a2 2 0 00-2-2 3 3 0 01-3 3H9a3 3 0 01-3-3z"/></svg> Copy`;
      this.classList.remove("copied");
    }, 2000);
  });
});

// ── Active nav link on scroll ─────────────────────────────────────────
const sections = document.querySelectorAll("section[id]");
const navLinks = document.querySelectorAll(".nav-link[data-section]");
const observer = new IntersectionObserver(entries => {
  entries.forEach(e => {
    if (e.isIntersecting) {
      navLinks.forEach(l => l.classList.remove("active"));
      const active = document.querySelector(`.nav-link[data-section="${e.target.id}"]`);
      if (active) active.classList.add("active");
    }
  });
}, { threshold: 0.4 });
sections.forEach(s => observer.observe(s));

// ── Network diagram canvas ────────────────────────────────────────────
function drawNetwork() {
  const container = document.getElementById("networkDiagram");
  const W = container.clientWidth || 400;
  const H = container.clientHeight || 300;
  const canvas = document.createElement("canvas");
  canvas.id = "nnCanvas"; canvas.width = W; canvas.height = H;
  container.appendChild(canvas);
  const ctx = canvas.getContext("2d");

  const layers = [
    { label: "Input\n3072", n: 5, color: "#8b949e" },
    { label: "Layer 1\n1024", n: 7, color: BLUE },
    { label: "Layer 2\n512",  n: 6, color: GREEN },
    { label: "Layer 3\n256",  n: 5, color: ORANGE },
    { label: "Output\n10",    n: 4, color: PURPLE },
  ];

  const margin = 36;
  const colW = (W - margin * 2) / (layers.length - 1);
  const radius = 7;
  const nodePositions = [];

  layers.forEach((layer, li) => {
    const x = margin + li * colW;
    const totalH = (layer.n - 1) * 32;
    const startY = H / 2 - totalH / 2;
    nodePositions.push([]);
    for (let ni = 0; ni < layer.n; ni++) {
      nodePositions[li].push({ x, y: startY + ni * 32 });
    }
  });

  // Draw edges with random pruning simulation
  ctx.save();
  layers.forEach((layer, li) => {
    if (li === layers.length - 1) return;
    nodePositions[li].forEach(from => {
      nodePositions[li + 1].forEach(to => {
        const pruned = Math.random() < 0.35;
        ctx.strokeStyle = pruned ? "#ffffff08" : layer.color + "30";
        ctx.lineWidth = pruned ? 0.5 : 1;
        ctx.beginPath(); ctx.moveTo(from.x, from.y); ctx.lineTo(to.x, to.y);
        ctx.stroke();
      });
    });
  });
  ctx.restore();

  // Labels
  layers.forEach((layer, li) => {
    const x = nodePositions[li][0].x;
    ctx.fillStyle = layer.color + "cc";
    ctx.font = "600 9px Inter, sans-serif";
    ctx.textAlign = "center";
    layer.label.split("\n").forEach((line, i) => {
      ctx.fillText(line, x, 18 + i * 11);
    });
  });

  // Draw nodes
  layers.forEach((layer, li) => {
    nodePositions[li].forEach((pos, ni) => {
      const pruned = li > 0 && li < layers.length - 1 && Math.random() < 0.25;
      if (pruned) {
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, radius, 0, Math.PI * 2);
        ctx.fillStyle = "#ffffff08";
        ctx.strokeStyle = "#ffffff15";
        ctx.lineWidth = 1;
        ctx.fill(); ctx.stroke();
        return;
      }
      // Glow
      const grd = ctx.createRadialGradient(pos.x, pos.y, 0, pos.x, pos.y, radius * 2.5);
      grd.addColorStop(0, layer.color + "55");
      grd.addColorStop(1, "transparent");
      ctx.beginPath(); ctx.arc(pos.x, pos.y, radius * 2.5, 0, Math.PI * 2);
      ctx.fillStyle = grd; ctx.fill();

      ctx.beginPath();
      ctx.arc(pos.x, pos.y, radius, 0, Math.PI * 2);
      ctx.fillStyle = layer.color;
      ctx.fill();
    });
  });

  // Gate legend
  ctx.fillStyle = ORANGE;
  ctx.font = "500 9px Inter, sans-serif";
  ctx.textAlign = "left";
  ctx.fillRect(margin, H - 22, 8, 8);
  ctx.fillText("Gate=0 (pruned)", margin + 12, H - 14);
  ctx.fillStyle = BLUE;
  ctx.fillRect(margin + 110, H - 22, 8, 8);
  ctx.fillStyle = TEXT;
  ctx.fillText("Gate=1 (active)", margin + 122, H - 14);
}

// ── Gate distribution histogram (mock) ───────────────────────────────
function drawGateHistogram() {
  const canvas = document.getElementById("histCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.fillStyle = "#010409"; ctx.fillRect(0, 0, W, H);

  // Simulate bimodal gate distribution
  const bins = 30;
  const binW = W / bins;
  const counts = [];
  for (let i = 0; i < bins; i++) {
    const pos = i / bins;
    // Big spike at 0 (pruned) + cluster at 1 (active)
    let val = 2 * Math.exp(-Math.pow((pos - 0.02) * 35, 2)) * 180
            + 1.2 * Math.exp(-Math.pow((pos - 0.92) * 12, 2)) * 100
            + (Math.random() * 4);
    counts.push(Math.max(0, val));
  }
  const maxC = Math.max(...counts);

  counts.forEach((c, i) => {
    const x = i * binW + 1;
    const barH = (c / maxC) * (H - 30);
    const color = i < 2 ? ORANGE : BLUE;
    ctx.fillStyle = color + "cc";
    ctx.fillRect(x, H - barH - 16, binW - 2, barH);
  });

  // Axes labels
  ctx.fillStyle = MUTED; ctx.font = "9px Inter,sans-serif"; ctx.textAlign = "center";
  ctx.fillText("0", 8, H - 4);
  ctx.fillText("0.5", W / 2, H - 4);
  ctx.fillText("1", W - 8, H - 4);
  ctx.fillText("Gate value", W / 2, H);
}

// ── Training curve canvas (mock) ─────────────────────────────────────
function drawTrainingCurves() {
  const canvas = document.getElementById("curveCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.fillStyle = "#010409"; ctx.fillRect(0, 0, W, H);

  const pad = 20;
  // Simulate 3 training curves (test acc over epochs)
  const lambdas = ["1e-4", "1e-3", "5e-3"];
  const colors  = [BLUE, GREEN, ORANGE];
  const epochs  = 30;

  lambdas.forEach((lam, li) => {
    const finalAcc  = LAMBDA_DATA[lam].acc / 100;
    const finalSpar = LAMBDA_DATA[lam].sparsity / 100;
    ctx.beginPath();
    for (let e = 0; e <= epochs; e++) {
      const t = e / epochs;
      // acc grows, sparsity also grows
      const acc = finalAcc * (1 - Math.exp(-4 * t)) + Math.sin(e * 1.2) * 0.01;
      const x = pad + (e / epochs) * (W - pad * 2);
      const y = (H - pad - 10) - acc * (H - pad * 2 - 10);
      e === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.strokeStyle = colors[li];
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Label
    ctx.fillStyle = colors[li];
    ctx.font = "9px JetBrains Mono, monospace";
    ctx.textAlign = "right";
    ctx.fillText(`λ=${lam}`, W - 4, pad + li * 14);
  });

  ctx.fillStyle = MUTED; ctx.font = "9px Inter,sans-serif"; ctx.textAlign = "center";
  ctx.fillText("Epoch →", W / 2, H - 2);
}

// ── Lambda comparison bar chart (mock) ─────────────────────────────────
function drawLambdaBar() {
  const canvas = document.getElementById("barCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.fillStyle = "#010409"; ctx.fillRect(0, 0, W, H);

  const lambdas = Object.keys(LAMBDA_DATA);
  const n = lambdas.length;
  const pad = 28;
  const groupW = (W - pad * 2) / n;
  const barW = groupW * 0.3;

  lambdas.forEach((lam, i) => {
    const d = LAMBDA_DATA[lam];
    const cx = pad + i * groupW + groupW / 2;
    const maxH = H - pad * 2;

    // Accuracy bar
    const aH = (d.acc / 100) * maxH;
    ctx.fillStyle = BLUE + "cc";
    ctx.fillRect(cx - barW - 2, H - pad - aH, barW, aH);

    // Sparsity bar
    const sH = (d.sparsity / 100) * maxH;
    ctx.fillStyle = ORANGE + "cc";
    ctx.fillRect(cx + 2, H - pad - sH, barW, sH);

    // Lambda label
    ctx.fillStyle = MUTED; ctx.font = "8px JetBrains Mono, monospace"; ctx.textAlign = "center";
    ctx.fillText(lam, cx, H - 4);
  });

  // Legend
  ctx.fillStyle = BLUE; ctx.fillRect(pad, 8, 10, 8);
  ctx.fillStyle = TEXT; ctx.font = "8px Inter,sans-serif"; ctx.textAlign = "left";
  ctx.fillText("Acc%", pad + 14, 16);
  ctx.fillStyle = ORANGE; ctx.fillRect(pad + 52, 8, 10, 8);
  ctx.fillStyle = TEXT; ctx.fillText("Sparsity%", pad + 66, 16);
}

// ── Initialise ────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  updateCmd();
  renderPredCards();

  // Draw network after layout is ready
  setTimeout(() => {
    drawNetwork();
    drawGateHistogram();
    drawTrainingCurves();
    drawLambdaBar();
  }, 100);

  initImageInference();
});

// ══════════════════════════════════════════════════════════════════════
//  IMAGE UPLOAD + INFERENCE
// ══════════════════════════════════════════════════════════════════════

const CIFAR10_CLASSES = [
  { name: "Airplane",   emoji: "✈️" },
  { name: "Automobile", emoji: "🚗" },
  { name: "Bird",       emoji: "🐦" },
  { name: "Cat",        emoji: "🐱" },
  { name: "Deer",       emoji: "🦌" },
  { name: "Dog",        emoji: "🐶" },
  { name: "Frog",       emoji: "🐸" },
  { name: "Horse",      emoji: "🐴" },
  { name: "Ship",       emoji: "🚢" },
  { name: "Truck",      emoji: "🚛" },
];

// ── API base (same origin when served via server.py) ─────────────────
const API_BASE = "http://127.0.0.1:5000";
let _inferMode   = "sim"; // "sim" | "live"
let _uploadedFile = null;

function initImageInference() {
  const dropZone    = document.getElementById("dropZone");
  const imgInput    = document.getElementById("imgInput");
  const browseBtn   = document.getElementById("browseBtn");
  const runInferBtn = document.getElementById("runInferBtn");
  const inferResults= document.getElementById("inferResults");
  const resetBtn    = document.getElementById("resetBtn");
  const modeSimBtn  = document.getElementById("modeSimBtn");
  const modeLiveBtn = document.getElementById("modeLiveBtn");
  const ckptRow     = document.getElementById("ckptRow");
  const ckptRefresh = document.getElementById("ckptRefresh");

  if (!dropZone) return;

  // ── Mode toggle ──────────────────────────────────────────────────────
  function setMode(mode) {
    _inferMode = mode;
    modeSimBtn.classList.toggle("active",  mode === "sim");
    modeLiveBtn.classList.toggle("active", mode === "live");
    ckptRow.style.display = mode === "live" ? "flex" : "none";
    if (mode === "live") fetchCheckpoints();
  }
  modeSimBtn.addEventListener("click",  () => setMode("sim"));
  modeLiveBtn.addEventListener("click", () => setMode("live"));

  // ── Checkpoint list ──────────────────────────────────────────────────
  async function fetchCheckpoints() {
    const sel = document.getElementById("ckptSelect");
    try {
      const res  = await fetch(`${API_BASE}/api/checkpoints`);
      const data = await res.json();
      sel.innerHTML = `<option value="">— auto-detect —</option>`;
      data.checkpoints.forEach(c => {
        const opt = document.createElement("option");
        opt.value = c.path; opt.textContent = c.label;
        sel.appendChild(opt);
      });
    } catch {
      sel.innerHTML = `<option value="">⚠ Server offline — run: python server.py</option>`;
    }
  }
  ckptRefresh && ckptRefresh.addEventListener("click", fetchCheckpoints);

  // ── File input ───────────────────────────────────────────────────────
  browseBtn.addEventListener("click", (e) => { e.stopPropagation(); imgInput.click(); });
  dropZone.addEventListener("click",  () => imgInput.click());
  imgInput.addEventListener("change", () => {
    if (imgInput.files[0]) { _uploadedFile = imgInput.files[0]; loadImage(_uploadedFile); }
  });

  // ── Drag-and-drop ────────────────────────────────────────────────────
  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault(); dropZone.classList.add("drag-over");
  });
  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
  dropZone.addEventListener("drop", (e) => {
    e.preventDefault(); dropZone.classList.remove("drag-over");
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) { _uploadedFile = file; loadImage(file); }
  });

  // ── Run button ───────────────────────────────────────────────────────
  runInferBtn.addEventListener("click", async () => {
    const svg = `<svg viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z" clip-rule="evenodd"/></svg>`;
    runInferBtn.innerHTML = "⏳ Processing…";
    runInferBtn.classList.add("running");
    try {
      if (_inferMode === "live") {
        await runInferenceAPI();
      } else {
        await new Promise(r => setTimeout(r, 600));
        runInferenceSimulated();
      }
    } catch(err) {
      console.error(err);
      alert("Inference error: " + err.message);
    } finally {
      runInferBtn.innerHTML = svg + " Run Inference";
      runInferBtn.classList.remove("running");
    }
  });

  // ── Reset ────────────────────────────────────────────────────────────
  resetBtn.addEventListener("click", () => {
    inferResults.style.display = "none";
    document.getElementById("previewArea").style.display = "none";
    dropZone.style.display = "flex";
    imgInput.value = "";
    _uploadedFile  = null;
  });
}

// ── Real API call ──────────────────────────────────────────────────────
async function runInferenceAPI() {
  if (!_uploadedFile) throw new Error("No image loaded.");
  const ckptSel = document.getElementById("ckptSelect");
  const form    = new FormData();
  form.append("image", _uploadedFile);
  if (ckptSel && ckptSel.value) form.append("checkpoint", ckptSel.value);

  const res  = await fetch(`${API_BASE}/api/infer`, { method: "POST", body: form });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);

  const pred = data.prediction;
  const ns   = data.neuron_stats;

  document.getElementById("resultBadge").textContent = pred.emoji;
  document.getElementById("resultClass").textContent =
    pred.class.charAt(0).toUpperCase() + pred.class.slice(1);
  document.getElementById("resultConf").textContent =
    `${(pred.confidence * 100).toFixed(1)}% confidence`;
  document.getElementById("rsActive").textContent   = ns.active.toLocaleString();
  document.getElementById("rsPruned").textContent   = ns.pruned.toLocaleString();
  document.getElementById("rsSparsity").textContent = `${(ns.sparsity * 100).toFixed(1)}%`;

  const modeTag = document.getElementById("resultModeTag");
  modeTag.textContent = "⚡ Real Model";
  modeTag.className   = "result-mode-tag live";

  // Build sorted prob list matching CIFAR10_CLASSES index
  const sorted = data.probabilities
    .map((p, i) => ({ p: p.prob, i: CIFAR10_CLASSES.findIndex(c => c.name.toLowerCase() === p.class) }))
    .sort((a, b) => b.p - a.p);
  renderProbBars(sorted, pred.class);

  document.getElementById("inferResults").style.display = "flex";
  document.getElementById("inferResults").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// ── Load & render uploaded image ──────────────────────────────────────
let _pixelData = null; // 32×32 RGBA flat array

function loadImage(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    const img = new Image();
    img.onload = () => {
      // Draw original (cropped square) on origCanvas
      const origCanvas = document.getElementById("origCanvas");
      const octx = origCanvas.getContext("2d");
      const s = Math.min(img.width, img.height);
      const ox = (img.width  - s) / 2;
      const oy = (img.height - s) / 2;
      octx.clearRect(0, 0, 160, 160);
      octx.drawImage(img, ox, oy, s, s, 0, 0, 160, 160);

      // Downsample to 32×32 via offscreen canvas
      const off = document.createElement("canvas");
      off.width = off.height = 32;
      const octx2 = off.getContext("2d");
      octx2.drawImage(img, ox, oy, s, s, 0, 0, 32, 32);
      const raw = octx2.getImageData(0, 0, 32, 32);
      _pixelData = raw.data; // RGBA, length = 32*32*4

      // Render 32×32 zoomed ×5 on cifar32Canvas
      const c32 = document.getElementById("cifar32Canvas");
      const cctx = c32.getContext("2d");
      cctx.imageSmoothingEnabled = false;
      cctx.clearRect(0, 0, 160, 160);
      const tmp = document.createElement("canvas");
      tmp.width = tmp.height = 32;
      tmp.getContext("2d").putImageData(raw, 0, 0);
      cctx.drawImage(tmp, 0, 0, 160, 160);

      // Pixel grid: luminance heatmap with purple→blue gradient
      drawPixelGrid(_pixelData);

      // Compute stats
      computeImgStats(_pixelData);

      // Show preview, hide drop zone
      document.getElementById("dropZone").style.display = "none";
      document.getElementById("previewArea").style.display = "flex";
    };
    img.src = e.target.result;
  };
  reader.readAsDataURL(file);
}

// ── Draw pixel-level intensity grid (purple heatmap) ──────────────────
function drawPixelGrid(data) {
  const gc = document.getElementById("gridCanvas");
  const gctx = gc.getContext("2d");
  const cellSize = 5; // 32 × 5 = 160
  gctx.clearRect(0, 0, 160, 160);
  for (let py = 0; py < 32; py++) {
    for (let px = 0; px < 32; px++) {
      const idx = (py * 32 + px) * 4;
      const r = data[idx], g = data[idx + 1], b = data[idx + 2];
      const lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
      // Map luminance to purple→blue palette
      const hue = 260 - lum * 60;
      const sat = 70 + lum * 20;
      const lig = 15 + lum * 65;
      gctx.fillStyle = `hsl(${hue},${sat}%,${lig}%)`;
      gctx.fillRect(px * cellSize, py * cellSize, cellSize, cellSize);
    }
  }
}

// ── Compute pixel statistics ──────────────────────────────────────────
let _imgFeatures = {};

function computeImgStats(data) {
  let rSum = 0, gSum = 0, bSum = 0, rSq = 0, gSq = 0, bSq = 0;
  const n = 32 * 32;
  for (let i = 0; i < data.length; i += 4) {
    const r = data[i] / 255, g = data[i + 1] / 255, b = data[i + 2] / 255;
    rSum += r; gSum += g; bSum += b;
    rSq  += r * r; gSq += g * g; bSq += b * b;
  }
  const rMean = rSum / n, gMean = gSum / n, bMean = bSum / n;
  const rStd = Math.sqrt(rSq / n - rMean * rMean);
  const gStd = Math.sqrt(gSq / n - gMean * gMean);
  const bStd = Math.sqrt(bSq / n - bMean * bMean);
  const lum  = 0.299 * rMean + 0.587 * gMean + 0.114 * bMean;
  const contrast = (rStd + gStd + bStd) / 3;

  // Entropy (histogram-based, grey level, 16 bins)
  const hist = new Array(16).fill(0);
  for (let i = 0; i < data.length; i += 4) {
    const grey = Math.round((0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]) / 16);
    hist[Math.min(grey, 15)]++;
  }
  let entropy = 0;
  hist.forEach(c => { if (c > 0) { const p = c / n; entropy -= p * Math.log2(p); } });

  _imgFeatures = { rMean, gMean, bMean, lum, contrast, entropy };

  document.getElementById("statR").textContent       = rMean.toFixed(3);
  document.getElementById("statG").textContent       = gMean.toFixed(3);
  document.getElementById("statB").textContent       = bMean.toFixed(3);
  document.getElementById("statLum").textContent     = lum.toFixed(3);
  document.getElementById("statContrast").textContent = contrast.toFixed(3);
  document.getElementById("statEntropy").textContent = entropy.toFixed(2) + " bit";
}

// ── Shared: render probability bars ──────────────────────────────────
function renderProbBars(sorted, topClassName) {
  const container = document.getElementById("probBars");
  container.innerHTML = "";
  sorted.forEach(({ p, i }) => {
    const cls   = CIFAR10_CLASSES[i];
    if (!cls) return;
    const isTop = cls.name.toLowerCase() === (topClassName || "").toLowerCase();
    const pct   = (p * 100).toFixed(1);
    const row   = document.createElement("div");
    row.className = "prob-row";
    row.innerHTML = `
      <div class="prob-class-name ${isTop ? 'top-class' : ''}">${cls.emoji} ${cls.name}</div>
      <div class="prob-track">
        <div class="prob-fill ${isTop ? 'top-fill' : ''}" style="width:0%" data-target="${p*100}%"></div>
      </div>
      <div class="prob-pct ${isTop ? 'top-pct' : ''}">${pct}%</div>
    `;
    container.appendChild(row);
  });
  requestAnimationFrame(() => requestAnimationFrame(() => {
    document.querySelectorAll(".prob-fill").forEach(el => { el.style.width = el.dataset.target; });
  }));
}

// ── Simulated inference (client-side colour stats) ────────────────────
function runInferenceSimulated() {
  const { rMean, gMean, bMean, lum, contrast, entropy } = _imgFeatures;

  // Colour bias vector — produces a soft prior over classes from RGB stats
  // Each row = [rWeight, gWeight, bWeight, lumWeight, contrastWeight, entropyWeight, bias]
  const weights = [
    [ 0.5, -0.3,  0.1,  -0.4,  0.3,  0.1,  0.1],  // airplane  (blue sky → low r, low lum)
    [-0.2, -0.1, -0.4,  -0.2, -0.2, -0.1, -0.2],  // automobile
    [-0.3,  0.4,  0.1,   0.1,  0.5,  0.3, -0.1],  // bird
    [ 0.4,  0.2, -0.2,   0.3, -0.3,  0.2,  0.1],  // cat
    [-0.1,  0.5, -0.4,   0.3,  0.3,  0.4, -0.2],  // deer
    [ 0.3, -0.1,  0.3,   0.1,  0.1,  0.2,  0.0],  // dog
    [-0.4,  0.6, -0.5,   0.2,  0.4,  0.5, -0.2],  // frog
    [ 0.2, -0.2,  0.2,  -0.1,  0.1, -0.1,  0.1],  // horse
    [-0.3, -0.4,  0.6,  -0.5, -0.1,  0.2, -0.1],  // ship
    [-0.1, -0.3,  0.0,  -0.3, -0.4, -0.3, -0.1],  // truck
  ];
  const inp = [rMean, gMean, bMean, lum, contrast, entropy / 4, 1];

  const logits = weights.map(row => row.reduce((s, w, i) => s + w * inp[i], 0));

  // Softmax
  const maxL = Math.max(...logits);
  const exps = logits.map(l => Math.exp(l - maxL));
  const sumE = exps.reduce((a, b) => a + b, 0);
  const probs = exps.map(e => e / sumE);

  // Find top class
  const topIdx    = probs.indexOf(Math.max(...probs));
  const topName   = CIFAR10_CLASSES[topIdx].name;
  const confidence = (probs[topIdx] * 100).toFixed(1);

  // Simulate sparsity (use current λ selection)
  const activeLams = getActiveLambdas();
  const lam = activeLams.length ? activeLams[Math.floor(activeLams.length / 2)] : "1e-3";
  const sparsityPct = LAMBDA_DATA[lam]?.sparsity ?? 60;
  const totalNeurons = 1792;
  const pruned = Math.round(totalNeurons * sparsityPct / 100);
  const active = totalNeurons - pruned;

  document.getElementById("resultBadge").textContent = CIFAR10_CLASSES[topIdx].emoji;
  document.getElementById("resultClass").textContent = CIFAR10_CLASSES[topIdx].name;
  document.getElementById("resultConf").textContent  = `${confidence}% confidence`;
  document.getElementById("rsActive").textContent    = active.toLocaleString();
  document.getElementById("rsPruned").textContent    = pruned.toLocaleString();
  document.getElementById("rsSparsity").textContent  = `${sparsityPct}%`;

  const modeTag = document.getElementById("resultModeTag");
  modeTag.textContent = "🧪 Simulated";
  modeTag.className   = "result-mode-tag sim";

  const sorted = probs.map((p, i) => ({ p, i })).sort((a, b) => b.p - a.p);
  renderProbBars(sorted, topName);

  document.getElementById("inferResults").style.display = "flex";
  document.getElementById("inferResults").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

