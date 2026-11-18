import { buildInputs } from "./preprocess.js";
import { loadModel, predict } from "./model.js";

const GITHUB_URL = "https://github.com/shruti1303-runwal/electricity-theft-detection";
const $ = (id) => document.getElementById(id);
let model, cfg, rows = [], ids = [], labels = null, cleaned = [], probs = [], selected = null;

// ---------------------------------------------------------------- CSV
function parseCSV(text) {
  const lines = text.trim().split(/\r?\n/);
  const header = lines[0].split(",").map((h) => h.trim().replace(/^"|"$/g, ""));
  const idCol = header.indexOf("CONS_NO");
  const flagCol = header.indexOf("FLAG");
  const dayCols = header.map((_, i) => i).filter((i) => i !== idCol && i !== flagCol);
  const out = { ids: [], flags: flagCol >= 0 ? [] : null, rows: [] };
  lines.slice(1).forEach((line, r) => {
    const c = line.split(",");
    out.ids.push(idCol >= 0 ? c[idCol] : `row ${r + 1}`);
    if (flagCol >= 0) out.flags.push(Number(c[flagCol]));
    out.rows.push(dayCols.map((i) => (c[i] === undefined || c[i].trim() === "" ? NaN : Number(c[i]))));
  });
  return out;
}

// ---------------------------------------------------------------- scoring
function score(data, sourceName) {
  if (data.rows[0].length !== cfg.n_days) {
    showError(`The model was trained on ${cfg.n_days} days per customer; this file has ${data.rows[0].length}.`);
    return;
  }
  showError("");
  ids = data.ids; labels = data.flags; rows = data.rows;
  const inputs = buildInputs(rows, cfg.n_days, cfg.n_weeks);
  cleaned = inputs.cleaned;
  probs = predict(model, inputs, rows.length, cfg);
  $("source").textContent = sourceName;
  $("results").hidden = false;
  render();
  const top = probs.indexOf(Math.max(...probs));
  selectCustomer(top);
}

function render() {
  const thr = Number($("thr").value);
  $("thrVal").textContent = thr.toFixed(2);
  const order = probs.map((p, i) => i).sort((a, b) => probs[b] - probs[a]);
  const flagged = order.filter((i) => probs[i] >= thr);
  $("kScored").textContent = probs.length;
  $("kFlagged").textContent = flagged.length;
  if (labels) {
    const hits = flagged.filter((i) => labels[i] === 1).length;
    const thieves = labels.filter((v) => v === 1).length;
    $("kPrec").textContent = flagged.length ? `${Math.round((hits / flagged.length) * 100)}%` : "–";
    $("kRec").textContent = thieves ? `${Math.round((hits / thieves) * 100)}%` : "–";
  } else {
    $("kPrec").textContent = $("kRec").textContent = "n/a";
  }
  $("tbody").innerHTML = order.map((i) => `
    <tr data-i="${i}" class="${i === selected ? "sel" : ""}">
      <td>${ids[i]}</td>
      <td><div class="pbar"><span style="width:${(probs[i] * 100).toFixed(1)}%" class="${probs[i] >= thr ? "hot" : ""}"></span></div>${(probs[i] * 100).toFixed(1)}%</td>
      <td>${probs[i] >= thr ? '<b class="flag">Inspect</b>' : '<span class="muted">OK</span>'}</td>
      ${labels ? `<td>${labels[i] === 1 ? "theft" : "normal"}</td>` : ""}
    </tr>`).join("");
  $("labelHead").hidden = !labels;
  $("tbody").querySelectorAll("tr").forEach((tr) => tr.addEventListener("click", () => selectCustomer(Number(tr.dataset.i))));
}

function selectCustomer(i) {
  selected = i;
  $("custName").textContent = ids[i];
  $("custProb").textContent = `${(probs[i] * 100).toFixed(1)}% theft probability`;
  $("custProb").className = probs[i] >= Number($("thr").value) ? "pill hot" : "pill";
  drawChart(cleaned[i]);
  render();
}

// ---------------------------------------------------------------- chart (plain SVG)
function drawChart(x) {
  const W = 760, H = 300, P = { l: 48, r: 14, t: 14, b: 34 };
  const n = x.length, ymax = Math.max(...x) * 1.08 || 1;
  const sx = (i) => P.l + (i / (n - 1)) * (W - P.l - P.r);
  const sy = (v) => H - P.b - (v / ymax) * (H - P.t - P.b);
  const avg = x.map((_, i) => { const s = x.slice(Math.max(0, i - 6), i + 1); return s.reduce((a, b) => a + b, 0) / s.length; });
  const path = (arr) => arr.map((v, i) => `${i ? "L" : "M"}${sx(i).toFixed(1)},${sy(v).toFixed(1)}`).join("");
  let grid = "";
  for (let k = 0; k <= 4; k++) {
    const v = (ymax / 4) * k;
    grid += `<line x1="${P.l}" x2="${W - P.r}" y1="${sy(v)}" y2="${sy(v)}" class="grid"/>
             <text x="${P.l - 8}" y="${sy(v) + 4}" text-anchor="end">${v.toFixed(0)}</text>`;
  }
  for (let d = 0; d < n; d += 60) grid += `<text x="${sx(d)}" y="${H - 12}" text-anchor="middle">day ${d + 1}</text>`;
  $("chart").innerHTML = `
    <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Daily consumption chart">
      ${grid}
      <path d="${path(x)}" class="raw"/>
      <path d="${path(avg)}" class="avg"/>
      <text x="${P.l}" y="${P.t - 2}" class="axis">kWh / day</text>
    </svg>`;
}

// ---------------------------------------------------------------- UI wiring
function showError(msg) { $("error").textContent = msg; $("error").hidden = !msg; }

async function init() {
  $("gh").href = GITHUB_URL;
  $("ghFoot").href = GITHUB_URL;
  try {
    ({ model, cfg } = await loadModel());
  } catch (e) {
    showError("Could not load the model. Run `python web/export_for_web.py` and redeploy.");
    return;
  }
  $("thr").value = cfg.threshold;
  $("modelStatus").textContent = "Model loaded · runs in your browser";
  $("btnSample").disabled = false;
  $("file").disabled = false;

  fetch("data/metrics.json").then((r) => r.json()).then((m) => {
    const t = m.dnn_tuned_threshold, b = m.baseline_logreg;
    $("mAcc").textContent = `${(t.accuracy * 100).toFixed(1)}%`;
    $("mAuc").textContent = t.auc.toFixed(3);
    $("mRec").textContent = `${(t.recall * 100).toFixed(1)}%`;
    $("mFp").textContent = `−${Math.round(m.fp_reduction_vs_baseline_pct)}%`;
    $("mNote").textContent = `Held-out test set of ${t.true_negatives + t.false_positives + t.true_positives + t.false_negatives} customers. Baseline logistic regression: AUC ${b.auc.toFixed(3)}, ${b.false_positives} false positives vs ${t.false_positives} for the DNN.`;
  }).catch(() => {});

  $("btnSample").onclick = async () => {
    $("btnSample").textContent = "Scoring…";
    const text = await (await fetch("data/sample_customers.csv")).text();
    const data = parseCSV(text);
    score(data, `${data.rows.length} sample customers`);
    $("btnSample").textContent = "Run on sample customers";
  };
  $("file").onchange = async (e) => {
    const f = e.target.files[0];
    if (f) score(parseCSV(await f.text()), f.name);
  };
  $("thr").oninput = () => { render(); if (selected !== null) selectCustomer(selected); };
}

init();
