// JavaScript port of src/preprocess.py — must stay identical to the Python version,
// otherwise the browser model would get different inputs than it was trained on.

function interpolateRow(row) {
  // Linear interpolation over missing values; edges take the nearest valid value
  // (same as pandas .interpolate(limit_direction="both")). All-missing row -> zeros.
  const x = row.slice();
  const valid = [];
  x.forEach((v, i) => { if (Number.isFinite(v)) valid.push(i); });
  if (valid.length === 0) return x.map(() => 0);
  for (let i = 0; i < valid[0]; i++) x[i] = x[valid[0]];
  for (let i = valid[valid.length - 1] + 1; i < x.length; i++) x[i] = x[valid[valid.length - 1]];
  for (let k = 0; k < valid.length - 1; k++) {
    const a = valid[k], b = valid[k + 1];
    for (let i = a + 1; i < b; i++) x[i] = x[a] + (x[b] - x[a]) * (i - a) / (b - a);
  }
  return x;
}

const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
const std = (a) => { const m = mean(a); return Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / a.length); };
function median(a) {
  const s = Float64Array.from(a).sort();
  const n = s.length;
  return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
}

function clean(row) {
  const x = interpolateRow(row);
  const cap = mean(x) + 2 * std(x);           // outlier capping
  return x.map((v) => Math.min(v, cap));
}

function features(x) {
  const eps = 1e-6;
  const n = x.length, half = Math.floor(n / 2);
  const m = mean(x), sd = std(x);
  const first = mean(x.slice(0, half)), second = mean(x.slice(half));
  const nWeeks = Math.floor(n / 7);
  const weekly = [];
  for (let w = 0; w < nWeeks; w++) weekly.push(x.slice(w * 7, w * 7 + 7).reduce((s, v) => s + v, 0));
  const diffs = [];
  for (let i = 1; i < n; i++) diffs.push(x[i] - x[i - 1]);
  const absd = diffs.map(Math.abs);
  // least-squares slope (same as np.polyfit(t, x, 1)[0])
  const tm = (n - 1) / 2;
  let num = 0, den = 0;
  for (let i = 0; i < n; i++) { num += (i - tm) * (x[i] - m); den += (i - tm) ** 2; }
  const slope = num / den;
  const f = [
    m, sd, Math.max(...x), median(x),
    sd / (m + eps),
    x.filter((v) => v < 0.05 * (m + eps)).length / n,
    second / (first + eps),
    std(weekly) / (mean(weekly) + eps),
    mean(absd) / (m + eps),
    slope / (m + eps),
    absd.filter((v) => v < 1e-3).length / absd.length,
  ];
  return f.map((v) => (Number.isFinite(v) ? v : 0));
}

function minmax(x) {
  const mn = Math.min(...x), mx = Math.max(...x);
  return x.map((v) => (v - mn) / (mx - mn + 1e-6));
}

// rows: array of daily-kWh arrays (NaN for missing). Returns flat Float32Arrays for tf.tensor.
export function buildInputs(rows, nDays, nWeeks) {
  const n = rows.length;
  const daily = new Float32Array(n * nDays);
  const weekly = new Float32Array(n * nWeeks * 7);
  const stats = [];
  const cleaned = [];
  rows.forEach((row, r) => {
    const x = clean(row);
    cleaned.push(x);
    stats.push(features(x));
    const s = minmax(x);
    daily.set(s.slice(0, nDays), r * nDays);
    weekly.set(s.slice(0, nWeeks * 7), r * nWeeks * 7);
  });
  return { daily, weekly, stats: Float32Array.from(stats.flat()), nFeats: stats[0].length, cleaned };
}
