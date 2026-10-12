/* HiGT-Bot training dashboard. Pure parsing helpers are exported for node tests. */
"use strict";

const QUEUE_C_STEPS = [
  ["rf_tuned", "Random Forest (tuned)"],
  ["gine_seed42", "GINE seed 42"],
  ["gine_seed1", "GINE seed 1"],
  ["gine_seed2", "GINE seed 2"],
  ["dirgine_seed42", "Directed GINE seed 42"],
  ["dirgine_seed1", "Directed GINE seed 1"],
  ["dirgine_seed2", "Directed GINE seed 2"],
  ["phase5_dir", "Directed T-GINE encoder"],
  ["cache_embeddings", "Cache embeddings"],
  ["hybrid_trees", "Hybrid trees (XGB/RF)"],
  ["phase6_dir", "Directed SAGPool"],
  ["edge", "HiGT-Bot v2 seed 42"],
  ["edge_seed1", "HiGT-Bot v2 seed 1"],
  ["edge_seed2", "HiGT-Bot v2 seed 2"],
  ["edge_undirected", "HiGT-Bot undirected (ablation)"],
  ["phase5_dir_seed1", "Directed T-GINE seed 1"],
  ["reeval", "Re-evaluation"],
];

function splitLines(text) {
  // CRLF files plus tqdm-style \r progress updates: treat any \r or \n as a break.
  return text.replace(/^﻿/, "").split(/\r\n|\r|\n/).filter((l) => l.trim() !== "");
}

/* Queue status file -> {start, done, steps: {name: {state, time}}, extra: [lines]} */
function parseQueue(text) {
  const out = { start: null, done: null, steps: {}, extra: [] };
  for (const line of splitLines(text || "")) {
    let m;
    if ((m = line.match(/^start\s+(.*)$/))) out.start = m[1].trim();
    else if ((m = line.match(/^DONE\b\s*(.*)$/))) out.done = m[1].trim() || "done";
    else if ((m = line.match(/^ok\s+(\S+)\s*(.*)$/))) out.steps[m[1]] = { state: "done", time: m[2].trim() };
    else if ((m = line.match(/^FAIL(?:ED)?\s+(\S+)\s*(.*)$/i))) out.steps[m[1]] = { state: "failed", time: m[2].trim() };
    else out.extra.push(line.trim());
  }
  return out;
}

/* Combine expected steps with queue status + which logs exist. */
function queueRows(expected, q, logNames) {
  const logs = new Set(logNames);
  const known = new Set(expected.map((e) => e[0]));
  const rows = expected.map(([name, label]) => {
    const s = q.steps[name];
    let state = "pending", time = "";
    if (s) { state = s.state; time = s.time; }
    else if (logs.has(name + ".log")) state = q.done ? "unfinished" : "running";
    return { name, label, state, time, log: logs.has(name + ".log") ? name + ".log" : null };
  });
  // Steps the status file mentions that are not in the expected list.
  for (const [name, s] of Object.entries(q.steps)) {
    if (!known.has(name)) rows.push({ name, label: name, state: s.state, time: s.time, log: logs.has(name + ".log") ? name + ".log" : null });
  }
  return rows;
}

const EPOCH_RE = /^\s*(?:\[([\w.-]+)\]\s*)?epoch\s+(\d+)\b(.*)$/i;

/* Training log -> {tag, epochs:[{epoch,valF1,best,lr,secs,improved,loss}], tail, wrote} */
function parseLog(text, tailN = 15) {
  const lines = splitLines(text || "");
  const epochs = [];
  let tag = null;
  for (const line of lines) {
    const m = line.match(EPOCH_RE);
    if (!m) continue;
    const rest = m[3];
    const num = (re) => { const x = rest.match(re); return x ? parseFloat(x[1]) : null; };
    const valF1 = num(/val[_ ]?f1[\s:=]+([\d.]+)/i);
    if (valF1 === null) continue;
    tag = m[1] || tag;
    epochs.push({
      epoch: parseInt(m[2], 10),
      valF1,
      best: num(/\bbest[\s:=]+([\d.]+)/i),
      loss: num(/(?:train_loss|loss|ce)[\s:=]+([\d.eE+-]+)/i),
      lr: num(/\blr[\s:=]+([\d.eE+-]+)/i),
      secs: num(/([\d.]+)s\s*$/),
      improved: /\s\*\s/.test(rest + " "),
    });
  }
  // A restarted run in the same log: keep only the last contiguous run.
  let startIdx = 0;
  for (let i = 1; i < epochs.length; i++) if (epochs[i].epoch <= epochs[i - 1].epoch) startIdx = i;
  const run = epochs.slice(startIdx);
  let best = null;
  for (const e of run) {
    const b = e.best !== null ? e.best : e.valF1;
    if (best === null || b > best) best = b;
  }
  const errLine = lines.find((l) => /Traceback|Error:|CUDA out of memory/.test(l)) || null;
  return {
    tag,
    epochs: run,
    lastEpoch: run.length ? run[run.length - 1] : null,
    best,
    tail: lines.slice(-tailN),
    wrote: lines.filter((l) => /^wrote\s/.test(l.trim())).map((l) => l.trim().slice(6)),
    error: errLine,
  };
}

/* Result JSON -> table row or null */
function resultRow(file, obj) {
  if (!obj || typeof obj !== "object" || !obj.test || typeof obj.test.f1 !== "number") return null;
  const t = obj.test;
  return {
    file,
    model: prettyResultName(file, obj),
    f1: t.f1, prAuc: t.pr_auc, fn: t.fn, fp: t.fp, tp: t.tp, tn: t.tn,
    recall: t.recall, precision: t.precision,
    isRef: file === "baseline_rf_tuned.json",
  };
}

function prettyResultName(file, obj) {
  const base = file.replace(/\.json$/, "");
  const rules = [
    [/^baseline_rf_tuned$/, "Random Forest (tuned)"],
    [/^baseline_rf$/, "Random Forest"],
    [/^baseline_gine_seed(\d+)$/, "GINE seed $1"],
    [/^baseline_dirgine_seed(\d+)$/, "Directed GINE seed $1"],
    [/^phase7_higt_bot_edge_undirected$/, "HiGT-Bot undirected (ablation)"],
    [/^phase5_temporal_gine_skip(?:_seed(\d+))?$/, (m) => "T-GINE encoder seed " + (m[1] || "42")],
    [/^phase6_sparse(?:_baseline)?(?:_seed(\d+))?$/, (m) => "SAGPool seed " + (m[1] || "42")],
    [/^phase7_higt_bot_edge(?:_seed(\d+))?$/, (m) => "HiGT-Bot v2 seed " + (m[1] || "42")],
  ];
  for (const [re, rep] of rules) {
    const m = base.match(re);
    if (m) return typeof rep === "function" ? rep(m) : base.replace(re, rep);
  }
  return base;
}

/* gpu_status.txt -> {ts: Date|null, gpus:[{util,memUsed,memTotal,temp,raw}], raw} */
function parseGpu(text) {
  const lines = splitLines(text || "");
  if (!lines.length) return null;
  const ts = new Date(lines[0].trim());
  const gpus = lines.slice(1).map((l) => {
    const p = l.split(",").map((s) => s.trim());
    const n = (s) => (s ? parseFloat(s) : NaN);
    return { util: n(p[0]), memUsed: n(p[1]), memTotal: n(p[2]), temp: n(p[3]), raw: l.trim() };
  });
  return { ts: isNaN(ts.getTime()) ? null : ts, gpus, raw: lines.join("\n") };
}

function ago(date, now = new Date()) {
  if (!date) return "unknown";
  const s = Math.max(0, Math.round((now - date) / 1000));
  if (s < 60) return s + " s ago";
  if (s < 3600) return Math.floor(s / 60) + " min ago";
  if (s < 86400) return Math.floor(s / 3600) + " h " + Math.floor((s % 3600) / 60) + " min ago";
  return Math.floor(s / 86400) + " d ago";
}

if (typeof module !== "undefined") {
  module.exports = { QUEUE_C_STEPS, splitLines, parseQueue, queueRows, parseLog, resultRow, prettyResultName, parseGpu, ago };
}

/* ------------------------------------------------------------------ browser */
if (typeof window !== "undefined") {
  const POLL_MS = 10000;
  const BASE = "logs/";
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const fmt = (x, d = 4) => (typeof x === "number" && isFinite(x) ? x.toFixed(d) : "–");
  const openLogs = new Set();

  async function getText(path) {
    const r = await fetch(BASE + path, { cache: "no-store" });
    if (!r.ok) throw new Error(r.status + " " + path);
    return r.text();
  }
  async function getList(dir) {
    try {
      const r = await fetch(BASE + dir, { cache: "no-store" });
      if (!r.ok) return [];
      return await r.json();
    } catch { return []; }
  }
  const tryText = (p) => getText(p).catch(() => null);

  function sparkline(epochs) {
    if (!epochs.length) return '<div class="muted small">no epochs yet</div>';
    const W = 320, H = 90, P = 4;
    const ys = epochs.map((e) => e.valF1);
    let lo = Math.min(...ys), hi = Math.max(...ys);
    if (hi - lo < 0.01) { lo -= 0.005; hi += 0.005; }
    const maxE = Math.max(epochs[epochs.length - 1].epoch, 2), minE = epochs[0].epoch;
    const x = (e) => P + ((e - minE) / Math.max(1, maxE - minE)) * (W - 2 * P);
    const y = (v) => H - P - ((v - lo) / (hi - lo)) * (H - 2 * P);
    const pts = epochs.map((e) => x(e.epoch).toFixed(1) + "," + y(e.valF1).toFixed(1)).join(" ");
    let bestIdx = 0;
    ys.forEach((v, i) => { if (v >= ys[bestIdx]) bestIdx = i; });
    const be = epochs[bestIdx];
    return `<svg class="chart" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" role="img" aria-label="validation F1 per epoch">
      <line x1="${P}" x2="${W - P}" y1="${y(ys[bestIdx]).toFixed(1)}" y2="${y(ys[bestIdx]).toFixed(1)}" class="bestline"/>
      <polyline points="${pts}" class="line"/>
      <circle cx="${x(be.epoch).toFixed(1)}" cy="${y(be.valF1).toFixed(1)}" r="3" class="dot"/>
    </svg>
    <div class="axis small muted"><span>ep ${minE}</span><span>val F1 ${lo.toFixed(3)}–${hi.toFixed(3)}</span><span>ep ${epochs[epochs.length - 1].epoch}</span></div>`;
  }

  function renderQueue(rows, q) {
    const counts = rows.reduce((a, r) => ((a[r.state] = (a[r.state] || 0) + 1), a), {});
    const summary = ["done", "running", "failed", "unfinished", "pending"]
      .filter((k) => counts[k]).map((k) => `<span class="pill ${k}">${counts[k]} ${k}</span>`).join(" ");
    $("queue-meta").innerHTML = `${summary}<span class="muted small"> · started ${esc(q.start || "?")}${q.done ? " · <b>DONE</b> " + esc(q.done) : ""}</span>`;
    $("queue").innerHTML = rows.map((r) => `<li class="step ${r.state}">
        <span class="badge ${r.state}">${r.state}</span>
        <span class="sname">${esc(r.label)}</span>
        <code class="muted small">${esc(r.name)}</code>
        <span class="muted small right">${esc(r.time)}</span></li>`).join("");
  }

  function renderLogs(logs) {
    $("logs").innerHTML = logs.map((l) => {
      const p = l.parsed, le = p.lastEpoch;
      const isOpen = openLogs.has(l.name) || (l.state === "running" && !openLogs.has("!" + l.name));
      return `<details class="card log ${l.state}" data-name="${esc(l.name)}" ${isOpen ? "open" : ""}>
        <summary>
          <span class="badge ${l.state}">${l.state}</span>
          <b>${esc(l.name)}</b>
          <span class="stats">
            <span>epoch <b>${le ? le.epoch : "–"}</b></span>
            <span>val F1 <b>${le ? fmt(le.valF1) : "–"}</b></span>
            <span>best <b>${fmt(p.best)}</b></span>
            <span class="muted">${l.mtime ? "written " + ago(l.mtime) : ""}</span>
          </span>
        </summary>
        <div class="logbody">
          <div class="chartbox">${sparkline(p.epochs)}${p.error ? `<div class="err small">${esc(p.error)}</div>` : ""}</div>
          <pre>${esc(p.tail.join("\n"))}</pre>
        </div>
      </details>`;
    }).join("") || '<p class="muted">No logs yet.</p>';
    document.querySelectorAll("details.log").forEach((d) => d.addEventListener("toggle", () => {
      const n = d.dataset.name;
      if (d.open) { openLogs.add(n); openLogs.delete("!" + n); } else { openLogs.delete(n); openLogs.add("!" + n); }
    }));
  }

  function renderResults(rows) {
    const ref = rows.find((r) => r.isRef);
    rows.sort((a, b) => (b.isRef - a.isRef) || b.f1 - a.f1);
    $("results").innerHTML = rows.length ? `<table>
      <thead><tr><th>Model</th><th>Test F1</th><th>Δ vs tuned RF</th><th>PR-AUC</th><th>Recall</th><th>Precision</th><th>FN</th><th>FP</th><th class="hide-sm">file</th></tr></thead>
      <tbody>${rows.map((r) => {
        const d = ref && !r.isRef ? r.f1 - ref.f1 : null;
        return `<tr class="${r.isRef ? "ref" : ""}">
          <td>${esc(r.model)}${r.isRef ? ' <span class="pill ref">reference bar</span>' : ""}</td>
          <td class="num"><b>${fmt(r.f1)}</b></td>
          <td class="num ${d === null ? "" : d >= 0 ? "pos" : "neg"}">${d === null ? "" : (d >= 0 ? "+" : "") + d.toFixed(4)}</td>
          <td class="num">${fmt(r.prAuc)}</td><td class="num">${fmt(r.recall, 3)}</td><td class="num">${fmt(r.precision, 3)}</td>
          <td class="num">${r.fn ?? "–"}</td><td class="num">${r.fp ?? "–"}</td>
          <td class="muted small hide-sm">${esc(r.file)}</td></tr>`;
      }).join("")}</tbody></table>` : '<p class="muted">No result JSONs yet.</p>';
  }

  function renderOldQueues(items) {
    $("oldqueues").innerHTML = items.map(({ name, q }) => {
      const steps = Object.entries(q.steps);
      return `<div class="card small-card"><b>${esc(name)}</b>
        <div class="muted small">${q.done ? "DONE " + esc(q.done) : "started " + esc(q.start || "?") + " · not done"}</div>
        <ul class="mini">${steps.map(([n, s]) => `<li><span class="badge ${s.state}">${s.state}</span> ${esc(n)} <span class="muted">${esc(s.time)}</span></li>`).join("")}
        ${q.extra.map((e) => `<li class="muted small">${esc(e)}</li>`).join("")}</ul></div>`;
    }).join("") || '<p class="muted">none</p>';
  }

  let lastGpu = null;
  function renderGpu() {
    const el = $("gpu");
    if (!lastGpu) { el.innerHTML = '<span class="muted">GPU: no gpu_status.txt — start dashboard/gpu_poller.sh on the host</span>'; return; }
    const ageS = lastGpu.ts ? (Date.now() - lastGpu.ts) / 1000 : Infinity;
    const stale = ageS > 60;
    el.innerHTML = lastGpu.gpus.map((g, i) => isFinite(g.util)
      ? `<span class="gpu-item"><b>GPU${lastGpu.gpus.length > 1 ? i : ""}</b>
          <span class="meter"><span style="width:${Math.min(100, g.util)}%"></span></span> ${g.util}% ·
          ${(g.memUsed / 1024).toFixed(1)}/${(g.memTotal / 1024).toFixed(1)} GiB · ${g.temp}°C</span>`
      : `<span class="err">${esc(g.raw)}</span>`).join(" ") +
      ` <span class="${stale ? "stale" : "muted"} small">(${ago(lastGpu.ts)}${stale ? " — stale, is the poller running?" : ""})</span>`;
  }

  async function refresh() {
    const t0 = new Date();
    try {
      const [relList, rootList, qcText, gpuText] = await Promise.all([
        getList("v2/"), getList(""), tryText("v2/queue_D_status.txt"), tryText("gpu_status.txt"),
      ]);
      const files = relList.filter((f) => f.type === "file");
      const mtimes = Object.fromEntries(files.map((f) => [f.name, new Date(f.mtime)]));
      const logNames = files.filter((f) => f.name.endsWith(".log")).map((f) => f.name);
      const jsonNames = files.filter((f) => f.name.endsWith(".json")).map((f) => f.name);

      const q = parseQueue(qcText || "");
      const rows = queueRows(QUEUE_C_STEPS, q, logNames);
      renderQueue(rows, q);
      const stateOf = Object.fromEntries(rows.filter((r) => r.log).map((r) => [r.log, r.state]));

      const [logTexts, jsons] = await Promise.all([
        Promise.all(logNames.map((n) => tryText("v2/" + n))),
        Promise.all(jsonNames.map((n) => tryText("v2/" + n).then((t) => { try { return JSON.parse(t); } catch { return null; } }))),
      ]);
      const order = Object.fromEntries(QUEUE_C_STEPS.map(([n], i) => [n + ".log", i]));
      const logs = logNames.map((name, i) => ({ name, mtime: mtimes[name], state: stateOf[name] || "log", parsed: parseLog(logTexts[i] || "") }))
        .sort((a, b) => (a.state === "running" ? -1 : 0) - (b.state === "running" ? -1 : 0) || (order[a.name] ?? 99) - (order[b.name] ?? 99));
      renderLogs(logs);
      renderResults(jsonNames.map((n, i) => resultRow(n, jsons[i])).filter(Boolean));

      const oldNames = rootList.filter((f) => f.type === "file" && /^queue_.*_status\.txt$/.test(f.name)).map((f) => f.name);
      const oldTexts = await Promise.all(oldNames.map((n) => tryText(n)));
      renderOldQueues(oldNames.map((n, i) => ({ name: n.replace(/_status\.txt$/, ""), q: parseQueue(oldTexts[i] || "") })));

      lastGpu = gpuText ? parseGpu(gpuText) : null;
      renderGpu();
      $("updated").textContent = "last updated " + t0.toLocaleTimeString();
      $("updated").classList.remove("stale");
    } catch (e) {
      $("updated").textContent = "update failed: " + e.message;
      $("updated").classList.add("stale");
    }
  }

  refresh();
  setInterval(refresh, POLL_MS);
  setInterval(renderGpu, 5000);
}
