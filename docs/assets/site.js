/* nba-xp visual documentation. Reads docs/data/*.json, fills numbers into the text, draws charts with Vega-Lite.
   No build step. All text that comes from data is inserted with textContent. */
(function () {
  "use strict";

  // ------------------------------------------------------------------------------------------ small helpers
  const $ = (s, r) => (r || document).querySelector(s);
  const $$ = (s, r) => Array.from((r || document).querySelectorAll(s));
  const el = (tag, attrs, ...kids) => {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (k === "class") e.className = v;
      else if (k === "text") e.textContent = v;
      else e.setAttribute(k, v);
    }
    for (const k of kids) if (k != null) e.append(k.nodeType ? k : document.createTextNode(String(k)));
    return e;
  };
  const MINUS = "−";
  const num = (v, d = 2) => (v < 0 ? MINUS : "") + Math.abs(v).toFixed(d);
  const signed = (v, d = 2) => (v < 0 ? MINUS : "+") + Math.abs(v).toFixed(d);
  const int = (v) => Math.round(v).toLocaleString("en-US");
  const pct = (v, d = 0) => (v * 100).toFixed(d) + "%";
  const compact = (n) => (n >= 1000 ? (n / 1000).toFixed(1) + "k" : String(n));
  const get = (o, path) => path.split(".").reduce((a, k) => (a == null ? a : a[k]), o);
  const MODEL_NAMES = { chain: "Champion chain", frozen: "Frozen v1", elo: "Elo baseline", home_only: "Home-only baseline" };
  let SAME = false; // true when the replay never promoted a challenger: the champion chain is then frozen v1

  const D = {};
  const FILES = ["meta", "timeline", "features", "dev", "test", "games", "checks", "drift"];

  async function load() {
    await Promise.all(
      FILES.map(async (f) => {
        const r = await fetch("data/" + f + ".json");
        if (!r.ok) throw new Error("data/" + f + ".json: HTTP " + r.status);
        D[f] = await r.json();
      })
    );
    // Values that the text uses and the JSON does not store.
    D.meta.mb = { model_inputs: (D.meta.bytes.staged + D.meta.bytes.features) / 1e6, raw_snapshot: D.meta.bytes.raw_snapshot / 1e6 };
    D.timeline.bubble_games = D.timeline.seasons.reduce((a, s) => a + s.n_bubble, 0);
    D.timeline.season_2019_train = D.timeline.seasons.find((s) => s.season === "2019-20").n_train;
    const fm = Object.values(D.dev.models.xgboost.folds).map((f) => f.mae);
    D.dev.fold_mae_range = Math.max(...fm) - Math.min(...fm);
    D.test.gain_home_only = D.test.models.home_only.mae - D.test.models.chain.mae;
    SAME = Object.keys(D.test.versions_used).length === 1;
    if (SAME) MODEL_NAMES.chain = "Model (frozen v1)";
  }

  function bindNumbers() {
    for (const n of $$("[data-k]")) {
      const v = get(D, n.dataset.k);
      if (v == null) { n.textContent = "n/a"; continue; }
      const d = n.dataset.d == null ? 2 : +n.dataset.d;
      const f = n.dataset.f || (n.dataset.d == null && Number.isInteger(v) ? "int" : undefined);
      n.textContent =
        typeof v === "string" ? v :
        f === "int" ? int(v) : f === "pct" ? pct(v, d) : f === "neg" ? num(-v, d) : f === "signed" ? signed(v, d) : num(v, d);
    }
  }

  // ------------------------------------------------------------------------------------------ theme
  let T = null;
  function readTheme() {
    const cs = getComputedStyle(document.documentElement);
    const g = (n) => cs.getPropertyValue(n).trim();
    const dark = cs.colorScheme.includes("dark") && !cs.colorScheme.includes("light dark");
    T = {
      dark, font: g("--font"), surface: g("--surface"), ink: g("--text"), ink2: g("--text2"), muted: g("--muted"), grid: g("--grid"),
      axis: g("--axis"), s1: g("--s1"), s2: g("--s2"), s3: g("--s3"), gray: g("--gray"),
    };
    // Sequential blue ramp (palette steps 100, 250, 400, 550, 700). On the dark surface "near zero" is the dark end.
    const ramp = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"];
    T.seq = dark ? ramp.slice().reverse() : ramp;
    T.modelColor = { chain: T.s1, xgboost: T.s1, frozen: T.s3, elo: T.s2, home_only: T.gray };
  }

  function vegaConfig() {
    return {
      background: "transparent",
      padding: { left: 18, right: 8, top: 6, bottom: 6 }, // room for axis labels that the browser draws wider than Vega measured
      font: T.font,
      view: { stroke: null },
      axis: {
        domainColor: T.axis, tickColor: T.axis, gridColor: T.grid, labelColor: T.ink2, titleColor: T.ink2,
        labelFontSize: 11, titleFontSize: 11, titleFontWeight: "normal", gridWidth: 1, domainWidth: 1, labelPadding: 4,
      },
      axisX: { grid: false },
      legend: { labelColor: T.ink2, titleColor: T.ink2, labelFontSize: 11, symbolStrokeWidth: 2, orient: "top", direction: "horizontal", padding: 0 },
      title: { color: T.ink, subtitleColor: T.ink2, fontSize: 13, subtitleFontSize: 11, fontWeight: 600, anchor: "start" },
      line: { strokeWidth: 2, strokeJoin: "round", strokeCap: "round" },
      text: { color: T.ink2, fontSize: 11 },
      range: { category: [T.s1, T.s2, T.s3, T.gray] },
    };
  }

  // ------------------------------------------------------------------------------------------ chart mounting
  const REG = [];

  async function draw(rec) {
    if (!rec.node.isConnected) return;
    const W = Math.max(rec.node.clientWidth || 300, 200);
    rec.w = W;
    rec.node.classList.add("busy");
    try {
      const spec = rec.build(W);
      await vegaEmbed(rec.node, spec, { actions: false, renderer: "svg", config: vegaConfig() });
    } catch (e) {
      console.error(e);
      rec.node.textContent = "This chart failed to draw: " + e.message;
    }
    rec.node.classList.remove("busy");
  }

  function mount(node, build) {
    const rec = { node, build, w: 0 };
    REG.push(rec);
    requestAnimationFrame(() => draw(rec)); // wait until the container sits in the page, so its width is real
    return rec;
  }

  function redrawAll(onlyIfWider) {
    for (let i = REG.length - 1; i >= 0; i--) if (!REG[i].node.isConnected) REG.splice(i, 1);
    for (const rec of REG) {
      if (onlyIfWider && Math.abs((rec.node.clientWidth || 0) - rec.w) < 24) continue;
      draw(rec);
    }
  }

  function tableEl(cols, rows) {
    const t = el("table");
    const head = el("tr");
    for (const c of cols) head.append(el("th", { class: c.num ? "n" : "", text: c.h }));
    t.append(el("thead", {}, head));
    const body = el("tbody");
    for (const r of rows) {
      const tr = el("tr");
      for (const c of cols) {
        const v = typeof c.k === "function" ? c.k(r) : r[c.k];
        tr.append(el("td", { class: c.num ? "n" : "", text: v == null ? "n/a" : c.f ? c.f(v) : String(v) }));
      }
      body.append(tr);
    }
    t.append(body);
    return t;
  }

  /* A figure: title, description, chart, and a table twin. `opts.table` = {cols, rows}. */
  function figure(slot, opts) {
    const node = typeof slot === "string" ? $(slot) : slot;
    const chart = el("div", { class: "chart", role: "img", "aria-label": opts.title + ". " + (opts.alt || "The table below the chart holds the same data.") });
    if (opts.minWidth) chart.style.minWidth = opts.minWidth + "px";
    const fig = el("figure", { class: "fig" }, el("figcaption", {}, el("div", { class: "t", text: opts.title }), opts.desc ? el("div", { class: "d", text: opts.desc }) : null), opts.scrollX ? el("div", { style: "overflow-x:auto" }, chart) : chart);
    if (opts.table) {
      fig.append(el("details", { class: "tv" }, el("summary", { text: "Table view" }), el("div", { class: "tvbody" }, tableEl(opts.table.cols, opts.table.rows))));
    }
    node.replaceChildren(fig);
    mount(chart, opts.build);
    return fig;
  }

  // ------------------------------------------------------------------------------------------ stat tiles
  function tiles() {
    const t = D.test, d = D.dev;
    const sw = (c) => el("i", { style: "background:" + c });
    const row = (color, label, v) => el("div", { class: "row" }, el("span", {}, sw(color), label), el("b", { text: v }));
    const savings = (b) => ({ m: -b.mean, lo: -b.hi, hi: -b.lo });
    const sT = savings(t.boot_chain_vs_elo.mae), sD = savings(d.boot_xgb_vs_elo.mae);
    const mk = (label, big, rows) => el("div", { class: "card tile" }, el("div", { class: "label", text: label }), el("div", { class: "big", text: big }), ...rows);
    const C = { m: "var(--s1)", e: "var(--s2)", h: "var(--gray)" };
    $("#tiles").replaceChildren(
      mk("MAE on the test season, points (lower is better)", num(t.models.chain.mae, 2), [
        row(C.m, (SAME ? "Model (frozen v1)" : "Model (champion chain)"), num(t.models.chain.mae, 2)), row(C.e, "Elo baseline", num(t.models.elo.mae, 2)), row(C.h, "Home-only baseline", num(t.models.home_only.mae, 2)),
      ]),
      mk("Brier score of P(home win), test season (lower is better)", num(t.models.chain.brier_p_home_win, 3), [
        row(C.m, (SAME ? "Model (frozen v1)" : "Model (champion chain)"), num(t.models.chain.brier_p_home_win, 4)), row(C.e, "Elo baseline", num(t.models.elo.brier_p_home_win, 4)), row(C.h, "Home-only baseline", num(t.models.home_only.brier_p_home_win, 4)),
      ]),
      mk("MAE saved against the Elo baseline, points", num(sT.m, 2), [
        el("div", { class: "row" }, el("span", { text: "Test season, 95% interval" }), el("b", { text: signed(sT.lo) + " to " + signed(sT.hi) })),
        el("div", { class: "row" }, el("span", { text: "Development" }), el("b", { text: num(sD.m, 2) })),
        el("div", { class: "row" }, el("span", { text: "Development, 95% interval" }), el("b", { text: signed(sD.lo) + " to " + signed(sD.hi) })),
      ]),
      mk("MAE on the development folds, points", num(d.models.xgboost.pooled.mae, 2), [
        row(C.m, "XGBoost model", num(d.models.xgboost.pooled.mae, 2)), row(C.e, "Elo baseline", num(d.models.elo.pooled.mae, 2)), row(C.h, "Home-only baseline", num(d.models.home_only.pooled.mae, 2)),
      ])
    );
  }

  // ------------------------------------------------------------------------------------------ example game
  const normPdf = (x, mu, s) => Math.exp(-0.5 * ((x - mu) / s) ** 2) / (s * Math.sqrt(2 * Math.PI));
  const REGIONS = {
    pw: { name: "P(home win)", sub: "margin above 0", f: (x) => x >= 0 },
    pb: { name: "P(blowout)", sub: "margin of 15 or more in either direction", f: (x) => Math.abs(x) >= 15 },
    pt: { name: "P(tight game)", sub: "margin of 5 or fewer in either direction", f: (x) => Math.abs(x) <= 5 },
  };

  function curveSpec(g, key, W) {
    const s = D.games.sigma, inR = REGIONS[key].f, pts = [];
    let grp = 0, prev = null;
    for (let x = -45; x <= 45.001; x += 0.25) {
      const r = inR(x);
      if (prev !== null && r !== prev) grp++;
      prev = r;
      pts.push({ x, d: normPdf(x, g.mu, s), r, grp });
    }
    const yc = Math.max(-45, Math.min(45, g.y));
    const xEnc = { field: "x", type: "quantitative", scale: { domain: [-45, 45], nice: false }, axis: { title: "Margin (home points minus away points)", tickCount: 7 } };
    const yEnc = { field: "d", type: "quantitative", axis: null };
    return {
      width: Math.max(160, W - 24), height: 140,
      title: { text: REGIONS[key].name + " = " + pct(g[key], 0), subtitle: "Shaded: " + REGIONS[key].sub },
      layer: [
        { data: { values: pts }, transform: [{ filter: "datum.r" }], mark: { type: "area", color: T.s1, opacity: 0.35 }, encoding: { x: xEnc, y: yEnc, detail: { field: "grp", type: "nominal" } } },
        { data: { values: pts }, mark: { type: "line", color: T.s1 }, encoding: { x: xEnc, y: yEnc } },
        { data: { values: [{ x: yc, t: "actual " + signed(g.y, 0) }] }, mark: { type: "rule", color: T.ink, strokeWidth: 2 }, encoding: { x: { field: "x", type: "quantitative" } } },
        { data: { values: [{ x: yc, t: "actual " + signed(g.y, 0) }] }, mark: { type: "text", align: yc > 0 ? "left" : "right", dx: yc > 0 ? 5 : -5, baseline: "top", color: T.ink, fontWeight: 600 }, encoding: { x: { field: "x", type: "quantitative" }, y: { value: 2 }, text: { field: "t" } } },
      ],
    };
  }

  let gameFilter = "typical";
  function pickGame(kind) {
    const G = D.games.games;
    const err = (g) => Math.abs(g.y - g.mu);
    if (kind === "random") return G[Math.floor(Math.random() * G.length)];
    if (kind === "close") return G.reduce((a, b) => (err(b) < err(a) ? b : a));
    if (kind === "worst") return G.reduce((a, b) => (err(b) > err(a) ? b : a));
    const reg = G.filter((g) => g.t === "Regular Season");
    const sorted = reg.map(err).sort((a, b) => a - b);
    const med = sorted[Math.floor(sorted.length / 2)];
    return reg.reduce((a, b) => (Math.abs(err(b) - med) < Math.abs(err(a) - med) ? b : a));
  }

  function showGame(kind) {
    gameFilter = kind;
    $$("#game-btns button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.pick === kind)));
    const g = pickGame(kind);
    $("#game-head").replaceChildren(
      el("span", { class: "teams", text: g.h + " (home) vs " + g.a }),
      el("span", { class: "small", text: g.d + " | " + g.t + " | model version " + g.v })
    );
    const box = $("#game-charts");
    box.replaceChildren();
    for (const key of ["pw", "pb", "pt"]) {
      const c = el("div", { class: "chart", role: "img", "aria-label": REGIONS[key].name + " for this game: " + pct(g[key], 0) });
      box.append(c);
      mount(c, (W) => curveSpec(g, key, W));
    }
    const won = g.y > 0 ? "home team won by " + g.y : "away team won by " + -g.y;
    $("#game-note").textContent =
      "Predicted mean margin (mu): " + signed(g.mu, 1) + ". Actual margin: " + signed(g.y, 0) + " (" + won + "). Error: " + Math.abs(g.y - g.mu).toFixed(1) +
      " points. Elo baseline predicted " + signed(g.muE, 1) + ".";
  }

  function gameCard() {
    $$("#game-btns button").forEach((b) => b.addEventListener("click", () => showGame(b.dataset.pick)));
    showGame("typical");
  }

  // ------------------------------------------------------------------------------------------ diagrams (SVG)
  function svgNS(w, h, inner, extra) {
    return '<svg viewBox="0 0 ' + w + " " + h + '" ' + (extra || "") + ' xmlns="http://www.w3.org/2000/svg">' + inner + "</svg>";
  }
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const FILL = { train: "var(--s1)", val: "var(--s2)", test: "var(--s3)" };
  const sshort = (s) => s; // season labels stay in full: 2013-14

  function timelineSVG() {
    const tl = D.timeline, seasons = tl.seasons.map((s) => s.season);
    const L = 214, cw = 62, rowH = 30, gap = 8, top = 64, N = seasons.length;
    const W = L + N * cw + 10;
    const inner = [];
    const defs = '<defs><pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="6" height="6" style="fill:var(--surface)"/><line x1="0" y1="0" x2="0" y2="6" style="stroke:var(--gray);stroke-width:2"/></pattern></defs>';
    const x0 = (i) => L + i * cw;
    // brackets
    const bracket = (i0, i1, text) => {
      const a = x0(i0) + 2, b = x0(i1 + 1) - 2;
      inner.push('<path d="M' + a + ",22 V16 H" + b + ' V22" fill="none" style="stroke:var(--muted);stroke-width:1.5"/>');
      inner.push('<text x="' + (a + b) / 2 + '" y="11" text-anchor="middle" class="small-t">' + esc(text) + "</text>");
    };
    bracket(0, 1, "warm-up");
    bracket(2, 11, "development seasons: 2015-16 to 2024-25");
    bracket(12, 12, "test");
    seasons.forEach((s, i) => inner.push('<text x="' + (x0(i) + cw / 2) + '" y="40" text-anchor="middle" class="strong small-t">' + s + "</text>"));
    const cell = (i, y, kind, text, trainShare) => {
      const x = x0(i) + 1, w = cw - 2;
      if (kind === "none") {
        inner.push('<rect x="' + x + '" y="' + y + '" width="' + w + '" height="' + rowH + '" rx="4" fill="none" style="stroke:var(--axis);stroke-width:1"/>');
        if (text) inner.push('<text x="' + (x + w / 2) + '" y="' + (y + 19) + '" text-anchor="middle" class="small-t">' + esc(text) + "</text>");
        return;
      }
      if (kind === "warm") {
        inner.push('<rect x="' + x + '" y="' + y + '" width="' + w + '" height="' + rowH + '" rx="4" class="lane"/>');
        inner.push('<text x="' + (x + w / 2) + '" y="' + (y + 19) + '" text-anchor="middle" class="small-t">' + esc(text || "warm-up") + "</text>");
        return;
      }
      const share = trainShare == null ? 1 : trainShare;
      const wf = w * share;
      inner.push('<rect x="' + x + '" y="' + y + '" width="' + wf + '" height="' + rowH + '" rx="4" style="fill:' + FILL[kind] + '"/>');
      if (share < 1) {
        inner.push('<rect x="' + (x + wf) + '" y="' + y + '" width="' + (w - wf) + '" height="' + rowH + '" rx="2" fill="url(#hatch)" style="stroke:var(--gray);stroke-width:1"/>');
      }
      inner.push('<text x="' + (x + wf / 2) + '" y="' + (y + 19) + '" text-anchor="middle" class="on-fill small-t">' + esc(text) + "</text>");
    };
    const label = (y, l1, l2) => {
      inner.push('<text x="4" y="' + (y + (l2 ? 13 : 19)) + '" class="strong">' + esc(l1) + "</text>");
      if (l2) inner.push('<text x="4" y="' + (y + 26) + '" class="small-t">' + esc(l2) + "</text>");
    };
    let y = top;
    // row: features
    label(y, "Features computed", "as-of each game date");
    inner.push('<rect x="' + (L + 1) + '" y="' + y + '" width="' + (N * cw - 2) + '" height="' + rowH + '" rx="4" class="lane"/>');
    inner.push('<text x="' + (L + (N * cw) / 2) + '" y="' + (y + 19) + '" text-anchor="middle" class="small-t">every game uses only games before its date (all 13 seasons)</text>');
    y += rowH + gap + 4;
    // row: final training
    label(y, "Final model", "train " + int(tl.final_n_train) + " games");
    seasons.forEach((s, i) => {
      if (tl.warmup.includes(s)) cell(i, y, "warm");
      else if (s === tl.test) cell(i, y, "none", "test");
      else {
        const info = tl.seasons[i];
        cell(i, y, "train", "train", info.n_bubble ? info.n_train / info.n : 1);
      }
    });
    y += rowH + gap + 4;
    // fold rows
    tl.folds.forEach((f, k) => {
      label(y, "Fold " + (k + 1) + ": validate " + f.val, "train " + int(f.n_train) + " | validate " + int(f.n_val));
      const vi = seasons.indexOf(f.val);
      seasons.forEach((s, i) => {
        if (tl.warmup.includes(s)) cell(i, y, "warm");
        else if (s === tl.test) cell(i, y, "none", "untouched");
        else if (i < vi) { const info = tl.seasons[i]; cell(i, y, "train", "train", info.n_bubble ? info.n_train / info.n : 1); }
        else if (i === vi) cell(i, y, "val", "validate");
        else cell(i, y, "none", "unused");
      });
      y += rowH + gap;
    });
    y += 4;
    // replay row
    label(y, "Replay (test season)", "one game day at a time");
    seasons.forEach((s, i) => { if (s === tl.test) cell(i, y, "test", "replay"); });
    y += rowH + gap + 10;
    // legend
    const lg = (x, kind, text) => {
      if (kind === "hatch") inner.push('<rect x="' + x + '" y="' + y + '" width="16" height="12" rx="2" fill="url(#hatch)" style="stroke:var(--gray)"/>');
      else if (kind === "warm") inner.push('<rect x="' + x + '" y="' + y + '" width="16" height="12" rx="2" class="lane"/>');
      else if (kind === "none") inner.push('<rect x="' + x + '" y="' + y + '" width="16" height="12" rx="2" fill="none" style="stroke:var(--axis)"/>');
      else inner.push('<rect x="' + x + '" y="' + y + '" width="16" height="12" rx="2" style="fill:' + FILL[kind] + '"/>');
      inner.push('<text x="' + (x + 22) + '" y="' + (y + 11) + '" class="small-t">' + esc(text) + "</text>");
    };
    lg(4, "train", "training games");
    lg(130, "val", "validation games");
    lg(262, "test", "test season");
    lg(370, "warm", "history only (warm-up)");
    lg(540, "hatch", "bubble games, excluded from training (" + tl.bubble_games + " games)");
    y += 26;
    $("#timeline-svg").innerHTML = svgNS(W, y, defs + inner.join(""), 'role="img" aria-label="Timeline of seasons, folds and the test season"');
  }

  function asofSVG() {
    const W = 460, H = 190, inner = [];
    const days = 13, x0 = 30, dx = 31, D0 = 7; // day D is column index 7
    inner.push('<line x1="14" y1="150" x2="448" y2="150" style="stroke:var(--axis);stroke-width:1.5"/>');
    inner.push('<text x="' + (x0 + D0 * dx) + '" y="170" text-anchor="middle" class="strong small-t">day D</text>');
    inner.push('<text x="' + (x0 + 3 * dx) + '" y="170" text-anchor="middle" class="small-t">earlier days</text>');
    inner.push('<text x="' + (x0 + 10.5 * dx) + '" y="170" text-anchor="middle" class="small-t">later days</text>');
    const games = [3, 2, 4, 3, 1, 2, 3, 3, 2, 4, 2, 3, 3];
    for (let i = 0; i < days; i++) {
      const cx = x0 + i * dx;
      for (let k = 0; k < games[i]; k++) {
        const cy = 138 - k * 17;
        const cls = i < D0 ? "fill:var(--s1)" : i === D0 ? "fill:var(--surface);stroke:var(--s3);stroke-width:2.5" : "fill:var(--grid);stroke:var(--axis);stroke-width:1";
        inner.push('<circle cx="' + cx + '" cy="' + cy + '" r="6" style="' + cls + '"/>');
      }
    }
    const bx = x0 + D0 * dx;
    inner.push('<rect x="' + (bx - 15) + '" y="36" width="30" height="120" rx="6" fill="none" style="stroke:var(--s3);stroke-width:1.5"/>');
    inner.push('<text x="' + bx + '" y="26" text-anchor="middle" class="small-t">features of these games</text>');
    inner.push('<text x="' + bx + '" y="14" text-anchor="middle" class="small-t">are computed here</text>');
    inner.push('<text x="' + (x0 + 2.5 * dx) + '" y="74" text-anchor="middle" class="small-t">used: games before D</text>');
    inner.push('<text x="' + (x0 + 10.5 * dx) + '" y="74" text-anchor="middle" class="small-t">not used</text>');
    inner.push('<text x="' + (x0 + 10.5 * dx) + '" y="88" text-anchor="middle" class="small-t">(not played yet)</text>');
    inner.push('<text x="' + (bx - 14) + '" y="186" text-anchor="start" class="small-t" style="display:none"></text>');
    $("#asof-svg").innerHTML = svgNS(W, H, inner.join(""), 'role="img" aria-label="Games before day D feed the features of games on day D. Games on day D and later do not."');
    $("#asof-svg").firstChild.style.minWidth = "420px";
  }

  function retrainSVG() {
    const W = 460, H = 190, inner = [];
    const segs = [
      { x: 14, w: 250, fill: "var(--s1)", t1: "challenger trains on", t2: "all games before D - 30 days" },
      { x: 266, w: 130, fill: "var(--s2)", t1: "scoring window", t2: "30 days before D" },
      { x: 398, w: 48, fill: "var(--s3)", t1: "day D", t2: "" },
    ];
    segs.forEach((s) => {
      inner.push('<rect x="' + s.x + '" y="46" width="' + s.w + '" height="44" rx="5" style="fill:' + s.fill + '"/>');
      inner.push('<text x="' + (s.x + s.w / 2) + '" y="' + (s.t2 ? 65 : 72) + '" text-anchor="middle" class="on-fill small-t">' + esc(s.t1) + "</text>");
      if (s.t2) inner.push('<text x="' + (s.x + s.w / 2) + '" y="80" text-anchor="middle" class="on-fill small-t">' + esc(s.t2) + "</text>");
    });
    inner.push('<text x="14" y="30" class="strong small-t">time runs left to right</text>');
    inner.push('<path d="M266,100 V108 H396 V100" fill="none" style="stroke:var(--muted);stroke-width:1.5"/>');
    inner.push('<text x="331" y="124" text-anchor="middle" class="small-t">champion and challenger are both scored here</text>');
    inner.push('<text x="331" y="138" text-anchor="middle" class="small-t">on the same games</text>');
    inner.push('<text x="14" y="168" class="small-t">The champion that exists at day D predicts the games of D.</text>');
    inner.push('<text x="14" y="182" class="small-t">A promoted challenger predicts day D too.</text>');
    $("#retrain-svg").innerHTML = svgNS(W, H, inner.join(""), 'role="img" aria-label="Retrain check: the challenger trains on games before D minus 30 days and is scored on the 30 days before D"');
    $("#retrain-svg").firstChild.style.minWidth = "420px";
  }

  // ------------------------------------------------------------------------------------------ features
  const DEFS = {
    elo: "A strength number for each franchise. Every franchise starts at 1505. After a game, the winner gains points and the loser loses the same number of points. The change is 20 times a margin multiplier times how surprising the result was. The multiplier is (|margin| + 3)^0.8 / (7.5 + 0.006 times the rating lead of the winner). The expected result includes a home advantage of 100 points, or 0 points at a neutral site (the NBA Cup final and the 2020 bubble). At the start of a new season, a rating keeps 75% of its value and moves 25% toward 1505. The setup follows the FiveThirtyEight NBA Elo.",
    record: "Wins, losses, and the win share (wins / (wins + losses)) of the team in the current regular season, before the game. Playoff and play-in games do not change the record, so a playoff game uses the final regular-season record. The win share is missing for the first game of a season.",
    streak: "The signed length of the current run. +3 means three wins in a row. -2 means two losses in a row. It counts every game type and resets to 0 at the start of a season.",
    pd: "The mean margin of the team (its points minus its opponents' points) over its last 5 games and over its last 10 games of the current season. It counts every game type. It is missing for the first game of a season.",
    rest: "Rest days are the days since the team's last game in the same season. They are missing for the first game of a season. The back-to-back flag is true when rest days equal 1.",
    travel: "The great-circle distance in km that the team travelled in the last 3 days and in the last 7 days, including the trip to this game. The game location is the home team's arena (Las Vegas for the NBA Cup final). A team starts a season at its home arena. The arena coordinates come from a hand-made table with one arena per franchise.",
    load: "The rotation is the 8 players with the most minutes in the team's last 10 games. Minute load is the total minutes that these 8 players played in the last 3, 7 and 14 days, in games with any team.",
    players: "Built from the players of the team's last game. The rating of a player is 48 times his plus-minus divided by (his minutes + 1,000), over his games with any team in the last 365 days. The 1,000 minutes pull a player with few minutes toward 0. The weight of a player is his mean minutes over his last 10 appearances, rescaled so that the weights of the team add to 240. Player-based team strength is the weighted mean rating. Its scale is plus-minus points per 48 minutes. Availability is the share of the team's rotation (the 8 players with the most minutes in its last 10 games) who played in its last game. Both columns are missing for the first game of the season of a team. The parameters were fixed before I looked at the test season.",
    eliminated: "Regular season only. True when the most wins the team can still reach is below the current wins of the team in the last postseason place of its conference. The last place is 10th from 2020-21 (the play-in era) and 8th before. The rule ignores tiebreakers.",
    win_or_go_home: "True when a loss ends the team's run. In the playoffs: the team has already lost 3 games in the series. In the play-in: the team plays the 9th-versus-10th game, or plays its second play-in game. Always false in the regular season.",
    game_type: "Three flags: play-in game, playoff game, NBA Cup game. A regular-season game has all three at 0. These flags are model inputs and have no home-minus-away version.",
  };

  function featureCharts(fam, p, W) {
    const B = D.features;
    const rows = p.bins.map((b, i) => ({ ...b, i, lo95: b.mean - 1.96 * b.se, hi95: b.mean + 1.96 * b.se, nlabel: compact(b.n) }));
    const yvals = rows.flatMap((r) => [r.lo95, r.hi95]).filter((v) => v != null);
    const pad = 0.8;
    const yDom = [Math.min(...yvals, B.mean_margin) - pad, Math.max(...yvals, B.mean_margin) + pad];
    const quant = p.kind === "numeric";
    const order = rows.map((r) => r.label);
    const xFull = quant
      ? { field: "x", type: "quantitative", axis: { title: p.label, tickCount: 6 } }
      : { field: "label", type: "ordinal", sort: order, axis: { title: p.label, labelAngle: 0, labelLimit: 120 } };
    const xEnc = quant ? xFull : { ...xFull, axis: { labels: false, title: null, ticks: false } };
    const tip = [
      { field: "label", title: "Bin" }, { field: "mean", title: "Mean margin", format: ".2f" }, { field: "se", title: "Standard error", format: ".2f" }, { field: "n", title: "Games", format: "," },
    ];
    const w = Math.max(200, W - 70);
    const yAxis = { title: "Mean home margin (points)", minExtent: 48, maxExtent: 48 };
    const main = {
      width: w, height: 190, data: { values: rows },
      layer: [
        { data: { values: [{ y: B.mean_margin }] }, mark: { type: "rule", color: T.axis, strokeWidth: 1 }, encoding: { y: { field: "y", type: "quantitative" } } },
        { mark: { type: "rule", color: T.s1, strokeWidth: 1.5, opacity: 0.8 }, encoding: { x: xEnc, y: { field: "lo95", type: "quantitative", scale: { domain: yDom }, axis: yAxis }, y2: { field: "hi95" } } },
        ...(p.kind === "category" ? [] : [{ mark: { type: "line", color: T.s1 }, encoding: { x: xEnc, y: { field: "mean", type: "quantitative", scale: { domain: yDom } } } }]),
        { mark: { type: "point", filled: true, color: T.s1, size: 70, stroke: T.surface, strokeWidth: 2, opacity: 1 }, encoding: { x: xEnc, y: { field: "mean", type: "quantitative", scale: { domain: yDom } }, tooltip: tip } },
      ],
    };
    if (quant) return main;
    // integer and category panels: a strip of game counts under the chart
    const strip = {
      width: w, height: 46, data: { values: rows },
      mark: { type: "bar", color: T.gray, cornerRadiusTopLeft: 3, cornerRadiusTopRight: 3, size: 16 },
      encoding: {
        x: xFull,
        y: { field: "n", type: "quantitative", axis: { title: "Games", tickCount: 2, format: ",d", minExtent: 48, maxExtent: 48 } },
        tooltip: tip,
      },
    };
    return { vconcat: [main, strip], spacing: 4 };
  }

  function famCards() {
    const abl = Object.fromEntries(D.dev.ablation.map((a) => [a.id, a]));
    const box = $("#fam-list");
    box.replaceChildren();
    D.features.families.forEach((fam) => {
      const a = abl[fam.id];
      const prim = fam.columns.find((c) => c.col.startsWith("diff_")) || fam.columns[0];
      const chips = el("div", { class: "stats" });
      const chip = (label, val) => chips.append(el("span", { class: "stat" }, label + " ", el("b", { text: val })));
      chip("Correlation of " + prim.col + " with margin:", num(prim.r, 2));
      chip("Share of gain:", pct(fam.gain_share, 1));
      if (a) {
        const clear = a.d_lo > 0;
        chip("MAE rise if removed:", signed(a.d_mae, 3) + " (" + signed(a.d_lo, 3) + " to " + signed(a.d_hi, 3) + (clear ? ", above zero" : ", includes zero") + ")");
      }
      const card = el("article", { class: "fam card", id: "fam-" + fam.id }, el("h3", { text: fam.name }), el("p", { class: "def", text: DEFS[fam.id] }),
        el("p", { class: "cols" }, "Columns: ", ...fam.columns.flatMap((c, i) => [i ? ", " : "", el("code", { text: c.col })])), chips);
      const figs = el("div", { class: fam.panels.length > 1 ? "grid2" : "" });
      card.append(figs);
      fam.panels.forEach((p, pi) => {
        const slot = el("div");
        figs.append(slot);
        const sub = p.kind === "numeric" ? "10 bins with equal game counts (about " + int(D.features.n_rows / 10) + " games each)." : "Games per bin are in the strip and in the table.";
        figure(slot, {
          title: p.label, desc: sub,
          build: (W) => featureCharts(fam, p, W),
          table: { cols: [{ h: "Bin", k: "label" }, { h: "Mean margin", k: "mean", f: (v) => num(v, 2), num: true }, { h: "Std. error", k: "se", f: (v) => num(v, 2), num: true }, { h: "Games", k: "n", f: int, num: true }], rows: p.bins },
        });
      });
      box.append(card);
    });
  }

  function importance() {
    const fams = D.features.families.map((f) => ({ name: f.name, share: f.gain_share })).sort((a, b) => b.share - a.share);
    figure("#f-importance-family", {
      title: "Share of gain by feature family",
      desc: "Gain summed over the home, away and difference columns of each family.",
      build: (W) => ({
        width: Math.max(120, W - 250), height: fams.length * 22,
        data: { values: fams },
        layer: [
          { mark: { type: "bar", color: T.s1, cornerRadiusTopRight: 4, cornerRadiusBottomRight: 4, height: 14 }, encoding: { y: { field: "name", type: "nominal", sort: fams.map((f) => f.name), axis: { title: null, labelLimit: 170 } }, x: { field: "share", type: "quantitative", axis: { title: "Share of total gain", format: "%", tickCount: 4 } }, tooltip: [{ field: "name", title: "Family" }, { field: "share", title: "Share of gain", format: ".1%" }] } },
          { mark: { type: "text", align: "left", dx: 4, color: T.ink2 }, encoding: { y: { field: "name", type: "nominal", sort: fams.map((f) => f.name) }, x: { field: "share", type: "quantitative" }, text: { field: "share", type: "quantitative", format: ".0%" } } },
        ],
      }),
      table: { cols: [{ h: "Family", k: "name" }, { h: "Share of gain", k: "share", f: (v) => pct(v, 1), num: true }], rows: fams },
    });
    const top = D.features.importance.slice(0, 15).map((r) => ({ col: r.col, share: r.share, family: D.features.families.find((f) => f.id === r.family).name }));
    figure("#f-importance-top", {
      title: "The 15 columns with the most gain",
      desc: D.features.n_used + " of " + D.features.n_feature_cols + " columns have any gain. The other " + (D.features.n_feature_cols - D.features.n_used) + " are never used by a split.",
      build: (W) => ({
        width: Math.max(120, W - 250), height: top.length * 20,
        data: { values: top },
        layer: [
          { mark: { type: "bar", color: T.s1, cornerRadiusTopRight: 4, cornerRadiusBottomRight: 4, height: 13 }, encoding: { y: { field: "col", type: "nominal", sort: top.map((t) => t.col), axis: { title: null, labelLimit: 180, labelFont: "ui-monospace, Menlo, monospace", labelFontSize: 10 } }, x: { field: "share", type: "quantitative", axis: { title: "Share of total gain", format: "%", tickCount: 4 } }, tooltip: [{ field: "col", title: "Column" }, { field: "family", title: "Family" }, { field: "share", title: "Share of gain", format: ".1%" }] } },
          { mark: { type: "text", align: "left", dx: 4, color: T.ink2 }, encoding: { y: { field: "col", type: "nominal", sort: top.map((t) => t.col) }, x: { field: "share", type: "quantitative" }, text: { field: "share", type: "quantitative", format: ".0%" } } },
        ],
      }),
      table: { cols: [{ h: "Column", k: "col" }, { h: "Family", k: "family" }, { h: "Share of gain", k: "share", f: (v) => pct(v, 1), num: true }], rows: top },
    });
    const ab = D.dev.ablation.filter((a) => a.id !== "all" && a.id !== "only elo+record").sort((a, b) => b.d_mae - a.d_mae);
    const all = D.dev.ablation[0], only = D.dev.ablation.find((a) => a.id === "only elo+record");
    figure("#f-ablation", {
      title: "Removal cost: rise in out-of-fold MAE when a family is removed",
      desc: "I retrained the XGBoost model with the same settings on the other columns and scored it on the same 5,283 out-of-fold games. The point is the change in MAE with all " + all.n_cols + " columns as the reference (" + num(all.mae, 3) + "). The bar is a 95% bootstrap interval of the change. A bar that includes zero means the family adds nothing measurable.",
      build: (W) => ({
        width: Math.max(100, W - 310), height: ab.length * 26, data: { values: ab },
        layer: [
          { data: { values: [{ x: 0 }] }, mark: { type: "rule", color: T.axis }, encoding: { x: { field: "x", type: "quantitative" } } },
          { mark: { type: "rule", color: T.s1, strokeWidth: 2 }, encoding: { y: { field: "name", type: "nominal", sort: ab.map((a) => a.name), axis: { title: null, labelLimit: 260 } }, x: { field: "d_lo", type: "quantitative", axis: { title: "Rise in MAE (points); positive means the model got worse" } }, x2: { field: "d_hi" } } },
          { mark: { type: "point", filled: true, color: T.s1, size: 80, stroke: T.surface, strokeWidth: 2, opacity: 1 }, encoding: { y: { field: "name", type: "nominal", sort: ab.map((a) => a.name) }, x: { field: "d_mae", type: "quantitative" }, tooltip: [{ field: "name", title: "Removed" }, { field: "d_mae", title: "Rise in MAE", format: "+.3f" }, { field: "d_lo", title: "Interval low", format: "+.3f" }, { field: "d_hi", title: "Interval high", format: "+.3f" }, { field: "n_cols", title: "Columns left" }, { field: "mae", title: "MAE", format: ".3f" }] } },
        ],
      }),
      table: { cols: [{ h: "Removed", k: "name" }, { h: "Columns left", k: "n_cols", num: true }, { h: "MAE", k: "mae", f: (v) => num(v, 3), num: true }, { h: "Rise in MAE", k: "d_mae", f: (v) => signed(v, 3), num: true }, { h: "Interval", k: (r) => signed(r.d_lo, 3) + " to " + signed(r.d_hi, 3) }], rows: D.dev.ablation.filter((a) => a.id !== "all") },
    });
    const clear = ab.filter((a) => a.d_lo > 0 && a.id !== "elo+record").map((a) => a.name.replace("Without ", ""));
    const noise = ab.filter((a) => a.d_lo <= 0 && a.id !== "elo+record").map((a) => a.name.replace("Without ", ""));
    const note = $("#signal-note");
    note.replaceChildren(
      el("p", {}, el("b", { text: "Player-based team strength and the Elo rating carry the signal. " }),
        "The strongest columns are diff_player_strength and diff_elo. Removing the player family costs " + signed(abl("players").d_mae, 3) + " MAE points (interval " + signed(abl("players").d_lo, 3) + " to " + signed(abl("players").d_hi, 3) + "). Removing the Elo rating alone costs " + signed(abl("elo").d_mae, 3) + ". Removing the record alone costs " + signed(abl("record").d_mae, 3) + (abl("record").d_lo <= 0 ? " (the interval includes zero)" : "") + ", because the Elo rating holds much of the same information. Removing Elo and record together costs " + signed(abl("elo+record").d_mae, 3) + ". A model with only Elo and record (" + only.n_cols + " columns) reaches MAE " + num(only.mae, 3) + ", against " + num(all.mae, 3) + " with all " + all.n_cols + " columns. The player family and the Elo rating both rate team strength, and diff_player_strength and diff_elo correlate at 0.93 on the training games."),
      el("p", {}, el("b", { text: "Most other families barely help. " }),
        "Removing a family has a measurable cost for: " + clear.join(", ") + ". The cost for the others includes zero: " + noise.join(", ") + ". Minute load has " + pct(D.features.families.find((f) => f.id === "load").gain_share, 1) + " of the gain, yet removing it changes MAE by " + signed(abl("load").d_mae, 3) + ". Gain shows where the trees split, not whether the splits help on new games."),
      el("p", { class: "small", text: "Removal cost for the other families is smaller than the noise of the estimate, so I cannot rank them. The model keeps them because they cost nothing to carry." })
    );
    function abl(id) { return D.dev.ablation.find((a) => a.id === id); }
  }

  // ------------------------------------------------------------------------------------------ decisions evidence
  function decisionEvidence() {
    const dv = D.dev, tl = D.timeline, ab = (id) => dv.ablation.find((a) => a.id === id);
    $("#ev-folds").textContent = tl.folds.map((f) => int(f.n_train)).join(" / ");
    const r = dv.residuals, s = dv.sigma;
    const sdm = r.sd_by_mu.map((g) => g.sd), type = Object.fromEntries(r.sd_by_type.map((g) => [g.label, g.sd]));
    $("#ev-normal").textContent =
      "Out-of-fold residuals: " + pct(r.share_within_1sd, 1) + " lie within 1 sigma (a Normal curve gives 68.3%) and " + pct(r.share_within_2sd, 1) + " within 2 sigma (95.4%). Excess kurtosis is " + num(r.excess_kurtosis, 2) +
      ". The residual spread is " + num(Math.min(...sdm), 1) + " to " + num(Math.max(...sdm), 1) + " across five groups of predicted mean. Playoff games have a spread of " + num(type["Playoffs"], 1) + " against " + num(type["Regular Season"], 1) +
      " in the regular season. On the test season the RMSE is " + num(D.test.models.chain.rmse, 2) + ", above sigma (" + num(D.test.sigma, 2) + "). Charts: section 5.";
    const g = dv.grid.map((x) => x.mean_val_mae), sel = dv.params;
    const best = {};
    dv.grid.forEach((x) => { const k = x.params.max_depth; best[k] = Math.min(best[k] ?? 99, x.mean_val_mae); });
    $("#ev-grid").textContent =
      "The 12 settings reach a mean validation MAE between " + num(Math.min(...g), 3) + " and " + num(Math.max(...g), 3) + ", a range of " + num(Math.max(...g) - Math.min(...g), 3) + " points. The best setting per tree depth: " +
      Object.keys(best).map((k) => "depth " + k + ", " + num(best[k], 3)).join("; ") + ". The choice of setting hardly matters. The chart below shows all 12.";
    const m = dv.models, red = (m.home_only.pooled.mae - m.elo.pooled.mae) / m.home_only.pooled.mae, red2 = (m.elo.pooled.mae - m.xgboost.pooled.mae) / m.home_only.pooled.mae;
    $("#ev-elo").textContent =
      "Out-of-fold MAE: home-only " + num(m.home_only.pooled.mae, 2) + ", Elo baseline " + num(m.elo.pooled.mae, 2) + ", XGBoost " + num(m.xgboost.pooled.mae, 2) + ". The Elo baseline removes " + pct(red, 1) + " of the home-only error. XGBoost removes " + pct(red2, 1) + " more.";
    const pl = ab("players"), tm = D.test.models;
    $("#ev-players").textContent =
      "Out-of-fold MAE with the family: " + num(m.xgboost.pooled.mae, 3) + " (XGBoost) against " + num(m.elo.pooled.mae, 3) + " (Elo baseline). Before the family, the two were 10.900 and 10.956. Removing the family raises MAE to " + num(pl.mae, 3) + " (" + signed(pl.d_mae, 3) + ", interval " + signed(pl.d_lo, 3) + " to " + signed(pl.d_hi, 3) + "). Test season: " + num(tm.chain.mae, 3) + " against " + num(tm.elo.mae, 3) + " (before: 11.445 against 11.514). On the training games, diff_player_strength and diff_elo correlate at 0.93, so most of the signal overlaps with Elo.";
    const e = ab("eliminated");
    $("#ev-elim").textContent =
      "The code switches the cutoff at season 2020-21. The eliminated columns hold " + pct(D.features.families.find((f) => f.id === "eliminated").gain_share, 1) + " of the gain. Removing them changes MAE by " + signed(e.d_mae, 3) + " (interval " + signed(e.d_lo, 3) + " to " + signed(e.d_hi, 3) + "), so the feature adds nothing measurable.";

    const C = D.checks, rows = C.rows, nPro = rows.filter((x) => x.promoted).length;
    const med = (a) => { const b = a.slice().sort((x, y) => x - y); return b.length ? b[Math.floor(b.length / 2)] : null; };
    const gainsPro = rows.filter((x) => x.promoted).map((x) => x.champion_mae - x.challenger_mae);
    if (C.has_p_better) {
      const best = rows.reduce((a, b) => (Math.min(b.p_better_mae, b.p_better_brier) > Math.min(a.p_better_mae, a.p_better_brier) ? b : a));
      $("#ev-gate").textContent =
        "In the latest replay " + rows.length + " checks ran and " + nPro + " promoted a challenger. " +
        (nPro === 0 ? "The closest check (" + best.check_date + ") had P(better) of " + num(best.p_better_mae, 2) + " on MAE and " + num(best.p_better_brier, 2) + " on Brier score, so it failed the 0.90 gate. " : "") +
        (() => {
          const early = rows.filter((x) => x.challenger_mae < x.champion_mae && x.challenger_brier < x.champion_brier);
          return early.length + " of " + rows.length + " checks show a challenger with a lower MAE and a lower Brier score, which was the earlier promotion rule. The smallest such MAE gain is " +
            (early.length ? num(Math.min(...early.map((x) => x.champion_mae - x.challenger_mae)), 3) : "n/a") + " points. The table in section 6 lists every check.";
        })();
    } else {
      $("#ev-gate").textContent =
        "The data on this page comes from the earlier rule: " + nPro + " of " + rows.length + " checks promoted a challenger, with a median MAE gain on the 30-day window of " + (gainsPro.length ? num(med(gainsPro), 3) : "n/a") +
        " points. The replay code now holds the bootstrap rule. The next run of the replay and of build_site.py will show its P(better) values.";
    }
    const nf = rows.map((x) => x.n_drift_flags), nfeat = D.drift.features.length;
    const seasonal = !D.drift.features.some((f) => f.startsWith("is_"));
    if (seasonal) {
      $("#ev-drift").textContent =
        "With the same-point-of-season reference, the replay flags between " + Math.min(...nf) + " and " + Math.max(...nf) + " of " + nfeat + " columns per check. The game-type flags are not checked. The heatmap in section 7 shows each flag.";
    } else {
      $("#ev-drift").textContent =
        "The data on this page comes from the earlier reference (all training games): it flagged between " + Math.min(...nf) + " and " + Math.max(...nf) + " of " + nfeat + " columns per check, including the win and loss counts and the game-type flags at nearly every check. The replay code now uses the same-point-of-season reference. The next run will show the effect.";
    }
    $("#ev-replay").textContent =
      "The replay covers " + D.test.n_game_days + " game days from " + D.test.first_date + " to " + D.test.last_date + " (" + int(D.test.n_games) + " games). It runs the retrain and promotion code in date order. It cannot test a live feed, a late result or a missing game.";
  }

  function gridChart() {
    const g = D.dev.grid.map((x) => ({ label: "depth " + x.params.max_depth + ", min child weight " + x.params.min_child_weight + ", " + x.params.n_estimators + " trees", mae: x.mean_val_mae, sel: x.params.max_depth === D.dev.params.max_depth && x.params.min_child_weight === D.dev.params.min_child_weight && x.params.n_estimators === D.dev.params.n_estimators }));
    g.sort((a, b) => a.mae - b.mae);
    figure("#f-grid", {
      title: "Grid search: mean validation MAE of the 12 settings",
      desc: "Mean of the four fold MAEs. Learning rate is 0.03 in every setting. The selected setting is blue. The whole range is under 0.05 points.",
      build: (W) => ({
        width: Math.max(120, W - 330), height: g.length * 22, data: { values: g },
        layer: [
          { mark: { type: "rule", strokeWidth: 1, color: T.grid }, encoding: { y: { field: "label", type: "nominal", sort: g.map((x) => x.label), axis: { title: null, labelLimit: 250, grid: true } }, x: { field: "mae", type: "quantitative", scale: { zero: false }, axis: { title: "Mean validation MAE (points)", format: ".3f" } }, x2: { datum: g[0].mae - 0.002 } } },
          { mark: { type: "point", filled: true, size: 80, opacity: 1, stroke: T.surface, strokeWidth: 2 }, encoding: { y: { field: "label", type: "nominal", sort: g.map((x) => x.label) }, x: { field: "mae", type: "quantitative", scale: { zero: false } }, color: { condition: { test: "datum.sel", value: T.s1 }, value: T.gray }, tooltip: [{ field: "label", title: "Setting" }, { field: "mae", title: "Mean validation MAE", format: ".4f" }] } },
        ],
      }),
      table: { cols: [{ h: "Setting", k: "label" }, { h: "Mean validation MAE", k: "mae", f: (v) => num(v, 4), num: true }, { h: "Selected", k: (r) => (r.sel ? "yes" : "") }], rows: g },
    });
  }

  // ------------------------------------------------------------------------------------------ validation
  function devTable() {
    const m = D.dev.models, folds = Object.keys(m.xgboost.folds);
    const t = el("table");
    t.append(el("thead", {}, el("tr", {}, ...["Fold (validation season)", "MAE home-only", "MAE Elo baseline", "MAE XGBoost", "Brier P(home win) home-only", "Brier Elo baseline", "Brier XGBoost"].map((h, i) => el("th", { class: i ? "n" : "", text: h })))));
    const body = el("tbody");
    const row = (label, sel, cls) => {
      const best = (k) => Math.min(sel("home_only")[k], sel("elo")[k], sel("xgboost")[k]);
      const cell = (name, k, d) => { const v = sel(name)[k]; return el("td", { class: "n", text: num(v, d) + (v === best(k) ? " *" : "") }); };
      body.append(el("tr", { class: cls || "" }, el("td", { text: label }), cell("home_only", "mae", 2), cell("elo", "mae", 2), cell("xgboost", "mae", 2), cell("home_only", "brier_p_home_win", 4), cell("elo", "brier_p_home_win", 4), cell("xgboost", "brier_p_home_win", 4)));
    };
    folds.forEach((f, i) => row(f + " (train " + int(D.timeline.folds[i].n_train) + ", validate " + int(D.timeline.folds[i].n_val) + ")", (name) => m[name].folds[f]));
    row("Pooled (" + int(D.dev.n_oof) + " games)", (name) => m[name].pooled, "sum");
    t.append(body);
    const wrap = $("#tbl-dev");
    wrap.replaceChildren(t, el("p", { class: "small", style: "padding:6px 10px 0", text: "* lowest value in the row. All models use the pooled sigma of their own out-of-fold residuals for the Brier score." }));
  }

  function relSpec(tables, target, names, W) {
    const rows = [];
    for (const [key, label] of names) for (const r of tables[key][target]) rows.push({ model: label, p: r.mean_predicted, obs: r.observed, lo: r.lo, hi: r.hi, n: r.n });
    const vals = rows.flatMap((r) => [r.p, r.lo, r.hi]);
    const lo = Math.max(0, Math.floor((Math.min(...vals) - 0.02) * 20) / 20), hi = Math.min(1, Math.ceil((Math.max(...vals) + 0.02) * 20) / 20);
    const dom = [lo, hi];
    const color = { field: "model", type: "nominal", scale: { domain: names.map((n) => n[1]), range: names.map((n) => T.modelColor[n[0]]) }, legend: { title: null, orient: "top" } };
    const w = Math.max(180, Math.min(W - 64, 360));
    return {
      width: w, height: w, data: { values: rows },
      layer: [
        { data: { values: [{ a: lo, b: lo }, { a: hi, b: hi }] }, mark: { type: "line", color: T.muted, strokeWidth: 1.5 }, encoding: { x: { field: "a", type: "quantitative", scale: { domain: dom } }, y: { field: "b", type: "quantitative", scale: { domain: dom } } } },
        { mark: { type: "rule", strokeWidth: 1.5, opacity: 0.75 }, encoding: { x: { field: "p", type: "quantitative", scale: { domain: dom, nice: false }, axis: { title: "Predicted probability", format: ".0%" } }, y: { field: "lo", type: "quantitative", scale: { domain: dom }, axis: { title: "Observed rate", format: ".0%" } }, y2: { field: "hi" }, color } },
        { mark: { type: "line", strokeWidth: 2 }, encoding: { x: { field: "p", type: "quantitative", scale: { domain: dom } }, y: { field: "obs", type: "quantitative", scale: { domain: dom } }, color } },
        { mark: { type: "point", filled: true, size: 60, opacity: 1, stroke: T.surface, strokeWidth: 2 }, encoding: { x: { field: "p", type: "quantitative", scale: { domain: dom } }, y: { field: "obs", type: "quantitative", scale: { domain: dom } }, color, tooltip: [{ field: "model", title: "Model" }, { field: "p", title: "Mean predicted", format: ".1%" }, { field: "obs", title: "Observed", format: ".1%" }, { field: "lo", title: "95% low", format: ".1%" }, { field: "hi", title: "95% high", format: ".1%" }, { field: "n", title: "Games in bin" }] } },
      ],
    };
  }

  function reliability(prefix, tables, names) {
    const TITLES = { p_home_win: "P(home win)", p_blowout: "P(blowout)", p_tight: "P(tight game)" };
    for (const target of Object.keys(TITLES)) {
      const rows = [];
      for (const [key, label] of names) for (const r of tables[key][target]) rows.push({ model: label, bin: r.bin + 1, p: r.mean_predicted, obs: r.observed, lo: r.lo, hi: r.hi, n: r.n });
      figure("#f-rel-" + prefix + "-" + target, {
        title: TITLES[target], desc: "Diagonal: perfect reliability. Each point is a bin of about " + int(rows[0].n) + " games.",
        build: (W) => relSpec(tables, target, names, W),
        table: { cols: [{ h: "Model", k: "model" }, { h: "Bin", k: "bin", num: true }, { h: "Predicted", k: "p", f: (v) => pct(v, 1), num: true }, { h: "Observed", k: "obs", f: (v) => pct(v, 1), num: true }, { h: "95% interval", k: (r) => pct(r.lo, 1) + " to " + pct(r.hi, 1) }, { h: "Games", k: "n", num: true }], rows },
      });
    }
  }

  function muVsMargin() {
    const names = [["xgboost", "XGBoost model"], ["elo", "Elo baseline"]];
    const rows = [];
    for (const [key, label] of names) D.dev.mu_vs_margin[key].forEach((r, i) => rows.push({ model: label, bin: i + 1, mu: r.mu, margin: r.margin, lo: r.margin - 1.96 * r.se, hi: r.margin + 1.96 * r.se, n: r.n }));
    const vals = rows.flatMap((r) => [r.mu, r.lo, r.hi]);
    const dom = [Math.floor(Math.min(...vals) - 1), Math.ceil(Math.max(...vals) + 1)];
    figure("#f-muvs", {
      title: "Mean actual margin against mean predicted margin (mu), 12 groups of games",
      desc: "Diagonal: perfect. Bars are 95% intervals of the mean margin in each group. Each group holds about 440 out-of-fold games.",
      build: (W) => {
        const w = Math.max(200, Math.min(W - 64, 520));
        const color = { field: "model", type: "nominal", scale: { domain: names.map((n) => n[1]), range: [T.s1, T.s2] }, legend: { title: null } };
        const sc = { domain: dom, nice: false };
        return {
          width: w, height: Math.min(w, 340), data: { values: rows },
          layer: [
            { data: { values: [{ a: dom[0] }, { a: dom[1] }] }, mark: { type: "line", color: T.muted, strokeWidth: 1.5 }, encoding: { x: { field: "a", type: "quantitative", scale: sc }, y: { field: "a", type: "quantitative", scale: sc } } },
            { mark: { type: "rule", strokeWidth: 1.5, opacity: 0.7 }, encoding: { x: { field: "mu", type: "quantitative", scale: sc, axis: { title: "Mean predicted margin, mu (points)" } }, y: { field: "lo", type: "quantitative", scale: sc, axis: { title: "Mean actual margin (points)" } }, y2: { field: "hi" }, color } },
            { mark: { type: "point", filled: true, size: 60, opacity: 1, stroke: T.surface, strokeWidth: 2 }, encoding: { x: { field: "mu", type: "quantitative", scale: sc }, y: { field: "margin", type: "quantitative", scale: sc }, color, tooltip: [{ field: "model", title: "Model" }, { field: "mu", title: "Mean mu", format: ".1f" }, { field: "margin", title: "Mean actual margin", format: ".1f" }, { field: "n", title: "Games" }] } },
          ],
        };
      },
      table: { cols: [{ h: "Model", k: "model" }, { h: "Group", k: "bin", num: true }, { h: "Mean mu", k: "mu", f: (v) => num(v, 1), num: true }, { h: "Mean actual margin", k: "margin", f: (v) => num(v, 1), num: true }, { h: "Games", k: "n", num: true }], rows },
    });
  }

  function residuals() {
    const r = D.dev.residuals, s = D.dev.sigma;
    const hist = r.hist.map((h) => ({ ...h, mid: (h.lo + h.hi) / 2 }));
    figure("#f-resid", {
      title: "Residuals against the Normal curve the model assumes",
      desc: "Residual = actual margin minus mu, out of fold. Bars: observed games per 2-point bin. Line: games expected under Normal(0, sigma = " + num(s, 2) + ").",
      build: (W) => {
        const xs = { domain: [-45, 45], nice: false };
        const color = { field: "series", type: "nominal", scale: { domain: ["Observed residuals", "Normal(0, sigma) expected"], range: [T.s1, T.ink2] }, legend: { title: null } };
        const obs = hist.map((h) => ({ ...h, series: "Observed residuals" })), exp = hist.map((h) => ({ ...h, series: "Normal(0, sigma) expected" }));
        return {
          width: Math.max(180, W - 70), height: 240,
          layer: [
            { data: { values: obs }, mark: { type: "bar", stroke: T.surface, strokeWidth: 1, opacity: 0.9, clip: true }, encoding: { x: { field: "lo", type: "quantitative", bin: { binned: true, step: 2 }, scale: xs, axis: { title: "Residual (points)", values: [-40, -30, -20, -10, 0, 10, 20, 30, 40] } }, x2: { field: "hi" }, y: { field: "count", type: "quantitative", axis: { title: "Games per 2-point bin" } }, color, tooltip: [{ field: "lo", title: "From" }, { field: "hi", title: "To" }, { field: "count", title: "Observed games" }, { field: "expected", title: "Expected games", format: ".0f" }] } },
            { data: { values: exp }, mark: { type: "line", interpolate: "monotone", clip: true }, encoding: { x: { field: "mid", type: "quantitative", scale: xs }, y: { field: "expected", type: "quantitative" }, color } },
          ],
        };
      },
      table: { cols: [{ h: "From", k: "lo", num: true }, { h: "To", k: "hi", num: true }, { h: "Observed games", k: "count", num: true }, { h: "Expected games", k: "expected", f: (v) => num(v, 1), num: true }], rows: hist },
    });
    const groups = [
      ...r.sd_by_mu.map((g, i) => ({ g: "Mu group " + (i + 1) + ": " + g.label.replace("mu ", "").replace(" to ", " to "), sd: g.sd, n: g.n })),
      ...r.sd_by_type.map((g) => ({ g: g.label, sd: g.sd, n: g.n })),
      ...r.sd_by_season.map((g) => ({ g: "Season " + g.label, sd: g.sd, n: g.n })),
    ];
    figure("#f-sd", {
      title: "Residual spread (root mean square) by group",
      desc: "The vertical line is the global sigma, " + num(s, 2) + ". A global sigma fits well when the bars end near the line.",
      build: (W) => ({
        width: Math.max(100, W - 330), height: groups.length * 20, data: { values: groups },
        layer: [
          { mark: { type: "bar", color: T.s1, cornerRadiusTopRight: 4, cornerRadiusBottomRight: 4, height: 12 }, encoding: { y: { field: "g", type: "nominal", sort: groups.map((x) => x.g), axis: { title: null, labelLimit: 280, labelFontSize: 10 } }, x: { field: "sd", type: "quantitative", scale: { domain: [0, 17] }, axis: { title: "Residual spread (points)" } }, tooltip: [{ field: "g", title: "Group" }, { field: "sd", title: "Spread", format: ".2f" }, { field: "n", title: "Games" }] } },
          { mark: { type: "text", align: "left", dx: 4, color: T.ink2, fontSize: 10 }, encoding: { y: { field: "g", type: "nominal", sort: groups.map((x) => x.g) }, x: { field: "sd", type: "quantitative" }, text: { field: "sd", type: "quantitative", format: ".1f" } } },
          { data: { values: [{ s }] }, mark: { type: "rule", color: T.ink, strokeWidth: 1.5 }, encoding: { x: { field: "s", type: "quantitative" } } },
        ],
      }),
      table: { cols: [{ h: "Group", k: "g" }, { h: "Games", k: "n", num: true }, { h: "Spread", k: "sd", f: (v) => num(v, 2), num: true }], rows: groups },
    });
    $("#resid-note").replaceChildren(
      el("p", {}, el("b", { text: "Reading. " }), "The shape is close to Normal: " + pct(r.share_within_1sd, 1) + " of the residuals fall within 1 sigma (Normal: 68.3%) and " + pct(r.share_within_2sd, 1) + " within 2 sigma (Normal: 95.4%). The mean residual is " + signed(r.mean, 2) + " points, so the model over-predicts the home margin by that amount on average. The spread changes little with the size of mu (" + num(Math.min(...r.sd_by_mu.map((g) => g.sd)), 1) + " to " + num(Math.max(...r.sd_by_mu.map((g) => g.sd)), 1) + "). It is larger in the playoffs, so one sigma is a compromise."),
      el("p", { class: "small", text: "I did not test a game-dependent sigma. The playoff result suggests it could help for playoff games." })
    );
  }

  // ------------------------------------------------------------------------------------------ test replay
  function limitRetrain() {
    const t = D.test, nPro = D.checks.rows.filter((x) => x.promoted).length, li = $("#limit-retrain");
    li.replaceChildren(el("b", { text: "Retraining is not shown to help. " }),
      nPro === 0
        ? "The replay ran " + D.checks.rows.length + " retrain checks and the gate promoted no challenger, so frozen v1 predicted the whole season. The check and promotion code ran on real timing, but it produced no gain in this season."
        : "The champion chain and frozen v1 differ by " + signed(t.boot_chain_vs_frozen.mae.mean, 3) + " MAE points over the season (95% interval " + signed(t.boot_chain_vs_frozen.mae.lo, 3) + " to " + signed(t.boot_chain_vs_frozen.mae.hi, 3) + "). The replay shows that the retrain machinery works. It does not show a gain from using it.");
  }

  function replayLead() {
    const t = D.test, m = t.models, C = D.checks.rows, nPro = C.filter((x) => x.promoted).length;
    $("#replay-lead").textContent =
      "Over " + int(t.n_games) + " games on " + t.n_game_days + " game days (" + t.first_date + " to " + t.last_date + "), the " + (SAME ? "model (frozen v1)" : "champion chain") + " has an MAE of " + num(m.chain.mae, 3) + (SAME ? "" : ", frozen v1 has " + num(m.frozen.mae, 3)) + ", and the Elo baseline has " + num(m.elo.mae, 3) +
      ". The replay ran " + C.length + " retrain checks and promoted a challenger " + nPro + (nPro === 1 ? " time. " : " times. ") +
      (SAME ? "No challenger passed the gate, so frozen v1 predicted the whole season. " : "") +
      "The replay follows these rules:";
  }

  function lineData() {
    const t = D.test, start = new Date(t.first_date + "T00:00:00Z").getTime() + 30 * 864e5;
    const rows = t.daily.filter((r) => new Date(r.date + "T00:00:00Z").getTime() >= start);
    return rows;
  }

  function timeSeriesSpec(W, rows, spec) {
    // Shared builder: lines for several series, a crosshair with a tooltip, and optional extra layers.
    const names = spec.series.map((s) => s.label);
    const long = [];
    for (const s of spec.series.slice().reverse()) for (const r of rows) long.push({ date: r.date, model: s.label, v: s.f(r) });
    const color = { field: "model", type: "nominal", scale: { domain: names, range: spec.series.map((s) => s.color) }, legend: { title: null } };
    const x = { field: "date", type: "temporal", axis: { title: null, format: "%b %Y", tickCount: 8 } };
    const tip = [{ field: "date", type: "temporal", title: "Date", format: "%d %b %Y" }, ...spec.series.map((s, i) => ({ field: "w" + i, title: s.label, format: ".2f" }))];
    const wide = rows.map((r) => { const o = { date: r.date }; spec.series.forEach((s, i) => (o["w" + i] = s.f(r))); return o; });
    return {
      width: Math.max(200, W - 70), height: spec.height || 260,
      layer: [
        { data: { values: long }, mark: { type: "line" }, encoding: { x, y: { field: "v", type: "quantitative", scale: { domain: spec.domain, nice: false }, axis: { title: spec.ytitle } }, color } },
        ...(spec.extra || []).map((l) => l(x)),
        { data: { values: wide }, mark: { type: "point", opacity: 0 }, encoding: { x, y: { field: "w0", type: "quantitative", scale: { domain: spec.domain } } }, params: [{ name: "hover", select: { type: "point", fields: ["date"], nearest: true, on: "pointerover", clear: "pointerout" } }] },
        { data: { values: wide }, transform: [{ filter: { param: "hover", empty: false } }], mark: { type: "rule", color: T.muted }, encoding: { x, tooltip: tip } },
      ],
    };
  }

  function rollingChart() {
    const rows = lineData(), t = D.test;
    const checks = D.checks.rows.map((c) => {
      const prev = t.daily.filter((r) => r.date < c.check_date).pop();
      return { date: prev ? prev.date : c.check_date, check_date: c.check_date, v: prev ? prev.roll_chain : c.rolling_mae, promoted: !!c.promoted, result: c.promoted ? "promoted" : "not promoted", trigger: c.trigger, champ: c.champion_mae, chall: c.challenger_mae };
    });
    const all = rows.flatMap((r) => [r.roll_chain, r.roll_frozen, r.roll_elo]);
    const dom = [Math.floor(Math.min(...all, t.dev_mae) - 0.5), Math.ceil(Math.max(...all) + 0.5)];
    const series = [
      { label: MODEL_NAMES.chain, f: (r) => r.roll_chain, color: "chain" },
      ...(SAME ? [] : [{ label: MODEL_NAMES.frozen, f: (r) => r.roll_frozen, color: "frozen" }]),
      { label: MODEL_NAMES.elo, f: (r) => r.roll_elo, color: "elo" },
    ];
    figure("#f-roll", {
      title: "Rolling 30-day MAE through the test season",
      desc: "Each point is the MAE of the games in the 30 days up to that date. The chart starts 30 days into the season, when the window first fills. Markers show retrain checks, placed at the line value of the day before the check. The grey line is the development MAE.",
      build: (W) => {
        const s = series.map((x) => ({ ...x, color: T.modelColor[x.color] }));
        return timeSeriesSpec(W, rows, {
          series: s, domain: dom, ytitle: "Rolling 30-day MAE (points)",
          extra: [
            () => ({ data: { values: [{ y: t.dev_mae }] }, mark: { type: "rule", color: T.muted, strokeWidth: 1.5 }, encoding: { y: { field: "y", type: "quantitative" } } }),
            () => ({ data: { values: [{ y: t.dev_mae, t: "development MAE " + num(t.dev_mae, 2) }] }, mark: { type: "text", align: "left", baseline: "bottom", dx: 4, dy: -2, color: T.ink2, fontSize: 10 }, encoding: { x: { value: 6 }, y: { field: "y", type: "quantitative" }, text: { field: "t" } } }),
            (x) => ({ data: { values: checks.filter((c) => c.promoted) }, mark: { type: "point", shape: "diamond", filled: true, size: 110, color: T.ink, opacity: 1, stroke: T.surface, strokeWidth: 2 }, encoding: { x, y: { field: "v", type: "quantitative" }, tooltip: [{ field: "check_date", title: "Check date" }, { field: "trigger", title: "Trigger" }, { field: "result", title: "Result" }] } }),
            (x) => ({ data: { values: checks.filter((c) => !c.promoted) }, mark: { type: "point", shape: "circle", filled: false, size: 80, color: T.ink, strokeWidth: 1.5, opacity: 1 }, encoding: { x, y: { field: "v", type: "quantitative" }, tooltip: [{ field: "check_date", title: "Check date" }, { field: "trigger", title: "Trigger" }, { field: "result", title: "Result" }] } }),
          ],
        });
      },
      table: { cols: [{ h: "Date", k: "date" }, { h: "Games in window", k: "n_window", num: true }, { h: "Champion chain", k: "roll_chain", f: (v) => num(v, 2), num: true }, { h: "Frozen v1", k: "roll_frozen", f: (v) => num(v, 2), num: true }, { h: "Elo baseline", k: "roll_elo", f: (v) => num(v, 2), num: true }], rows },
    });
    const mk = el("p", { class: "small", style: "margin-top:-0.6rem" }, "Marker key: a filled diamond is a check that promoted a challenger. An open circle is a check that kept the champion.");
    $("#f-roll").append(mk);
  }

  function gapChart() {
    const t = D.test, rows = t.daily.filter((r) => r.n_cum >= 100);
    const series = [
      { label: SAME ? "Elo MAE minus model MAE" : "Elo MAE minus champion chain MAE", f: (r) => r.cum_elo - r.cum_chain, color: "chain" },
      ...(SAME ? [] : [{ label: "Elo MAE minus frozen v1 MAE", f: (r) => r.cum_elo - r.cum_frozen, color: "frozen" }]),
    ];
    const all = rows.flatMap((r) => series.map((s) => s.f(r)));
    const dom = [Math.floor((Math.min(...all, 0) - 0.05) * 10) / 10, Math.ceil((Math.max(...all, 0) + 0.05) * 10) / 10];
    figure("#f-gap", {
      title: "Cumulative MAE saved against the Elo baseline",
      desc: "At each date: the MAE of the Elo baseline minus the MAE of the model, over all games so far. Above zero means the model is better. The chart starts when 100 games are in.",
      build: (W) => {
        const s = series.map((x) => ({ ...x, color: T.modelColor[x.color] }));
        return timeSeriesSpec(W, rows, {
          series: s, domain: dom, height: 220, ytitle: "Points of MAE saved (cumulative)",
          extra: [() => ({ data: { values: [{ y: 0 }] }, mark: { type: "rule", color: T.muted, strokeWidth: 1.5 }, encoding: { y: { field: "y", type: "quantitative" } } })],
        });
      },
      table: { cols: [{ h: "Date", k: "date" }, { h: "Games so far", k: "n_cum", num: true }, { h: "Elo MAE minus chain MAE", k: (r) => r.cum_elo - r.cum_chain, f: (v) => signed(v, 3), num: true }, { h: "Elo MAE minus frozen v1 MAE", k: (r) => r.cum_elo - r.cum_frozen, f: (v) => signed(v, 3), num: true }], rows },
    });
    const last = t.daily[t.daily.length - 1];
    $("#roll-note").replaceChildren(
      el("b", { text: "Reading. " }),
      (SAME ? "No challenger was promoted, so one line stands for the champion chain and for frozen v1. " : "The champion chain and frozen v1 lines overlap: they differ by " + signed(t.boot_chain_vs_frozen.mae.mean, 3) + " MAE points over the season (95% interval " + signed(t.boot_chain_vs_frozen.mae.lo, 3) + " to " + signed(t.boot_chain_vs_frozen.mae.hi, 3) + "). ") +
      (t.boot_chain_vs_elo.mae.hi < 0 ? "The model beats the Elo baseline by a small, clear margin" : "The model stays close to the Elo baseline") + ": the saving is " + num(last.cum_elo - last.cum_chain, 3) + " points at the end, and the 95% interval of the model-minus-Elo difference is " + signed(t.boot_chain_vs_elo.mae.lo, 3) + " to " + signed(t.boot_chain_vs_elo.mae.hi, 3) + ". The rolling lines rise at the end of the season, when the window holds mostly playoff games. Playoff MAE is higher (see the chart below)."
    );
  }

  function typeChart() {
    const bt = D.test.mae_by_type, types = ["Regular Season", "Playoffs"].filter((k) => bt[k]);
    const models = [["chain", MODEL_NAMES.chain], ...(SAME ? [] : [["frozen", MODEL_NAMES.frozen]]), ["elo", MODEL_NAMES.elo]];
    const rows = [];
    for (const k of types) for (const [m, label] of models) rows.push({ type: k + " (" + int(bt[k].n) + " games)", model: label, mae: bt[k][m] });
    const tableRows = Object.keys(bt).map((k) => ({ type: k, n: bt[k].n, chain: bt[k].chain, frozen: bt[k].frozen, elo: bt[k].elo, home_only: bt[k].home_only, sd: D.test.residual_sd_by_type[k] }));
    figure("#f-type", {
      title: "MAE by game type on the test season",
      desc: "Regular-season and playoff games. The table lists all four game types, including the NBA Cup final (1 game) and the play-in (6 games), which are too few to read.",
      build: (W) => ({
        width: Math.max(200, Math.min(W - 70, 560)), height: 240, data: { values: rows },
        layer: [
          { mark: { type: "bar", cornerRadiusTopLeft: 4, cornerRadiusTopRight: 4, size: 28 }, encoding: { x: { field: "type", type: "nominal", sort: rows.map((r) => r.type), axis: { title: null, labelAngle: 0, labelLimit: 220 } }, xOffset: { field: "model", type: "nominal", sort: models.map((m) => m[1]) }, y: { field: "mae", type: "quantitative", axis: { title: "MAE (points)" } }, color: { field: "model", type: "nominal", scale: { domain: models.map((m) => m[1]), range: models.map((m) => T.modelColor[m[0]]) }, legend: { title: null } }, tooltip: [{ field: "type", title: "Game type" }, { field: "model", title: "Model" }, { field: "mae", title: "MAE", format: ".2f" }] } },
          { mark: { type: "text", dy: -6, color: T.ink2, fontSize: 10 }, encoding: { x: { field: "type", type: "nominal", sort: rows.map((r) => r.type) }, xOffset: { field: "model", type: "nominal", sort: models.map((m) => m[1]) }, y: { field: "mae", type: "quantitative" }, text: { field: "mae", type: "quantitative", format: ".1f" } } },
        ],
      }),
      table: { cols: [{ h: "Game type", k: "type" }, { h: "Games", k: "n", num: true }, { h: "Champion chain", k: "chain", f: (v) => num(v, 2), num: true }, { h: "Frozen v1", k: "frozen", f: (v) => num(v, 2), num: true }, { h: "Elo baseline", k: "elo", f: (v) => num(v, 2), num: true }, { h: "Home-only", k: "home_only", f: (v) => num(v, 2), num: true }, { h: "Residual spread, chain", k: "sd", f: (v) => num(v, 2), num: true }], rows: tableRows },
    });
  }

  function checksTable() {
    const C = D.checks, hasP = C.has_p_better;
    const cols = [
      { h: "Check date", k: "check_date" }, { h: "Trigger", k: "trigger" },
      { h: "Rolling MAE before check", k: "rolling_mae", f: (v) => num(v, 2), num: true },
      { h: "Champion MAE (30 days)", k: "champion_mae", f: (v) => num(v, 3), num: true },
      { h: "Challenger MAE (30 days)", k: "challenger_mae", f: (v) => num(v, 3), num: true },
      { h: "Challenger minus champion", k: (r) => r.challenger_mae - r.champion_mae, f: (v) => signed(v, 3), num: true },
      ...(hasP ? [{ h: "P(better) MAE", k: "p_better_mae", f: (v) => num(v, 2), num: true }, { h: "P(better) Brier", k: "p_better_brier", f: (v) => num(v, 2), num: true }] : []),
      { h: "Promoted", k: (r) => (r.promoted ? "yes, v" + r.new_version : "no") },
      { h: "Columns flagged for drift", k: "n_drift_flags", num: true },
    ];
    $("#tbl-checks").replaceChildren(tableEl(cols, C.rows),
      el("p", { class: "small", style: "padding:6px 10px 0", text: hasP ? "P(better) is the share of 1,000 bootstrap resamples of the 30-day window in which the challenger beats the champion. The gate needs at least 0.90 on both MAE and Brier score." : "This table comes from the earlier rule (promote when the challenger is lower on both MAE and Brier score). The columns P(better) MAE and P(better) Brier appear after the next replay run." }));
  }

  // ------------------------------------------------------------------------------------------ monitoring
  function driftCharts() {
    const dr = D.drift, famOf = {}, famOrder = {};
    D.features.families.forEach((f, fi) => f.columns.forEach((c, ci) => { famOf[c.col] = f.name; famOrder[c.col] = fi * 100 + ci; }));
    const feats = dr.features.slice().sort((a, b) => (famOrder[a] ?? 9999) - (famOrder[b] ?? 9999));
    const label = (d) => { const [y, m, dd] = d.split("-"); return dd + " " + ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][+m - 1]; };
    const rows = dr.rows.map(([f, d, psi, flag]) => ({ feature: f, family: famOf[f] || "other", date: d, dlabel: label(d), psi, flagged: flag }));
    const dates = dr.dates.map(label);
    const counts = dr.dates.map((d, i) => ({ date: dates[i], n: rows.filter((r) => r.date === d && r.flagged).length, total: feats.length }));
    const seasonal = !dr.features.some((f) => f.startsWith("is_"));
    $("#drift-note").replaceChildren(
      el("div", { class: "callout info" },
        el("p", {}, el("b", { text: "Reference used for the data below. " }), seasonal
          ? "Training games at the same point of the season as the 30-day window: their day number within the season lies between the first and the last day number of the window. The game-type flags are not checked."
          : "All training games. The replay code now compares with training games at the same point of the season and skips the game-type flags. The next replay run will produce that version.")));
    const fl = counts.map((c) => c.n), firstHigh = counts.find((c) => c.n >= feats.length / 4);
    $("#drift-note").append(el("p", {}, el("b", { text: "What the heatmap shows. " }),
      "The replay flagged between " + Math.min(...fl) + " and " + Math.max(...fl) + " of " + feats.length + " columns per check. " +
      (firstHigh ? "From the check of " + firstHigh.date + " on, a quarter or more of the columns are flagged. " : "") +
      "I did not investigate the cause. One candidate is that playoff games enter the 30-day window late in the season. A flag does not mean the model got worse: the MAE chart above rises over the same weeks, but I did not test a link."));
    figure("#f-drift-count", {
      title: "Columns flagged (PSI above 0.2) at each retrain check",
      desc: "Out of " + feats.length + " checked columns.",
      build: (W) => ({
        width: Math.max(200, W - 70), height: 150, data: { values: counts },
        layer: [
          { mark: { type: "bar", color: T.s1, cornerRadiusTopLeft: 4, cornerRadiusTopRight: 4, size: 18 }, encoding: { x: { field: "date", type: "ordinal", sort: dates, axis: { title: null, labelAngle: -60 } }, y: { field: "n", type: "quantitative", scale: { domain: [0, feats.length] }, axis: { title: "Columns flagged" } }, tooltip: [{ field: "date", title: "Check date" }, { field: "n", title: "Columns flagged" }, { field: "total", title: "Columns checked" }] } },
          { mark: { type: "text", dy: -6, color: T.ink2, fontSize: 10 }, encoding: { x: { field: "date", type: "ordinal", sort: dates }, y: { field: "n", type: "quantitative" }, text: { field: "n", type: "quantitative" } } },
        ],
      }),
      table: { cols: [{ h: "Check date", k: "date" }, { h: "Columns flagged", k: "n", num: true }, { h: "Columns checked", k: "total", num: true }], rows: counts },
    });
    figure("#f-drift", {
      title: "PSI of each input column at each retrain check",
      desc: "Rows are input columns, grouped by feature family. Columns are check dates. A larger PSI has more contrast with the background: darker blue on the light theme, lighter blue on the dark theme. A dot marks a flagged cell (PSI above 0.2).",
      scrollX: true, minWidth: 560,
      build: (W) => {
        const w = Math.max(380, W - 235);
        const sort = dates;
        return {
          width: w, height: feats.length * 14, data: { values: rows },
          layer: [
            { mark: { type: "rect", stroke: T.surface, strokeWidth: 1 }, encoding: { x: { field: "dlabel", type: "ordinal", sort, axis: { title: null, labelAngle: -60, orient: "top" } }, y: { field: "feature", type: "nominal", sort: feats, axis: { title: null, labelFontSize: 9, labelLimit: 175, labelPadding: 2, minExtent: 150 } }, color: { field: "psi", type: "quantitative", scale: { type: "threshold", domain: [0.1, 0.2, 0.5, 1], range: T.seq }, legend: { title: "PSI", orient: "bottom", direction: "horizontal", format: ".1f", gradientLength: 220 } }, tooltip: [{ field: "feature", title: "Column" }, { field: "family", title: "Family" }, { field: "dlabel", title: "Check date" }, { field: "psi", title: "PSI", format: ".3f" }, { field: "flagged", title: "Flagged" }] } },
            { transform: [{ filter: "datum.flagged" }], mark: { type: "circle", size: 16, color: T.surface, opacity: 1 }, encoding: { x: { field: "dlabel", type: "ordinal", sort }, y: { field: "feature", type: "nominal", sort: feats } } },
          ],
        };
      },
      table: { cols: [{ h: "Column", k: "feature" }, { h: "Check date", k: "date" }, { h: "PSI", k: "psi", f: (v) => num(v, 3), num: true }, { h: "Flagged", k: (r) => (r.flagged ? "yes" : "no") }], rows },
    });
  }

  // ------------------------------------------------------------------------------------------ page chrome
  function themeButton() {
    const btn = $("#themebtn"), order = ["auto", "light", "dark"];
    let cur = "auto";
    try { cur = localStorage.getItem("nbaxp-theme") || "auto"; } catch (e) { /* storage blocked */ }
    const apply = () => {
      if (cur === "auto") document.documentElement.removeAttribute("data-theme");
      else document.documentElement.setAttribute("data-theme", cur);
      btn.textContent = "Theme: " + cur;
    };
    apply();
    btn.addEventListener("click", () => {
      cur = order[(order.indexOf(cur) + 1) % 3];
      try { localStorage.setItem("nbaxp-theme", cur); } catch (e) { /* ignore */ }
      apply();
      themeChanged();
    });
  }
  function themeChanged() {
    readTheme();
    redrawAll(false);
  }
  function scrollSpy() {
    const links = $$("nav.toc a"), map = new Map(links.map((a) => [a.getAttribute("href").slice(1), a]));
    const io = new IntersectionObserver((es) => {
      es.forEach((e) => { if (e.isIntersecting) { links.forEach((a) => a.classList.remove("on")); map.get(e.target.id).classList.add("on"); } });
    }, { rootMargin: "-20% 0px -70% 0px" });
    map.forEach((_, id) => io.observe(document.getElementById(id)));
  }

  // ------------------------------------------------------------------------------------------ main
  async function main() {
    // The theme button must work before the data arrives.
    try { const s = localStorage.getItem("nbaxp-theme"); if (s && s !== "auto") document.documentElement.setAttribute("data-theme", s); } catch (e) { /* ignore */ }
    readTheme();
    try {
      await load();
    } catch (e) {
      $("#status").textContent = "The data files did not load: " + e.message;
      return;
    }
    if (typeof vegaEmbed === "undefined") $("#status").textContent = "The chart library did not load. The numbers still show; the charts do not.";
    else $("#status").textContent = "";
    bindNumbers();
    limitRetrain();
    themeButton();
    scrollSpy();
    tiles();
    timelineSVG(); asofSVG(); retrainSVG();
    decisionEvidence();
    replayLead();
    checksTable();
    if (typeof vegaEmbed === "undefined") return;
    gameCard();
    famCards();
    importance();
    gridChart();
    devTable();
    reliability("dev", D.dev.calibration, [["xgboost", "XGBoost model"], ["elo", "Elo baseline"]]);
    muVsMargin();
    residuals();
    rollingChart();
    gapChart();
    reliability("test", D.test.calibration, [["chain", MODEL_NAMES.chain], ["elo", MODEL_NAMES.elo]]);
    typeChart();
    driftCharts();
    let timer = null;
    window.addEventListener("resize", () => { clearTimeout(timer); timer = setTimeout(() => redrawAll(true), 200); });
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", themeChanged);
  }
  main();
})();
