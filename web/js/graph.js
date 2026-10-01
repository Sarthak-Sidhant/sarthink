// Memory graph: a 2D view of the precomputed 3D layout, projected onto its two widest axes (PCA).
// The bulk of the graph (~100k nodes and edges) draws with WebGL, so every frame is cheap; a 2D canvas
// on top draws only what needs text or emphasis: "you", highlighted nodes, hover ring and labels.
import { css, esc, num, platformOf, platformLabel, cleanName } from './util.js';

const PERSON_GROUPS = /_user$|^me$/;

// ─── tiny CSV reader (quoted fields, "" escapes, newlines inside quotes) ──────
function parseCSV(text) {
  const rows = []; let row = [], field = '', i = 0, q = false;
  const n = text.length;
  while (i < n) {
    const c = text.charCodeAt(i);
    if (q) {
      if (c === 34) { if (text.charCodeAt(i + 1) === 34) { field += '"'; i += 2; continue; } q = false; i++; continue; }
      const j = text.indexOf('"', i);
      if (j < 0) { field += text.slice(i); break; }
      field += text.slice(i, j); i = j; continue;
    }
    if (c === 34) { q = true; i++; }
    else if (c === 44) { row.push(field); field = ''; i++; }
    else if (c === 10 || c === 13) {
      if (c === 13 && text.charCodeAt(i + 1) === 10) i++;
      row.push(field); field = ''; if (row.length > 1 || row[0] !== '') rows.push(row); row = []; i++;
    } else {
      let j = i; while (j < n) { const d = text.charCodeAt(j); if (d === 44 || d === 10 || d === 13 || d === 34) break; j++; }
      field += text.slice(i, j); i = j;
    }
  }
  if (field !== '' || row.length) { row.push(field); rows.push(row); }
  return rows;
}
async function fetchCSV(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url.split('/').pop()}: HTTP ${r.status}`);
  const rows = parseCSV(await r.text());
  const head = rows.shift();
  return { col: Object.fromEntries(head.map((h, i) => [h, i])), rows };
}

// colour strings → [r, g, b, a] in 0..1
function rgba(s) {
  s = String(s).trim();
  let m = s.match(/^#([0-9a-f]{3}|[0-9a-f]{6})$/i);
  if (m) {
    let h = m[1]; if (h.length === 3) h = [...h].map(c => c + c).join('');
    return [parseInt(h.slice(0, 2), 16) / 255, parseInt(h.slice(2, 4), 16) / 255, parseInt(h.slice(4, 6), 16) / 255, 1];
  }
  m = s.match(/rgba?\(([^)]+)\)/);
  if (m) { const p = m[1].split(',').map(parseFloat); return [p[0] / 255, p[1] / 255, p[2] / 255, p[3] ?? 1]; }
  return [0.5, 0.5, 0.5, 1];
}

// Level of detail: every node/edge carries a "lod" weight (its message count); anything under u_lod is
// pushed off-screen in the vertex shader, so zooming out shows the meaningful skeleton, zooming in the rest.
// points: solid core, darker rim for contrast, a faint halo
const VS_POINTS = `
attribute vec2 a_pos; attribute vec4 a_col; attribute float a_size; attribute float a_lod;
uniform vec2 u_center; uniform float u_scale; uniform vec2 u_half; uniform float u_dpr; uniform float u_zoom; uniform float u_dim; uniform float u_lod;
varying vec4 v_col; varying float v_px;
void main() {
  if (a_lod < u_lod) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); gl_PointSize = 0.0; v_col = vec4(0.0); v_px = 1.0; return; }
  vec2 p = (a_pos - u_center) * u_scale / u_half;
  gl_Position = vec4(p.x, -p.y, 0.0, 1.0);
  float px = max(2.4, a_size * u_zoom * 2.0) * u_dpr * 1.5;
  gl_PointSize = px; v_px = px;
  v_col = vec4(a_col.rgb, a_col.a * u_dim);
}`;
const FS_POINTS = `
precision mediump float;
varying vec4 v_col; varying float v_px;
void main() {
  float d = length(gl_PointCoord * 2.0 - 1.0);
  float edge = 2.0 / v_px;
  float core = 1.0 - smoothstep(0.66 - edge, 0.66 + edge, d);
  float rim = smoothstep(0.46, 0.62, d) * core;
  float halo = (1.0 - smoothstep(0.6, 1.0, d)) * 0.16;
  float a = v_col.a * max(core, halo);
  if (a <= 0.004) discard;
  vec3 rgb = mix(v_col.rgb, v_col.rgb * 0.62, rim);
  gl_FragColor = vec4(rgb * a, a);
}`;
// edges: coloured by their endpoints' platforms, very faint
const VS_LINES = `
attribute vec2 a_pos; attribute vec4 a_col; attribute float a_lod;
uniform vec2 u_center; uniform float u_scale; uniform vec2 u_half; uniform float u_lod;
varying vec4 v_col;
void main() {
  if (a_lod < u_lod) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); v_col = vec4(0.0); return; }
  vec2 p = (a_pos - u_center) * u_scale / u_half; gl_Position = vec4(p.x, -p.y, 0.0, 1.0); v_col = a_col;
}`;
const FS_LINES = `
precision mediump float; uniform float u_alpha; varying vec4 v_col;
void main() { gl_FragColor = vec4(v_col.rgb * u_alpha, u_alpha); }`;

export class Graph {
  constructor(canvas, { onClick, onHover, onInfo } = {}) {
    this.cv = canvas;
    this.ov = document.createElement('canvas');
    this.ov.id = 'graph-overlay';
    canvas.after(this.ov);
    this.ctx = this.ov.getContext('2d');
    this.gl = canvas.getContext('webgl', { antialias: true, premultipliedAlpha: true, alpha: true });
    this.onClick = onClick; this.onHover = onHover; this.onInfo = onInfo;
    this.n = 0; this.view = { cx: 0, cy: 0, s: 1 }; this.hi = null; this.focus = null; this.hover = -1;
    this.anim = null; this.dpr = Math.min(2, window.devicePixelRatio || 1); this.frameReq = 0; this.glDirty = true;
    this._bindInput();
    new ResizeObserver(() => this.resize()).observe(canvas.parentElement);
  }

  // ─── data ────────────────────────────────────────────────────────────────
  async load(nodesUrl, edgesUrl) {
    if (!this.gl) throw new Error('this browser has WebGL turned off');
    const [N, E] = await Promise.all([fetchCSV(nodesUrl), fetchCSV(edgesUrl)]);
    const n = this.n = N.rows.length, c = N.col;
    this.ids = new Array(n); this.labels = new Array(n); this.groups = new Array(n); this.plat = new Array(n);
    this.size = new Float32Array(n); this.x = new Float32Array(n); this.y = new Float32Array(n);
    this.isPerson = new Uint8Array(n); this.index = new Map();
    const P = new Float64Array(n * 3);
    for (let i = 0; i < n; i++) {
      const d = N.rows[i];
      this.ids[i] = d[c.id]; this.labels[i] = d[c.label]; this.groups[i] = d[c.group];
      this.size[i] = +d[c.size] || 1;
      this.isPerson[i] = PERSON_GROUPS.test(d[c.group]) ? 1 : 0;
      this.plat[i] = platformOf(d[c.group]);
      P[i * 3] = +d[c.layout_x] || 0; P[i * 3 + 1] = +d[c.layout_y] || 0; P[i * 3 + 2] = +d[c.layout_z] || 0;
      this.index.set(d[c.id], i);
    }
    this._project(P);
    this.me = this.groups.indexOf('me');

    // edges + adjacency (CSR) for highlight edges
    const es = new Int32Array(E.rows.length), ed = new Int32Array(E.rows.length);
    let m = 0;
    for (const e of E.rows) {
      const a = this.index.get(e[E.col.source]), b = this.index.get(e[E.col.target]);
      if (a !== undefined && b !== undefined) { es[m] = a; ed[m] = b; m++; }
    }
    this.es = es.subarray(0, m); this.ed = ed.subarray(0, m);
    const start = new Int32Array(n + 1);
    for (let k = 0; k < m; k++) { start[this.es[k] + 1]++; start[this.ed[k] + 1]++; }
    for (let i = 0; i < n; i++) start[i + 1] += start[i];
    this.adjStart = start; this.adj = new Int32Array(m * 2);
    const fill = start.slice(0, n);
    for (let k = 0; k < m; k++) { this.adj[fill[this.es[k]]++] = this.ed[k]; this.adj[fill[this.ed[k]]++] = this.es[k]; }

    // radius (css px at the fitted zoom) by message count
    this.r = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      this.r[i] = this.isPerson[i] ? Math.min(9, 1.3 + Math.sqrt(Math.log1p(this.size[i])) * 1.25)
        : 0.9 + Math.min(2, Math.log1p(this.size[i]) * 0.18);
    }
    if (this.me >= 0) this.r[this.me] = 7;
    // lod weight: people need 2+ messages, chats 3+, to appear at the overview zoom
    this.lod = new Float32Array(n);
    for (let i = 0; i < n; i++) this.lod[i] = this.isPerson[i] ? this.size[i] : this.size[i] / 1.5;
    if (this.me >= 0) this.lod[this.me] = 1e9;
    this.byImportance = [];
    for (let i = 0; i < n; i++) if (this.isPerson[i]) this.byImportance.push(i);
    this.byImportance.sort((a, b) => this.size[b] - this.size[a]);
    this._buildGrid();
    this._initGL();
    this.people = this.byImportance.map(i => ({
      id: this.ids[i], pid: +this.ids[i].slice(2), name: this.labels[i], platform: this.plat[i],
      group: this.groups[i], messages: this.size[i],
    }));
    this.resize();
    this.fit(false);
    if (this._pending) { const [ids, o] = this._pending; this._pending = null; this.highlight(ids, o); }
    this.onInfo?.(`${num(this.people.length)} people · ${num(n - this.people.length)} chats`);
    return this;
  }

  _project(P) {
    // PCA: the two directions the 3D layout spreads most along ("me" stays at the origin)
    const n = this.n, step = Math.max(1, Math.floor(n / 20000));
    const C = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    for (let i = 0; i < n; i += step) {
      const v0 = P[i * 3], v1 = P[i * 3 + 1], v2 = P[i * 3 + 2], v = [v0, v1, v2];
      for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) C[a][b] += v[a] * v[b];
    }
    const mul = v => [0, 1, 2].map(a => C[a][0] * v[0] + C[a][1] * v[1] + C[a][2] * v[2]);
    const norm = v => { const l = Math.hypot(...v) || 1; return v.map(x => x / l); };
    const power = against => {
      let v = against ? [0.2, 1, 0.3] : [1, 0.3, 0.2];
      for (let it = 0; it < 60; it++) {
        v = mul(v);
        if (against) { const d = v[0] * against[0] + v[1] * against[1] + v[2] * against[2]; v = v.map((x, k) => x - d * against[k]); }
        v = norm(v);
      }
      return v;
    };
    const e1 = power(), e2 = power(e1);
    for (let i = 0; i < n; i++) {
      const a = P[i * 3], b = P[i * 3 + 1], c = P[i * 3 + 2];
      this.x[i] = a * e1[0] + b * e1[1] + c * e1[2];
      this.y[i] = a * e2[0] + b * e2[1] + c * e2[2];
    }
  }

  _buildGrid() {
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (let i = 0; i < this.n; i++) {
      const x = this.x[i], y = this.y[i];
      if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
    }
    const cells = 256, cw = (x1 - x0) / cells || 1, ch = (y1 - y0) / cells || 1;
    const cell = new Int32Array(this.n), start = new Int32Array(cells * cells + 1);
    for (let i = 0; i < this.n; i++) {
      const cx = Math.min(cells - 1, Math.floor((this.x[i] - x0) / cw)), cy = Math.min(cells - 1, Math.floor((this.y[i] - y0) / ch));
      cell[i] = cy * cells + cx; start[cell[i] + 1]++;
    }
    for (let k = 0; k < cells * cells; k++) start[k + 1] += start[k];
    const items = new Int32Array(this.n), fill = start.slice(0, cells * cells);
    for (let i = 0; i < this.n; i++) items[fill[cell[i]]++] = i;
    this.grid = { x0, y0, cw, ch, cells, start, items };
    this.bounds = { x0, y0, x1, y1 };
  }

  // ─── WebGL (the static bulk: every edge and node) ──────────────────────────
  _initGL() {
    const gl = this.gl;
    const prog = (vs, fs) => {
      const p = gl.createProgram();
      for (const [type, src] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]]) {
        const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
        if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
        gl.attachShader(p, s);
      }
      gl.linkProgram(p);
      const u = {}, a = {};
      for (let i = 0; i < gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS); i++) { const nm = gl.getActiveUniform(p, i).name; u[nm] = gl.getUniformLocation(p, nm); }
      for (let i = 0; i < gl.getProgramParameter(p, gl.ACTIVE_ATTRIBUTES); i++) { const nm = gl.getActiveAttrib(p, i).name; a[nm] = gl.getAttribLocation(p, nm); }
      return { p, u, a };
    };
    this.P = prog(VS_POINTS, FS_POINTS);
    this.L = prog(VS_LINES, FS_LINES);
    // draw order: chats first, then people from smallest to biggest ("me" is drawn on the overlay)
    const order = [];
    for (let i = 0; i < this.n; i++) if (!this.isPerson[i]) order.push(i);
    for (let k = this.byImportance.length - 1; k >= 0; k--) if (this.byImportance[k] !== this.me) order.push(this.byImportance[k]);
    this.order = order;
    const pos = new Float32Array(order.length * 2), sz = new Float32Array(order.length);
    order.forEach((i, k) => { pos[k * 2] = this.x[i]; pos[k * 2 + 1] = this.y[i]; sz[k] = this.r[i]; });
    const lod = new Float32Array(order.length);
    order.forEach((i, k) => { lod[k] = this.lod[i]; });
    this.bufPos = this._buf(pos); this.bufSize = this._buf(sz); this.bufLod = this._buf(lod);
    const lp = new Float32Array(this.es.length * 4);
    for (let k = 0; k < this.es.length; k++) {
      lp[k * 4] = this.x[this.es[k]]; lp[k * 4 + 1] = this.y[this.es[k]]; lp[k * 4 + 2] = this.x[this.ed[k]]; lp[k * 4 + 3] = this.y[this.ed[k]];
    }
    this.bufLines = this._buf(lp);
    const ll = new Float32Array(this.es.length * 2);
    for (let k = 0; k < this.es.length; k++) ll[k * 2] = ll[k * 2 + 1] = Math.min(this.lod[this.es[k]], this.lod[this.ed[k]]);
    this.bufLineLod = this._buf(ll);
    this.themeChanged(false);
  }
  _buf(data) { const gl = this.gl, b = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW); return b; }

  themeChanged(redraw = true) {
    this.colors = {
      me: css('--me'), bg: css('--graph-bg'), text: css('--text'), text2: css('--text-2'), muted: css('--muted'),
      accent: css('--accent'), accent2: css('--accent-2'), surface: css('--surface'),
      edgeAlpha: parseFloat(css('--graph-edge-alpha')) || 0.05,
      dark: document.documentElement.dataset.theme === 'dark',
      plat: { twitter: css('--series-1'), reddit: css('--series-2'), discord: css('--series-3'), facebook: css('--series-4'), instagram: css('--series-5'), other: css('--muted') },
    };
    if (this.order) {
      const platRGB = Object.fromEntries(Object.entries(this.colors.plat).map(([k, v]) => [k, rgba(v)]));
      const sky = rgba(this.colors.accent);
      const col = new Float32Array(this.order.length * 4);
      this.order.forEach((i, k) => {
        const c = platRGB[this.plat[i]] || platRGB.other;
        col.set([c[0], c[1], c[2], this.isPerson[i] ? 1 : 0.6], k * 4);
      });
      // each edge end takes its node's platform colour, so fans read as soft colour, not a grey wedge
      const lc = new Float32Array(this.es.length * 8);
      for (let k = 0; k < this.es.length; k++) {
        for (const [o, i] of [[0, this.es[k]], [4, this.ed[k]]]) {
          const c = i === this.me ? sky : (platRGB[this.plat[i]] || platRGB.other);
          lc[k * 8 + o] = c[0]; lc[k * 8 + o + 1] = c[1]; lc[k * 8 + o + 2] = c[2]; lc[k * 8 + o + 3] = 1;
        }
      }
      for (const b of [this.bufCol, this.bufLineCol]) if (b) this.gl.deleteBuffer(b);
      this.bufCol = this._buf(col); this.bufLineCol = this._buf(lc);
    }
    this.glDirty = true;
    if (redraw) this.draw();
  }

  _drawGL() {
    const gl = this.gl, v = this.view, dim = !!this.hi;
    gl.viewport(0, 0, this.cv.width, this.cv.height);
    gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);   // transparent: the CSS wash shows through
    gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    const common = pr => {
      gl.useProgram(pr.p);
      gl.uniform2f(pr.u.u_center, v.cx, v.cy); gl.uniform1f(pr.u.u_scale, v.s); gl.uniform2f(pr.u.u_half, this.w / 2, this.h / 2);
    };
    const attr = (pr, name, buf, size) => {
      gl.bindBuffer(gl.ARRAY_BUFFER, buf); gl.enableVertexAttribArray(pr.a[name]); gl.vertexAttribPointer(pr.a[name], size, gl.FLOAT, false, 0, 0);
    };
    common(this.L);
    const lodT = this.lodThreshold();
    gl.uniform1f(this.L.u.u_alpha, this.colors.edgeAlpha * (dim ? 0.12 : 1));
    gl.uniform1f(this.L.u.u_lod, lodT);
    attr(this.L, 'a_pos', this.bufLines, 2); attr(this.L, 'a_col', this.bufLineCol, 4); attr(this.L, 'a_lod', this.bufLineLod, 1);
    gl.drawArrays(gl.LINES, 0, this.es.length * 2);
    for (const a of Object.values(this.L.a)) gl.disableVertexAttribArray(a);
    const P = this.P;
    common(P);
    gl.uniform1f(P.u.u_dpr, this.dpr); gl.uniform1f(P.u.u_zoom, this.zoomFactor()); gl.uniform1f(P.u.u_dim, dim ? 0.07 : 1);
    gl.uniform1f(P.u.u_lod, lodT);
    attr(P, 'a_pos', this.bufPos, 2); attr(P, 'a_col', this.bufCol, 4); attr(P, 'a_size', this.bufSize, 1); attr(P, 'a_lod', this.bufLod, 1);
    gl.drawArrays(gl.POINTS, 0, this.order.length);
    for (const a of Object.values(P.a)) gl.disableVertexAttribArray(a);
  }

  // ─── view ────────────────────────────────────────────────────────────────
  resize() {
    const p = this.cv.parentElement, w = p.clientWidth, h = p.clientHeight;
    if (!w || !h || (w === this.w && h === this.h)) return;
    this.w = w; this.h = h;
    for (const c of [this.cv, this.ov]) { c.width = Math.round(w * this.dpr); c.height = Math.round(h * this.dpr); }
    if (this.n && !this.fitted) this.fit(false);
    this.glDirty = true; this.draw();
  }
  toScreen(i) { const v = this.view; return [(this.x[i] - v.cx) * v.s + this.w / 2, (this.y[i] - v.cy) * v.s + this.h / 2]; }
  // zoomed out: only the meaningful skeleton; each doubling of zoom halves the cut until everything shows
  lodThreshold() { const k = this.view.s / (this.fitScale || this.view.s); return k >= 3 ? 0 : 2 / Math.max(1, k); }
  visibleCount() { const t = this.lodThreshold(); let c = 0; for (let i = 0; i < this.n; i++) if (this.lod[i] >= t) c++; return c; }
  zoomFactor() { return Math.max(0.6, Math.min(3.2, Math.sqrt(this.view.s / (this.fitScale || this.view.s)))); }
  setView(v) { this.view = v; this.glDirty = true; this.draw(); }

  fit(animate = true) {
    if (!this.n || !this.w) return;
    // robust extent from a sample: ignore the outermost stragglers so they don't shrink everything
    const step = Math.max(1, Math.floor(this.n / 5000)), xs = [], ys = [];
    for (let i = 0; i < this.n; i += step) { xs.push(this.x[i]); ys.push(this.y[i]); }
    xs.sort((a, b) => a - b); ys.sort((a, b) => a - b);
    const q = (a, p) => a[Math.floor(p * (a.length - 1))];
    const box = { x0: q(xs, 0.004), x1: q(xs, 0.996), y0: q(ys, 0.004), y1: q(ys, 0.996) };
    const s = Math.min(this.w / (box.x1 - box.x0 || 1), this.h / (box.y1 - box.y0 || 1)) * 0.84;
    this.fitScale = s; this.fitted = true;
    this.goTo({ cx: (box.x0 + box.x1) / 2, cy: (box.y0 + box.y1) / 2, s }, animate);
  }
  frame(indices, animate = true) {
    let pts = indices.filter(i => i !== this.me);
    if (!pts.length) pts = indices;
    if (!pts.length || !this.w) return;
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const i of pts) { x0 = Math.min(x0, this.x[i]); x1 = Math.max(x1, this.x[i]); y0 = Math.min(y0, this.y[i]); y1 = Math.max(y1, this.y[i]); }
    const minSpan = (this.bounds.x1 - this.bounds.x0) / 14;
    const sw = Math.max(x1 - x0, minSpan), sh = Math.max(y1 - y0, minSpan);
    const s = Math.min(this.w * 0.6 / sw, this.h * 0.6 / sh, this.fitScale * 10);
    this.goTo({ cx: (x0 + x1) / 2, cy: (y0 + y1) / 2, s }, animate);
  }
  goTo(target, animate = true) {
    cancelAnimationFrame(this.anim);
    if (!animate) { this.setView({ ...target }); return; }
    const from = { ...this.view }, t0 = performance.now(), dur = 800;
    const ease = t => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
    const step = now => {
      const t = Math.min(1, (now - t0) / dur), e = ease(t);
      const ls = Math.log(from.s) + (Math.log(target.s) - Math.log(from.s)) * e;
      this.setView({ cx: from.cx + (target.cx - from.cx) * e, cy: from.cy + (target.cy - from.cy) * e, s: Math.exp(ls) });
      if (t < 1) this.anim = requestAnimationFrame(step);
    };
    this.anim = requestAnimationFrame(step);
  }

  // ─── highlight ───────────────────────────────────────────────────────────
  highlight(nodeIds, { fly = true, focus = null } = {}) {
    if (!this.n) { this._pending = [nodeIds, { fly, focus }]; return 0; }
    const idx = [...new Set((nodeIds || []).map(id => this.index.get(id)).filter(i => i !== undefined))];
    const now = performance.now(), born = new Map();
    for (const i of idx) born.set(i, this.born?.get(i) ?? now);   // newly found nodes pop in
    this.born = born;
    this.hi = idx.length ? new Set(idx) : null;
    this.focus = focus ? new Set(focus.map(id => this.index.get(id)).filter(i => i !== undefined)) : null;
    this.glDirty = true;
    if (fly && idx.length) this.frame(this.focus?.size ? [...this.focus] : idx); else this.draw();
    return idx.length;
  }
  clear() { this._pending = null; this.hi = null; this.focus = null; this.born = null; this.glDirty = true; this.fit(); }

  // ─── drawing: WebGL only when the view changed; the overlay animates on its own clock ─────
  draw() {
    if (this.frameReq) return;
    this.frameReq = requestAnimationFrame(t => { this.frameReq = 0; this._render(t); });
  }
  _render(t = performance.now()) {
    if (!this.n || !this.w || !this.order) return;
    if (this.glDirty) { this._drawGL(); this.glDirty = false; }
    const c = this.ctx;
    c.setTransform(1, 0, 0, 1, 0, 0); c.clearRect(0, 0, this.ov.width, this.ov.height);
    c.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    this._renderOverlay(c, t);
    // keep animating (pulse, flowing links) while visible; ~30 fps is plenty for ambient motion
    if (this.cv.offsetParent !== null && !document.hidden) {
      clearTimeout(this._tick);
      this._tick = setTimeout(() => this.draw(), 33);
    }
  }

  _renderOverlay(c, t) {
    const z = this.zoomFactor(), C = this.colors, labels = [];
    const me = this.me >= 0 ? this.toScreen(this.me) : null;
    if (this.hi) {
      // links: the graph's own edges among lit nodes, plus a curved line from you to each lit person
      const pairs = [];
      for (const a of this.hi) {
        for (let k = this.adjStart[a]; k < this.adjStart[a + 1]; k++) {
          const b = this.adj[k];
          if (b > a && this.hi.has(b)) pairs.push([a, b]);
        }
        if (this.isPerson[a] && a !== this.me && this.me >= 0) pairs.push([this.me, a]);
      }
      c.lineCap = 'round';
      for (const [a, b] of pairs) {
        const [x1, y1] = this.toScreen(a), [x2, y2] = this.toScreen(b);
        const dx = x2 - x1, dy = y2 - y1, len = Math.hypot(dx, dy) || 1;
        const bend = Math.min(80, len * 0.18);
        const mx = (x1 + x2) / 2 - dy / len * bend, my = (y1 + y2) / 2 + dx / len * bend;
        const ca = a === this.me ? C.accent : C.plat[this.plat[a]] || C.accent, cb = C.plat[this.plat[b]] || C.accent2;
        const g = c.createLinearGradient(x1, y1, x2, y2);
        g.addColorStop(0, ca); g.addColorStop(1, cb);
        const foc = this.focus && (this.focus.has(a) || this.focus.has(b));
        c.beginPath(); c.moveTo(x1, y1); c.quadraticCurveTo(mx, my, x2, y2);
        c.strokeStyle = g; c.globalAlpha = foc ? 0.55 : 0.28; c.lineWidth = foc ? 2.2 : 1.4; c.setLineDash([]); c.stroke();
        // a bright dash flowing outward along the link
        c.globalAlpha = foc ? 0.95 : 0.6; c.lineWidth = foc ? 2.4 : 1.6;
        c.setLineDash([10, len]); c.lineDashOffset = -((t / 14) % (len + 10));
        c.stroke();
      }
      c.setLineDash([]); c.globalAlpha = 1;
      const order = [...this.hi].sort((a, b) => this.isPerson[a] - this.isPerson[b] || this.size[a] - this.size[b]);
      for (const i of order) {
        if (i === this.me) continue;
        const [x, y] = this.toScreen(i), foc = this.focus?.has(i);
        const age = Math.min(1, (t - (this.born?.get(i) ?? 0)) / 420);
        const pop = age < 1 ? 1 + 0.6 * Math.sin(age * Math.PI) : 1;
        const r = (Math.max(this.isPerson[i] ? 5 : 3.2, this.r[i] * z) + (foc ? 2 : 0)) * pop;
        const col = C.plat[this.plat[i]] || C.muted;
        c.save();
        c.shadowColor = col; c.shadowBlur = foc ? 22 : 12;
        c.beginPath(); c.arc(x, y, r, 0, Math.PI * 2); c.fillStyle = col; c.fill();
        c.restore();
        c.beginPath(); c.arc(x, y, r, 0, Math.PI * 2); c.strokeStyle = C.dark ? 'rgba(255,255,255,.7)' : '#fff'; c.lineWidth = 1.6; c.stroke();
        if (foc) {
          const pr = r + 5 + 3 * Math.sin(t / 260);
          c.beginPath(); c.arc(x, y, pr, 0, Math.PI * 2); c.strokeStyle = C.accent2; c.globalAlpha = 0.7; c.lineWidth = 1.6; c.stroke(); c.globalAlpha = 1;
        }
      }
      const lab = [...this.hi].filter(i => i !== this.me && (this.isPerson[i] || this.hi.size <= 12 || this.focus?.has(i)))
        .sort((a, b) => (this.focus?.has(b) ? 1 : 0) - (this.focus?.has(a) ? 1 : 0) || this.isPerson[b] - this.isPerson[a] || this.size[b] - this.size[a]);
      for (const i of lab) { const [x, y] = this.toScreen(i); labels.push({ i, x, y, r: Math.max(5, this.r[i] * z) + 2, strong: true, pill: true, col: C.plat[this.plat[i]] }); }
    } else {
      let shown = 0;
      for (const i of this.byImportance) {
        if (shown >= 14) break;
        if (i === this.me) continue;
        const [x, y] = this.toScreen(i);
        if (x < 0 || x > this.w || y < 0 || y > this.h) continue;
        labels.push({ i, x, y, r: this.r[i] * z, strong: false }); shown++;
      }
    }
    // "you": a gradient orb with a slow breathing halo
    if (me) {
      const [x, y] = me, r = 7.5 * Math.min(1.35, z), b = (Math.sin(t / 700) + 1) / 2;
      const halo = c.createRadialGradient(x, y, r * 0.5, x, y, r * (3.2 + b));
      halo.addColorStop(0, 'rgba(120, 180, 255, .45)'); halo.addColorStop(0.55, 'rgba(240, 140, 195, .18)'); halo.addColorStop(1, 'rgba(240, 140, 195, 0)');
      c.beginPath(); c.arc(x, y, r * (3.2 + b), 0, Math.PI * 2); c.fillStyle = halo; c.fill();
      const g = c.createLinearGradient(x - r, y - r, x + r, y + r);
      g.addColorStop(0, '#4aa8f0'); g.addColorStop(0.5, '#9a8cf0'); g.addColorStop(1, '#ee7fb6');
      c.beginPath(); c.arc(x, y, r, 0, Math.PI * 2); c.fillStyle = g; c.fill();
      c.lineWidth = 2; c.strokeStyle = '#fff'; c.stroke();
      labels.unshift({ i: this.me, x, y, r: r + 2, strong: true, text: 'You' });
    }
    if (this.hover >= 0) {
      const [x, y] = this.toScreen(this.hover), r = Math.max(5, this.r[this.hover] * z);
      c.beginPath(); c.arc(x, y, r + 4, 0, Math.PI * 2); c.strokeStyle = C.accent2; c.lineWidth = 2; c.stroke();
    }
    this._drawLabels(c, labels);
  }

  _drawLabels(c, labels) {
    const placed = [], C = this.colors;
    c.textBaseline = 'middle'; c.lineJoin = 'round';
    for (const L of labels) {
      let t = L.text || cleanName(this.labels[L.i]);
      if (t.length > 26) t = t.slice(0, 25) + '…';
      c.font = `${L.strong ? 600 : 500} ${L.strong ? 12 : 11}px Inter, system-ui, sans-serif`;
      const w = c.measureText(t).width, pad = L.pill ? 8 : 3, x = L.x + L.r + 6, y = L.y;
      const box = [x - 2, y - 10, x + w + pad * 2 + 2, y + 10];
      if (placed.some(b => !(box[2] < b[0] || box[0] > b[2] || box[3] < b[1] || box[1] > b[3]))) continue;
      placed.push(box);
      if (L.pill) {
        c.save();
        c.shadowColor = 'rgba(40, 60, 110, .18)'; c.shadowBlur = 10; c.shadowOffsetY = 2;
        c.beginPath(); c.roundRect(x, y - 10, w + pad * 2, 20, 10);
        c.fillStyle = C.dark ? 'rgba(23, 30, 47, .92)' : 'rgba(255, 255, 255, .94)'; c.fill();
        c.restore();
        c.fillStyle = C.text; c.fillText(t, x + pad, y + 0.5);
      } else {
        c.lineWidth = 4; c.strokeStyle = C.dark ? 'rgba(16, 22, 42, .85)' : 'rgba(255, 255, 255, .9)'; c.strokeText(t, x, y);
        c.fillStyle = L.strong ? C.text : C.text2; c.fillText(t, x, y);
      }
    }
  }

  // ─── input ───────────────────────────────────────────────────────────────
  pick(mx, my) {
    if (!this.n || !this.w) return -1;
    const v = this.view, wx = (mx - this.w / 2) / v.s + v.cx, wy = (my - this.h / 2) / v.s + v.cy;
    const rad = 14 / v.s, g = this.grid, z = this.zoomFactor(), lodT = this.lodThreshold();
    const cx0 = Math.max(0, Math.floor((wx - rad - g.x0) / g.cw)), cx1 = Math.min(g.cells - 1, Math.floor((wx + rad - g.x0) / g.cw));
    const cy0 = Math.max(0, Math.floor((wy - rad - g.y0) / g.ch)), cy1 = Math.min(g.cells - 1, Math.floor((wy + rad - g.y0) / g.ch));
    let best = -1, bd = Infinity;
    for (let cy = cy0; cy <= cy1; cy++) {
      for (let cx = cx0; cx <= cx1; cx++) {
        const k = cy * g.cells + cx;
        for (let j = g.start[k]; j < g.start[k + 1]; j++) {
          const i = g.items[j];
          if (this.hi && !this.hi.has(i) && i !== this.me) continue;   // while highlighted, only lit nodes are pickable
          if (!this.hi && this.lod[i] < lodT) continue;                // culled at this zoom
          const dd = Math.hypot(this.x[i] - wx, this.y[i] - wy) * v.s - this.r[i] * z - (this.isPerson[i] ? 3 : 0);
          if (dd < 8 && dd < bd) { bd = dd; best = i; }
        }
      }
    }
    return best;
  }
  neighbours(id) {
    const i = this.index?.get(id);
    if (i === undefined) return [];
    const out = [];
    for (let k = this.adjStart[i]; k < this.adjStart[i + 1]; k++) out.push(this.ids[this.adj[k]]);
    return out;
  }
  // people (among the ~3000 you talk to most) whose name appears in the text, e.g. an answer
  peopleIn(text) {
    if (!this.n || !text) return [];
    const out = [];
    for (const i of this.byImportance.slice(0, 3000)) {
      if (i === this.me) continue;
      const name = cleanName(this.labels[i]);
      if (name.length < 4 || /^(Deleted|Deactivated)/.test(name)) continue;
      const k = text.indexOf(name);
      if (k < 0) continue;
      const before = text[k - 1], after = text[k + name.length];
      if ((!before || !/[\w@]/.test(before)) && (!after || !/\w/.test(after))) out.push(this.ids[i]);
      if (out.length >= 12) break;
    }
    return out;
  }
  infoById(id) { const i = this.index?.get(id); return i === undefined ? null : this.nodeInfo(i); }
  get meId() { return this.me >= 0 ? this.ids[this.me] : null; }
  nodeInfo(i) {
    return { id: this.ids[i], label: this.labels[i], group: this.groups[i], platform: this.plat[i], size: this.size[i], person: !!this.isPerson[i] };
  }
  tooltipHtml(i) {
    const p = this.isPerson[i];
    return `<b>${esc(cleanName(this.labels[i]))}${i === this.me ? ' (you)' : ''}</b><span>${p ? `${num(this.size[i])} messages` : 'conversation'} · ${esc(platformLabel(this.plat[i]))}</span>`;
  }

  _bindInput() {
    const cv = this.ov;
    let drag = null, moved = false;
    const pos = e => { const r = cv.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
    cv.addEventListener('pointerdown', e => {
      cv.setPointerCapture(e.pointerId);
      drag = { p: pos(e), v: { ...this.view } }; moved = false;
      cancelAnimationFrame(this.anim);
    });
    cv.addEventListener('pointermove', e => {
      const [mx, my] = pos(e);
      if (drag) {
        const dx = mx - drag.p[0], dy = my - drag.p[1];
        if (Math.abs(dx) + Math.abs(dy) > 3) { moved = true; cv.classList.add('dragging'); this.onHover?.(null); }
        if (moved) this.setView({ ...drag.v, cx: drag.v.cx - dx / drag.v.s, cy: drag.v.cy - dy / drag.v.s });
        return;
      }
      const h = this.pick(mx, my);
      if (h !== this.hover) { this.hover = h; cv.classList.toggle('hovering', h >= 0); this.draw(); }
      this.onHover?.(h >= 0 ? { html: this.tooltipHtml(h), x: mx, y: my } : null);
    });
    cv.addEventListener('pointerup', e => {
      if (!drag) return;
      cv.classList.remove('dragging');
      if (!moved) { const [mx, my] = pos(e); const i = this.pick(mx, my); if (i >= 0) this.onClick?.(this.nodeInfo(i)); }
      drag = null;
    });
    cv.addEventListener('pointercancel', () => { drag = null; cv.classList.remove('dragging'); });
    cv.addEventListener('pointerleave', () => { if (this.hover >= 0) { this.hover = -1; this.draw(); } this.onHover?.(null); });
    cv.addEventListener('wheel', e => {
      e.preventDefault();
      cancelAnimationFrame(this.anim);
      const [mx, my] = pos(e), v = this.view, fs = this.fitScale || v.s;
      const k = Math.exp(-e.deltaY * (e.ctrlKey ? 0.01 : 0.0018));
      const s = Math.max(fs * 0.4, Math.min(fs * 60, v.s * k));
      const wx = (mx - this.w / 2) / v.s + v.cx, wy = (my - this.h / 2) / v.s + v.cy;
      this.setView({ s, cx: wx - (mx - this.w / 2) / s, cy: wy - (my - this.h / 2) / s });
    }, { passive: false });
  }
}
