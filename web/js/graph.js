// Memory graph: a 2D view of the precomputed 3D layout, projected onto its two widest axes (PCA).
// The same renderer can tilt into 3D: the layout rotates about its centre with perspective, and the
// flat map is just the 3D view looked at straight down its narrowest axis, so the two morph smoothly.
// The bulk of the graph (~100k nodes and edges) draws with WebGL, so every frame is cheap; a 2D canvas
// on top draws only what needs text or emphasis: "you", highlighted nodes, hover ring and labels.
// Accounts and chats come from the graph files (rebuilt on import); people (groups of accounts, see
// docs/accounts-and-people.md) are a layer on top: one node per person at the centre of their accounts, linked
// to each. setPeople() rebuilds just that layer, so linking an account shows up at once.
import { css, esc, num, platformOf, platformLabel, cleanName } from './util.js';

const PERSON_GROUPS = /_user$|_me$|^me$|^person$/;   // drawn as people-sized nodes: accounts, my accounts, people
const ME_PERSON = 'P_1';
const ANGLE_3D = { yaw: 0.55, pitch: -0.5, p: 1 };   // the tilt the map swings to when 3D is turned on

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
// Shared projection, one camera for both modes: rotate about the target (u_r1..3 rows), then perspective
// from a camera u_dist away (u_persp 0 = the flat map, 1 = full 3D), then zoom. w < 0: behind the camera.
const PROJECT = `
uniform vec3 u_r1; uniform vec3 u_r2; uniform vec3 u_r3; uniform vec3 u_target;
uniform float u_persp; uniform float u_dist; uniform float u_focal; uniform float u_scale; uniform vec2 u_half;
vec4 project(vec3 a) {
  vec3 d = a - u_target;
  vec3 q = vec3(dot(u_r1, d), dot(u_r2, d), dot(u_r3, d));
  float depth = u_dist - q.z * u_persp, near = u_focal * 0.01;
  float f = u_focal / max(near, depth);
  return vec4(q.xy * f * u_scale / u_half, f, depth - near);
}`;
// points. Flat map: solid core, darker rim for contrast, a faint halo. 3D: the classic viewer's soft
// glowing orbs in its vivid palette, sized in world units so nearer nodes really are bigger.
const VS_POINTS = `
attribute vec3 a_pos; attribute vec4 a_col; attribute vec3 a_col3; attribute float a_size; attribute float a_glow; attribute float a_lod;
uniform float u_dpr; uniform float u_zoom; uniform float u_dim; uniform float u_lod; uniform float u_glowk;
varying vec4 v_col; varying float v_px; varying float v_glow;
${PROJECT}
void main() {
  v_glow = u_persp;
  vec4 p = project(a_pos);
  if (a_lod < u_lod || p.w <= 0.0) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); gl_PointSize = 0.0; v_col = vec4(0.0); v_px = 1.0; return; }
  gl_Position = vec4(p.x, -p.y, 0.0, 1.0);
  float flatPx = max(2.4, a_size * u_zoom * 2.0) * 1.5;
  float glowPx = clamp(a_glow * u_glowk * p.z, 1.5, 64.0);
  float px = mix(flatPx, glowPx, u_persp) * u_dpr;
  gl_PointSize = px; v_px = px;
  v_col = vec4(mix(a_col.rgb, a_col3, u_persp), mix(a_col.a, 1.0, u_persp) * u_dim);
}`;
const FS_POINTS = `
precision mediump float;
varying vec4 v_col; varying float v_px; varying float v_glow;
void main() {
  vec2 uv = gl_PointCoord * 2.0 - 1.0;
  float r2 = dot(uv, uv), d = sqrt(r2);
  float edge = 2.0 / v_px;
  float core = 1.0 - smoothstep(0.66 - edge, 0.66 + edge, d);
  float rim = smoothstep(0.46, 0.62, d) * core;
  float halo = (1.0 - smoothstep(0.6, 1.0, d)) * 0.16;
  float glow = r2 > 1.0 ? 0.0 : clamp(exp(-r2 * 2.0) + (1.0 - d) * 0.5, 0.0, 1.0);
  float a = v_col.a * mix(max(core, halo), glow, v_glow);
  if (a <= 0.004) discard;
  vec3 rgb = mix(mix(v_col.rgb, v_col.rgb * 0.62, rim), v_col.rgb, v_glow);
  gl_FragColor = vec4(rgb * a, a);
}`;
// edges: coloured by their endpoints' platforms, very faint; in 3D they fade out entirely, as in the classic view
const VS_LINES = `
attribute vec3 a_pos; attribute vec4 a_col; attribute float a_lod;
uniform float u_lod;
varying vec4 v_col;
${PROJECT}
void main() {
  vec4 p = project(a_pos);
  if (a_lod < u_lod || p.w <= 0.0) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); v_col = vec4(0.0); return; }
  gl_Position = vec4(p.x, -p.y, 0.0, 1.0);
  v_col = vec4(a_col.rgb, a_col.a * (1.0 - u_persp));
}`;
const FS_LINES = `
precision mediump float; uniform float u_alpha; varying vec4 v_col;
void main() { float a = u_alpha * v_col.a; gl_FragColor = vec4(v_col.rgb * a, a); }`;
const GLOW_WORLD = 26;   // a node's glow radius per log2(messages), in layout units (the classic viewer's scale)

export class Graph {
  constructor(canvas, { onClick, onHover, onInfo } = {}) {
    this.cv = canvas;
    this.ov = document.createElement('canvas');
    this.ov.id = 'graph-overlay';
    canvas.after(this.ov);
    this.ctx = this.ov.getContext('2d');
    this.gl = canvas.getContext('webgl', { antialias: true, premultipliedAlpha: true, alpha: true });
    this.onClick = onClick; this.onHover = onHover; this.onInfo = onInfo;
    this.n = 0; this.view = { t: [0, 0, 0], s: 1, D: 1, yaw: 0, pitch: 0, p: 0 }; this.vel = null; this.is3D = false; this.spin = false; this.hi = null; this.focus = null; this.hover = -1;
    this.anim = null; this.dpr = Math.min(2, window.devicePixelRatio || 1); this.frameReq = 0; this.glDirty = true;
    this._bindInput();
    new ResizeObserver(() => this.resize()).observe(canvas.parentElement);
  }

  // ─── data ────────────────────────────────────────────────────────────────
  async load(nodesUrl, edgesUrl, people = []) {
    if (!this.gl) throw new Error('this browser has WebGL turned off');
    const [N, E] = await Promise.all([fetchCSV(nodesUrl), fetchCSV(edgesUrl)]);
    const n = N.rows.length, c = N.col;
    const B = this.base = {
      n, ids: new Array(n), labels: new Array(n), groups: new Array(n), plat: new Array(n), size: new Float32Array(n),
      col3: new Float32Array(n * 3), P: new Float64Array(n * 3), index: new Map(), platRaw: {},
    };
    for (let i = 0; i < n; i++) {
      const d = N.rows[i];
      B.col3.set(rgba(d[c.color] || '#8899aa').slice(0, 3), i * 3);
      B.ids[i] = d[c.id]; B.labels[i] = d[c.label]; B.groups[i] = d[c.group];
      B.size[i] = +d[c.size] || 1;
      B.plat[i] = platformOf(d[c.group]);
      if (PERSON_GROUPS.test(d[c.group]) && !/_me$/.test(d[c.group]) && !B.platRaw[B.plat[i]]) B.platRaw[B.plat[i]] = d[c.color];
      B.P[i * 3] = +d[c.layout_x] || 0; B.P[i * 3 + 1] = +d[c.layout_y] || 0; B.P[i * 3 + 2] = +d[c.layout_z] || 0;
      B.index.set(d[c.id], i);
    }
    const es = [], ed = [];
    for (const e of E.rows) {
      const a = B.index.get(e[E.col.source]), b = B.index.get(e[E.col.target]);
      if (a !== undefined && b !== undefined) { es.push(a); ed.push(b); }
    }
    B.es = es; B.ed = ed;
    this.n = n; this.X = new Float32Array(n * 3); this.x = new Float32Array(n); this.y = new Float32Array(n);
    this._project(B.P);                                    // the projection comes from the accounts and chats only
    this.platRaw = B.platRaw;
    this._compose(people);
    this.resize();
    this.fit(false, this.is3D ? ANGLE_3D : null);
    if (this._pending) { const [ids, o] = this._pending; this._pending = null; this.highlight(ids, o); }
    return this;
  }

  // people changed (an account attached or detached, a person renamed or deleted): rebuild only their layer
  setPeople(people) {
    if (!this.base) { this._people = people; return; }
    const hiIds = this.hi ? [...this.hi].map(i => this.ids[i]) : null, focusIds = this.focus ? [...this.focus].map(i => this.ids[i]) : null;
    this._compose(people);
    if (hiIds) { this.hi = new Set(hiIds.map(id => this.index.get(id)).filter(i => i !== undefined)); this.born = null; }
    if (focusIds) this.focus = new Set(focusIds.map(id => this.index.get(id)).filter(i => i !== undefined));
    this.glDirty = true; this.draw();
  }

  // the base (accounts, chats, their edges) plus one node per person, linked to each of their accounts
  _compose(people) {
    const B = this.base, bn = B.n;
    const persons = (people || []).map(p => ({ ...p, members: (p.accounts || []).map(a => B.index.get(a.node)).filter(i => i !== undefined) }))
      .filter(p => p.members.length || p.node === ME_PERSON);
    const n = this.n = bn + persons.length;
    const grow = (Arr, src, k = 1) => { const a = new Arr(n * k); a.set(src.subarray(0, bn * k)); return a; };
    this.ids = B.ids.concat(persons.map(p => p.node)); this.labels = B.labels.concat(persons.map(p => p.name));
    this.groups = B.groups.concat(persons.map(() => 'person')); this.plat = B.plat.concat(persons.map(() => 'person'));
    this.size = new Float32Array(n); this.size.set(B.size);
    this.col3 = grow(Float32Array, B.col3, 3);
    this.X = grow(Float32Array, this.X, 3); this.x = grow(Float32Array, this.x); this.y = grow(Float32Array, this.y);
    this.index = new Map(B.index); this.personOf = new Map(); this.accountsOf = new Map();
    const pc = rgba(css('--person')).slice(0, 3);
    const es = B.es.slice(), ed = B.ed.slice();
    persons.forEach((p, k) => {
      const i = bn + k;
      this.index.set(p.node, i);
      this.size[i] = p.members.reduce((t, m) => t + B.size[m], 0) || p.messages || 1;
      this.col3.set(pc, i * 3);
      // where: the centre of their accounts (me: the origin); a one-account person sits just outside it
      let raw = [0, 0, 0];
      if (p.node !== ME_PERSON && p.members.length) {
        for (const m of p.members) for (let a = 0; a < 3; a++) raw[a] += B.P[m * 3 + a] / p.members.length;
        if (p.members.length === 1) { const l = Math.hypot(...raw) || 1; raw = raw.map(v => v + v / l * 60); }
      }
      this._place(i, raw);
      this.accountsOf.set(p.node, p.members.map(m => B.ids[m]));
      for (const m of p.members) { this.personOf.set(B.ids[m], p.node); es.push(i); ed.push(m); }
    });
    this.es = Int32Array.from(es); this.ed = Int32Array.from(ed);
    this.isPerson = new Uint8Array(n);
    for (let i = 0; i < n; i++) this.isPerson[i] = PERSON_GROUPS.test(this.groups[i]) ? 1 : 0;
    this.me = this.index.get(ME_PERSON) ?? -1;
    this.peopleList = persons;

    // adjacency (CSR) for highlight edges
    const m = this.es.length, start = new Int32Array(n + 1);
    for (let k = 0; k < m; k++) { start[this.es[k] + 1]++; start[this.ed[k] + 1]++; }
    for (let i = 0; i < n; i++) start[i + 1] += start[i];
    this.adjStart = start; this.adj = new Int32Array(m * 2);
    const fill = start.slice(0, n);
    for (let k = 0; k < m; k++) { this.adj[fill[this.es[k]]++] = this.ed[k]; this.adj[fill[this.ed[k]]++] = this.es[k]; }

    // radius (css px at the fitted zoom) by message count; people a touch bigger than their biggest account
    this.r = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      this.r[i] = this.isPerson[i] ? Math.min(9, 1.3 + Math.sqrt(Math.log1p(this.size[i])) * 1.25)
        : 0.9 + Math.min(2, Math.log1p(this.size[i]) * 0.18);
      if (i >= bn) this.r[i] = Math.min(11, this.r[i] + 1.5);
    }
    if (this.me >= 0) this.r[this.me] = 7;
    // lod weight: accounts need 2+ messages, chats 3+, to appear at the overview zoom; people always show
    this.lod = new Float32Array(n);
    for (let i = 0; i < n; i++) this.lod[i] = i >= bn ? 1e9 : this.isPerson[i] ? this.size[i] : this.size[i] / 1.5;
    this.byImportance = [];
    for (let i = 0; i < n; i++) if (this.isPerson[i]) this.byImportance.push(i);
    this.byImportance.sort((a, b) => this.size[b] - this.size[a]);
    this._buildGrid();
    this._initGL();
    this.people = [];      // accounts (for the legend: which platforms are present)
    for (let i = 0; i < bn; i++) if (this.isPerson[i]) this.people.push({ id: this.ids[i], platform: this.plat[i], messages: this.size[i] });
    const accounts = this.people.length;
    this.onInfo?.(`${num(persons.length)} people · ${num(accounts)} accounts · ${num(bn - accounts)} chats`);
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
    const e3 = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
    this.basis = [e1, e2, e3];
    // 3D: centred on the layout's median so it orbits about the middle of the cloud
    const med = this.med = [0, 1, 2].map(a => { const v = []; for (let i = a; i < n * 3; i += step * 3) v.push(P[i]); v.sort((p, q) => p - q); return v[v.length >> 1] || 0; });
    const rs = [];
    for (let i = 0; i < n; i++) {
      this._place(i, [P[i * 3], P[i * 3 + 1], P[i * 3 + 2]]);
      if (i % step === 0) rs.push(Math.hypot(this.X[i * 3], this.X[i * 3 + 1], this.X[i * 3 + 2]));
    }
    rs.sort((p, q) => p - q);
    this.dist = 3 * (rs[Math.floor(rs.length * 0.98)] || 1);   // focal length: the camera's resting distance
    this.view.D = this.dist; this._tf = [0, 0];
    this.R = this._rot(this.view);
  }

  // a layout position (raw 3D) -> this node's centred 3D position and its flat-map coordinates
  _place(i, raw) {
    const [e1, e2] = this.basis, med = this.med;
    const a = raw[0] - med[0], b = raw[1] - med[1], c = raw[2] - med[2];
    this.X[i * 3] = a; this.X[i * 3 + 1] = b; this.X[i * 3 + 2] = c;
    this.x[i] = a * e1[0] + b * e1[1] + c * e1[2];
    this.y[i] = a * e2[0] + b * e2[1] + c * e2[2];
  }

  // rotation rows for a view: orbit (yaw, pitch) applied on top of the PCA basis
  _rot(v) {
    const [e1, e2, e3] = this.basis, cy = Math.cos(v.yaw || 0), sy = Math.sin(v.yaw || 0), cp = Math.cos(v.pitch || 0), sp = Math.sin(v.pitch || 0);
    const r1 = e1.map((_, k) => cy * e1[k] + sy * e3[k]);
    const t3 = e1.map((_, k) => -sy * e1[k] + cy * e3[k]);
    const r2 = e2.map((_, k) => cp * e2[k] - sp * t3[k]);
    const r3 = e2.map((_, k) => sp * e2[k] + cp * t3[k]);
    return [r1, r2, r3];
  }
  get flat() { const v = this.view; return !v.p && !v.yaw && !v.pitch; }
  // pixels per layout unit at the target's depth
  get pxPerUnit() { return this.view.s * this.dist / this.view.D; }
  // a node relative to the camera target, after rotation and perspective, before zoom:
  // [x, y, depth scale, in front of the camera?]
  _proj(i, R = this.R, v = this.view) {
    const t = v.t, F = this.dist;
    if (R === this.R && v === this.view && this.flat) { const f = F / v.D; return [(this.x[i] - this._tf[0]) * f, (this.y[i] - this._tf[1]) * f, f, true]; }
    const a = this.X[i * 3] - t[0], b = this.X[i * 3 + 1] - t[1], c = this.X[i * 3 + 2] - t[2];
    const depth = v.D - (R[2][0] * a + R[2][1] * b + R[2][2] * c) * v.p, near = F * 0.01;
    const f = F / Math.max(near, depth);
    return [(R[0][0] * a + R[0][1] * b + R[0][2] * c) * f, (R[1][0] * a + R[1][1] * b + R[1][2] * c) * f, f, depth > near];
  }
  // on-screen glow diameter (css px) of node i at depth scale f, matching the shader
  _glowPx(i, f) { return Math.max(1.5, Math.min(64, Math.log2(this.size[i] + 1) * this._glowK() * f)); }
  _glowK() { return GLOW_WORLD * this.view.s * 2; }

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
    for (const b of [this.bufPos, this.bufSize, this.bufLod, this.bufGlow, this.bufLines, this.bufLineLod]) if (b) gl.deleteBuffer(b);
    if (this.P) return this._uploadGL();
    const prog = (vs, fs) => {
      const p = gl.createProgram();
      for (const [type, src] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]]) {
        const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
        if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
        gl.attachShader(p, s);
      }
      gl.linkProgram(p);
      if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
      const u = {}, a = {};
      for (let i = 0; i < gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS); i++) { const nm = gl.getActiveUniform(p, i).name; u[nm] = gl.getUniformLocation(p, nm); }
      for (let i = 0; i < gl.getProgramParameter(p, gl.ACTIVE_ATTRIBUTES); i++) { const nm = gl.getActiveAttrib(p, i).name; a[nm] = gl.getAttribLocation(p, nm); }
      return { p, u, a };
    };
    this.P = prog(VS_POINTS, FS_POINTS);
    this.L = prog(VS_LINES, FS_LINES);
    this._uploadGL();
  }
  _uploadGL() {
    // draw order: chats first, then people from smallest to biggest ("me" is drawn on the overlay)
    const order = [];
    for (let i = 0; i < this.n; i++) if (!this.isPerson[i]) order.push(i);
    for (let k = this.byImportance.length - 1; k >= 0; k--) if (this.byImportance[k] !== this.me) order.push(this.byImportance[k]);
    this.order = order;
    const pos = new Float32Array(order.length * 3), sz = new Float32Array(order.length);
    order.forEach((i, k) => { pos.set(this.X.subarray(i * 3, i * 3 + 3), k * 3); sz[k] = this.r[i]; });
    const lod = new Float32Array(order.length), glow = new Float32Array(order.length);
    order.forEach((i, k) => { lod[k] = this.lod[i]; glow[k] = Math.log2(this.size[i] + 1); });
    this.bufPos = this._buf(pos); this.bufSize = this._buf(sz); this.bufLod = this._buf(lod); this.bufGlow = this._buf(glow);
    const lp = new Float32Array(this.es.length * 6);
    for (let k = 0; k < this.es.length; k++) {
      lp.set(this.X.subarray(this.es[k] * 3, this.es[k] * 3 + 3), k * 6); lp.set(this.X.subarray(this.ed[k] * 3, this.ed[k] * 3 + 3), k * 6 + 3);
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
      plat: { twitter: css('--series-1'), reddit: css('--series-2'), discord: css('--series-3'), facebook: css('--series-4'), instagram: css('--series-5'), whatsapp: css('--series-6'), person: css('--person'), other: css('--muted') },
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
      // 3D palette: the classic viewer's vivid colours. Its darker chat shades were made to sink into black,
      // so on a light background chats become light tints of their platform colour and white Facebook turns violet.
      const dark = this.colors.dark;
      this.platColor3 = { ...this.platRaw, person: this.colors.plat.person };
      if (!dark && /^#f/i.test(this.platColor3.facebook || '')) this.platColor3.facebook = this.colors.plat.facebook;
      const plat3 = Object.fromEntries(Object.entries(this.platColor3).map(([k, v]) => [k, rgba(v)]));
      const col3 = new Float32Array(this.order.length * 3);
      this.order.forEach((i, k) => {
        if (dark || !plat3[this.plat[i]]) { col3.set(this.col3.subarray(i * 3, i * 3 + 3), k * 3); return; }
        const c = plat3[this.plat[i]], w = this.isPerson[i] ? 0 : 0.45;
        col3.set([c[0] + (1 - c[0]) * w, c[1] + (1 - c[1]) * w, c[2] + (1 - c[2]) * w], k * 3);
      });
      for (const b of [this.bufCol, this.bufLineCol, this.bufCol3]) if (b) this.gl.deleteBuffer(b);
      this.bufCol = this._buf(col); this.bufLineCol = this._buf(lc); this.bufCol3 = this._buf(col3);
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
      gl.uniform3fv(pr.u.u_target, v.t); gl.uniform1f(pr.u.u_scale, v.s); gl.uniform2f(pr.u.u_half, this.w / 2, this.h / 2);
      gl.uniform3fv(pr.u.u_r1, this.R[0]); gl.uniform3fv(pr.u.u_r2, this.R[1]); gl.uniform3fv(pr.u.u_r3, this.R[2]);
      gl.uniform1f(pr.u.u_persp, v.p || 0); gl.uniform1f(pr.u.u_dist, v.D); gl.uniform1f(pr.u.u_focal, this.dist);
    };
    const attr = (pr, name, buf, size) => {
      gl.bindBuffer(gl.ARRAY_BUFFER, buf); gl.enableVertexAttribArray(pr.a[name]); gl.vertexAttribPointer(pr.a[name], size, gl.FLOAT, false, 0, 0);
    };
    common(this.L);
    const lodT = this.lodThreshold();
    gl.uniform1f(this.L.u.u_alpha, this.colors.edgeAlpha * (dim ? 0.12 : 1));
    gl.uniform1f(this.L.u.u_lod, lodT);
    attr(this.L, 'a_pos', this.bufLines, 3); attr(this.L, 'a_col', this.bufLineCol, 4); attr(this.L, 'a_lod', this.bufLineLod, 1);
    if (v.p < 0.999) gl.drawArrays(gl.LINES, 0, this.es.length * 2);
    for (const a of Object.values(this.L.a)) gl.disableVertexAttribArray(a);
    const P = this.P;
    common(P);
    gl.uniform1f(P.u.u_dpr, this.dpr); gl.uniform1f(P.u.u_zoom, this.zoomFactor()); gl.uniform1f(P.u.u_dim, dim ? 0.07 : 1);
    gl.uniform1f(P.u.u_lod, lodT); gl.uniform1f(P.u.u_glowk, this._glowK());
    attr(P, 'a_col3', this.bufCol3, 3); attr(P, 'a_glow', this.bufGlow, 1);
    attr(P, 'a_pos', this.bufPos, 3); attr(P, 'a_col', this.bufCol, 4); attr(P, 'a_size', this.bufSize, 1); attr(P, 'a_lod', this.bufLod, 1);
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
  toScreen(i) { const s = this.view.s, [x, y] = this._proj(i); return [x * s + this.w / 2, y * s + this.h / 2]; }
  zoomK() { return this.pxPerUnit / (this.fitScale || this.pxPerUnit); }
  // zoomed out: only the meaningful skeleton; each doubling of zoom halves the cut until everything shows
  lodThreshold() { if (this.view.p > 0.5) return 0; const k = this.zoomK(); return k >= 3 ? 0 : 2 / Math.max(1, k); }
  visibleCount() { const t = this.lodThreshold(); let c = 0; for (let i = 0; i < this.n; i++) if (this.lod[i] >= t) c++; return c; }
  zoomFactor() { return Math.max(0.6, Math.min(3.2, Math.sqrt(this.zoomK()))); }
  setView(v) {
    this.view = { yaw: 0, pitch: 0, p: 0, D: this.dist, ...v };
    this.R = this._rot(this.view);
    const t = this.view.t, [e1, e2] = this.basis || [[1, 0, 0], [0, 1, 0]];
    this._tf = [t[0] * e1[0] + t[1] * e1[1] + t[2] * e1[2], t[0] * e2[0] + t[1] * e2[1] + t[2] * e2[2]];
    this.glDirty = true; this.draw();
  }
  // move the target by a screen-space offset (px), at the target's depth
  _shift(t, R, dx, dy, k) { return t.map((x, a) => x + (dx * R[0][a] + dy * R[1][a]) * k); }

  // flat map ⇄ 3D: tilt and add perspective (or straighten back out), refitting at the new angle
  set3D(on, animate = true) {
    this.is3D = !!on; this.spin = this.is3D && !this.hi; this.vel = null;
    if (!this.n || !this.w) return;
    this.fit(animate, this.is3D ? ANGLE_3D : { yaw: 0, pitch: 0, p: 0 });
  }

  fit(animate = true, angle = null) {
    if (!this.n || !this.w) return;
    const a = angle || { yaw: this.view.yaw, pitch: this.view.pitch, p: this.view.p };
    const R = this._rot(a), v0 = { ...this.view, ...a, D: this.dist };
    // robust extent from a sample: ignore the outermost stragglers so they don't shrink everything
    const step = Math.max(1, Math.floor(this.n / 5000)), xs = [], ys = [];
    for (let i = 0; i < this.n; i += step) { const [x, y] = this._proj(i, R, v0); xs.push(x); ys.push(y); }
    xs.sort((a, b) => a - b); ys.sort((a, b) => a - b);
    const q = (a, p) => a[Math.floor(p * (a.length - 1))];
    const box = { x0: q(xs, 0.004), x1: q(xs, 0.996), y0: q(ys, 0.004), y1: q(ys, 0.996) };
    const s = Math.min(this.w / (box.x1 - box.x0 || 1), this.h / (box.y1 - box.y0 || 1)) * (a.p ? 1.05 : 0.84);
    this.fitScale = s; this.fitted = true;
    const t = this._shift(this.view.t, R, (box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2, 1);
    this.goTo({ ...a, t, s, D: this.dist }, animate);
  }
  frame(indices, animate = true) {
    let pts = indices.filter(i => i !== this.me);
    if (!pts.length) pts = indices;
    if (!pts.length || !this.w) return;
    // centre on the points' 3D bounding box, then fit their spread as seen from the current angle
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (const i of pts) for (let a = 0; a < 3; a++) { lo[a] = Math.min(lo[a], this.X[i * 3 + a]); hi[a] = Math.max(hi[a], this.X[i * 3 + a]); }
    const t = lo.map((l, a) => (l + hi[a]) / 2), flatV = { ...this.view, t, p: 0, D: this.dist };
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const i of pts) { const [x, y] = this._proj(i, this.R, flatV); x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); }
    const minSpan = (this.bounds.x1 - this.bounds.x0) / 14;
    const sw = Math.max(x0 < 0 ? -x0 * 2 : 0, x1 * 2, minSpan), sh = Math.max(y0 < 0 ? -y0 * 2 : 0, y1 * 2, minSpan);
    const ppu = Math.min(this.w * 0.6 / sw, this.h * 0.6 / sh, this.fitScale * 10);   // wanted px per unit
    if (this.is3D) this.goTo({ ...this.view, t, D: this.view.s * this.dist / ppu }, animate);   // dolly in
    else this.goTo({ ...this.view, t, s: ppu, D: this.dist }, animate);
  }
  goTo(target, animate = true) {
    cancelAnimationFrame(this.anim);
    if (!animate) { this.setView({ ...target }); return; }
    const from = { ...this.view }, t0 = performance.now(), dur = 800;
    // take the short way round when straightening a spun 3D view
    if (target.yaw !== undefined) from.yaw = target.yaw + Math.atan2(Math.sin(from.yaw - target.yaw), Math.cos(from.yaw - target.yaw));
    const ease = t => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
    const step = now => {
      const t = Math.min(1, (now - t0) / dur), e = ease(t);
      const ls = Math.log(from.s) + (Math.log(target.s) - Math.log(from.s)) * e;
      const lerp = k => (from[k] || 0) + ((target[k] ?? from[k] ?? 0) - (from[k] || 0)) * e;
      const tt = target.t || from.t, D0 = from.D || this.dist, D1 = target.D || D0;
      this.setView({
        t: from.t.map((x, a) => x + (tt[a] - x) * e), s: Math.exp(ls), D: Math.exp(Math.log(D0) + (Math.log(D1) - Math.log(D0)) * e),
        yaw: lerp('yaw'), pitch: lerp('pitch'), p: lerp('p'),
      });
      if (t < 1) this.anim = requestAnimationFrame(step); else this.anim = null;
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
    if (idx.length) this.spin = false;
    if (fly && idx.length) this.frame(this.focus?.size ? [...this.focus] : idx); else this.draw();
    return idx.length;
  }
  clear() { this._pending = null; this.hi = null; this.focus = null; this.born = null; this.glDirty = true; this.spin = this.is3D; this.fit(); }

  // ─── drawing: WebGL only when the view changed; the overlay animates on its own clock ─────
  draw() {
    if (this.frameReq) return;
    this.frameReq = requestAnimationFrame(t => { this.frameReq = 0; this._render(t); });
  }
  _render(t = performance.now()) {
    if (!this.n || !this.w || !this.order) return;
    // 3D idles with a slow turn until you grab it
    const dt = Math.min(100, t - (this._spunAt || t));
    this._spunAt = t;
    if (!this.anim && !this._dragging) {
      if (this.vel) {   // let a flicked orbit glide to a stop
        const v = this.view, k = Math.pow(0.9, dt / 16);
        v.yaw += this.vel[0] * dt; v.pitch = Math.max(-1.5, Math.min(1.5, v.pitch + this.vel[1] * dt));
        this.vel = this.vel.map(x => x * k);
        if (Math.abs(this.vel[0]) + Math.abs(this.vel[1]) < 1e-5) this.vel = null;
        this.R = this._rot(v); this.glDirty = true;
      } else if (this.spin) {
        this.view.yaw += dt * 0.00012; this.R = this._rot(this.view); this.glDirty = true;
      }
    }
    if (this.glDirty) { this._drawGL(); this.glDirty = false; }
    const c = this.ctx;
    c.setTransform(1, 0, 0, 1, 0, 0); c.clearRect(0, 0, this.ov.width, this.ov.height);
    c.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    this._renderOverlay(c, t);
    // keep animating (pulse, flowing links) while visible; ~30 fps is plenty for ambient motion
    if (this.cv.offsetParent !== null && !document.hidden) {
      clearTimeout(this._tick);
      if (this.is3D && (this.spin || this.vel)) this.draw();   // camera moving: every frame
      else this._tick = setTimeout(() => this.draw(), 33);
    }
  }

  _renderOverlay(c, t) {
    const z = this.zoomFactor(), labels = [];
    const space = this.view.p > 0.5, C = space ? { ...this.colors, plat: { ...this.colors.plat, ...this.platColor3 } } : this.colors;
    this._C = C;
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
    // people: a clear line from each person to each of their accounts (the faint GL edges are for chats)
    if (this.peopleList?.length) {
      c.lineCap = 'round'; c.setLineDash([]);
      for (const p of this.peopleList) {
        const i = this.index.get(p.node);
        if (i === undefined || (this.hi && !this.hi.has(i) && !p.members.some(m => this.hi.has(m)))) continue;
        const [px, py] = this.toScreen(i);
        c.strokeStyle = i === this.me ? C.accent : C.plat.person; c.lineWidth = i === this.me ? 1.4 : 1.6;
        c.globalAlpha = this.hi ? 0.75 : 0.5;
        c.beginPath();
        for (const m of p.members) { const [ax, ay] = this.toScreen(m); c.moveTo(px, py); c.lineTo(ax, ay); }
        c.stroke();
      }
      c.globalAlpha = 1;
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
    const placed = [], C = this._C || this.colors;
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
    if (!this.flat) return this._pickProjected(mx, my);
    const v = { s: this.pxPerUnit }, wx = (mx - this.w / 2) / v.s + this._tf[0], wy = (my - this.h / 2) / v.s + this._tf[1];
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
  // rotated / perspective view: no grid, just test every visible node on screen
  _pickProjected(mx, my) {
    const z = this.zoomFactor(), lodT = this.lodThreshold();
    let best = -1, bd = Infinity;
    const s = this.view.s, glow = this.view.p > 0.5;
    for (let i = 0; i < this.n; i++) {
      if (this.hi ? !this.hi.has(i) && i !== this.me : this.lod[i] < lodT) continue;
      const [px, py, f, front] = this._proj(i);
      if (!front) continue;
      const x = px * s + this.w / 2, y = py * s + this.h / 2;
      if (Math.abs(x - mx) > 80 || Math.abs(y - my) > 80) continue;
      const r = glow && !this.hi ? this._glowPx(i, f) * 0.35 : this.r[i] * z;
      // prefer the nearer of overlapping nodes, as you'd expect when looking into the cloud
      const dd = Math.hypot(x - mx, y - my) - r - (this.isPerson[i] ? 3 : 0) - (glow ? f * 2 : 0);
      if (dd < 8 && dd < bd) { bd = dd; best = i; }
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
    const id = this.ids[i], kind = id.startsWith('P_') ? 'person' : id.startsWith('A_') ? 'account' : 'chat';
    return { id, kind, label: this.labels[i], group: this.groups[i], platform: this.plat[i], size: this.size[i],
             person: !!this.isPerson[i], personOf: this.personOf?.get(id) || null, accounts: this.accountsOf?.get(id) || null };
  }
  // the person an account belongs to (P_ id), if any; mine() = it is one of my own accounts
  personOfNode(id) { return this.personOf?.get(id) || null; }
  isMine(id) { return id === ME_PERSON || this.personOf?.get(id) === ME_PERSON; }
  tooltipHtml(i) {
    const id = this.ids[i], name = esc(cleanName(this.labels[i]));
    if (id.startsWith('P_')) {
      const k = this.accountsOf.get(id)?.length || 0;
      return `<b>${name}${i === this.me ? ' (you)' : ''}</b><span>person · ${k} account${k === 1 ? '' : 's'} · ${num(this.size[i])} messages</span>`;
    }
    const owner = this.personOf?.get(id), on = owner && owner !== ME_PERSON ? ` · ${esc(cleanName(this.labels[this.index.get(owner)]))}` : '';
    return `<b>${name}</b><span>${this.isPerson[i] ? `${num(this.size[i])} messages` : 'conversation'} · ${esc(platformLabel(this.plat[i]))}${on}</span>`;
  }

  _bindInput() {
    const cv = this.ov;
    let drag = null, moved = false;
    const pos = e => { const r = cv.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
    cv.addEventListener('contextmenu', e => { if (this.is3D) e.preventDefault(); });
    cv.addEventListener('pointerdown', e => {
      cv.setPointerCapture(e.pointerId);
      // in 3D: drag orbits, shift/right/middle-drag pans
      drag = { p: pos(e), v: { ...this.view }, R: this.R, ppu: this.pxPerUnit, orbit: this.is3D && e.button === 0 && !e.shiftKey, last: [...pos(e), performance.now()] }; moved = false;
      cancelAnimationFrame(this.anim); this.anim = null; this._dragging = true; this.vel = null;
    });
    cv.addEventListener('pointermove', e => {
      const [mx, my] = pos(e);
      if (drag) {
        const dx = mx - drag.p[0], dy = my - drag.p[1];
        if (Math.abs(dx) + Math.abs(dy) > 3) { moved = true; cv.classList.add('dragging'); this.onHover?.(null); }
        if (moved && drag.orbit) {
          this.spin = false;
          const now = performance.now(), ddt = Math.max(1, now - drag.last[2]);
          drag.vel = [(mx - drag.last[0]) * 0.006 / ddt, -(my - drag.last[1]) * 0.006 / ddt]; drag.last = [mx, my, now];
          this.setView({ ...this.view, yaw: drag.v.yaw + dx * 0.006, pitch: Math.max(-1.5, Math.min(1.5, drag.v.pitch - dy * 0.006)) });
        } else if (moved) this.setView({ ...this.view, t: this._shift(drag.v.t, drag.R, -dx, -dy, 1 / drag.ppu) });
        return;
      }
      const h = this.pick(mx, my);
      if (h !== this.hover) { this.hover = h; cv.classList.toggle('hovering', h >= 0); this.draw(); }
      this.onHover?.(h >= 0 ? { html: this.tooltipHtml(h), x: mx, y: my } : null);
    });
    cv.addEventListener('pointerup', e => {
      if (!drag) return;
      cv.classList.remove('dragging');
      if (!moved && e.button === 0) { const [mx, my] = pos(e); const i = this.pick(mx, my); if (i >= 0) this.onClick?.(this.nodeInfo(i)); }
      // flicked: keep turning and ease out (only if the pointer was still moving when released)
      if (moved && drag.orbit && drag.vel && performance.now() - drag.last[2] < 80) { this.vel = drag.vel; this.draw(); }
      drag = null; this._dragging = false;
    });
    cv.addEventListener('pointercancel', () => { drag = null; this._dragging = false; cv.classList.remove('dragging'); });
    cv.addEventListener('pointerleave', () => { if (this.hover >= 0) { this.hover = -1; this.draw(); } this.onHover?.(null); });
    cv.addEventListener('wheel', e => {
      e.preventDefault();
      cancelAnimationFrame(this.anim); this.anim = null;
      const [mx, my] = pos(e), v = this.view, fs = this.fitScale || v.s, ppu = this.pxPerUnit;
      // zoom toward the cursor: 2D scales the map, 3D dollies the camera (fly into the cloud)
      const ppu2 = Math.max(fs * 0.4, Math.min(fs * (this.is3D ? 200 : 60), ppu * Math.exp(-e.deltaY * (e.ctrlKey ? 0.01 : 0.0018))));
      const k = ppu2 / ppu;
      const t = this._shift(v.t, this.R, mx - this.w / 2, my - this.h / 2, (1 - 1 / k) / ppu);
      if (this.is3D) { this.spin = false; this.setView({ ...v, t, D: v.D / k }); }
      else this.setView({ ...v, t, s: v.s * k });
    }, { passive: false });
  }
}
