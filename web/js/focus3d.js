// Focus view: a small, animated three.js constellation of exactly what is being looked at —
// you, the people, their chats and the conversation sessions (the cited sources) — instead of the
// whole 80k-node map. Shape says what a node is, colour says which platform it's from.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { css, esc, fmtDate, cleanName, platformLabel, sessionTitle } from './util.js';

const KIND_SIZE = { me: 34, person: 26, theme: 22, chat: 15, session: 15 };
const REST = {   // spring rest lengths between kinds
  'me-person': 150, 'me-chat': 110, 'me-session': 120, 'me-theme': 120,
  'chat-person': 70, 'chat-session': 58, 'person-session': 75, 'person-theme': 85, 'session-theme': 48,
};
const rest = (a, b) => REST[[a, b].sort().join('-')] || 80;

// dates arrive as ISO ("2025-08-29") or already formatted ("29 Aug 2025"); show "29 Aug"
const shortDate = d => !d ? '' : /^\d{4}-/.test(d) ? fmtDate(d, { year: false }) : String(d).replace(/\s\d{4}$/, '').replace(/^0/, '');
const fullDate = d => !d ? '' : /^\d{4}-/.test(d) ? fmtDate(d) : String(d).replace(/^0/, '');
// "Tweet Thread 1978…" says nothing; name chats by what they are
const chatName = (label, platform) => /^tweet thread \d+/i.test(label || '') ? 'Tweet thread'
  : /^reddit (thread|post) /i.test(label || '') ? 'Reddit thread' : (label || platformLabel(platform));

// ─── builders: turn app data into a scene spec {nodes, links} ──────────────────
// who to show for a set of node ids: people (P_), and accounts (A_) that belong to no person in the set;
// never me or my own accounts (I'm the centre of every scene)
function whoIn(ids, graph, meId) {
  const set = new Set(ids), out = [];
  for (const x of ids) {
    if (x === meId || graph.isMine(x)) continue;
    if (x.startsWith('P_') || (x.startsWith('A_') && !set.has(graph.personOfNode(x)))) out.push(x);
  }
  return out;
}
export function specFromSources(sources, graph) {
  const nodes = new Map(), links = [];
  const add = (id, n) => { if (!nodes.has(id)) nodes.set(id, { id, ...n }); return id; };
  const meId = graph.meId;
  add(meId, { kind: 'me', label: 'You' });
  for (const s of sources) {
    const prev = nodes.get(`S_${s.session_id}`);
    if (prev && s.n != null) { prev.label = `${prev.label} · ${s.n}`; continue; }
    const sid = add(`S_${s.session_id}`, {
      kind: 'session', platform: s.platform, label: s.n != null ? String(s.n) : shortDate(s.date),
      title: sessionTitle(s.header), date: s.date, snippet: s.snippet, session: s.session_id, n: s.n,
    });
    const people = whoIn(s.nodes, graph, meId).slice(0, 6);
    const chats = s.nodes.filter(x => x.startsWith('T_')).slice(0, 2);
    for (const t of chats) {
      const info = graph.infoById(t);
      add(t, { kind: 'chat', platform: info?.platform || s.platform, label: chatName(info?.label, info?.platform || s.platform), size: info?.size });
      links.push([t, sid]);
      if (people.length <= 2) links.push([meId, t]);
    }
    for (const p of people) {
      const info = graph.infoById(p);
      add(p, { kind: 'person', platform: info?.platform || s.platform, label: cleanName(info?.label || '?'), size: info?.size, ref: p });
      if (chats.length) chats.forEach(t => links.push([t, p])); else links.push([p, sid]);
      links.push([meId, p]);
    }
    if (!chats.length && !people.length) links.push([meId, sid]);
  }
  return { nodes: [...nodes.values()], links: dedupe(links) };
}

export function specFromNodes(ids, graph) {
  const meId = graph.meId, nodes = [{ id: meId, kind: 'me', label: 'You' }], links = [];
  const people = whoIn(ids, graph, meId).slice(0, 24);
  const chats = ids.filter(x => x.startsWith('T_')).slice(0, 24);
  for (const p of people) {
    const info = graph.infoById(p);
    if (!info) continue;
    nodes.push({ id: p, kind: 'person', platform: info.platform, label: cleanName(info.label), size: info.size, ref: p });
    links.push([meId, p]);
  }
  const has = new Set(nodes.map(n => n.id));
  for (const t of chats) {
    const info = graph.infoById(t);
    if (!info) continue;
    nodes.push({ id: t, kind: 'chat', platform: info.platform, label: chatName(info.label, info.platform), size: info.size });
    const linked = graph.neighbours(t).filter(p => has.has(p));
    if (linked.length) linked.forEach(p => links.push([p, t])); else links.push([meId, t]);
  }
  return { nodes, links: dedupe(links) };
}

// add a person's dossier (themes, their conversations, top chats) to a thin scene
export function enrichSpec(spec, dossiers, graph) {
  const nodes = [...spec.nodes], links = [...spec.links], ids = new Set(nodes.map(n => n.id));
  for (const d of dossiers) {
    const extra = specFromPerson(d, graph);
    for (const n of extra.nodes) if (!ids.has(n.id)) { nodes.push(n); ids.add(n.id); }
    links.push(...extra.links);
  }
  return { ...spec, nodes, links: dedupe(links) };
}

// a person: you → them → each of their accounts → that account's themes → example conversations.
// Themes belong to accounts; a faint link joins two themes on different accounts that share a subject.
// An account (or a person dossier without account_themes) hangs its themes off itself.
export function specFromPerson(d, graph) {
  const meId = graph.meId, pid = d.node, info = graph.infoById(pid);
  const nodes = [
    { id: meId, kind: 'me', label: 'You' },
    { id: pid, kind: 'person', platform: info?.platform, label: cleanName(d.name), size: d.messages_total, ref: d.node },
  ];
  const links = [[meId, pid]], has = id => nodes.some(n => n.id === id);
  const accts = (d.account_themes || []).filter(a => a.messages_total);
  for (const a of accts) {
    nodes.push({ id: a.node, kind: 'person', platform: a.platform, label: cleanName(a.name), size: a.messages_total, ref: a.node });
    links.push([pid, a.node]);
  }
  const ownerOf = chat => accts.find(a => graph.neighbours(chat).includes(a.node))?.node || pid;
  for (const c of d.top_chats.slice(0, 5)) {
    if (has(c.node)) continue;
    nodes.push({ id: c.node, kind: 'chat', platform: c.platform, label: chatName(c.title, c.platform), size: c.messages });
    links.push([accts.length ? ownerOf(c.node) : pid, c.node]);
  }
  const groups = accts.length ? accts.map(a => ({ node: a.node, platform: a.platform, themes: a.themes }))
    : [{ node: pid, platform: info?.platform, themes: d.themes }];
  const total = groups.reduce((n, g) => n + Math.min(4, g.themes.length), 0);
  const perTheme = total > 8 ? 2 : 3;               // keep the scene readable for people with many accounts
  for (const g of groups) {
    g.themes.slice(0, accts.length ? 4 : 6).forEach((t, k) => {
      const tid = `TH_${g.node}_${k}`;
      nodes.push({ id: tid, kind: 'theme', label: t.theme, title: t.description, size: t.sessions });
      links.push([g.node, tid]);
      for (const e of t.examples.slice(-perTheme)) {
        const sid = `S_${e.session_id}`;
        if (!has(sid)) {
          nodes.push({ id: sid, kind: 'session', platform: g.platform, label: shortDate(e.date), title: sessionTitle(e.header), date: e.date, snippet: e.summary, session: e.session_id });
        }
        links.push([tid, sid]);
      }
    });
  }
  for (const l of d.shared_themes || []) {
    const [a, b] = [l.a, l.b].map(x => x.replace(/^(.*):(\d+)$/, 'TH_$1_$2'));
    if (has(a) && has(b)) links.push([a, b, { faint: true }]);
  }
  return { nodes, links: dedupe(links), focus: pid };
}

function dedupe(links) {
  const seen = new Set();
  return links.filter(([a, b]) => { const k = a < b ? a + '|' + b : b + '|' + a; if (a === b || seen.has(k)) return false; seen.add(k); return true; });
}

// ─── textures (drawn once per kind/colour/theme) ─────────────────────────────
const texCache = new Map();
function mix(hex, to, t) {
  const p = h => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16));
  const a = p(hex), b = p(to);
  return `rgb(${a.map((v, i) => Math.round(v + (b[i] - v) * t)).join(',')})`;
}
function tex(kind, color, dark) {
  const key = `${kind}|${color}|${dark}`;
  if (texCache.has(key)) return texCache.get(key);
  const S = 128, c = document.createElement('canvas'); c.width = c.height = S;
  const g = c.getContext('2d'), m = S / 2, rim = dark ? 'rgba(255,255,255,.85)' : '#ffffff';
  if (kind === 'glow') {
    const r = g.createRadialGradient(m, m, 0, m, m, m);
    r.addColorStop(0, color); r.addColorStop(1, 'rgba(255,255,255,0)');
    g.globalAlpha = 0.55; g.fillStyle = r; g.fillRect(0, 0, S, S);
  } else if (kind === 'me' || kind === 'person' || kind === 'theme') {
    const R = 52;
    g.shadowColor = 'rgba(30,45,90,.28)'; g.shadowBlur = 10; g.shadowOffsetY = 3;
    let fill;
    if (kind === 'me') {
      fill = g.createLinearGradient(m - R, m - R, m + R, m + R);
      fill.addColorStop(0, '#4aa8f0'); fill.addColorStop(0.5, '#9a8cf0'); fill.addColorStop(1, '#ee7fb6');
    } else if (kind === 'theme') {
      fill = g.createLinearGradient(m - R, m - R, m + R, m + R);
      fill.addColorStop(0, '#9a8cf0'); fill.addColorStop(1, '#ee7fb6');
    } else {
      fill = g.createRadialGradient(m - R * 0.35, m - R * 0.4, R * 0.1, m, m, R);
      fill.addColorStop(0, mix(color, '#ffffff', 0.45)); fill.addColorStop(0.55, color); fill.addColorStop(1, mix(color, '#101828', 0.28));
    }
    g.beginPath(); g.arc(m, m, R, 0, Math.PI * 2); g.fillStyle = fill; g.fill();
    g.shadowColor = 'transparent';
    g.lineWidth = 7; g.strokeStyle = rim; g.stroke();
    if (kind !== 'person') {   // a glossy highlight
      const h = g.createRadialGradient(m - 16, m - 20, 2, m - 16, m - 20, 30);
      h.addColorStop(0, 'rgba(255,255,255,.55)'); h.addColorStop(1, 'rgba(255,255,255,0)');
      g.fillStyle = h; g.beginPath(); g.arc(m, m, R - 4, 0, Math.PI * 2); g.fill();
    }
  } else if (kind === 'chat') {
    const R = 46;
    g.shadowColor = 'rgba(30,45,90,.22)'; g.shadowBlur = 8; g.shadowOffsetY = 2;
    g.beginPath(); g.arc(m, m, R, 0, Math.PI * 2); g.fillStyle = dark ? '#1d2539' : '#ffffff'; g.fill();
    g.shadowColor = 'transparent';
    g.lineWidth = 14; g.strokeStyle = color; g.stroke();
    g.beginPath(); g.arc(m, m, 12, 0, Math.PI * 2); g.fillStyle = color; g.fill();
  } else if (kind === 'session') {
    const R = 46;
    g.translate(m, m); g.rotate(Math.PI / 4);
    g.shadowColor = 'rgba(30,45,90,.25)'; g.shadowBlur = 8; g.shadowOffsetY = 2;
    g.beginPath(); g.roundRect(-R * 0.72, -R * 0.72, R * 1.44, R * 1.44, 14);
    const f = g.createLinearGradient(-R, -R, R, R);
    f.addColorStop(0, mix(color, '#ffffff', 0.35)); f.addColorStop(1, color);
    g.fillStyle = f; g.fill();
    g.shadowColor = 'transparent'; g.lineWidth = 7; g.strokeStyle = rim; g.stroke();
  }
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 4;
  texCache.set(key, t);
  return t;
}
let dotTex;
function particleTex() {
  if (dotTex) return dotTex;
  const c = document.createElement('canvas'); c.width = c.height = 32;
  const g = c.getContext('2d'), r = g.createRadialGradient(16, 16, 0, 16, 16, 16);
  r.addColorStop(0, 'rgba(255,255,255,1)'); r.addColorStop(0.4, 'rgba(255,255,255,.8)'); r.addColorStop(1, 'rgba(255,255,255,0)');
  g.fillStyle = r; g.fillRect(0, 0, 32, 32);
  return (dotTex = new THREE.CanvasTexture(c));
}

// ─── the scene ───────────────────────────────────────────────────────────────
export class Focus3D {
  constructor(el, { onSession, onPerson } = {}) {
    this.el = el; this.onSession = onSession; this.onPerson = onPerson;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    this.renderer.setClearColor(0x000000, 0);
    el.prepend(this.renderer.domElement);
    this.labels = el.querySelector('.f3-labels');
    this.tip = el.querySelector('.f3-tip');
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(42, 1, 1, 5000);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    Object.assign(this.controls, { enableDamping: true, dampingFactor: 0.08, autoRotate: true, autoRotateSpeed: 0.55, minDistance: 120, maxDistance: 2600, enablePan: true });
    this.controls.addEventListener('start', () => { this.controls.autoRotate = false; clearTimeout(this._resume); });
    this.controls.addEventListener('end', () => { clearTimeout(this._resume); this._resume = setTimeout(() => { this.controls.autoRotate = true; }, 7000); });
    this.ray = new THREE.Raycaster(); this.mouse = new THREE.Vector2(); this.hover = null; this.focusId = null;
    this.group = new THREE.Group(); this.scene.add(this.group);
    this._bind();
    new ResizeObserver(() => this.resize()).observe(el);
    this.loop = this.loop.bind(this);
  }

  get visible() { return !this.el.hidden; }

  show(spec) {
    this.el.hidden = false;
    this.resize();
    this._build(spec);
    if (!this.raf) this.raf = requestAnimationFrame(this.loop);
    if (spec.focus) setTimeout(() => this.focus(spec.focus, { fly: false }), 50);
  }
  hide() { this.el.hidden = true; cancelAnimationFrame(this.raf); this.raf = 0; this.spec = null; this.labels.innerHTML = ''; this.tip.hidden = true; }
  has(id) { return !!this.byId?.has(id); }
  rebuild() { if (this.spec && this.visible) this._build(this.spec, { keepCamera: true }); }

  _clear() {
    for (const o of [...this.group.children]) {
      this.group.remove(o);
      o.geometry?.dispose();
      o.material?.dispose();   // shared canvas textures stay cached
    }
    this.labels.innerHTML = '';
  }

  _build(spec, { keepCamera = false } = {}) {
    this._clear();
    this.spec = spec; this.focusId = null;
    const dark = document.documentElement.dataset.theme === 'dark';
    const plat = { twitter: css('--series-1'), reddit: css('--series-2'), discord: css('--series-3'), facebook: css('--series-4'), instagram: css('--series-5'), whatsapp: css('--series-6') };
    const colorOf = n => n.kind === 'me' ? '#9a8cf0' : n.kind === 'theme' ? '#c48ae6' : n.ref?.startsWith('P_') ? css('--person') : plat[n.platform] || css('--accent');
    const nodes = spec.nodes.map((n, k) => ({ ...n, k, color: colorOf(n) }));
    this.nodes = nodes; this.byId = new Map(nodes.map(n => [n.id, n]));
    const links = spec.links.map(([a, b, o]) => [this.byId.get(a), this.byId.get(b), o?.faint]).filter(([a, b]) => a && b);
    this.links = links;
    this._layout(nodes, links);

    // entry order: breadth-first from "you", so the constellation unfolds outward
    const order = []; const seen = new Set([nodes[0].id]); let frontier = [nodes[0]];
    const adj = new Map(nodes.map(n => [n.id, []]));
    links.forEach(([a, b]) => { adj.get(a.id).push(b); adj.get(b.id).push(a); });
    this.adj = adj;
    while (frontier.length) {
      order.push(...frontier);
      const next = [];
      for (const n of frontier) for (const m of adj.get(n.id)) if (!seen.has(m.id)) { seen.add(m.id); m.parent = n; next.push(m); }
      frontier = next;
    }
    nodes.forEach(n => { if (!seen.has(n.id)) order.push(n); });
    const t0 = performance.now();
    order.forEach((n, i) => { n.born = t0 + 120 + i * 45; n.phase = Math.random() * Math.PI * 2; });

    for (const n of nodes) {
      const sz = KIND_SIZE[n.kind] * (n.kind === 'person' ? 0.85 + Math.min(0.6, Math.log10((n.size || 10)) * 0.15) : 1);
      n.base = sz;
      const glow = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex('glow', n.color, dark), depthWrite: false, transparent: true, opacity: 0.0 }));
      glow.scale.setScalar(sz * 3.2); glow.renderOrder = 1;
      const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex(n.kind, n.color, dark), depthWrite: false, transparent: true }));
      sp.scale.setScalar(0.001); sp.renderOrder = 3; sp.userData.node = n;
      this.group.add(glow, sp);
      n.sprite = sp; n.glow = glow; n.pos = new THREE.Vector3();
      const lab = document.createElement('div');
      lab.className = `f3-label k-${n.kind}`;
      lab.innerHTML = n.kind === 'session'
        ? `<i style="background:${n.color}"></i>${esc(n.label || '')}`
        : esc(String(n.label || '').length > 30 ? String(n.label).slice(0, 29) + '…' : n.label || '');
      this.labels.append(lab); n.lab = lab;
    }

    // links: curved, coloured end to end; each carries a particle that flows along it
    this.curves = [];
    const pPos = new Float32Array(links.length * 3), pCol = new Float32Array(links.length * 3);
    links.forEach(([a, b, faint], i) => {
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(20 * 3), 3));
      const cols = new Float32Array(20 * 3), ca = new THREE.Color(a.color), cb = new THREE.Color(b.color);
      for (let k = 0; k < 20; k++) { const c = ca.clone().lerp(cb, k / 19); cols.set([c.r, c.g, c.b], k * 3); }
      geo.setAttribute('color', new THREE.BufferAttribute(cols, 3));
      const line = new THREE.Line(geo, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0, depthWrite: false }));
      line.renderOrder = 0;
      this.group.add(line);
      this.curves.push({ a, b, line, faint, bend: (Math.random() - 0.5) * 0.5, speed: 0.00018 + Math.random() * 0.00022, off: Math.random() });
      const c = cb.clone().lerp(new THREE.Color('#ffffff'), 0.25); pCol.set([c.r, c.g, c.b], i * 3);
    });
    const pg = new THREE.BufferGeometry();
    pg.setAttribute('position', new THREE.BufferAttribute(pPos, 3));
    pg.setAttribute('color', new THREE.BufferAttribute(pCol, 3));
    this.particles = new THREE.Points(pg, new THREE.PointsMaterial({ size: 7, map: particleTex(), vertexColors: true, transparent: true, depthWrite: false, opacity: 0.95, sizeAttenuation: true }));
    this.particles.renderOrder = 2;
    this.group.add(this.particles);

    if (!keepCamera) {
      // fit the constellation's radius (ignoring the farthest 10% of stragglers) to the pane
      const rs = nodes.map(n => n.target.length()).sort((a, b) => a - b);
      const r = rs[Math.floor((rs.length - 1) * 0.9)] || 100;
      const half = (this.camera.fov * Math.PI / 180) / 2, aspect = Math.min(1, this.camera.aspect || 1);
      const dist = Math.max(420, (r + 30) / Math.tan(half) / aspect * 0.95);
      this.camera.position.set(dist * 0.35, dist * 0.32, dist * 0.88);
      this.controls.target.set(0, 0, 0);
      this.camTo = null;
    }
    this.controls.autoRotate = true;
  }

  // a small 3D force layout (n is tiny, so O(n²) is fine)
  _layout(nodes, links) {
    nodes.forEach((n, i) => {
      n.target = n.kind === 'me' ? new THREE.Vector3() : new THREE.Vector3().randomDirection().multiplyScalar(90 + Math.random() * 90);
      n.v = new THREE.Vector3();
    });
    const tmp = new THREE.Vector3();
    for (let it = 0; it < 360; it++) {
      const cool = 1 - it / 400;
      for (let i = 0; i < nodes.length; i++) for (let j = i + 1; j < nodes.length; j++) {
        tmp.subVectors(nodes[i].target, nodes[j].target);
        const d2 = Math.max(tmp.lengthSq(), 25), f = 9000 / d2;
        tmp.normalize().multiplyScalar(f * cool);
        nodes[i].v.add(tmp); nodes[j].v.sub(tmp);
      }
      for (const [a, b, faint] of links) {
        tmp.subVectors(b.target, a.target);
        const d = tmp.length() || 1, f = (d - rest(a.kind, b.kind)) * (faint ? 0.008 : 0.06) * cool;
        tmp.multiplyScalar(f / d);
        a.v.add(tmp); b.v.sub(tmp);
      }
      for (const n of nodes) {
        if (n.kind === 'me') { n.v.set(0, 0, 0); continue; }
        n.v.addScaledVector(n.target, -0.004);
        n.v.clampLength(0, 18);
        n.target.add(n.v); n.v.multiplyScalar(0.6);
      }
    }
  }

  focus(id, { fly = true } = {}) {
    const n = this.byId?.get(id);
    this.focusId = n ? id : null;
    if (!n) return false;
    this.focusSet = new Set([id, ...this.adj.get(id).map(m => m.id)]);
    if (fly) {
      const dir = this.camera.position.clone().sub(this.controls.target).normalize();
      this.camTo = { target: n.target.clone(), pos: n.target.clone().add(dir.multiplyScalar(320)), t0: performance.now(), from: { target: this.controls.target.clone(), pos: this.camera.position.clone() } };
      this.controls.autoRotate = false;
      clearTimeout(this._resume); this._resume = setTimeout(() => { this.controls.autoRotate = true; }, 9000);
    }
    return true;
  }
  unfocus() { this.focusId = null; this.focusSet = null; }

  resize() {
    const w = this.el.clientWidth, h = this.el.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.renderer.domElement.style.width = w + 'px'; this.renderer.domElement.style.height = h + 'px';
    this.camera.aspect = w / h; this.camera.updateProjectionMatrix();
    this.w = w; this.h = h;
  }

  loop(now) {
    this.raf = this.visible ? requestAnimationFrame(this.loop) : 0;
    if (!this.nodes || !this.w) return;
    const ease = t => 1 - Math.pow(1 - Math.min(1, Math.max(0, t)), 3);
    const back = t => { t = Math.min(1, Math.max(0, t)); const c = 1.7; return 1 + (c + 1) * Math.pow(t - 1, 3) + c * Math.pow(t - 1, 2); };
    const F = this.focusSet;
    for (const n of this.nodes) {
      const p = ease((now - n.born) / 900), s = back((now - n.born) / 700);
      const from = n.parent?.pos || new THREE.Vector3();
      n.pos.lerpVectors(from, n.target, p);
      if (n.kind !== 'me') n.pos.y += Math.sin(now * 0.0009 + n.phase) * 2.2 * p;
      const dim = F && !F.has(n.id), isF = n.id === this.focusId, hov = this.hover === n;
      const k = (isF ? 1.35 : hov ? 1.2 : 1) * (n.kind === 'me' ? 1 + Math.sin(now / 700) * 0.04 : 1);
      n.sprite.position.copy(n.pos); n.glow.position.copy(n.pos);
      n.sprite.scale.setScalar(Math.max(0.001, n.base * s * k));
      n.sprite.material.opacity = dim ? 0.18 : 1;
      n.glow.material.opacity = (isF ? 0.75 + Math.sin(now / 240) * 0.2 : n.kind === 'me' ? 0.6 : 0.3) * p * (dim ? 0.15 : 1);
      n.glow.scale.setScalar(n.base * (isF ? 4.4 : 3.2) * s);
    }
    const pp = this.particles.geometry.attributes.position;
    const mid = new THREE.Vector3(), q = new THREE.Vector3();
    this.curves.forEach((c, i) => {
      const a = c.a.pos, b = c.b.pos;
      mid.addVectors(a, b).multiplyScalar(0.5);
      const len = a.distanceTo(b);
      q.copy(mid).multiplyScalar(1 + c.bend * 0.4); q.y += len * c.bend * 0.6;
      const arr = c.line.geometry.attributes.position.array;
      for (let k = 0; k < 20; k++) {
        const t = k / 19, u = 1 - t;
        arr[k * 3] = u * u * a.x + 2 * u * t * q.x + t * t * b.x;
        arr[k * 3 + 1] = u * u * a.y + 2 * u * t * q.y + t * t * b.y;
        arr[k * 3 + 2] = u * u * a.z + 2 * u * t * q.z + t * t * b.z;
      }
      c.line.geometry.attributes.position.needsUpdate = true;
      c.line.geometry.computeBoundingSphere();
      const live = Math.min(ease((now - c.b.born) / 900), ease((now - c.a.born) / 900));
      const lit = !F || (F.has(c.a.id) && F.has(c.b.id));
      c.line.material.opacity = live * (lit ? (F ? 0.85 : 0.5) : 0.05) * (c.faint ? 0.4 : 1);
      const t = (now * c.speed + c.off) % 1, u = 1 - t;
      pp.array[i * 3] = u * u * a.x + 2 * u * t * q.x + t * t * b.x;
      pp.array[i * 3 + 1] = u * u * a.y + 2 * u * t * q.y + t * t * b.y;
      pp.array[i * 3 + 2] = (live < 0.9 || !lit || c.faint) ? 1e5 : u * u * a.z + 2 * u * t * q.z + t * t * b.z;
    });
    pp.needsUpdate = true;
    if (this.camTo) {
      const t = ease((now - this.camTo.t0) / 1100);
      this.controls.target.lerpVectors(this.camTo.from.target, this.camTo.target, t);
      this.camera.position.lerpVectors(this.camTo.from.pos, this.camTo.pos, t);
      if (t >= 1) this.camTo = null;
    }
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
    this._labels(now);
  }

  _labels(now) {
    const v = new THREE.Vector3(), F = this.focusSet;
    const chats = this.nodes.filter(n => n.kind === 'chat').length, sessions = this.nodes.filter(n => n.kind === 'session').length;
    const themed = this.nodes.some(n => n.kind === 'theme');
    const rank = n => (n === this.hover || n.id === this.focusId ? 0 : { me: 1, person: 2, theme: 3, session: 4, chat: 5 }[n.kind]);
    const placed = [];
    for (const n of [...this.nodes].sort((a, b) => rank(a) - rank(b))) {
      v.copy(n.pos).project(this.camera);
      const lit = n === this.hover || n.id === this.focusId || (F && F.has(n.id));
      const show = n.born < now - 300 && v.z < 1 && (
        n.kind === 'me' || n.kind === 'person' || n.kind === 'theme' || lit ||
        (n.kind === 'chat' && chats <= 4) || (n.kind === 'session' && sessions <= (themed ? 0 : 24)));
      if (!show) { n.lab.style.opacity = 0; continue; }
      const x = (v.x + 1) / 2 * this.w, y = (1 - v.y) / 2 * this.h + this._screenRadius(n) + 4;
      if (!n.lw) { n.lw = n.lab.offsetWidth || 60; n.lh = n.lab.offsetHeight || 18; }
      const box = [x - n.lw / 2 - 2, y - 1, x + n.lw / 2 + 2, y + n.lh + 1];
      if (rank(n) > 0 && placed.some(b => !(box[2] < b[0] || box[0] > b[2] || box[3] < b[1] || box[1] > b[3]))) { n.lab.style.opacity = 0; continue; }
      placed.push(box);
      n.lab.style.transform = `translate(${x.toFixed(1)}px, ${y.toFixed(1)}px) translateX(-50%)`;
      n.lab.style.opacity = F && !F.has(n.id) ? 0.25 : 1;
      n.lab.classList.toggle('on', n.id === this.focusId || this.hover === n);
    }
  }
  _screenRadius(n) {
    const d = this.camera.position.distanceTo(n.pos);
    return (n.sprite.scale.x / 2) / (d * Math.tan((this.camera.fov * Math.PI / 180) / 2)) * (this.h / 2) * 0.82;
  }

  _pick(e) {
    const r = this.renderer.domElement.getBoundingClientRect();
    this.mouse.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    this.ray.setFromCamera(this.mouse, this.camera);
    const hits = this.ray.intersectObjects(this.nodes.map(n => n.sprite), false);
    return hits.length ? hits[0].object.userData.node : null;
  }

  _bind() {
    const cv = this.renderer.domElement;
    let down = null;
    cv.addEventListener('pointermove', e => {
      if (!this.nodes) return;
      const n = this._pick(e);
      this.hover = n;
      cv.style.cursor = n && (n.kind === 'session' || n.kind === 'person') ? 'pointer' : 'grab';
      if (n && (n.kind === 'session' || n.kind === 'theme' || n.kind === 'chat' || n.kind === 'person')) {
        const r = this.el.getBoundingClientRect();
        this.tip.innerHTML = n.kind === 'session'
          ? `<b>${esc(n.title || 'Conversation')}</b><span>${esc(platformLabel(n.platform))} · ${esc(fullDate(n.date))}${n.n != null ? ` · source ${n.n}` : ''}</span>${n.snippet ? `<p>${esc(n.snippet.slice(0, 220))}${n.snippet.length > 220 ? '…' : ''}</p>` : ''}<em>Click to read</em>`
          : n.kind === 'theme' ? `<b>${esc(n.label)}</b><span>${n.size} conversations</span>${n.title ? `<p>${esc(n.title)}</p>` : ''}`
            : `<b>${esc(n.label)}</b><span>${esc(platformLabel(n.platform))}${n.size ? ` · ${Number(n.size).toLocaleString()} messages` : ''}</span>${n.kind === 'person' ? '<em>Click for profile</em>' : ''}`;
        this.tip.hidden = false;
        const x = e.clientX - r.left, y = e.clientY - r.top;
        this.tip.style.left = Math.min(r.width - this.tip.offsetWidth - 8, x + 14) + 'px';
        this.tip.style.top = Math.min(r.height - this.tip.offsetHeight - 8, y + 14) + 'px';
      } else this.tip.hidden = true;
    });
    cv.addEventListener('pointerleave', () => { this.hover = null; this.tip.hidden = true; });
    cv.addEventListener('pointerdown', e => { down = [e.clientX, e.clientY]; });
    cv.addEventListener('pointerup', e => {
      if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 4) { down = null; return; }
      down = null;
      const n = this._pick(e);
      if (!n) { this.unfocus(); return; }
      if (n.kind === 'session') { this.focus(n.id); this.onSession?.(n); }
      else if (n.kind === 'person') { this.focus(n.id); this.onPerson?.(n); }
      else this.focus(n.id);
    });
  }
}
