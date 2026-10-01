// The right-hand stage: the full 2D memory map, or — once there's something specific to look at —
// the 3D focus constellation of just those people, chats and sessions.
import { $, esc, num, cleanName } from './util.js';
import { getDossier } from './people.js';

export function showGraphError(msg) {
  const el = $('graph-error');
  el.innerHTML = `<span>${esc(msg)}</span><button aria-label="Dismiss">×</button>`;
  el.hidden = false;
  el.querySelector('button').onclick = () => { el.hidden = true; };
}

export class Stage {
  constructor(graph, { onSession, onPerson } = {}) {
    this.graph = graph;
    this.handlers = { onSession, onPerson };
    // stand-in until three.js has loaded (it loads on first use, so it can never break the map)
    this.focus = { visible: false, spec: null, has: () => false, hide() {}, rebuild() {}, resize() {}, focus() {}, show() {} };
    $('graph-map').addEventListener('click', () => this.clear());
    $('graph-expand').addEventListener('click', () => this.expand(!$('graph-pane').classList.contains('expanded')));
    addEventListener('keydown', e => { if (e.key === 'Escape' && $('graph-pane').classList.contains('expanded')) this.expand(false); });
    $('graph-clear').addEventListener('click', () => this.clear());
  }
  get ready() { return !!this.graph.n; }
  get inFocus() { return this.focus.visible; }

  _mapMode() {
    if (!this.focus.visible && !$('overview').hidden) return;
    this.focus.hide();
    $('overview').hidden = false; $('focus-bar').hidden = true; $('graph-map').hidden = true;
    this.graph.resize(); this.graph.glDirty = true; this.graph.draw();
  }
  async _load() {
    if (!this._mod) {
      this._mod = import('./focus3d.js').then(m => {
        this.focus = new m.Focus3D($('focus3d'), this.handlers);
        return m;
      });
    }
    try { return await this._mod; } catch (e) {
      this._mod = null;
      showGraphError(`The 3D view couldn't load (${e.message}). Showing the map instead.`);
      return null;
    }
  }

  async _focusMode(spec, what) {
    if (!spec) return;
    $('overview').hidden = true;
    $('graph-clear').hidden = true; $('graph-map').hidden = false;
    $('focus-what').innerHTML = what; $('focus-bar').hidden = false;
    this.focus.show(spec);
  }

  // live progress while the assistant reads: light the nodes up on the full map
  highlight(nodes, opts) {
    this._mapMode();
    const n = this.graph.highlight(nodes, opts);
    $('graph-clear').hidden = !n;
  }

  async showSources(sources) {
    if (!this.ready) return;   // chat re-selects its answer once the graph has loaded
    const m = await this._load();
    if (!m) { this.highlight([...new Set(sources.flatMap(s => s.nodes))]); return; }
    const spec = m.specFromSources(sources, this.graph);
    const people = spec.nodes.filter(n => n.kind === 'person').length, sessions = spec.nodes.filter(n => n.kind === 'session').length;
    this._focusMode(spec, `<b>This answer</b> · ${sessions} conversation${sessions === 1 ? '' : 's'} · ${people} ${people === 1 ? 'person' : 'people'}`);
  }

  // one conversation opened from somewhere else (e.g. the reader): keep it if it's already on stage
  async showSession(s) {
    if (!this.ready) return;
    const m = await this._load();
    if (!m) { this.highlight(s.nodes); return; }
    const id = `S_${s.session_id}`;
    if (this.focus.visible && this.focus.has(id)) { this.focus.focus(id); return; }
    const src = { session_id: s.session_id, header: s.header, date: s.start, platform: s.platform, nodes: s.nodes, snippet: s.summary?.summary };
    this._focusMode(m.specFromSources([src], this.graph), '<b>One conversation</b>');
    setTimeout(() => this.focus.focus(id, { fly: false }), 60);
  }

  focusSession(sessionId) {
    if (this.focus.visible) this.focus.focus(`S_${sessionId}`);
  }

  async showPerson(pid, name) {
    if (!this.ready) { this._pending = () => this.showPerson(pid, name); return; }
    $('focus-what').innerHTML = `<b>${esc(cleanName(name || ''))}</b> · reading your history…`;
    $('focus-bar').hidden = false;
    let d;
    try { d = await getDossier(pid); } catch (e) { $('focus-what').textContent = `Couldn't load (${e.message})`; return; }
    const m = await this._load();
    if (!m) { $('focus-bar').hidden = true; this.highlight([d.node, ...d.top_chats.map(c => c.node)], { focus: [d.node] }); return; }
    const spec = m.specFromPerson(d, this.graph);
    const sessions = spec.nodes.filter(n => n.kind === 'session').length;
    this._focusMode(spec, `<b>${esc(cleanName(d.name))}</b> · ${num(d.messages_total)} messages · ${d.themes.length} themes · ${sessions} conversations`);
  }

  clear() {
    this._pending = null;
    this._mapMode();
    this.graph.clear();
    $('graph-clear').hidden = true;
  }

  // full screen graph (also how "Explore graph" shows it on narrow windows, where the panel is off-screen)
  expand(on = true) {
    $('graph-pane').classList.toggle('expanded', on);
    requestAnimationFrame(() => { this.graph.resize(); this.focus.resize(); });
  }

  // an answer that cited no conversations: the people and chats it looked at / named
  async showNodes(nodes, label = 'This answer') {
    if (!this.ready || !nodes?.length) return false;
    const m = await this._load();
    if (!m) { this.highlight(nodes); return true; }
    const spec = m.specFromNodes(nodes, this.graph);
    if (spec.nodes.length < 2) return false;
    const people = spec.nodes.filter(n => n.kind === 'person').length;
    this._focusMode(spec, `<b>${esc(label)}</b> · ${people} ${people === 1 ? 'person' : 'people'}`);
    return true;
  }

  graphLoaded() { const p = this._pending; this._pending = null; p?.(); }
  themeChanged() { this.graph.themeChanged(); this.focus.rebuild(); }
}
