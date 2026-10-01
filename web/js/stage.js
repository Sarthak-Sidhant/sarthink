// The right-hand stage: the full 2D memory map, or — once there's something specific to look at —
// the 3D focus constellation of just those people, chats and sessions.
import { $, esc, num, cleanName } from './util.js';
import { Focus3D, specFromSources, specFromPerson } from './focus3d.js';
import { getDossier } from './people.js';

export class Stage {
  constructor(graph, { onSession, onPerson } = {}) {
    this.graph = graph;
    this.focus = new Focus3D($('focus3d'), { onSession, onPerson });
    $('graph-map').addEventListener('click', () => this.clear());
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
  _focusMode(spec, what) {
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

  showSources(sources) {
    if (!this.ready) return;   // chat re-selects its answer once the graph has loaded
    const spec = specFromSources(sources, this.graph);
    const people = spec.nodes.filter(n => n.kind === 'person').length, sessions = spec.nodes.filter(n => n.kind === 'session').length;
    this._focusMode(spec, `<b>This answer</b> · ${sessions} conversation${sessions === 1 ? '' : 's'} · ${people} ${people === 1 ? 'person' : 'people'}`);
  }

  // one conversation opened from somewhere else (e.g. the reader): keep it if it's already on stage
  showSession(s) {
    if (!this.ready) return;
    const id = `S_${s.session_id}`;
    if (this.focus.visible && this.focus.has(id)) { this.focus.focus(id); return; }
    const src = { session_id: s.session_id, header: s.header, date: s.start, platform: s.platform, nodes: s.nodes, snippet: s.summary?.summary };
    this._focusMode(specFromSources([src], this.graph), '<b>One conversation</b>');
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
    const spec = specFromPerson(d, this.graph);
    const sessions = spec.nodes.filter(n => n.kind === 'session').length;
    this._focusMode(spec, `<b>${esc(cleanName(d.name))}</b> · ${num(d.messages_total)} messages · ${d.themes.length} themes · ${sessions} conversations`);
  }

  clear() {
    this._pending = null;
    this._mapMode();
    this.graph.clear();
    $('graph-clear').hidden = true;
  }

  graphLoaded() { const p = this._pending; this._pending = null; p?.(); }
  themeChanged() { this.graph.themeChanged(); this.focus.rebuild(); }
}
