// Ask: chat on the left; the selected answer drives the graph highlight and the sources pane on the right.
import { $, esc, markdown, dot, num, platformLabel, sessionTitle } from './util.js';
import { renderReader } from './reader.js';

const SUGGESTIONS = [
  'Who do I talk to the most, and where?',
  'What did I promise people in the last few months?',
  'What have I been talking about most this year?',
  'Summarise what happened in my chats last week',
];
const STORE = 'sarthink.chat.v1';

export class Chat {
  constructor(graph) {
    this.graph = graph; this.turns = []; this.busy = false; this.sel = null;
    this.log = $('chat-log'); this.input = $('chat-in');
    $('suggest').innerHTML = SUGGESTIONS.map(s => `<button type="button">${esc(s)}</button>`).join('');
    $('suggest').addEventListener('click', e => { const b = e.target.closest('button'); if (b) this.ask(b.textContent); });
    $('composer').addEventListener('submit', e => { e.preventDefault(); this.ask(); });
    this.input.addEventListener('keydown', e => {
      if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); this.ask(); }
    });
    this.input.addEventListener('input', () => this.autosize());
    $('new-chat').addEventListener('click', () => this.reset());
    this.log.addEventListener('click', e => this.onLogClick(e));
    $('sources-body').addEventListener('click', e => this.onSourcesClick(e));
    this.restore();
    this.updateSend();
  }

  autosize() { this.input.style.height = 'auto'; this.input.style.height = Math.min(200, this.input.scrollHeight) + 'px'; this.updateSend(); }
  updateSend() { $('send').disabled = this.busy || !this.input.value.trim(); }
  focus() { this.input.focus(); }
  prefill(text) { this.input.value = text; this.autosize(); this.input.focus(); }

  history() {
    return this.turns.filter(t => t.data?.answer).slice(-3)
      .flatMap(t => [{ role: 'user', content: t.q }, { role: 'assistant', content: t.data.answer }]);
  }

  async ask(text) {
    const q = (text ?? this.input.value).trim();
    if (!q || this.busy) return;
    this.busy = true; this.input.value = ''; this.autosize();
    $('chat-empty').hidden = true;
    const turn = { q, data: null, steps: [], t0: Date.now() };
    const history = this.history();
    this.turns.push(turn);
    turn.el = this.renderTurn(turn);
    this.log.append(turn.el);
    this.scroll();
    const seen = new Set();
    const tick = setInterval(() => this.paintProgress(turn), 1000);
    try {
      const r = await fetch('/api/ask_stream', {
        method: 'POST', headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ question: q, history, multi: $('deep').checked }),
      });
      if (!r.ok || !r.body) throw new Error(`${r.status} ${(await r.text()).slice(0, 200)}`);
      const reader = r.body.getReader(), dec = new TextDecoder();
      let buf = '';
      while (!turn.data) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        let cut;
        while ((cut = buf.indexOf('\n\n')) >= 0) {
          const line = buf.slice(0, cut).split('\n').find(l => l.startsWith('data: '));
          buf = buf.slice(cut + 2);
          if (!line) continue;
          const ev = JSON.parse(line.slice(6));
          if (ev.type === 'step') {
            turn.steps.push(ev.text);
            (ev.nodes || []).forEach(n => seen.add(n));
            if (seen.size) this.graph.highlight([...seen], { fly: false });   // the graph lights up as it reads
            this.paintProgress(turn);
          } else if (ev.type === 'answer') turn.data = ev.data;
          else if (ev.type === 'error') throw new Error(ev.message);
        }
      }
      if (!turn.data) throw new Error('the answer stream ended early');
      this.paintAnswer(turn);
      this.select(turn);
    } catch (e) {
      turn.error = e.message;
      turn.el.querySelector('.a').innerHTML = `<div class="a-err">Something went wrong: ${esc(e.message)}</div>
        <div class="a-meta"><button data-retry>Try again</button></div>`;
    } finally {
      clearInterval(tick);
      this.busy = false; this.updateSend(); this.save(); this.scroll();
    }
  }

  renderTurn(turn) {
    const el = document.createElement('div');
    el.className = 'turn';
    el.innerHTML = `<div class="q">${esc(turn.q)}</div><div class="a"></div>`;
    el._turn = turn;
    if (turn.data) this.paintAnswer(turn, el); else if (!turn.error) this.paintProgress(turn, el);
    return el;
  }

  paintProgress(turn, el = turn.el) {
    if (turn.data || turn.error || !el) return;
    const s = Math.round((Date.now() - turn.t0) / 1000);
    const last = turn.steps.slice(-4);
    el.querySelector('.a').innerHTML = `<div class="progress">
      ${last.length ? last.map((t, i) => `<div class="step${i === last.length - 1 ? ' cur' : ''}">${esc(t)}</div>`).join('') : '<div class="step cur">Thinking</div>'}
      <div class="elapsed"><span class="dots"><i></i><i></i><i></i></span> ${s}s</div></div>`;
  }

  paintAnswer(turn, el = turn.el) {
    const d = turn.data, a = el.querySelector('.a');
    const lookups = (d.steps || []).filter(x => !['plan', 'verify'].includes(x.tool)).length;
    const nSrc = (d.sources || []).length;
    a.innerHTML = `<div class="a-body">${markdown(d.answer)}</div>
      ${numbersHtml(d.steps)}
      ${turn.steps.length ? `<details class="trace"><summary>${lookups} lookups · ${d.seconds}s${d.mode === 'multi' ? ' · deep research' : ''}</summary><ol>${turn.steps.map(t => `<li>${esc(t)}</li>`).join('')}</ol></details>` : ''}
      <div class="a-meta">
        ${nSrc ? `<button data-show-sources>${nSrc} source${nSrc > 1 ? 's' : ''} →</button>` : ''}
        <button data-copy title="Copy answer">Copy</button>
      </div>`;
  }

  select(turn, { fly = true } = {}) {
    this.sel = turn;
    this.log.querySelectorAll('.a.selected').forEach(x => x.classList.remove('selected'));
    turn.el?.querySelector('.a').classList.add('selected');
    this.renderSources();
    const nodes = [...new Set((turn.data?.sources || []).flatMap(s => s.nodes))];
    if (nodes.length) this.graph.highlight(nodes, { fly }); else this.graph.clear();
    $('graph-clear').hidden = !nodes.length;
  }

  renderSources(activeN = null) {
    const body = $('sources-body'), srcs = this.sel?.data?.sources || [];
    $('sources-title').textContent = 'Sources';
    $('sources-sub').textContent = srcs.length ? `${srcs.length} cited` : '';
    if (!srcs.length) {
      body.innerHTML = `<p class="placeholder">${this.sel ? 'This answer didn\'t cite any conversations.' : 'Citations from the selected answer appear here. Click one to read the conversation.'}</p>`;
      return;
    }
    body.innerHTML = srcs.map(s => `
      <div class="src${activeN === s.n ? ' on' : ''}" data-n="${s.n}" tabindex="0">
        <span class="n">${s.n}</span>
        <div class="h">${esc(sessionTitle(s.header))}</div>
        <div class="d">${dot(s.platform)}${esc(platformLabel(s.platform))} · ${esc(s.date || '')}${s.read_full ? ' · read in full' : ''}</div>
        <div class="sn">${esc(s.snippet || '')}</div>
      </div>`).join('');
  }

  focusSource(n) {
    const s = (this.sel?.data?.sources || []).find(x => x.n === +n);
    if (!s) return;
    const all = [...new Set(this.sel.data.sources.flatMap(x => x.nodes))];
    this.graph.highlight(all, { focus: s.nodes });
    this.log.querySelectorAll('.cite.on').forEach(c => c.classList.remove('on'));
    this.sel.el.querySelectorAll(`.cite[data-n="${n}"]`).forEach(c => c.classList.add('on'));
    if (!$('sources-body').querySelector('.src')) this.renderSources(+n);
    $('sources-body').querySelectorAll('.src').forEach(x => x.classList.toggle('on', x.dataset.n === String(n)));
    $('sources-body').querySelector(`.src[data-n="${n}"]`)?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }

  openSource(n) {
    const s = (this.sel?.data?.sources || []).find(x => x.n === +n);
    if (!s) return;
    this.focusSource(n);
    $('sources-title').textContent = `Source ${n}`;
    $('sources-sub').textContent = '';
    const body = $('sources-body');
    body.scrollTop = 0;
    renderReader(body, s.session_id, {
      backLabel: 'All sources',
      onBack: () => { this.renderSources(+n); this.focusSource(n); },
    });
  }

  // open any conversation in the right pane (from the graph, or elsewhere)
  openSession(sessionId, label = 'Conversation') {
    $('sources-title').textContent = label; $('sources-sub').textContent = '';
    renderReader($('sources-body'), sessionId, {
      backLabel: this.sel ? 'All sources' : 'Back',
      onBack: () => this.renderSources(),
      onNodes: nodes => { this.graph.highlight(nodes); $('graph-clear').hidden = false; },
    });
  }

  onLogClick(e) {
    const turnEl = e.target.closest('.turn'), turn = turnEl?._turn;
    if (!turn) return;
    if (e.target.closest('[data-retry]')) { turnEl.remove(); this.turns.splice(this.turns.indexOf(turn), 1); this.ask(turn.q); return; }
    if (e.target.closest('[data-copy]')) { navigator.clipboard?.writeText(turn.data.answer); e.target.textContent = 'Copied'; setTimeout(() => { e.target.textContent = 'Copy'; }, 1200); return; }
    if (!turn.data) return;
    const cite = e.target.closest('.cite[data-n]');
    if (this.sel !== turn) this.select(turn, { fly: !cite });
    if (cite) this.focusSource(cite.dataset.n);
    if (e.target.closest('[data-show-sources]')) $('sources-pane').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  onSourcesClick(e) {
    const src = e.target.closest('.src[data-n]');
    if (src) this.openSource(src.dataset.n);
  }

  reset() {
    if (this.busy) return;
    this.turns = []; this.sel = null;
    this.log.querySelectorAll('.turn').forEach(t => t.remove());
    $('chat-empty').hidden = false;
    this.renderSources(); this.graph.clear(); $('graph-clear').hidden = true;
    this.save(); this.focus();
  }

  scroll() { requestAnimationFrame(() => { this.log.scrollTop = this.log.scrollHeight; }); }

  // the conversation survives a reload (this browser only)
  save() {
    try {
      const keep = this.turns.filter(t => t.data).slice(-20).map(t => ({
        q: t.q, steps: t.steps, data: {
          answer: t.data.answer, sources: t.data.sources, seconds: t.data.seconds, mode: t.data.mode,
          steps: (t.data.steps || []).map(s => s.tool === 'run_sql' ? { tool: s.tool, args: s.args, result: s.result } : { tool: s.tool }),
        },
      }));
      localStorage.setItem(STORE, JSON.stringify(keep));
    } catch { /* storage full or blocked: fine */ }
  }
  restore() {
    let saved = [];
    try { saved = JSON.parse(localStorage.getItem(STORE) || '[]'); } catch { /* ignore */ }
    if (!saved.length) return;
    $('chat-empty').hidden = true;
    for (const t of saved) { this.turns.push(t); t.el = this.renderTurn(t); this.log.append(t.el); }
    this.scroll();
  }
  afterGraphLoad() { const last = this.turns[this.turns.length - 1]; if (last?.data) this.select(last, { fly: true }); }
}


// "How these numbers were computed": the agent's read-only SQL, as a bar list or table
function numbersHtml(steps) {
  const qs = (steps || []).filter(x => x.tool === 'run_sql' && x.result?.rows?.length);
  if (!qs.length) return '';
  const parts = qs.map(x => {
    const { columns, rows } = x.result;
    const shown = rows.slice(0, 12);
    const numCol = columns.findIndex((c, i) => i > 0 && shown.every(r => typeof r[i] === 'number'));
    let body;
    if (shown.length >= 2 && numCol > 0 && columns.length <= 3) {
      const max = Math.max(...shown.map(r => r[numCol]), 1);
      body = shown.map(r => `<div class="hbar" title="${esc(r[0])}: ${num(r[numCol])}"><span class="l">${esc(r[0])}</span>` +
        `<span class="t" style="width:${Math.max(1, 100 * r[numCol] / max)}%"></span><span class="v">${num(r[numCol])}</span></div>`).join('');
    } else {
      body = `<table><tr>${columns.map(c => `<th>${esc(c)}</th>`).join('')}</tr>${shown.map(r => `<tr>${r.map(v => `<td>${esc(typeof v === 'number' ? num(v) : v ?? '')}</td>`).join('')}</tr>`).join('')}</table>`;
    }
    return `<div class="nums-q"><code>${esc(x.args?.sql || '')}</code>${body}${rows.length > 12 ? `<div class="muted" style="font-size:11.5px;margin-top:4px">… ${rows.length - 12} more rows</div>` : ''}</div>`;
  });
  return `<details class="nums"><summary>How these numbers were computed · ${qs.length} ${qs.length > 1 ? 'queries' : 'query'}</summary>${parts.join('')}</details>`;
}
