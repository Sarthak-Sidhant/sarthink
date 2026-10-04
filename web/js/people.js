// People and accounts (docs/accounts-and-people.md). An account is one profile on one platform, with its own
// summary and graph node. A person groups accounts ("Riya" = her WhatsApp + Instagram); you are person 1.
// Attaching or detaching an account is one row on the server: the page and the graph update in place.
import { $, api, esc, dot, fmtDate, fmtMonth, num, initials, cleanName, platformColor, platformLabel, PLATFORMS, toast } from './util.js';
import { openDrawer } from './reader.js';

const SHOW = 150;
const ME = 'P_1';
const dossiers = new Map();   // this tab's copy, keyed by node ref ("P_3" / "A_12"); the server caches them too
const url = ref => ref.startsWith('P_') ? `/api/people/${ref.slice(2)}` : `/api/account/${ref.slice(2)}`;
export async function getDossier(ref) {
  ref = String(ref);
  if (!dossiers.has(ref)) dossiers.set(ref, api(`${url(ref)}/dossier`).catch(e => { dossiers.delete(ref); throw e; }));
  return dossiers.get(ref);
}
const forget = ref => dossiers.delete(ref);

export class People {
  constructor({ onAsk, onShowInGraph, onPeopleChanged }) {
    this.people = []; this.accounts = []; this.suggestions = [];
    this.tab = 'people'; this.q = ''; this.plat = null; this.current = null; this.info = null;
    this.onAsk = onAsk; this.onShowInGraph = onShowInGraph; this.onPeopleChanged = onPeopleChanged;
    let t;
    $('ppl-q').addEventListener('input', e => { clearTimeout(t); t = setTimeout(() => { this.q = e.target.value.trim(); this.refreshList(); }, 150); });
    $('ppl-platforms').addEventListener('click', e => {
      const b = e.target.closest('button');
      if (!b) return;
      if (b.dataset.tab) { this.tab = b.dataset.tab; this.plat = null; }
      else if (b.dataset.p) this.plat = this.plat === b.dataset.p ? null : b.dataset.p;
      this.refreshList();
    });
    $('ppl-items').addEventListener('click', e => {
      const r = e.target.closest('[data-ref]');
      if (r) location.hash = `#people/${r.dataset.ref}`;
    });
    $('person').addEventListener('click', e => this.onClick(e));
    let ft;
    $('person').addEventListener('input', e => {
      if (e.target.id === 'acct-q') { clearTimeout(ft); ft = setTimeout(() => this.findAccounts(e.target.value.trim()), 200); }
      if (e.target.id === 'pp-q') this.findPeople(e.target.value.trim());
    });
    $('person').addEventListener('keydown', e => {
      if (e.key === 'Enter' && e.target.closest('.inline-form')) e.target.closest('.inline-form').querySelector('[data-save]')?.click();
    });
  }

  // ─── data ─────────────────────────────────────────────────────────────
  async load() {
    await Promise.all([this.loadPeople(), this.loadSuggestions()]);
    this.refreshList();
  }
  async loadPeople() {
    try { this.people = (await api('/api/people')).people; } catch { return; }
    this.onPeopleChanged?.(this.people);
  }
  async loadSuggestions() {
    try { this.suggestions = (await api('/api/suggestions')).suggestions; } catch { this.suggestions = []; }
  }
  async loadAccounts() {
    const q = new URLSearchParams({ q: this.q, limit: SHOW });
    if (this.plat) q.set('platform', this.plat);
    try { this.accounts = (await api(`/api/accounts?${q}`)).accounts; } catch { this.accounts = []; }
  }

  // after any change: people, suggestions, the list, the open page and the graph
  async changed(refs = []) {
    refs.forEach(forget);
    await Promise.all([this.loadPeople(), this.loadSuggestions()]);
    this.refreshList();
    if (this.current) this.open(this.current, { quiet: true });
  }

  // ─── list ─────────────────────────────────────────────────────────────
  async refreshList() {
    if (this.tab === 'accounts') await this.loadAccounts();
    this.renderChips(); this.renderList();
  }
  renderChips() {
    const tab = (k, label, n) => `<button data-tab="${k}" class="${this.tab === k ? 'on' : ''}">${label}${n != null ? ` <span class="muted">${num(n)}</span>` : ''}</button>`;
    $('ppl-platforms').innerHTML = tab('people', 'People', this.people.length) + tab('accounts', 'Accounts') +
      (this.suggestions.length ? tab('suggestions', 'Same person?', this.suggestions.length) : '') +
      (this.tab === 'accounts' ? `<div class="chips-row">${PLATFORMS.map(p =>
        `<button data-p="${p.key}" class="${this.plat === p.key ? 'on' : ''}">${dot(p.key)}${esc(p.label)}</button>`).join('')}</div>` : '');
    $('ppl-q').placeholder = this.tab === 'accounts' ? 'Search accounts' : 'Search people';
  }
  renderList() {
    const q = this.q.toLowerCase(), cur = this.current;
    let html = '';
    if (this.tab === 'people') {
      const list = this.people.filter(p => !q || p.name.toLowerCase().includes(q));
      html = list.map(p => row({ ref: p.node, name: p.name, isMe: p.node === ME, color: p.node === ME ? 'var(--me);color:var(--bg)' : 'var(--person)',
        sub: `${p.platforms.map(dot).join('')} ${p.accounts.length} account${p.accounts.length === 1 ? '' : 's'}`, n: p.messages }, cur)).join('') +
        (list.length ? '' : `<p class="placeholder">${q ? 'No person by that name.' : 'No people yet.'}</p>`) +
        `<p class="ppl-note">A person groups the accounts of one human. Open any account (Accounts) to start one.</p>`;
    } else if (this.tab === 'accounts') {
      html = this.accounts.map(a => row({ ref: a.node, name: a.name, isMe: a.is_me, color: a.is_me ? 'var(--me);color:var(--bg)' : platformColor(a.platform),
        sub: `${esc(platformLabel(a.platform))}${a.person ? ` · <span class="tag">${esc(cleanName(a.person))}</span>` : ''}`, n: a.messages }, cur)).join('') +
        (this.accounts.length >= SHOW ? `<p class="ppl-note">The ${SHOW} biggest. Search to find anyone.</p>` : '') +
        (this.accounts.length ? '' : '<p class="placeholder">No account matches.</p>');
    } else {
      html = this.suggestions.map((s, i) => `<div class="ppl sugg${cur === `S_${i}` ? ' on' : ''}" data-ref="suggestions">
          <span class="nm"><b>${esc(cleanName(s.a.name))}</b><span>${dot(s.a.platform)}${esc(platformLabel(s.a.platform))} ↔ ${dot(s.b.platform)}${esc(platformLabel(s.b.platform))}</span></span>
          <span class="cnt">${num(s.a.messages + s.b.messages)}</span></div>`).join('') ||
        '<p class="placeholder">No open suggestions.</p>';
    }
    $('ppl-items').innerHTML = html;
  }

  // ─── a person or an account ───────────────────────────────────────────
  async open(ref, { quiet = false } = {}) {
    if (ref === 'suggestions') return this.openSuggestions();
    this.current = ref;
    $('ppl-items').querySelectorAll('[data-ref]').forEach(r => r.classList.toggle('on', r.dataset.ref === ref));
    const el = $('person');
    if (!quiet) {
      el.scrollTop = 0;
      el.innerHTML = `<div class="p-head"><span class="avatar" style="background:var(--muted)"></span>
        <div><h1>Loading…</h1><div class="sub">Reading your history…</div></div></div>
        <div class="tiles">${'<div class="tile"><div class="skeleton" style="height:26px;width:60%"></div><div class="skeleton" style="height:12px;width:80%;margin-top:8px"></div></div>'.repeat(4)}</div>
        <div class="skeleton" style="height:180px;border-radius:12px"></div>`;
    }
    let info, d;
    try { [info, d] = await Promise.all([api(url(ref)), getDossier(ref)]); } catch (e) {
      if (this.current === ref) el.innerHTML = `<p class="placeholder">Couldn't load (${esc(e.message)}).</p>`;
      return;
    }
    if (this.current !== ref) return;
    this.info = info; this.dossier = d;
    this.render(ref, info, d);
  }

  render(ref, info, d) {
    const el = $('person'), isPerson = ref.startsWith('P_'), isMe = ref === ME || info.is_me;
    const pct = d.messages_total ? Math.round(100 * d.messages_by_you / d.messages_total) : 0;
    const name = cleanName(isPerson ? info.name : info.name);
    const platforms = String(d.platforms || info.platform || '').split(/,\s*/).filter(Boolean);
    const span = d.first_day && d.last_day ? `${fmtDate(d.first_day)} – ${fmtDate(d.last_day)}` : '';
    const avatar = isMe ? 'var(--me);color:var(--bg)' : isPerson ? 'var(--person)' : platformColor(info.platform);
    el.innerHTML = `
      <div class="p-head">
        <span class="avatar" style="background:${avatar}">${esc(initials(name))}</span>
        <div><h1>${esc(name)}</h1><div class="sub">${isPerson ? `Person · ${info.accounts.length} account${info.accounts.length === 1 ? '' : 's'} &nbsp;·&nbsp; ` : ''}${platforms.map(p => `${dot(p)} ${esc(platformLabel(p))}`).join(' &nbsp; ')}${span ? ' &nbsp;·&nbsp; ' + esc(span) : ''}</div></div>
        <div class="acts">
          <button class="ghost-btn" data-graph>Show in graph</button>
          ${isMe ? '' : `<button class="primary-btn" data-ask>Ask about ${esc(name.length > 18 ? 'them' : name)}</button>`}
        </div>
      </div>
      ${isPerson ? personPanel(info) : accountPanel(info)}
      <div class="tiles">
        <div class="tile"><div class="v">${num(d.messages_total)}</div><div class="l">messages ${isMe ? 'in total' : 'together'}</div></div>
        <div class="tile"><div class="v">${isMe ? num(d.top_chats?.length || 0) : pct + '%'}</div><div class="l">${isMe ? 'top chats shown' : 'written by you'}</div></div>
        <div class="tile"><div class="v">${num(d.months?.length || 0)}</div><div class="l">active months</div></div>
        <div class="tile"><div class="v">${d.last_day ? esc(fmtDate(d.last_day)) : '—'}</div><div class="l">last talked</div></div>
      </div>
      <div class="section" id="p-facts" hidden></div>
      ${d.months?.length ? `<div class="section">
        <h2>Activity <span class="legend"><span><i style="background:var(--series-1)"></i>You</span><span><i style="background:var(--ramp-light)"></i>${isMe ? 'Others' : esc(name)}</span></span></h2>
        <div class="chart" id="act-chart">${activityChart(d.months)}</div>
      </div>` : ''}
      ${d.themes?.length ? `<div class="section"><h2>What you talk about</h2><div class="themes">${d.themes.map(t => themeHtml(t)).join('')}</div></div>` : ''}
      ${d.account_themes?.length ? `<div class="section" id="p-by-account"><h2>What you talk about <span class="muted">on each account</span></h2>
        <div class="by-acct">${d.account_themes.map(a => accountThemes(a, d.shared_themes || [], d.account_themes)).join('')}</div></div>` : ''}
      ${d.top_chats?.length ? `<div class="section"><h2>Where you talk</h2><div class="chats">${d.top_chats.map(c => `
        <div class="chat-row">
          <div><div class="ttl">${dot(c.platform)}<span>${esc(c.title || platformLabel(c.platform))}</span></div>
          <div class="sub">${esc(platformLabel(c.platform))} · ${esc(fmtDate(c.from))} – ${esc(fmtDate(c.to))}</div></div>
          <span class="n">${num(c.messages)} msgs</span>
        </div>`).join('')}</div></div>` : ''}`;
    bindChartTips($('act-chart'));
    this.loadFacts(isMe ? 'me' : ref, ref, isMe);
  }

  // the memory layer: what Sarthink knows about this person (or you), by category, with dates and sources
  async loadFacts(ref, page, isMe) {
    let d;
    try { d = await api(`/api/facts?ref=${encodeURIComponent(ref)}`); } catch { return; }
    const el = $('p-facts');
    if (!el || this.current !== page || !d.facts?.length) return;      // moved to another page meanwhile
    const now = d.facts.filter(f => f.status !== 'past'), past = d.facts.filter(f => f.status === 'past');
    const by = {};
    for (const f of now) (by[f.category] = by[f.category] || []).push(f);
    const order = Object.keys(FACT_CATS).filter(k => by[k]).concat(Object.keys(by).filter(k => !FACT_CATS[k]));
    const PER = 4;
    el.innerHTML = `<h2>What I know <span class="muted">${isMe ? 'about you' : ''} · ${num(now.length)} facts from your conversations</span></h2>
      <div class="facts">${order.map(k => `
        <div class="fact-cat">
          <h3>${esc(FACT_CATS[k] || k)} <span class="muted">${by[k].length}</span></h3>
          ${by[k].map((f, i) => factRow(f, i >= PER)).join('')}
          ${by[k].length > PER ? `<button class="ghost-btn sm" data-more-facts>Show ${by[k].length - PER} more</button>` : ''}
        </div>`).join('')}</div>
      ${past.length ? `<details class="fact-history"><summary>History · ${num(past.length)} things that used to be true</summary>
        ${past.sort((a, b) => String(b.until || '').localeCompare(String(a.until || ''))).map(f => factRow(f)).join('')}</details>` : ''}`;
    el.hidden = false;
    el.addEventListener('click', async e => {          // your verdict on a fact
      const b = e.target.closest('[data-fact-v]'), row = b?.closest('[data-fact]');
      if (!b || !row?.dataset.fact) return;
      try {
        await api(`/api/facts/${encodeURIComponent(row.dataset.fact)}/feedback`, { body: { verdict: b.dataset.factV } });
        if (b.dataset.factV === 'wrong') { row.classList.add('gone'); setTimeout(() => row.remove(), 250); toast('Hidden — it won\'t come back'); }
        else b.outerHTML = '<span class="ok">✓ you confirmed</span>';
      } catch (err) { toast(`Couldn't save (${err.message})`); }
    });
    el.querySelectorAll('[data-more-facts]').forEach(b => b.addEventListener('click', () => {
      b.parentElement.querySelectorAll('.fact.extra').forEach(x => x.classList.remove('extra')); b.remove();
    }));
  }

  // ─── same-name suggestions in one place ───────────────────────────────
  openSuggestions() {
    this.current = 'suggestions';
    const side = a => `<div class="m-side" data-open="${a.node}">${dot(a.platform)}<div><b>${esc(cleanName(a.name))}</b>
      <span>${esc(platformLabel(a.platform))} · ${num(a.messages)} msgs${a.person ? ` · in ${esc(cleanName(a.person))}` : ''}</span></div></div>`;
    $('person').innerHTML = `<div class="p-head"><div><h1>Same person?</h1>
        <div class="sub">Accounts with the same name on different platforms. Nothing is grouped until you say so.</div></div></div>
      ${this.suggestions.length ? `<div class="chats">${this.suggestions.map((s, i) => `<div class="chat-row match">
        <div class="m-pair">${side(s.a)}<span class="muted">↔</span>${side(s.b)}</div>
        <span class="acct-acts"><button class="primary-btn sm" data-same="${i}">Same person</button><button class="ghost-btn sm" data-not="${i}">Not them</button></span>
      </div>`).join('')}</div>` : '<p class="placeholder">No open suggestions.</p>'}
      <div class="acct-status muted"></div>`;
  }

  // ─── actions (each a single row on the server) ────────────────────────
  status(html) { const s = $('person').querySelector('.acct-status'); if (s) s.innerHTML = html; }
  async act(fn, refs) {
    $('person').querySelectorAll('.p-panel button, .p-panel input, .chat-row button').forEach(b => { b.disabled = true; });
    try { const r = await fn(); await this.changed(refs); return r; } catch (e) {
      this.status(`<span class="bad">${esc(e.message)}</span>`);
      $('person').querySelectorAll('button, input').forEach(b => { b.disabled = false; });
    }
  }
  // put account `aid` and the account/person on the other side together, the way a human would expect
  async sameHuman(a, b) {
    const pa = a.person_id, pb = b.person_id;
    if (pa && !pb) return api(`/api/people/${pa}/accounts`, { body: { account_id: b.account_id } }).then(() => `P_${pa}`);
    if (pb && !pa) return api(`/api/people/${pb}/accounts`, { body: { account_id: a.account_id } }).then(() => `P_${pb}`);
    if (pa && pb) {      // two people: move b's accounts into a's person, then drop b's person
      const pB = await api(`/api/people/${pb}`);
      for (const x of pB.accounts) await api(`/api/people/${pa}/accounts`, { body: { account_id: x.account_id } });
      await api(`/api/people/${pb}`, { method: 'DELETE' });
      return `P_${pa}`;
    }
    const p = await api('/api/people', { body: { name: cleanName(a.name), account_ids: [a.account_id, b.account_id] } });
    return p.node;
  }

  async onClick(e) {
    const t = e.target, b = t.closest('button, [data-open]');
    const s = t.closest('[data-session]');
    if (s) { openDrawer(s.dataset.session); return; }
    if (b?.dataset.open) { location.hash = `#people/${b.dataset.open}`; return; }
    if (this.current === 'suggestions' && b) {
      const sg = this.suggestions[+(b.dataset.same ?? b.dataset.not)];
      if (!sg) return;
      if (b.dataset.not !== undefined) {
        await api('/api/suggestions/reject', { body: { a: sg.a.account_id, b: sg.b.account_id } });
        await this.loadSuggestions(); this.renderChips(); this.openSuggestions(); return;
      }
      this.status('Grouping…');
      const ref = await this.act(() => this.sameHuman(sg.a, sg.b), [sg.a.node, sg.b.node]);
      if (ref) location.hash = `#people/${ref}`;
      return;
    }
    const info = this.info, ref = this.current;
    if (!b || !info) {
      if (t.closest('[data-ask]') && this.dossier) this.onAsk(`What do I usually talk about with ${cleanName(this.dossier.name)}, and what's the latest between us?`);
      return;
    }
    if (b.hasAttribute('data-ask')) return this.onAsk(`What do I usually talk about with ${cleanName(this.dossier.name)}, and what's the latest between us?`);
    if (b.hasAttribute('data-graph')) return this.onShowInGraph(ref, this.dossier?.name);
    const pid = ref.startsWith('P_') ? +ref.slice(2) : info.person_id;
    // person page
    if (b.dataset.remove) return this.act(() => api(`/api/people/${pid}/accounts/${b.dataset.remove}`, { method: 'DELETE' }), [ref, `A_${b.dataset.remove}`]);
    if (b.dataset.add) return this.act(() => api(`/api/people/${pid}/accounts`, { body: { account_id: +b.dataset.add } }), [ref, `A_${b.dataset.add}`]);
    if (b.dataset.sugg) {
      const other = info.suggestions[+b.dataset.sugg];
      return this.act(async () => {
        const into = await this.sameHuman(ref.startsWith('P_') ? { ...info, person_id: pid, account_id: info.accounts?.[0]?.account_id, name: info.name } : info, other);
        if (into && into !== ref) location.hash = `#people/${into}`;
      }, [ref, other.node, other.person_node].filter(Boolean));
    }
    if (b.dataset.reject) {
      const other = info.suggestions[+b.dataset.reject];
      const mine = ref.startsWith('P_') ? info.accounts.map(a => a.account_id) : [info.account_id];
      return this.act(async () => { for (const a of mine) await api('/api/suggestions/reject', { body: { a, b: other.account_id } }); }, []);
    }
    if (b.hasAttribute('data-rename')) { const f = $('p-rename'); f.hidden = !f.hidden; if (!f.hidden) f.querySelector('input').select(); return; }
    if (b.hasAttribute('data-save') && b.closest('#p-rename')) {
      const v = $('p-rename').querySelector('input').value.trim();
      if (v) return this.act(() => api(`/api/people/${pid}/name`, { body: { name: v } }), [ref]);
      return;
    }
    if (b.hasAttribute('data-delete')) {
      if (!confirm(`Remove the person “${cleanName(info.name)}”? Their accounts stay, just without a person.`)) return;
      await this.act(() => api(`/api/people/${pid}`, { method: 'DELETE' }), [ref, ...info.accounts.map(a => a.node)]);
      location.hash = '#people';
      return;
    }
    // account page
    if (b.hasAttribute('data-create')) {
      const v = $('p-create').querySelector('input').value.trim() || cleanName(info.name);
      const p = await this.act(() => api('/api/people', { body: { name: v, account_ids: [info.account_id] } }), [ref]);
      if (p) location.hash = `#people/${p.node}`;
      return;
    }
    if (b.hasAttribute('data-new')) { const f = $('p-create'); f.hidden = !f.hidden; $('p-join').hidden = true; if (!f.hidden) f.querySelector('input').select(); return; }
    if (b.hasAttribute('data-join')) { const f = $('p-join'); f.hidden = !f.hidden; $('p-create').hidden = true; if (!f.hidden) { f.querySelector('input').focus(); this.findPeople(''); } return; }
    if (b.dataset.into) return this.act(() => api(`/api/people/${b.dataset.into}/accounts`, { body: { account_id: info.account_id } }), [ref, `P_${b.dataset.into}`]);
    if (b.hasAttribute('data-leave')) return this.act(() => api(`/api/people/${info.person_id}/accounts/${info.account_id}`, { method: 'DELETE' }), [ref, info.person_node]);
  }

  async findAccounts(q) {
    const out = $('acct-found');
    if (!out) return;
    if (q.length < 2) { out.innerHTML = ''; return; }
    const { accounts } = await api(`/api/accounts?${new URLSearchParams({ q, limit: 12 })}`);
    const mine = new Set(this.info.accounts.map(a => a.account_id));
    const list = accounts.filter(a => !a.is_me && !mine.has(a.account_id));
    out.innerHTML = list.length ? `<div class="chats">${list.map(a => acctRow(a,
      `<button class="ghost-btn sm" data-add="${a.account_id}">${a.person ? 'Move here' : 'Add'}</button>`)).join('')}</div>`
      : '<p class="placeholder">No account by that name.</p>';
  }

  findPeople(q) {
    const out = $('pp-found');
    if (!out) return;
    const list = this.people.filter(p => p.node !== ME && (!q || p.name.toLowerCase().includes(q.toLowerCase()))).slice(0, 12);
    out.innerHTML = list.length ? `<div class="chats">${list.map(p => `<div class="chat-row"><div><div class="ttl"><span>${esc(cleanName(p.name))}</span></div>
      <div class="sub">${p.platforms.map(dot).join('')} ${p.accounts.length} account${p.accounts.length === 1 ? '' : 's'}</div></div>
      <span class="acct-acts"><button class="ghost-btn sm" data-into="${p.person_id}">Add here</button></span></div>`).join('')}</div>`
      : `<p class="placeholder">${this.people.length > 1 ? 'No person by that name.' : 'No people yet: create one instead.'}</p>`;
  }
}

// ─── pieces ─────────────────────────────────────────────────────────────
function row({ ref, name, isMe, color, sub, n }, cur) {
  return `<div class="ppl${cur === ref ? ' on' : ''}" data-ref="${ref}">
    <span class="avatar" style="background:${color}">${esc(initials(name))}</span>
    <span class="nm"><b>${esc(cleanName(name))}${isMe ? ' <span class="muted" style="font-weight:400">(you)</span>' : ''}</b><span>${sub}</span></span>
    <span class="cnt" title="messages">${num(n)}</span></div>`;
}

function acctRow(a, action) {
  return `<div class="chat-row">
    <div><div class="ttl">${dot(a.platform)}<span class="link" data-open="${a.node}">${esc(cleanName(a.name))}</span></div>
    <div class="sub">${esc(platformLabel(a.platform))} · ${num(a.messages)} msgs${a.person ? ` · in ${esc(cleanName(a.person))}` : ''}${a.why ? ' · ' + esc(a.why) : ''}</div></div>
    <span class="acct-acts">${action}</span></div>`;
}

function suggestionsHtml(sugs, label) {
  if (!sugs?.length) return '';
  return `<div class="acct-sugs"><h3>${label}</h3><div class="chats sug">${sugs.map((s, i) => acctRow(s,
    `<button class="primary-btn sm" data-sugg="${i}">Same person</button><button class="ghost-btn sm" data-reject="${i}">Not them</button>`)).join('')}</div></div>`;
}

function personPanel(p) {
  const isMe = p.node === ME;
  const rows = p.accounts.map(a => acctRow(a, isMe ? '<span class="muted">yours</span>'
    : `<button class="ghost-btn sm" data-remove="${a.account_id}" title="Not this person: take this account out">Remove</button>`)).join('');
  return `<div class="section p-panel">
    <h2>Accounts <span class="muted acct-count">${p.accounts.length}</span>
      ${isMe ? '' : '<span class="h-acts"><button class="ghost-btn sm" data-rename>Rename</button><button class="ghost-btn sm" data-delete>Remove person</button></span>'}</h2>
    ${isMe ? '' : `<div class="acct-rename inline-form" id="p-rename" hidden><input value="${esc(p.name)}" placeholder="Name shown everywhere"><button class="primary-btn sm" data-save>Save</button></div>`}
    <div class="chats">${rows || '<p class="placeholder">No accounts yet: add one below.</p>'}</div>
    ${isMe ? '' : suggestionsHtml(p.suggestions, 'Same person?')}
    ${isMe ? '' : `<div class="acct-find"><input id="acct-q" type="search" placeholder="Add an account to ${esc(cleanName(p.name))}: type a name"></div><div id="acct-found"></div>`}
    <div class="acct-status muted"></div>
  </div>`;
}

function accountPanel(a) {
  if (a.is_me) return `<div class="section p-panel"><p class="muted">One of your own accounts (it belongs to <a href="#people/${ME}">you</a>).</p></div>`;
  const owner = a.person
    ? `<div class="owner">Part of <a href="#people/${a.person_node}"><b>${esc(cleanName(a.person))}</b></a>
        <button class="ghost-btn sm" data-leave title="Not this person: take this account out">Remove from person</button></div>`
    : `<div class="owner">Not part of a person yet.
        <button class="primary-btn sm" data-new>Create a person</button><button class="ghost-btn sm" data-join>Add to a person…</button></div>
       <div class="acct-rename inline-form" id="p-create" hidden><input value="${esc(cleanName(a.name))}" placeholder="Their name"><button class="primary-btn sm" data-save data-create>Create</button></div>
       <div id="p-join" hidden><div class="acct-find"><input id="pp-q" type="search" placeholder="Which person? Type a name"></div><div id="pp-found"></div></div>`;
  return `<div class="section p-panel">${owner}${suggestionsHtml(a.suggestions, 'Same person on another platform?')}<div class="acct-status muted"></div></div>`;
}

const FACT_CATS = {
  relationship: 'Relationships', family: 'Family', education: 'Studies', work: 'Work & projects', location: 'Where',
  health: 'Health', money: 'Money', possession: 'Has', goal: 'Goals', plan: 'Plans', skill: 'Skills',
  preference: 'Likes & dislikes', belief: 'Views', identity: 'Who', social: 'Social', event: 'Events',
};
// "As of 2026-08-03, Sarthak is …" → "Sarthak is …": the dates are shown alongside
const factText = s => s.replace(/^(As of|On|By|Since) \d{4}-\d{2}(-\d{2})?( to \d{4}-\d{2}(-\d{2})?)?,\s*/i, '');
const span = f => f.since && f.until ? `${fmtDate(f.since)} – ${fmtDate(f.until)}` : f.since ? `since ${fmtDate(f.since)}` : f.until ? `until ${fmtDate(f.until)}` : '';
// how fresh a fact is: when the messages last backed it, and whether it may have changed since
const freshness = f => {
  if (f.status === 'past') return f.until && f.last_seen && f.until <= f.last_seen && f.durability === 'transient' ? `<span>not mentioned since ${esc(fmtDate(f.last_seen))}</span>` : '';
  if (!f.last_seen) return '';
  return f.stale ? `<span class="stale" title="Not mentioned for over six months: it may have changed">last mentioned ${esc(fmtDate(f.last_seen))} · may be out of date</span>`
    : `<span>last mentioned ${esc(fmtDate(f.last_seen))}</span>`;
};
function factRow(f, extra = false) {
  return `<div class="fact${extra ? ' extra' : ''}${f.status === 'disputed' ? ' disputed' : ''}${f.stale ? ' stale' : ''}" data-fact="${esc(f.id || '')}">
    <div>${esc(factText(f.fact))}</div>
    <div class="meta">${span(f) ? `<span>${esc(span(f))}</span>` : ''}${freshness(f)}
      <span title="Separate conversations that say this">${f.conversations} conversation${f.conversations === 1 ? '' : 's'}</span>
      ${f.status === 'disputed' ? '<span class="tag">sources disagree</span>' : ''}
      ${f.checked ? '' : '<span class="unchecked" title="Not yet checked against the messages">unchecked</span>'}
      ${f.source ? `<button class="link" data-session="${esc(f.source)}">source</button>` : ''}
      <span class="fact-acts">${f.yours === 'right' ? '<span class="ok">✓ you confirmed</span>' : '<button data-fact-v="right" title="This is right">✓</button>'}<button data-fact-v="wrong" title="This is wrong: hide it and never bring it back">✕ wrong</button></span></div>
  </div>`;
}

function themeHtml(t, also = []) {
  return `<div class="theme">
    <div class="t">${esc(t.theme)}<span>${t.sessions} chat${t.sessions === 1 ? '' : 's'}</span></div>
    ${t.description ? `<div class="d">${esc(t.description)}</div>` : ''}
    ${t.example ? `<div class="eg">“${esc(t.example)}”</div>` : ''}
    ${also.length ? `<div class="also">${also.map(x => `<span title="Also on ${esc(x.label)}: ${esc(x.theme)}">${dot(x.platform)}also on ${esc(x.label)}</span>`).join('')}</div>` : ''}
    ${t.examples?.length ? `<div class="ex">${t.examples.map(x => `<button data-session="${esc(x.session_id)}" title="${esc(x.summary || x.header)}">${esc(fmtDate(x.date))}</button>`).join('')}</div>` : ''}
  </div>`;
}

// One account of a person: its stats, then its own themes (themes belong to accounts, not to the person).
// A theme that shares a subject with a theme on another of their accounts says so.
function accountThemes(a, links, all) {
  const pct = a.messages_total ? Math.round(100 * a.messages_by_you / a.messages_total) : 0;
  const span = a.first_day && a.last_day ? `${fmtDate(a.first_day)} – ${fmtDate(a.last_day)}` : '';
  const byNode = Object.fromEntries(all.map(x => [x.node, x]));
  const also = i => links.flatMap(l => {
    const [me, other] = l.a === `${a.node}:${i}` ? [l.a, l.b] : l.b === `${a.node}:${i}` ? [l.b, l.a] : [];
    if (!me) return [];
    const [node, k] = other.split(':'), o = byNode[node];
    return o?.themes[k] ? [{ platform: o.platform, label: platformLabel(o.platform), theme: o.themes[k].theme }] : [];
  });
  return `<div class="acct-card" data-acct="${a.node}">
    <div class="acct-h">
      <span class="ttl">${dot(a.platform)}<b title="${esc(cleanName(a.name))}">${esc(cleanName(a.name))}</b><span class="muted">${esc(platformLabel(a.platform))}</span></span>
      <span class="acct-stats"><span><b>${num(a.messages_total)}</b> msgs</span><span><b>${pct}%</b> by you</span>${span ? `<span>${esc(span)}</span>` : ''}<a href="#people/${a.node}">Open</a></span>
    </div>
    ${a.themes?.length ? `<div class="themes">${a.themes.map((t, i) => themeHtml(t, also(i))).join('')}</div>` : '<p class="placeholder">Too few conversations here for recurring themes.</p>'}
  </div>`;
}

// Monthly bars, one column per calendar month (gaps included), "you" stacked under "them".
function activityChart(months) {
  const by = Object.fromEntries(months.map(m => [m.month, m]));
  const all = [];
  let [y, m] = months[0].month.split('-').map(Number);
  const [ly, lm] = months[months.length - 1].month.split('-').map(Number);
  while (y < ly || (y === ly && m <= lm)) {
    const k = `${y}-${String(m).padStart(2, '0')}`;
    all.push(by[k] || { month: k, total: 0, you: 0 });
    if (++m > 12) { m = 1; y++; }
  }
  const max = Math.max(...all.map(x => x.total), 1);
  const cols = all.map(x => {
    const them = x.total - x.you, h = v => (100 * v / max).toFixed(2);
    return `<div class="col" data-m="${x.month}" data-t="${x.total}" data-y="${x.you}">
      ${them ? `<div class="t" style="height:${h(them)}%"></div>` : ''}${x.you ? `<div class="y" style="height:${h(x.you)}%"></div>` : ''}</div>`;
  }).join('');
  const ticks = [all[0], all[Math.floor(all.length / 2)], all[all.length - 1]].map(x => fmtMonth(x.month));
  return `<div class="bars">${cols}</div><div class="bars-x"><span>${ticks[0]}</span><span>${all.length > 2 ? ticks[1] : ''}</span><span>${ticks[2]}</span></div>`;
}

function bindChartTips(el) {
  if (!el) return;
  const tip = document.createElement('div');
  tip.className = 'chart-tip'; tip.hidden = true; el.append(tip);
  el.addEventListener('mousemove', e => {
    const c = e.target.closest('.col');
    if (!c) { tip.hidden = true; return; }
    const t = +c.dataset.t, y = +c.dataset.y;
    tip.innerHTML = `<b>${fmtMonth(c.dataset.m)}</b>${num(t)} messages${t ? ` · ${num(y)} yours` : ''}`;
    tip.hidden = false;
    const r = el.getBoundingClientRect(), cr = c.getBoundingClientRect();
    const x = Math.min(r.width - tip.offsetWidth - 6, Math.max(6, cr.left - r.left + cr.width / 2 - tip.offsetWidth / 2));
    tip.style.left = x + 'px'; tip.style.top = '6px';
  });
  el.addEventListener('mouseleave', () => { tip.hidden = true; });
}
