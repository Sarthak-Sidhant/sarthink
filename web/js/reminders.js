// Reminders: commitments found in conversations, grouped by when they're due; plus "on this day".
import { $, api, esc, dot, fmtDate, relDay, daysBetween, todayIST, toast, markdown, sessionTitle } from './util.js';
import { openDrawer } from './reader.js';

const PAGE = 200;

export class Reminders {
  constructor() {
    this.owner = 'me'; this.status = 'open'; this.q = ''; this.items = []; this.offset = 0; this.loaded = false;
    seg('rem-owner', v => { this.owner = v; this.load(); });
    seg('rem-status', v => { this.status = v; this.load(); });
    let t;
    $('rem-q').addEventListener('input', e => { clearTimeout(t); t = setTimeout(() => { this.q = e.target.value.trim(); this.load(); }, 220); });
    $('rem-list').addEventListener('click', e => this.onClick(e));
    $('otd-date').value = todayIST();
    $('otd-date').addEventListener('change', () => this.loadOnThisDay());
    $('otd-body').addEventListener('click', e => this.onOtdClick(e));
  }

  show() { if (!this.loaded) { this.loaded = true; this.load(); this.loadOnThisDay(); } }

  async badge() {
    try {
      const d = await api('/api/commitments?owner=me&status=open&limit=1');
      $('rem-badge').textContent = d.counts.total || '';
    } catch { /* backend down: no badge */ }
  }

  async load(more = false) {
    const list = $('rem-list');
    if (!more) { this.offset = 0; this.items = []; list.innerHTML = '<div class="rem-card">' + '<div class="rem"><span></span><div class="skeleton" style="height:16px;width:60%"></div><span></span></div>'.repeat(5) + '</div>'; }
    const qs = new URLSearchParams({ owner: this.owner, status: this.status, q: this.q, limit: PAGE, offset: this.offset });
    let d;
    try { d = await api('/api/commitments?' + qs); } catch (e) {
      list.innerHTML = `<p class="placeholder">Couldn't load reminders (${esc(e.message)}).</p>`; return;
    }
    this.items.push(...d.items); this.offset += d.items.length; this.total = d.counts.total;
    $('rem-sub').textContent = `${d.counts.total.toLocaleString()} ${this.status === 'all' ? '' : labelOf(this.status) + ' '}· ${d.counts.mine.toLocaleString()} yours, ${d.counts.theirs.toLocaleString()} from others · ${d.counts.with_due.toLocaleString()} with a date`;
    this.render();
  }

  render() {
    const list = $('rem-list');
    if (!this.items.length) {
      list.innerHTML = `<p class="placeholder">${this.q ? 'Nothing matches that filter.' : 'Nothing here.'}</p>`; return;
    }
    const today = todayIST(), groups = { overdue: [], week: [], later: [], none: [] };
    for (const it of this.items) {
      if (!it.due) groups.none.push(it);
      else {
        const dd = daysBetween(today, it.due);
        (dd < 0 ? groups.overdue : dd <= 7 ? groups.week : groups.later).push(it);
      }
    }
    groups.overdue.sort((a, b) => b.due.localeCompare(a.due));
    groups.week.sort((a, b) => a.due.localeCompare(b.due));
    groups.later.sort((a, b) => a.due.localeCompare(b.due));
    groups.none.sort((a, b) => b.said_on.localeCompare(a.said_on));
    const open = this.status === 'open';
    const names = { overdue: open ? 'Past due' : 'Was due', week: 'This week', later: 'Later', none: 'No date' };
    list.innerHTML = Object.entries(groups).filter(([, v]) => v.length).map(([k, v]) => `
      <div class="rem-group ${open ? k : ''}">
        <h3>${names[k]} <span class="c">${v.length}</span></h3>
        <div class="rem-card">${v.map(it => this.row(it, today)).join('')}</div>
      </div>`).join('') +
      (this.offset < this.total ? `<button class="ghost-btn more" data-more>Show more (${(this.total - this.offset).toLocaleString()} left)</button>` : '');
  }

  row(it, today) {
    const dd = it.due ? daysBetween(today, it.due) : null;
    const cls = it.status !== 'open' || dd === null ? '' : dd < 0 ? 'over' : dd <= 3 ? 'soon' : '';
    const who = it.owner === 'me' ? 'You' : it.who;
    const others = it.people.filter(p => p !== it.who).slice(0, 3);
    const acts = it.status === 'open'
      ? `${it.due ? '<button data-act="calendar" title="Add to the calendar export">Calendar</button>' : ''}<button data-act="dismissed" title="Not a real reminder">Dismiss</button>`
      : '<button data-act="open" title="Move back to open">Reopen</button>';
    return `<div class="rem" data-cid="${esc(it.cid)}">
      <button class="check${it.status === 'done' ? ' done' : ''}" data-act="${it.status === 'done' ? 'open' : 'done'}" title="${it.status === 'done' ? 'Mark as open' : 'Mark done'}" aria-label="Mark done">
        <svg viewBox="0 0 24 24" width="11" height="11" fill="none" stroke="currentColor" stroke-width="3"><path d="M5 12l5 5 9-10"/></svg>
      </button>
      <div>
        <div class="title">${esc(it.title)}</div>
        ${it.what && it.what.toLowerCase() !== it.title.toLowerCase() ? `<div class="what">${esc(it.what)}</div>` : ''}
        <div class="meta">
          <span class="owner">${esc(who)}</span>
          ${others.length ? `<span>with ${esc(others.join(', '))}${it.people.length - 1 > others.length ? ' +' + (it.people.length - 1 - others.length) : ''}</span>` : ''}
          <span>${dot(it.platform)} said ${esc(fmtDate(it.said_on))}</span>
          ${it.status === 'calendar' ? '<span>· in calendar</span>' : ''}
        </div>
      </div>
      <div class="right">
        ${it.due ? `<span class="due ${cls}" title="${esc(fmtDate(it.due))}">${esc(relDay(it.due))}</span>` : ''}
        <div class="acts"><button data-act="read" title="Read the conversation">Read</button>${acts}</div>
      </div>
    </div>`;
  }

  async onClick(e) {
    if (e.target.closest('[data-more]')) { this.load(true); return; }
    const row = e.target.closest('.rem'), btn = e.target.closest('[data-act]');
    if (!row) return;
    const it = this.items.find(x => x.cid === row.dataset.cid);
    if (!btn) { if (!e.target.closest('button')) openDrawer(it.session_id); return; }
    const act = btn.dataset.act;
    if (act === 'read') { openDrawer(it.session_id); return; }
    const prev = it.status;
    try {
      await api(`/api/commitments/${encodeURIComponent(it.cid)}`, { body: { status: act } });
    } catch (err) { toast(`Couldn't update: ${err.message}`); return; }
    it.status = act;
    const msg = { done: 'Marked done', dismissed: 'Dismissed', calendar: 'Added to calendar export', open: 'Moved back to open' }[act];
    if (this.status !== 'all' && act !== this.status) {
      row.classList.add('leaving');
      setTimeout(() => { this.items = this.items.filter(x => x !== it); this.offset--; this.total--; this.render(); }, 250);
    } else this.render();
    undoToast(msg, async () => {
      await api(`/api/commitments/${encodeURIComponent(it.cid)}`, { body: { status: prev } });
      this.load(); this.badge();
    });
    this.badge();
  }

  async loadOnThisDay() {
    const body = $('otd-body'), date = $('otd-date').value || todayIST();
    body.innerHTML = '<div class="skeleton" style="height:40px;margin:8px 0"></div>'.repeat(3);
    let d, dayList;
    try { [d, dayList] = await Promise.all([api(`/api/on_this_day?date=${date}`), api(`/api/day?date=${date}`)]); } catch (e) {
      body.innerHTML = `<p class="placeholder">Couldn't load (${esc(e.message)}).</p>`; return;
    }
    const years = Object.entries(d.years);
    const same = dayList.sessions;
    const item = s => `<div class="otd-item" data-session="${esc(s.session_id)}">
        <div class="h">${dot(s.platform)}<span>${esc(sessionTitle(s.header))}</span></div>
        ${s.summary ? `<div class="s">${esc(s.summary)}</div>` : ''}</div>`;
    body.innerHTML = `
      <div class="otd-review">
        <button class="ghost-btn sm" data-review="${esc(date)}" ${same.length ? '' : 'disabled'}>Review ${esc(fmtDate(date))}</button>
        <div class="txt" id="otd-review"></div>
      </div>
      ${same.length ? `<div class="otd-year">${esc(fmtDate(date))} · ${same.length} conversation${same.length > 1 ? 's' : ''}</div>${same.slice(0, 6).map(item).join('')}` : ''}
      ${years.length ? years.map(([y, list]) => `<div class="otd-year">${y}</div>${list.slice(0, 4).map(item).join('')}`).join('')
        : `<p class="placeholder">No conversations on ${esc(fmtDate(date, { year: false }))} in earlier years.</p>`}`;
  }

  async onOtdClick(e) {
    const it = e.target.closest('.otd-item');
    if (it) { openDrawer(it.dataset.session); return; }
    const rv = e.target.closest('[data-review]');
    if (!rv) return;
    rv.disabled = true;
    const out = $('otd-review');
    out.innerHTML = '<span class="dots"><i></i><i></i><i></i></span> Writing the day…';
    try {
      const v = await api(`/api/day_review?date=${rv.dataset.review}`, { method: 'POST' });
      out.innerHTML = markdown(v.review);
      out.querySelectorAll('.cite[data-n]').forEach(c => {
        const s = v.sources.find(x => x.n === +c.dataset.n);
        if (s) c.onclick = () => openDrawer(s.session_id);
      });
    } catch (err) { out.textContent = `Couldn't write the review: ${err.message}`; }
    rv.disabled = false;
  }
}

function seg(id, onChange) {
  const el = $(id);
  el.addEventListener('click', e => {
    const b = e.target.closest('button[data-v]');
    if (!b || b.classList.contains('on')) return;
    el.querySelectorAll('button').forEach(x => x.classList.toggle('on', x === b));
    onChange(b.dataset.v);
  });
}
const labelOf = s => ({ open: 'open', done: 'done', dismissed: 'dismissed', calendar: 'in calendar' }[s] || s);

function undoToast(msg, undo) {
  const t = $('toast');
  t.innerHTML = `${esc(msg)} <button style="color:inherit;text-decoration:underline;margin-left:10px;font-weight:600">Undo</button>`;
  t.hidden = false;
  t.querySelector('button').onclick = async () => { t.hidden = true; try { await undo(); } catch (e) { toast(`Undo failed: ${e.message}`); } };
  clearTimeout(t._h); t._h = setTimeout(() => { t.hidden = true; }, 4000);
}
