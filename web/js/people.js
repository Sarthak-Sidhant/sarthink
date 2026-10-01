// People: searchable directory (from the graph) + a dossier for whoever is selected.
import { $, api, esc, dot, fmtDate, fmtMonth, num, initials, cleanName, platformColor, platformLabel, PLATFORMS } from './util.js';
import { openDrawer } from './reader.js';

const SHOW = 150;
const dossiers = new Map();   // this tab's copy; the server keeps them in profile_cache.db

export class People {
  constructor({ onAsk, onShowInGraph }) {
    this.all = []; this.q = ''; this.plat = null; this.current = null;
    this.onAsk = onAsk; this.onShowInGraph = onShowInGraph;
    let t;
    $('ppl-q').addEventListener('input', e => { clearTimeout(t); t = setTimeout(() => { this.q = e.target.value.trim().toLowerCase(); this.renderList(); }, 120); });
    $('ppl-platforms').addEventListener('click', e => {
      const b = e.target.closest('button[data-p]');
      if (!b) return;
      this.plat = this.plat === b.dataset.p ? null : b.dataset.p;
      this.renderChips(); this.renderList();
    });
    $('ppl-items').addEventListener('click', e => { const r = e.target.closest('.ppl'); if (r) location.hash = `#people/${r.dataset.pid}`; });
    $('person').addEventListener('click', e => this.onPersonClick(e));
  }

  setPeople(list) {
    this.all = list;
    this.renderChips(); this.renderList();
  }

  renderChips() {
    const counts = {};
    for (const p of this.all) counts[p.platform] = (counts[p.platform] || 0) + 1;
    $('ppl-platforms').innerHTML = PLATFORMS.filter(p => counts[p.key]).map(p =>
      `<button data-p="${p.key}" class="${this.plat === p.key ? 'on' : ''}">${dot(p.key)}${esc(p.label)} <span class="muted">${num(counts[p.key])}</span></button>`).join('');
  }

  renderList() {
    const q = this.q;
    const match = this.all.filter(p => (!this.plat || p.platform === this.plat || p.group === 'me') && (!q || p.name.toLowerCase().includes(q)));
    const items = match.slice(0, SHOW);
    $('ppl-items').innerHTML = items.map(p => `
      <div class="ppl${this.current === p.pid ? ' on' : ''}" data-pid="${p.pid}">
        <span class="avatar" style="background:${p.group === 'me' ? 'var(--me);color:var(--bg)' : platformColor(p.platform)}">${esc(initials(p.name))}</span>
        <span class="nm"><b>${esc(cleanName(p.name))}${p.group === 'me' ? ' <span class="muted" style="font-weight:400">(you)</span>' : ''}</b><span>${esc(platformLabel(p.platform))}</span></span>
        <span class="cnt" title="messages">${num(p.messages)}</span>
      </div>`).join('') +
      (match.length > SHOW ? `<div class="ppl-note">Showing the ${SHOW} most talked-to of ${num(match.length)}. Search to find anyone.</div>` : '') +
      (!match.length ? '<p class="placeholder">No one matches.</p>' : '');
  }

  async open(pid) {
    pid = +pid;
    this.current = pid;
    $('ppl-items').querySelectorAll('.ppl').forEach(r => r.classList.toggle('on', +r.dataset.pid === pid));
    const meta = this.all.find(p => p.pid === pid);
    const el = $('person');
    el.scrollTop = 0;
    if (dossiers.has(pid)) { this.dossier = dossiers.get(pid); this.renderDossier(this.dossier, meta); return; }
    el.innerHTML = `<div class="p-head">
        <span class="avatar" style="background:${meta ? platformColor(meta.platform) : 'var(--muted)'}">${esc(initials(meta?.name))}</span>
        <div><h1>${esc(cleanName(meta?.name || 'Loading…'))}</h1><div class="sub">Reading your history together…</div></div>
      </div>
      <div class="tiles">${'<div class="tile"><div class="skeleton" style="height:26px;width:60%"></div><div class="skeleton" style="height:12px;width:80%;margin-top:8px"></div></div>'.repeat(4)}</div>
      <div class="skeleton" style="height:180px;border-radius:12px"></div>`;
    let d = dossiers.get(pid);
    if (!d) {
      try { d = await api(`/api/person/${pid}/dossier`); } catch (e) {
        if (this.current === pid) el.querySelector('.sub').textContent = `Couldn't load (${e.message}).`;
        return;
      }
      dossiers.set(pid, d);
    }
    if (this.current !== pid) return;
    this.dossier = d;
    this.renderDossier(d, meta);
  }

  renderDossier(d, meta) {
    const el = $('person'), isMe = meta?.group === 'me';
    const pct = d.messages_total ? Math.round(100 * d.messages_by_you / d.messages_total) : 0;
    const name = cleanName(d.name);
    const platforms = String(d.platforms || meta?.platform || '').split(/,\s*/).filter(Boolean);
    const span = d.first_day && d.last_day ? `${fmtDate(d.first_day)} – ${fmtDate(d.last_day)}` : '';
    el.innerHTML = `
      <div class="p-head">
        <span class="avatar" style="background:${isMe ? 'var(--me);color:var(--bg)' : platformColor(meta?.platform || platforms[0])}">${esc(initials(name))}</span>
        <div><h1>${esc(name)}</h1><div class="sub">${platforms.map(p => `${dot(p)} ${esc(platformLabel(p))}`).join(' &nbsp; ')}${span ? ' &nbsp;·&nbsp; ' + esc(span) : ''}</div></div>
        <div class="acts">
          <button class="ghost-btn" data-graph>Show in graph</button>
          ${isMe ? '' : `<button class="primary-btn" data-ask>Ask about ${esc(name.length > 18 ? 'them' : name)}</button>`}
        </div>
      </div>
      <div class="tiles">
        <div class="tile"><div class="v">${num(d.messages_total)}</div><div class="l">messages ${isMe ? 'in total' : 'together'}</div></div>
        <div class="tile"><div class="v">${isMe ? num(d.top_chats.length) : pct + '%'}</div><div class="l">${isMe ? 'top chats shown' : 'written by you'}</div></div>
        <div class="tile"><div class="v">${num(d.months.length)}</div><div class="l">active months</div></div>
        <div class="tile"><div class="v">${d.last_day ? esc(fmtDate(d.last_day)) : '—'}</div><div class="l">last talked</div></div>
      </div>
      ${d.months.length ? `<div class="section">
        <h2>Activity <span class="legend"><span><i style="background:var(--series-1)"></i>You</span><span><i style="background:var(--ramp-light)"></i>${isMe ? 'Others' : esc(name)}</span></span></h2>
        <div class="chart" id="act-chart">${activityChart(d.months)}</div>
      </div>` : ''}
      ${d.themes.length ? `<div class="section"><h2>What you talk about</h2><div class="themes">${d.themes.map(t => `
        <div class="theme">
          <div class="t">${esc(t.theme)}<span>${t.sessions} chat${t.sessions === 1 ? '' : 's'}</span></div>
          ${t.description ? `<div class="d">${esc(t.description)}</div>` : ''}
          ${t.examples.length ? `<div class="ex">${t.examples.map(x => `<button data-session="${esc(x.session_id)}" title="${esc(x.summary || x.header)}">${esc(fmtDate(x.date))}</button>`).join('')}</div>` : ''}
        </div>`).join('')}</div></div>` : ''}
      ${d.top_chats.length ? `<div class="section"><h2>Where you talk</h2><div class="chats">${d.top_chats.map(c => `
        <div class="chat-row">
          <div><div class="ttl">${dot(c.platform)}<span>${esc(c.title || platformLabel(c.platform))}</span></div>
          <div class="sub">${esc(platformLabel(c.platform))} · ${esc(fmtDate(c.from))} – ${esc(fmtDate(c.to))}</div></div>
          <span class="n">${num(c.messages)} msgs</span>
        </div>`).join('')}</div></div>` : ''}`;
    bindChartTips($('act-chart'));
  }

  onPersonClick(e) {
    const s = e.target.closest('[data-session]');
    if (s) { openDrawer(s.dataset.session); return; }
    const d = this.dossier;
    if (!d) return;
    if (e.target.closest('[data-ask]')) this.onAsk(`What do I usually talk about with ${cleanName(d.name)}, and what's the latest between us?`);
    if (e.target.closest('[data-graph]')) this.onShowInGraph([d.node, ...d.top_chats.map(c => c.node)], d.node);
  }
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
