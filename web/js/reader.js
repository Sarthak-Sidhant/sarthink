// Conversation reader: one session (summary, topics, transcript, previous/next), rendered into any container.
import { api, esc, dot, fmtDate, parseTranscript, platformLabel, sessionTitle } from './util.js';

const cache = new Map();
export async function getSession(id) {
  if (!cache.has(id)) cache.set(id, api(`/api/session/${encodeURIComponent(id)}`).catch(e => { cache.delete(id); throw e; }));
  return cache.get(id);
}

export async function renderReader(el, sessionId, { onNodes, onBack, backLabel } = {}) {
  el.innerHTML = `${onBack ? backBtn(backLabel) : ''}<div class="reader-head"><div class="skeleton" style="height:18px;width:70%"></div>
    <div class="skeleton" style="height:12px;width:40%;margin-top:8px"></div></div>
    ${'<div class="skeleton" style="height:14px;margin:12px 4px"></div>'.repeat(6)}`;
  bindBack(el, onBack);
  let s;
  try { s = await getSession(sessionId); } catch (e) {
    el.innerHTML = `${onBack ? backBtn(backLabel) : ''}<p class="placeholder">Couldn't load this conversation (${esc(e.message)}).</p>`;
    bindBack(el, onBack); return;
  }
  onNodes?.(s.nodes, s);
  const sum = s.summary || {};
  const meName = (s.people.find(p => /\(me\)$/.test(p.name)) || {}).name;
  const lines = parseTranscript(s.text);
  const msgs = []; let last = null;
  for (const l of lines) {
    if (!l.who) { msgs.push(`<div class="m plain"><div class="txt">${esc(l.text)}</div></div>`); last = null; continue; }
    const same = last && last.who === l.who && last.ts.slice(0, 13) === l.ts.slice(0, 13);
    msgs.push(`<div class="m${l.who === meName ? ' me' : ''}">${same ? '' : `<div class="who">${esc(l.who.replace(/ \(me\)$/, ''))}${l.who === meName ? ' <span class="muted" style="font-weight:400">(you)</span>' : ''}<time>${esc(l.ts.slice(11))}${l.ts.slice(0, 10) !== (last?.ts || '').slice(0, 10) ? ' · ' + fmtDate(l.ts) : ''}</time></div>`}<div class="txt">${esc(l.text)}</div></div>`);
    last = l;
  }
  el.innerHTML = `${onBack ? backBtn(backLabel) : ''}
    <div class="reader-head">
      <h3>${esc(sessionTitle(s.header))}</h3>
      <div class="meta">${dot(s.platform)}<span>${esc(platformLabel(s.platform))}</span><span>·</span><span>${esc(fmtDate(s.start))}</span><span>·</span><span>${s.message_count} messages</span></div>
      ${sum.summary ? `<div class="reader-sum"><span class="lbl">Summary</span>${esc(sum.summary)}</div>` : ''}
      ${sum.topics?.length ? `<div class="topics">${sum.topics.slice(0, 8).map(t => `<span>${esc(t)}</span>`).join('')}</div>` : ''}
    </div>
    <div class="msgs">${msgs.join('')}</div>
    <div class="reader-nav">
      <button class="ghost-btn sm" data-go="${esc(s.previous_session || '')}" ${s.previous_session ? '' : 'disabled style="visibility:hidden"'}>← Earlier</button>
      <button class="ghost-btn sm" data-go="${esc(s.next_session || '')}" ${s.next_session ? '' : 'disabled style="visibility:hidden"'}>Later →</button>
    </div>`;
  bindBack(el, onBack);
  el.querySelectorAll('[data-go]').forEach(b => b.addEventListener('click', () => {
    if (b.dataset.go) { renderReader(el, b.dataset.go, { onNodes, onBack, backLabel }); el.scrollTop = 0; el.closest('.pane-body, .drawer-body')?.scrollTo(0, 0); }
  }));
}

const backBtn = label => `<button class="ghost-btn sm back-btn" style="margin-bottom:12px">← ${esc(label || 'Back')}</button>`;
function bindBack(el, onBack) { el.querySelector('.back-btn')?.addEventListener('click', onBack); }

// Slide-over drawer (Reminders / People)
export function openDrawer(sessionId, opts = {}) {
  const d = document.getElementById('drawer'), body = document.getElementById('drawer-body');
  d.hidden = false; body.scrollTop = 0;
  renderReader(body, sessionId, { ...opts, onBack: closeDrawer, backLabel: 'Close' });
}
export function closeDrawer() { document.getElementById('drawer').hidden = true; }
document.getElementById('drawer').addEventListener('click', e => { if (e.target.closest('[data-close]')) closeDrawer(); });
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeDrawer(); });
