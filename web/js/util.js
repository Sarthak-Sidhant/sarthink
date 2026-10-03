// Small shared helpers: DOM, escaping, API, formatting, markdown, platforms.

export const $ = id => document.getElementById(id);
export const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
export const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

export async function api(path, opts = {}) {
  const r = await fetch(path, opts.body !== undefined && typeof opts.body !== 'string'
    ? { ...opts, method: opts.method || 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(opts.body) }
    : opts);
  if (!r.ok) {
    let msg = `${r.status}`;
    try { const j = await r.json(); msg = j.detail || JSON.stringify(j); } catch { /* not json */ }
    throw new Error(msg);
  }
  return r.json();
}

// Platforms take categorical slots in a fixed order; "me" wears ink, not a series hue.
export const PLATFORMS = [
  { key: 'twitter', label: 'Twitter / X', slot: '--series-1' },
  { key: 'reddit', label: 'Reddit', slot: '--series-2' },
  { key: 'discord', label: 'Discord', slot: '--series-3' },
  { key: 'facebook', label: 'Facebook', slot: '--series-4' },
  { key: 'instagram', label: 'Instagram', slot: '--series-5' },
  { key: 'whatsapp', label: 'WhatsApp', slot: '--series-6' },
];
export function platformOf(s) {
  s = String(s || '').toLowerCase();
  if (s === 'me') return 'me';
  if (s === 'person') return 'person';      // a person: a group of accounts, not one platform
  return (PLATFORMS.find(p => s.startsWith(p.key)) || {}).key || 'other';
}
export function platformColor(p) {
  const k = platformOf(p);
  if (k === 'me') return css('--me');
  if (k === 'person') return css('--person');
  const slot = PLATFORMS.find(x => x.key === k);
  return slot ? css(slot.slot) : css('--muted');
}
export const platformLabel = p => platformOf(p) === 'person' ? 'Person'
  : (PLATFORMS.find(x => x.key === platformOf(p)) || { label: p || 'Other' }).label;
export const dot = p => `<i class="pdot" style="background:${platformColor(p)}" title="${esc(platformLabel(p))}"></i>`;

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
export function fmtDate(iso, { year = true } = {}) {
  if (!iso) return '';
  const [y, m, d] = iso.slice(0, 10).split('-').map(Number);
  return `${d} ${MONTHS[m - 1]}${year ? ' ' + y : ''}`;
}
export const fmtMonth = ym => { const [y, m] = ym.split('-').map(Number); return `${MONTHS[m - 1]} ${y}`; };
export const num = n => Number(n || 0).toLocaleString();
export function todayIST() {
  return new Date(Date.now() + 5.5 * 3600e3).toISOString().slice(0, 10);
}
export function daysBetween(a, b) {
  return Math.round((Date.parse(b) - Date.parse(a)) / 86400e3);
}
export function relDay(iso) {
  const d = daysBetween(todayIST(), iso);
  if (d === 0) return 'Today';
  if (d === 1) return 'Tomorrow';
  if (d === -1) return 'Yesterday';
  if (d < 0 && d > -45) return `${-d} days ago`;
  if (d > 0 && d < 45) return `In ${d} days`;
  return fmtDate(iso);
}
export const initials = name => String(name || '?').replace(/^u\//, '').replace(/[^\p{L}\p{N} ]/gu, ' ').trim()
  .split(/\s+/).slice(0, 2).map(w => w[0] || '').join('').toUpperCase() || '?';
export const cleanName = n => String(n || '').replace(/\s*\(me\)$/, '');

let toastTimer;
export function toast(msg) {
  const t = $('toast');
  t.textContent = msg; t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.hidden = true; }, 2600);
}

// ─── Markdown (input escaped first): headings, lists, tables, paragraphs, citations ───
function inline(s) {
  return s
    .replace(/\*\*(.+?)\*\*/g, '<b>$1</b>')
    .replace(/(^|[\s(])\*(?!\s)(.+?)\*(?=[\s).,;:!?]|$)/g, '$1<i>$2</i>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\[(\d{1,3})\]/g, '<span class="cite" data-n="$1">$1</span>')
    .replace(/\[sql\]/gi, '<span class="cite db" title="Computed from your message database">db</span>')
    .replace(/\[note\]/gi, '<span class="cite note" title="Something you asked Sarthink to remember">note</span>');
}
export function markdown(text) {
  const lines = esc(text || '').split('\n');
  let html = '', list = null, table = [];
  const flushList = () => { if (list) { html += `</${list}>`; list = null; } };
  const flushTable = () => {
    if (!table.length) return;
    html += '<table>' + table.map((r, i) => {
      const tag = i === 0 ? 'th' : 'td';
      return '<tr>' + r.replace(/^\||\|$/g, '').split('|').map(c => `<${tag}>${inline(c.trim())}</${tag}>`).join('') + '</tr>';
    }).join('') + '</table>';
    table = [];
  };
  for (const raw of lines) {
    const line = raw.trimEnd();
    if (/^\s*\|/.test(line)) {
      flushList();
      if (!/^\s*\|?[\s:|-]+\|?\s*$/.test(line) || !line.includes('-')) table.push(line.trim());
      continue;
    }
    flushTable();
    let m;
    if ((m = line.match(/^#{1,4}\s+(.*)/))) { flushList(); html += `<h3>${inline(m[1])}</h3>`; }
    else if ((m = line.match(/^\s*[-*•]\s+(.*)/))) { if (list !== 'ul') { flushList(); html += '<ul>'; list = 'ul'; } html += `<li>${inline(m[1])}</li>`; }
    else if ((m = line.match(/^\s*\d+[.)]\s+(.*)/))) { if (list !== 'ol') { flushList(); html += '<ol>'; list = 'ol'; } html += `<li>${inline(m[1])}</li>`; }
    else if (!line.trim() || /^-{3,}$/.test(line.trim())) flushList();
    else { flushList(); html += `<p>${inline(line)}</p>`; }
  }
  flushList(); flushTable();
  return html;
}

// Session headers look like "Twitter thread · Me (me), @a, @b +3 · 1 Oct 2025 · Group name".
// Title: the people (if few), else the chat's own name, else the kind of conversation.
export function sessionTitle(header) {
  const parts = String(header || '').split(' · ').map(x => x.trim());
  const [kind, people = '', , name = ''] = parts;
  const others = people.split(', ').filter(p => p && !/\(me\)$/.test(p));
  const many = /\+\d+$/.test(people) || others.length > 3;
  const ownName = name && !/^(DM |Direct Messages)/.test(name) ? name : '';
  if (others.length && !many) return others.join(', ') + (ownName && !others.some(o => ownName.includes(o)) ? ` — ${ownName}` : '');
  if (ownName) return ownName;
  return others.length ? `${kind} with ${others.slice(0, 2).join(', ')} and others` : kind || 'Conversation';
}

// "[2025-08-31 12:07] Name: text" lines → [{ts, who, text}] (lines that don't match stay as plain text)
export function parseTranscript(text) {
  const out = [];
  for (const line of String(text || '').split('\n')) {
    const m = line.match(/^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2})\]\s+([^:]{1,80}?):\s?(.*)$/);
    if (m) out.push({ ts: m[1], who: m[2], text: m[3].replace(/ \/ /g, '\n') });
    else if (line.trim()) out.push({ ts: null, who: null, text: line });
  }
  return out;
}
