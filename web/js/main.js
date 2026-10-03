// App shell: routing between Ask / Reminders / People, theme, backend status, stage wiring.
import { $, api, esc, num, dot, platformLabel, cleanName, PLATFORMS } from './util.js';
import { Graph } from './graph.js';
import { Stage, showGraphError } from './stage.js';
import { Chat } from './chat.js';
import { Reminders } from './reminders.js';
import { People } from './people.js';

addEventListener('error', e => showGraphError(`Something broke: ${e.message}`));
addEventListener('unhandledrejection', e => showGraphError(`Something broke: ${e.reason?.message || e.reason}`));

const graph = new Graph($('graph'), {
  onHover: h => {
    const tip = $('graph-tip');
    if (!h) { tip.hidden = true; return; }
    tip.innerHTML = h.html; tip.hidden = false;
    const pane = $('graph-pane').getBoundingClientRect();
    tip.style.left = Math.min(pane.width - tip.offsetWidth - 8, h.x + 14) + 'px';
    tip.style.top = Math.min(pane.height - tip.offsetHeight - 8, h.y + 14) + 'px';
  },
  // a person or an account on the full map: a card, and their constellation
  onClick: n => {
    if (n.kind !== 'person' && n.kind !== 'account') return;
    const p = { ref: n.id, name: n.label, platform: n.platform, size: n.size, isMe: graph.isMine(n.id), accounts: n.accounts };
    personCard(p);
    if (!p.isMe) stage.showPerson(p.ref, p.name);
  },
  onInfo: t => { $('graph-info').textContent = t; },
});
const stage = new Stage(graph, {
  // a session in the 3D view: open it in the sources pane (as its citation, when it is one)
  onSession: n => (n.n != null && chat.sel ? chat.openSource(n.n) : chat.openSession(n.session, 'Conversation')),
  onPerson: n => personCard({ ref: n.ref, name: n.label, platform: n.platform, size: n.size, isMe: graph.isMine(n.ref) }),
});
const chat = new Chat(stage);
const reminders = new Reminders();
const people = new People({
  onAsk: q => { location.hash = '#ask'; chat.prefill(q); },
  onShowInGraph: (ref, name) => { location.hash = '#ask'; requestAnimationFrame(() => stage.showPerson(ref, name)); },
  onPeopleChanged: list => graph.setPeople(list),     // a link shows up in the graph at once
});

// a person or account picked on the map or in the 3D view: a small card in the sources pane
function personCard({ ref, name, platform, size, isMe, accounts }) {
  name = cleanName(name);
  const owner = ref.startsWith('A_') ? graph.personOfNode(ref) : null, ownerInfo = owner && owner !== 'P_1' ? graph.infoById(owner) : null;
  const what = ref.startsWith('P_') ? `${dot('person')}Person · ${accounts?.length || 0} account${accounts?.length === 1 ? '' : 's'}`
    : `${dot(platform)}${esc(platformLabel(platform))}${ownerInfo ? ` · part of ${esc(cleanName(ownerInfo.label))}` : ''}`;
  $('sources-title').textContent = name; $('sources-sub').textContent = '';
  $('sources-body').innerHTML = `
    <div style="padding:10px 8px">
      <div class="d muted" style="display:flex;gap:6px;align-items:center;font-size:12.5px">${what}${size ? ` · ${num(size)} messages` : ''}</div>
      <div style="display:flex;gap:8px;margin-top:14px;flex-wrap:wrap">
        <a class="primary-btn" href="#people/${ref}">Open profile</a>
        ${ownerInfo ? `<a class="ghost-btn" href="#people/${owner}">Open ${esc(cleanName(ownerInfo.label))}</a>` : ''}
        ${isMe ? '' : `<button class="ghost-btn" data-ask-about>Ask about ${esc(name.length > 20 ? 'them' : name)}</button>`}
        ${isMe ? '' : '<button class="ghost-btn" data-explore>Explore in 3D</button>'}
        ${chat.sel ? '<button class="ghost-btn" data-back>Back to sources</button>' : ''}
      </div>
    </div>`;
  $('sources-body').querySelector('[data-ask-about]')?.addEventListener('click', () => chat.prefill(`What's the story with ${name}? What do we talk about?`));
  $('sources-body').querySelector('[data-explore]')?.addEventListener('click', () => stage.showPerson(ref, name));
  $('sources-body').querySelector('[data-back]')?.addEventListener('click', () => chat.select(chat.sel));
}

$('graph-fit').addEventListener('click', () => (stage.inFocus ? stage.focus.show(stage.focus.spec) : graph.fit()));

// flat map ⇄ 3D (remembered per browser)
function set3D(on, animate = true) {
  graph.set3D(on, animate);
  legend3D();
  $('graph-3d').textContent = on ? '2D' : '3D';
  $('graph-3d').setAttribute('aria-pressed', String(on));
  $('graph-3d').title = on ? 'Back to the flat map' : 'Show the map in 3D — drag to orbit, shift-drag to pan';
  try { localStorage.setItem('sarthink.map3d', on ? '1' : '0'); } catch (e) { /* storage blocked */ }
}
// the legend follows the 3D view's vivid palette
function legend3D() {
  for (const p of PLATFORMS) $('graph-pane').style.setProperty(p.slot, graph.is3D && graph.platColor3?.[p.key] || '');
}
$('graph-3d').addEventListener('click', () => set3D(!graph.is3D));
try { if (localStorage.getItem('sarthink.map3d') === '1') set3D(true, false); } catch (e) { /* storage blocked */ }

// ─── routing ──────────────────────────────────────────────────────────────
function route() {
  const [view, arg] = (location.hash.slice(1) || 'ask').split('/');
  const v = ['ask', 'reminders', 'people'].includes(view) ? view : 'ask';
  for (const s of ['ask', 'reminders', 'people']) $(`view-${s}`).hidden = s !== v;
  document.querySelectorAll('.tabs a').forEach(a => a.classList.toggle('on', a.dataset.view === v));
  if (v === 'ask') { graph.resize(); stage.focus.resize(); chat.focus(); }
  if (v === 'reminders') reminders.show();
  if (v === 'people' && arg) people.open(arg);
  document.title = { ask: 'Sarthink', reminders: 'Reminders · Sarthink', people: 'People · Sarthink' }[v];
}
addEventListener('hashchange', route);
route();

// ─── theme ────────────────────────────────────────────────────────────────
$('theme-btn').addEventListener('click', () => {
  const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem('sarthink.theme', next); } catch { /* ignore */ }
  stage.themeChanged();
  legend3D();
});

// ─── backend status ───────────────────────────────────────────────────────
async function health() {
  const st = $('status'), label = st.querySelector('span');
  try {
    const h = await api('/api/health');
    const enc = h.encoder;
    const semantic = h.dense && (!enc || enc.ok);
    st.className = 'status ' + (semantic ? 'ok' : 'warn');
    label.textContent = semantic ? `${num(h.chunks)} memories` : 'keyword search only';
    st.title = semantic ? 'Backend connected · semantic search ready' : 'Embedding encoder unreachable: search falls back to keywords';
  } catch (e) {
    st.className = 'status bad'; label.textContent = 'backend offline';
    st.title = `Can't reach the Sarthink API: ${e.message}`;
  }
}
health(); setInterval(health, 30000);
reminders.badge();

// ─── graph data ───────────────────────────────────────────────────────────
api('/api/people').catch(() => ({ people: [] }))
  .then(r => graph.load('/processed_data/graph/cosmograph_nodes.csv', '/processed_data/graph/cosmograph_edges.csv', r.people))
  .then(g => {
    const present = new Set(g.people.map(p => p.platform));
    $('graph-legend').innerHTML = `<span><i style="background:var(--grad)"></i>You</span><span><i style="background:var(--person)"></i>People</span>` +
      PLATFORMS.filter(p => present.has(p.key)).map(p => `<span><i style="background:var(${p.slot})"></i>${esc(p.label)}</span>`).join('');
    people.load();
    legend3D();
    stage.graphLoaded();
    chat.afterGraphLoad();
  })
  .catch(e => {
    $('graph-info').textContent = 'Graph unavailable';
    showGraphError(`The memory map couldn't load: ${e.message || e}`);
    people.load();                         // the people pages don't need the graph
  });
