/** The shell: routing, loading, and the run page.
 *
 * The route lives in the hash so a link to an attempt is a link anyone can
 * paste. `#/run/<run_id>` and `#/run/<run_id>/<trajectory_id>` are the whole
 * routing table.
 */

import { h, mount } from './lib/dom.mjs';
import { api, pooled } from './lib/api.mjs';
import { num, stamp } from './lib/format.mjs';
import { summarise, runHealth } from './lib/diagnose.mjs';
import { renderSidebar } from './components/sidebar.mjs';
import { renderHealth, renderTable } from './components/table.mjs';
import { Drawer } from './components/drawer.mjs';
import { rowsSignature, runsSignature } from './lib/changed.mjs';

const sidebarEl = document.getElementById('sidebar');
const mainEl = document.getElementById('main');
const drawer = new Drawer(document.getElementById('drawer'), document.getElementById('scrim'));

/** Rows per page.
 *
 * A page costs one `/paths` call per row to draw its strips, so this is not a
 * display preference -- it is how much work opening a run does. A thousand-row
 * run used to load one capped page and filter it in the browser, which made
 * every step but the newest look empty. */
const PAGE_SIZE = 100;

const state = {
  runs: [],
  runId: null,
  filter: '',
  rows: [],
  step: null,
  flag: null,
  // Paging. `cursors[n]` opens page n; index 0 is always null, and the rest
  // arrive from the server as we go, which is why paging back is a lookup
  // rather than a second kind of request.
  page: 0,
  cursors: [null],
  total: 0,
  // Whether the server is still reading the record directory. While it is,
  // `total` is null, a page boundary can move, and the pager says so rather
  // than printing a number that is about to change.
  indexing: false,
  hasMore: false,
  loading: false,
  source: { kind: 'down', label: 'connecting', api: '' },
  error: null,
};

drawer.onClose = () => {
  if (state.runId) setRoute(`#/run/${encodeURIComponent(state.runId)}`, true);
};

function setRoute(hash, replace = false) {
  if (location.hash === hash) return;
  if (replace) history.replaceState(null, '', hash);
  else location.hash = hash;
}

function parseRoute() {
  const parts = decodeURIComponent(location.hash.replace(/^#\/?/, '')).split('/');
  if (parts[0] !== 'run' || !parts[1]) return {};
  return { runId: parts[1], trajectoryId: parts[2] || null };
}

/** Which capture process is on the other end, said plainly.
 *
 * A live service and a record directory answer the same routes, and the whole
 * point of that is that this viewer does not care -- but the person reading it
 * very much does, because one of them is still being written to. */
async function identify() {
  const viewer = await api.viewer().catch(() => ({ api: 'unknown' }));
  state.source.api = viewer.api;
  try {
    const health = await api.health();
    if (health.source === 'record') {
      state.source = { kind: 'record', label: `record  ${health.record || ''}`, api: viewer.api };
    } else {
      state.source = { kind: 'live', label: `live  ${health.status}`, api: viewer.api };
    }
  } catch {
    state.source = { kind: 'down', label: 'cannot reach the capture API', api: viewer.api };
  }
}

/** What was last drawn, so a poll that changed nothing draws nothing. */
const drawn = { runs: null, rows: null };

function drawSidebar() {
  drawn.runs = runsSignature(state.runs);
  renderSidebar(sidebarEl, {
    runs: state.runs,
    active: state.runId,
    source: state.source,
    filter: state.filter,
    onFilter: (value) => {
      state.filter = value;
      drawSidebar();
    },
    onPick: (runId) => setRoute(`#/run/${encodeURIComponent(runId)}`),
    onTheme: () => {
      const next = document.documentElement.dataset.theme === 'light' ? 'dark' : 'light';
      document.documentElement.dataset.theme = next;
      try {
        localStorage.setItem('theme', next);
      } catch {
        /* a private window is not a reason to fail */
      }
    },
  });
}

/** How many trajectories sit at a step, from the run summary. */
function stepCount(run, step) {
  if (step === null) return run?.trajectory_count ?? 0;
  return run?.step_counts?.[String(step)] ?? 0;
}

/** The step picker.
 *
 * A slider rather than a chip per step, because a run has as many steps as it
 * has training steps and a row of two hundred chips is not a control. The
 * leftmost position is "all", so there is somewhere to go back to.
 *
 * Dragging only re-labels; the fetch waits for the release. A request per
 * slider tick would put a hundred listings on the wire to answer one question.
 */
function stepSlider(run, steps) {
  const index = state.step === null ? -1 : steps.indexOf(state.step);
  const describe = (at) => {
    const value = at < 0 ? null : steps[at];
    return value === null
      ? `all steps · ${num(run?.trajectory_count ?? 0)} trajectories`
      : `step ${value} · ${num(stepCount(run, value))} trajectories`;
  };
  const readout = h('span', { class: 'step-readout mono' }, describe(index));
  const slider = h('input', {
    type: 'range',
    id: 'step-slider',
    min: '-1',
    max: String(steps.length - 1),
    step: '1',
    value: String(index),
    'aria-label': 'step',
    oninput: (event) => {
      readout.textContent = describe(Number(event.target.value));
    },
    onchange: (event) => {
      const at = Number(event.target.value);
      selectStep(at < 0 ? null : steps[at]);
    },
  });
  return h(
    'div',
    { class: 'steps' },
    h('span', { class: 'label' }, 'step'),
    h(
      'button',
      {
        class: `chip${state.step === null ? ' on' : ''}`,
        onclick: () => selectStep(null),
      },
      h('span', { class: 'n' }, 'all')
    ),
    slider,
    readout
  );
}

/** Which page of which filter, and the two buttons that move it.
 *
 * While the server is still indexing there is no total to print: the count
 * would move under the reader, and a pager renders whatever it is given as
 * fact. It says what is actually true instead -- how many rows are in hand,
 * and that more are still being found. */
function pager() {
  const first = state.rows.length ? state.page * PAGE_SIZE + 1 : 0;
  const last = state.page * PAGE_SIZE + state.rows.length;
  const canPrev = state.page > 0 && !state.loading;
  const canNext = state.hasMore && !state.loading;
  const counted =
    state.total === null || state.total === undefined
      ? `${num(first)}-${num(last)} (indexing...)`
      : `${num(first)}-${num(last)} of ${num(state.total)}`;
  return h(
    'div',
    { class: 'pager' },
    h(
      'button',
      { class: 'chip', disabled: !canPrev, onclick: () => canPrev && loadPage(state.page - 1) },
      '‹ prev'
    ),
    h('span', { class: 'mono' }, state.loading ? 'loading...' : counted),
    h(
      'button',
      { class: 'chip', disabled: !canNext, onclick: () => canNext && loadPage(state.page + 1) },
      'next ›'
    )
  );
}

function drawRun() {
  drawn.rows = rowsSignature(state.rows, state);
  const run = state.runs.find((candidate) => candidate.id === state.runId);
  // Health is over the rows in hand, which is this page. The flag filter is
  // still client-side -- a flag is derived from `/paths` and is not a column
  // the server can filter on -- so it narrows the page, and the pager says
  // which page that is.
  const health = runHealth(state.rows);
  const visible = state.flag
    ? state.rows.filter((row) => row.summary?.flags.includes(state.flag))
    : state.rows;
  const steps = run?.steps || [];

  const healthEl = h('div');
  const tableEl = h('div');
  const scope =
    state.step === null ? 'across all steps' : `at step ${state.step}`;

  mount(
    mainEl,
    h(
      'div',
      { class: 'page' },
      h(
        'div',
        { class: 'page-head' },
        h('h1', {}, state.runId),
        h('span', { class: 'crumb' }, run?.project || '')
      ),
      h(
        'div',
        { class: 'page-sub' },
        run
          ? `${num(run.trajectory_count)} trajectories · ${num(run.task_count)} tasks · ` +
            `steps ${run.steps?.length ? `${run.steps[0]}..${run.steps[run.steps.length - 1]}` : '-'} · ` +
            `${stamp(run.created_at)}${run.upstream?.model ? ` · ${run.upstream.model}` : ''}`
          : ''
      ),
      h('h2', { class: 'section' }, `record health · page ${num(state.page + 1)} ${scope}`),
      healthEl,
      steps.length > 1 ? stepSlider(run, steps) : null,
      h(
        'h2',
        { class: 'section' },
        [
          state.step === null ? 'trajectories' : `trajectories at step ${state.step}`,
          state.flag ? ` flagged ${state.flag}` : '',
        ].join('')
      ),
      pager(),
      tableEl
    )
  );

  renderHealth(healthEl, {
    health,
    active: state.flag,
    onToggle: (flag) => {
      state.flag = state.flag === flag ? null : flag;
      drawRun();
    },
  });
  renderTable(tableEl, { rows: visible, selected: drawer.id, onOpen: openTrajectory });
}

/** Pick a step, or `null` for all of them.
 *
 * Re-queries from the first page rather than filtering what is loaded. That is
 * the whole fix: with a thousand trajectories and one capped page, every step
 * but the newest looked empty, because none of its rows had been fetched.
 */
function selectStep(step) {
  if (state.step === step) return;
  state.step = step;
  state.flag = null;
  state.cursors = [null];
  loadPage(0);
}

function openTrajectory(trajectoryId) {
  setRoute(`#/run/${encodeURIComponent(state.runId)}/${trajectoryId}`);
}

/** Fill in each row's mask strip and flags.
 *
 * One `/paths` call per trajectory, a few at a time. The table is drawn first
 * and fills in behind, because the ids and rewards are useful immediately and
 * waiting for every path in a run to decode before showing anything would make
 * the viewer feel broken on a large run. */
async function scan(rows) {
  const mine = state.runId;
  await pooled(rows, async (row) => {
    // Shape without words: the strip colours by kind and sizes by token count,
    // and never reads a character. On a large run the text and the logprobs
    // are almost all of the response, so this is most of the wait.
    const paths = await api.paths(row.id, { text: false });
    row.paths = paths;
    row.summary = summarise(paths, row);
    if (state.runId === mine) drawRun();
  });
}

/** Fetch one page of the current run, step and filter.
 *
 * Paging forward uses the cursor the last page returned; paging back uses the
 * one that opened that page, kept in `state.cursors`. A cursor is opaque --
 * the live service returns a keyset and a record returns an offset -- so this
 * stores them and never reads them.
 */
async function loadPage(index, { refresh = false } = {}) {
  const runId = state.runId;
  const cursor = state.cursors[index] ?? null;
  state.loading = true;
  drawRun();
  try {
    const listing = await api.trajectories({
      run_id: runId,
      step: state.step,
      limit: PAGE_SIZE,
      cursor,
      refresh: refresh ? 'true' : null,
    });
    if (state.runId !== runId) return;
    state.rows = listing.data;
    state.page = index;
    state.hasMore = Boolean(listing.has_more);
    // `total` is after filtering, and null while the server is still reading
    // the record directory. Either way the pager is told the truth rather
    // than a count it would print as settled.
    state.indexing = Boolean(listing.indexing);
    state.total = listing.total ?? null;
    if (listing.next_cursor) state.cursors[index + 1] = listing.next_cursor;
    state.loading = false;
    drawRun();
    scan(state.rows);
  } catch (error) {
    state.loading = false;
    mount(mainEl, h('div', { class: 'page' }, h('div', { class: 'err' }, String(error.message || error))));
  }
}

async function loadRun(runId) {
  state.runId = runId;
  state.rows = [];
  state.step = null;
  state.flag = null;
  state.page = 0;
  state.cursors = [null];
  state.total = 0;
  state.hasMore = false;
  drawSidebar();
  mount(mainEl, h('div', { class: 'page' }, h('div', { class: 'spin' }, 'loading run...')));
  await loadPage(0);
}

async function route() {
  let { runId, trajectoryId } = parseRoute();
  if (!runId) {
    if (!state.runs.length) {
      mount(
        mainEl,
        h(
          'div',
          { class: 'page' },
          h('div', { class: 'empty-state' }, 'No runs yet. Capture one and it appears here.')
        )
      );
      return;
    }
    // Land on the newest run. `replaceState` fires no `hashchange`, so this
    // falls through to load it rather than waiting for an event that will
    // never arrive -- which is a blank page on every visit to `/`.
    runId = state.runs[0].id;
    setRoute(`#/run/${encodeURIComponent(runId)}`, true);
  }
  if (runId !== state.runId) await loadRun(runId);
  if (trajectoryId && drawer.id !== trajectoryId) drawer.open(trajectoryId);
  else if (!trajectoryId && drawer.id) drawer.close();
  drawSidebar();
}

async function boot() {
  try {
    document.documentElement.dataset.theme = localStorage.getItem('theme') || '';
  } catch {
    /* ignore */
  }
  await identify();
  try {
    state.runs = await api.runs({ limit: 200 });
  } catch (error) {
    state.error = error;
  }
  drawSidebar();
  await route();
  window.addEventListener('hashchange', route);

  // A record directory is written to while it is being read -- by the capture
  // process that owns it, and possibly by several of them sharing a volume.
  // So this polls whatever the source is; the assumption that a record never
  // changes was true of a finished export and is not true of this.
  setInterval(refreshNow, 5000);
}

/** Pick up what has changed: new runs, new trajectories, and new statuses.
 *
 * `refresh` makes the server rescan before answering, which is what reaches a
 * trajectory that has just moved from `active/` to `committed/`. Only the
 * first page is refetched -- new trajectories sort to the front, so a later
 * page is a window into what is already behind them and refreshing it would
 * splice rows into the middle of a listing nobody moved. */
async function refreshNow() {
  try {
    state.runs = await api.runs({ limit: 200, refresh: 'true' });
    // Redraw only what changed. Every one of these draws replaces its
    // container's children, which resets the scroll inside it -- and a poll
    // that rebuilds the page every five seconds while somebody is reading it
    // takes their place away. Most polls change nothing at all.
    if (runsSignature(state.runs) !== drawn.runs) drawSidebar();
    if (state.runId && state.page === 0) {
      const listing = await api.trajectories({
        run_id: state.runId,
        step: state.step,
        limit: PAGE_SIZE,
        refresh: 'true',
      });
      // Rows are replaced rather than merged, because a row changes: a
      // trajectory finishes, a reward lands, a capture gap appears. What is
      // kept is the decoded path summary, which costs a call per row.
      const summaries = new Map(state.rows.map((row) => [row.id, row]));
      const rows = listing.data.map((row) => {
        const known = summaries.get(row.id);
        return known ? { ...row, paths: known.paths, summary: known.summary } : row;
      });
      const fresh = rows.filter((row) => !summaries.has(row.id));
      state.rows = rows;
      state.indexing = Boolean(listing.indexing);
      state.total = listing.total ?? null;
      state.hasMore = Boolean(listing.has_more);
      // The cursors named offsets into a listing that may have grown.
      if (fresh.length) state.cursors = [null];
      if (rowsSignature(rows, state) !== drawn.rows) drawRun();
      if (fresh.length) scan(fresh);
    }
    // And the trajectory somebody has open, which is the one they are
    // watching. A run in progress changes there first.
    if (drawer.id) await drawer.reload();
  } catch {
    /* the capture process going away mid-run is normal; keep what we have */
  }
}

boot();
