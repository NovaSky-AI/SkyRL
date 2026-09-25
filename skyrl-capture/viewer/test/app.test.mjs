/** Boot the whole viewer against a fake `/v1` and drive it.
 *
 * The render tests check each view in isolation. This one checks the thing
 * isolation cannot: that the shell asks for the right URLs, routes on the
 * hash, fills the table in behind the first paint, and opens the drawer -- the
 * path a person actually takes.
 *
 * `fetch` is answered from the same fixtures, so this needs no server and no
 * network.
 */

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { install } from './dom-shim.mjs';
import { rowsSignature, runsSignature, trajectorySignature } from '../public/lib/changed.mjs';

const shim = install();
const HERE = fileURLToPath(new URL('.', import.meta.url));
const fixture = (name) => JSON.parse(readFileSync(join(HERE, 'fixtures', `${name}.json`), 'utf8'));

const runs = fixture('runs');
const trajectory = fixture('trajectory');
const trajectoryPlain = fixture('trajectory-plain');
const pathsForked = fixture('paths-forked');
const pathsPlain = fixture('paths-plain');
const graph = fixture('graph');
const exchanges = fixture('exchanges');

// The listing envelope the server actually sends: `indexing` says whether the
// record directory has been read all the way through, and `total` is null
// until it has.
const listing = {
  data: [trajectory, trajectoryPlain],
  next_cursor: null,
  has_more: false,
  total: 2,
  indexing: false,
  indexed_trajectories: 2,
};
const pathsFor = { [trajectory.id]: pathsForked, [trajectoryPlain.id]: pathsPlain };

const asked = [];

globalThis.fetch = async (url) => {
  asked.push(url);
  const { pathname } = new URL(url, 'http://viewer.test');
  const json = (value) => ({ ok: true, status: 200, json: async () => value });

  if (pathname === '/__viewer') return json({ api: 'http://127.0.0.1:8601' });
  if (pathname === '/api/healthz') return json({ status: 'ok', source: 'record', record: '/tmp/rec' });
  if (pathname === '/api/v1/runs') return json({ data: runs });
  if (pathname === '/api/v1/trajectories') return json(listing);
  const match = pathname.match(/^\/api\/v1\/trajectories\/([^/]+)(\/(\w+))?$/);
  if (match) {
    const [, id, , leaf] = match;
    if (!leaf) return json(pathsFor[id] ? (id === trajectory.id ? trajectory : trajectoryPlain) : {});
    if (leaf === 'paths') return json(pathsFor[id]);
    if (leaf === 'graph') return json(graph);
    if (leaf === 'exchanges') return json(exchanges);
  }
  return { ok: false, status: 404, json: async () => ({ detail: `unhandled ${pathname}` }) };
};

globalThis.location = { hash: '' };
globalThis.history = { replaceState(_state, _title, hash) { globalThis.location.hash = hash; } };
globalThis.setInterval = () => 0;

const settle = () => new Promise((resolve) => setTimeout(resolve, 60));

await import('../public/app.mjs');
await settle();

test('it identifies which capture process it is reading', () => {
  assert.ok(asked.includes('/__viewer'));
  assert.ok(asked.includes('/api/healthz'));
  const sidebar = shim.byId('sidebar');
  assert.match(sidebar.textContent, /record/, 'a record and a live service must not look alike');
});

test('with no route it opens the newest run rather than an empty page', () => {
  assert.equal(globalThis.location.hash, `#/run/${encodeURIComponent(runs[0].id)}`);
  const main = shim.byId('main');
  assert.match(main.textContent, new RegExp(runs[0].id));
});

test('the run page draws the table, and a step slider over the run\'s steps', () => {
  const main = shim.byId('main');
  assert.ok(main.findAll((node) => node.tagName === 'TR').length > 1, 'the table');
  // A run is a flat list narrowed by step, not the columns of a matrix. A
  // slider rather than a chip each, because a real run has as many steps as it
  // has training steps and a row of two hundred chips is not a control.
  const steps = runs[0].steps || [];
  const sliders = main.findAll((node) => node.tagName === 'INPUT' && node.getAttribute('type') === 'range');
  if (steps.length > 1) {
    assert.equal(sliders.length, 1, 'one slider');
    assert.equal(sliders[0].getAttribute('min'), '-1', 'and a position for "all"');
    assert.equal(sliders[0].getAttribute('max'), String(steps.length - 1));
  } else {
    assert.equal(sliders.length, 0, 'one step means nothing to choose');
  }
});

test('the pager says which rows of how many are on screen', () => {
  const main = shim.byId('main');
  const pager = main.findAll((node) => node.hasClass('pager'));
  assert.equal(pager.length, 1);
  assert.match(pager[0].textContent, /1-2 of 2/, 'counted after filtering, not from the page');
});

test('it fills each row in behind the first paint, one /paths call per row', () => {
  for (const row of listing.data) {
    assert.ok(
      asked.some((url) => url.startsWith(`/api/v1/trajectories/${row.id}/paths`)),
      `scanned ${row.id}`
    );
  }
  const main = shim.byId('main');
  assert.ok(main.findAll((node) => node.hasClass('strip')).length > 0, 'strips arrived');
});

test('the strip is fetched without the text it does not render', () => {
  // The strip colours by kind and sizes by token count and never reads a
  // character, so asking for the decoded text and the logprobs would be most
  // of the response spent on nothing.
  const scans = asked.filter((url) => url.includes('/paths'));
  assert.ok(scans.length > 0);
  assert.ok(
    scans.every((url) => url.includes('text=false')),
    `every row scan drops the text: ${scans.join(', ')}`
  );
});

test('the step filter is a query, not a filter over what happened to load', () => {
  // The bug this guards: one capped page filtered in the browser made every
  // step but the newest look empty on a run larger than the page.
  const listings = asked.filter((url) => url.startsWith('/api/v1/trajectories?'));
  assert.ok(listings.length > 0, 'the run page lists through the API');
  assert.ok(
    listings.every((url) => url.includes('limit=')),
    'and asks for a bounded page'
  );
});

test('the health band reports what the scan found', () => {
  const main = shim.byId('main');
  assert.match(main.textContent, /replayed|nothing flagged/);
});

test('it never asks for a URL outside the documented API', () => {
  for (const url of asked) {
    assert.ok(
      url.startsWith('/api/v1/') || url === '/api/healthz' || url === '/__viewer',
      `unexpected ${url}`
    );
  }
});

test('a deep link to a trajectory opens the drawer on it', async () => {
  globalThis.location.hash = `#/run/${encodeURIComponent(runs[0].id)}/${trajectory.id}`;
  // The app listens on `hashchange`; the shim's window swallows listeners, so
  // this drives the drawer the way that event would.
  const drawer = shim.byId('drawer');
  const { Drawer } = await import('../public/components/drawer.mjs');
  const instance = new Drawer(drawer, shim.byId('scrim'));
  await instance.open(trajectory.id);
  await settle();
  assert.match(drawer.textContent, new RegExp(trajectory.id));
  assert.match(drawer.textContent, /replayed/, 'the case the design exists for is visible');
  assert.ok(drawer.hasClass('open'));
});


// -- polling must not take the reader's place away -----------------------------
test('an unchanged poll is recognised, so nothing is redrawn', () => {
  // Every draw replaces its container's children, and that resets the scroll
  // inside it. A trajectory being read is usually a finished one, which will
  // never change again -- so the poll has to be able to say "nothing here".
  const runs = [{ id: 'grpo-step-1', project: 'demo', trajectory_count: 8 }];
  assert.equal(runsSignature(runs), runsSignature(structuredClone(runs)));

  const view = { indexing: false, total: 2 };
  const rows = [
    { id: 'tr_a', status: 'finished', revision: 0, summary: {} },
    { id: 'tr_b', status: 'finished', revision: 0, summary: {} },
  ];
  assert.equal(rowsSignature(rows, view), rowsSignature(structuredClone(rows), view));

  const bundle = {
    trajectory: { status: 'finished', revision: 0, capture: { exchange_count: 4, node_count: 10 } },
    paths: { paths: [{}, {}] },
    exchanges: { data: [{}, {}, {}, {}] },
  };
  assert.equal(trajectorySignature(bundle), trajectorySignature(structuredClone(bundle)));
});

test('everything that changes what is on screen is noticed', () => {
  const runs = [{ id: 'r', project: 'demo', trajectory_count: 8 }];
  const base = runsSignature(runs);
  assert.notEqual(base, runsSignature([{ ...runs[0], trajectory_count: 9 }]), 'a new trajectory');
  assert.notEqual(base, runsSignature([]), 'the run going away');

  const view = { indexing: false, total: 1 };
  const rows = [{ id: 'tr_a', status: 'running', revision: 0, summary: null }];
  const row = rowsSignature(rows, view);
  assert.notEqual(row, rowsSignature([{ ...rows[0], status: 'finished' }], view), 'it finished');
  assert.notEqual(row, rowsSignature([{ ...rows[0], revision: 1 }], view), 'a reward landed');
  assert.notEqual(row, rowsSignature([{ ...rows[0], summary: {} }], view), 'its paths were scanned');
  assert.notEqual(row, rowsSignature(rows, { indexing: true, total: null }), 'the pager changed');

  const capture = { exchange_count: 4, node_count: 10, calls_missing: 0 };
  const bundle = { trajectory: { status: 'running', revision: 0, capture } };
  const one = trajectorySignature(bundle);
  assert.notEqual(
    one,
    trajectorySignature({ trajectory: { ...bundle.trajectory, status: 'finished' } }),
    'it finished'
  );
  assert.notEqual(
    one,
    trajectorySignature({
      trajectory: { ...bundle.trajectory, capture: { ...capture, exchange_count: 5 } },
    }),
    'a turn landed'
  );
  assert.notEqual(
    one,
    trajectorySignature({
      trajectory: { ...bundle.trajectory, capture: { ...capture, calls_missing: 1 } },
    }),
    'a capture gap appeared'
  );
  assert.equal(trajectorySignature({}), null, 'nothing open is not a state to compare');
});
