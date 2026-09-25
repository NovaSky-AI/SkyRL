/** Render every view, against payloads a real `/v1` returned.
 *
 * The fixtures in `fixtures/` are verbatim responses from a
 * `skyrl-capture view` over a record of a real GPU run -- a tokens-mode
 * trajectory with a fork and a replayed assistant turn, and a text-mode one.
 * Refresh them with `node test/capture-fixtures.mjs <api-url>`.
 *
 *     node --test viewer/test/
 */

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { install, el } from './dom-shim.mjs';

install();

const HERE = fileURLToPath(new URL('.', import.meta.url));
const fixture = (name) => JSON.parse(readFileSync(join(HERE, 'fixtures', `${name}.json`), 'utf8'));

const { summarise, runHealth } = await import('../public/lib/diagnose.mjs');
const { strip } = await import('../public/components/strip.mjs');
const { renderPath } = await import('../public/components/path.mjs');
const { renderTree } = await import('../public/components/tree.mjs');
const { renderCalls } = await import('../public/components/calls.mjs');
const { renderTable, renderHealth } = await import('../public/components/table.mjs');
const { renderSidebar } = await import('../public/components/sidebar.mjs');

const forked = fixture('paths-forked');
const plain = fixture('paths-plain');
const text = fixture('paths-text');
const trajectory = fixture('trajectory');
const graph = fixture('graph');
const exchanges = fixture('exchanges');
const runs = fixture('runs');

const STATE = {
  pathIndex: 0,
  showGiven: true,
  showControls: false,
  showLogprobs: false,
  openNode: null,
};

// -- the four kinds ----------------------------------------------------------
test('a replayed assistant turn is its own kind, not "untrainable"', () => {
  const kinds = forked.paths.flatMap((path) => path.blocks.map((block) => block.kind));
  assert.ok(kinds.includes('replayed'), 'the fixture must contain the case the design exists for');
  assert.ok(kinds.includes('scaffold'));
  assert.ok(kinds.includes('sampled'));

  const root = el();
  renderPath(root, { data: forked, state: { ...STATE, pathIndex: 1 }, onState() {} });
  const badges = root.findAll((node) => node.hasClass('badge') && node.textContent === 'replayed');
  assert.ok(badges.length >= 1, 'the replayed block is labelled as such on screen');
});

test('every block shows the token range it occupies', () => {
  const root = el();
  renderPath(root, { data: forked, state: STATE, onState() {} });
  const ranges = root.findAll((node) => node.hasClass('range')).map((node) => node.textContent);
  assert.ok(ranges.length > 0);
  for (const range of ranges) assert.match(range, /^\[\d+:\d+\]$/);
});

test('special tokens and the scaffold are rendered, not stripped', () => {
  const root = el();
  renderPath(root, { data: forked, state: STATE, onState() {} });
  const body = root.textContent;
  assert.ok(body.includes('<|im_start|>'), 'boundary bugs live at these markers');
});

// -- the strip ---------------------------------------------------------------
test('a thin scaffold still gets a segment', () => {
  const path = forked.paths[0];
  const node = strip(path.blocks);
  assert.equal(node.children.length, path.blocks.length);
  const scaffold = path.blocks.findIndex((block) => block.kind === 'scaffold');
  assert.ok(scaffold >= 0);
  assert.ok(node.children[scaffold].hasClass('scaffold'));
});

test('a path with nothing in it gets a strip that says so rather than an empty bar', () => {
  assert.ok(strip([]).hasClass('none'));
});

// -- diagnosis ---------------------------------------------------------------
test('a forked trajectory with replayed text is flagged as both', () => {
  const summary = summarise(forked, trajectory);
  assert.ok(summary.flags.includes('replayed'));
  assert.ok(summary.flags.includes('forked'));
  assert.equal(summary.paths, forked.paths.length);
  assert.ok(summary.trainable > 0 && summary.trainable < summary.tokens);
});

test('a clean trajectory is flagged as nothing', () => {
  const summary = summarise(plain, fixture('trajectory-plain'));
  assert.deepEqual(summary.flags, []);
});

test('missing calls outrank anything the paths say', () => {
  const summary = summarise(plain, {
    ...trajectory,
    capture: { ...trajectory.capture, calls_missing: 2 },
  });
  assert.ok(summary.flags.includes('calls-missing'));
});

test('a path that trains on nothing is called out', () => {
  const empty = {
    ...plain,
    paths: [{ ...plain.paths[0], trainable_count: 0, blocks: [], logprobs: [] }],
  };
  assert.ok(summarise(empty, trajectory).flags.includes('no-train'));
});

test('the run health band counts each flag once per trajectory', () => {
  const rows = [
    { summary: summarise(forked, trajectory) },
    { summary: summarise(plain, fixture('trajectory-plain')) },
  ];
  const health = runHealth(rows);
  assert.equal(health.scanned, 2);
  assert.equal(health.total, 2);
  const counts = Object.fromEntries(health.counts);
  assert.equal(counts.replayed, 1);
});

// -- text mode ---------------------------------------------------------------
test('a text-mode trajectory says what it cannot show instead of showing nothing', () => {
  const root = el();
  renderPath(root, { data: text, state: STATE, onState() {} });
  assert.match(root.textContent, /Text mode/);
  assert.ok(root.findAll((node) => node.hasClass('block')).length > 0 || text.paths[0].blocks.length === 0);
});

test('text mode carries no token flags it cannot know', () => {
  const summary = summarise(text, fixture('trajectory-text'));
  assert.ok(!summary.flags.includes('no-logprobs'), 'there are no tokens, so there is nothing to weight');
});

// -- forks, read in the tree -------------------------------------------------
test('an opened fork names where the branches stopped matching', () => {
  const [point] = graph.branch_points;
  const root = el();
  renderTree(root, { graph, paths: forked, state: { ...STATE, openNode: point.node_id }, onState() {} });
  assert.match(root.textContent, /they agree for \d+ characters, then part|differ from their first character/);
});

test('train-once is checked, not assumed', () => {
  const root = el();
  renderTree(root, { graph, paths: forked, state: STATE, onState() {} });
  assert.match(root.textContent, /Train-once/);
});

test('a node sampled in two paths is reported as a violation', () => {
  const shared = JSON.parse(JSON.stringify(forked));
  // Force the case: make both paths train the same node.
  const victim = shared.paths[0].blocks.find((block) => block.kind === 'sampled');
  const other = shared.paths[1].blocks.find((block) => block.kind === 'sampled');
  assert.ok(victim && other);
  other.node_id = victim.node_id;
  const root = el();
  renderTree(root, { graph, paths: shared, state: STATE, onState() {} });
  assert.match(root.textContent, /Train-once violated/);
});

// -- the tree, which is where forks are read ---------------------------------
test('the tree marks branch points', () => {
  const root = el();
  renderTree(root, { graph, paths: forked, state: STATE, onState() {} });
  assert.equal(root.findAll((node) => node.hasClass('node')).length, graph.nodes.length);
  const forks = root.findAll((node) => node.hasClass('node') && node.hasClass('fork'));
  assert.equal(forks.length, graph.branch_points.length, 'a marked node per branch point');
  assert.ok(forks.length > 0, 'the fixture must actually branch');
});

test('opening a node shows it and what follows it', () => {
  const [point] = graph.branch_points;
  const root = el();
  renderTree(root, { graph, paths: forked, state: { ...STATE, openNode: point.node_id }, onState() {} });
  const panel = root.findAll((node) => node.hasClass('node-panel'));
  assert.equal(panel.length, 1, 'one panel, under the node that was opened');
  assert.match(panel[0].textContent, /this node/);
  assert.match(panel[0].textContent, new RegExp(`branches into ${point.child_count}`));
  const columns = panel[0].findAll((node) => node.hasClass('continuation'));
  assert.equal(columns.length, point.child_count, 'one column per branch');
});

test('a node with one child reads as a continuation, and a leaf says so', () => {
  const linear = graph.nodes.find(
    (node) =>
      graph.nodes.filter((other) => other.parent_node_id === node.node_id).length === 1
  );
  assert.ok(linear, 'the fixture must contain an unbranched node');
  let root = el();
  renderTree(root, { graph, paths: forked, state: { ...STATE, openNode: linear.node_id }, onState() {} });
  let panel = root.findAll((node) => node.hasClass('node-panel'))[0];
  assert.match(panel.textContent, /then/);
  assert.doesNotMatch(panel.textContent, /branches into/);

  const leaf = graph.leaf_node_ids[0];
  root = el();
  renderTree(root, { graph, paths: forked, state: { ...STATE, openNode: leaf }, onState() {} });
  panel = root.findAll((node) => node.hasClass('node-panel'))[0];
  assert.match(panel.textContent, /Nothing follows/);
});

test('a node shared by several paths is rendered once, not once per path', () => {
  // `textByNode` joins a node's blocks *within* a path -- an assistant turn is
  // a scaffold plus a sampled span -- but a shared prefix node appears in every
  // path that runs through it, and joining those too doubled its text.
  const root = el();
  const shared = forked.paths[0].node_ids.find((id) =>
    forked.paths.every((path) => path.node_ids.includes(id))
  );
  assert.ok(shared, 'the fixture must have a node on the shared prefix');
  renderTree(root, { graph, paths: forked, state: { ...STATE, openNode: shared }, onState() {} });
  const body = root.findAll((node) => node.hasClass('node-panel'))[0].findAll((n) => n.hasClass('text'))[0];
  const opens = (body.textContent.match(/<\|im_start\|>/g) || []).length;
  assert.ok(opens <= 1, `a node's own text opens at most one turn, saw ${opens}`);
});

test('the call table flags a truncated generation', () => {
  const truncated = {
    ...exchanges,
    data: exchanges.data.map((row) => ({ ...row, completion_reason: 'length' })),
  };
  const root = el();
  renderCalls(root, { exchanges: truncated });
  assert.match(root.textContent, /truncated/);
});

test('the table renders a row per trajectory before any scan has finished', () => {
  const root = el();
  const rows = [{ ...trajectory, summary: null }];
  renderTable(root, { rows, selected: null, onOpen() {} });
  assert.match(root.textContent, /trainable/);
});

test('a clean run says so rather than showing an empty band', () => {
  const root = el();
  renderHealth(root, { health: { scanned: 3, total: 3, counts: [] }, active: null, onToggle() {} });
  assert.match(root.textContent, /nothing flagged/);
});

test('the sidebar groups runs under their project', () => {
  const root = el();
  renderSidebar(root, {
    runs,
    active: runs[0].id,
    source: { kind: 'record', label: 'record', api: 'http://x' },
    filter: '',
    onFilter() {},
    onPick() {},
    onTheme() {},
  });
  assert.match(root.textContent, new RegExp(runs[0].project));
  assert.match(root.textContent, new RegExp(runs[0].id));
});

test('the sidebar filter actually filters', () => {
  const root = el();
  renderSidebar(root, {
    runs,
    active: null,
    source: { kind: 'live', label: 'live', api: 'http://x' },
    filter: 'nothing-matches-this',
    onFilter() {},
    onPick() {},
    onTheme() {},
  });
  assert.match(root.textContent, /no match/);
});

// -- inspecting a token ------------------------------------------------------
test('a block with offsets renders one span per token', () => {
  // Text is what a person reads; the token is what is trained on. Without
  // offsets there is nothing linking the two, which is why the server sends
  // them rather than the browser guessing.
  const text = 'The retry decorator';
  const withOffsets = {
    ...plain,
    paths: [
      {
        ...plain.paths[0],
        logprobs: [-0.1, -2.0, -0.01],
        blocks: [
          {
            kind: 'sampled', role: null, trainable: true, start: 0, end: 3,
            token_count: 3, text,
            token_ids: [785, 22683, 50678],
            token_offsets: [0, 3, 9, 19],
          },
        ],
      },
    ],
  };
  const root = el();
  renderPath(root, { data: withOffsets, state: { ...STATE, showLogprobs: true }, onState() {} });

  const spans = root.findAll((node) => node.hasClass('tok'));
  assert.equal(spans.length, 3, 'one span per token');
  assert.deepEqual(spans.map((s) => s.textContent), ['The', ' retry', ' decorator']);
  assert.equal(spans.map((s) => s.textContent).join(''), text, 'and they tile the text');

  assert.ok(root.findAll((node) => node.hasClass('token-readout')).length, 'somewhere to say what a token is');
});

test('tokens sharing one character each keep a span to hover', () => {
  // A byte-level BPE splits a multi-byte character across tokens, so the
  // character goes to the first of them and the rest own none. They must not
  // collapse to nothing: an empty span has no width and cannot be hovered,
  // and the id and logprob of a token in the loss are exactly what someone
  // opened this view to see.
  const text = 'ok \u{1F680} done';
  const shared = {
    ...plain,
    paths: [
      {
        ...plain.paths[0],
        logprobs: [-0.1, -2.0, -0.5, -0.01],
        blocks: [
          {
            kind: 'sampled', role: null, trainable: true, start: 0, end: 4,
            token_count: 4, text,
            token_ids: [562, 11162, 248, 2814],
            // Non-decreasing, not strictly increasing: tokens 1 and 2 share
            // the emoji, and token 2 owns no characters of its own. Counted in
            // UTF-16 code units, which is what `slice` here indexes by -- the
            // emoji is two of them, so code points would read `[0,3,4,4,9]`
            // and shift every token after it.
            token_offsets: [0, 3, 5, 5, 10],
          },
        ],
      },
    ],
  };
  const root = el();
  renderPath(root, { data: shared, state: { ...STATE, showLogprobs: true }, onState() {} });

  const spans = root.findAll((node) => node.hasClass('tok'));
  assert.equal(spans.length, 4, 'every token still has a span');
  assert.deepEqual(spans.map((s) => s.textContent), ['ok ', '\u{1F680}', '', ' done']);
  assert.equal(spans.map((s) => s.textContent).join(''), text, 'and they tile the text');

  // The one that owns no characters is marked, so it can be given width and
  // read as what it is rather than as a rendering slip.
  const joined = root.findAll((node) => node.hasClass('joined'));
  assert.equal(joined.length, 1, 'exactly the token that shares a character');
  assert.equal(joined[0].textContent, '');
});

test('a block without offsets is still one run of text', () => {
  // Any span whose per-token decode did not reassemble exactly. Built by
  // stripping the offsets rather than by finding a fixture that happens to
  // lack them: capture records them at write time, so a fixture without them
  // would be an accident this test came to depend on.
  const withoutOffsets = {
    ...plain,
    paths: plain.paths.map((path) => ({
      ...path,
      blocks: path.blocks.map(({ token_offsets, ...block }) => block),
    })),
  };
  const root = el();
  renderPath(root, { data: withoutOffsets, state: STATE, onState() {} });
  assert.equal(root.findAll((node) => node.hasClass('tok')).length, 0);
  assert.ok(root.findAll((node) => node.hasClass('text')).length, 'the text is still there');
});
