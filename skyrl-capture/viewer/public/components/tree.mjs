/** The message graph as a tree, with any node openable.
 *
 * Forks used to live on their own tab, as two paths picked from dropdowns.
 * That asked the wrong question: by the time a trace has branched, "path 0
 * against path 1" is two nearly identical transcripts, and the line that
 * differs is buried in the middle of both.
 *
 * A branch happens at a node, so that is where it is read. Clicking any node
 * shows its text and the text that follows it. One child is a continuation;
 * several are laid side by side with the text they share dimmed, so the point
 * they part is the thing on screen. Nothing special-cases a fork -- a fork is
 * just a node whose "what comes next" has more than one answer, which is also
 * the honest description of what a branch point is.
 *
 * The train-once rule rides along, because it is a statement about paths in
 * the plural and this is the only view holding several at once: a sampled node
 * reachable from more than one branch must be in the loss exactly once.
 */

import { h, mount } from '../lib/dom.mjs';
import { num, short } from '../lib/format.mjs';

/** The decoded text of each node, from whichever path attributed it.
 *
 * A node appears in every path that runs through it with the same text, so the
 * first one wins. Text mode attributes nothing, and then this map is empty and
 * the panel says so rather than showing blank boxes.
 */
function textByNode(paths) {
  const text = new Map();
  for (const path of paths || []) {
    // Per path first. A node is cut into several blocks *within* one path --
    // an assistant turn is a scaffold and a sampled span -- and those join.
    // Across paths it is the same node with the same text, so joining those
    // too would render every shared node once per path that runs through it.
    const local = new Map();
    for (const block of path.blocks || []) {
      if (!block.node_id || block.text === undefined) continue;
      local.set(block.node_id, (local.get(block.node_id) || '') + block.text);
    }
    for (const [nodeId, body] of local) {
      if (!text.has(nodeId)) text.set(nodeId, body);
    }
  }
  return text;
}

/** Which paths run through each node, and which of them train it. */
function pathsByNode(paths) {
  const through = new Map();
  for (const path of paths || []) {
    for (const nodeId of path.node_ids || []) {
      if (!through.has(nodeId)) through.set(nodeId, { paths: [], trains: [] });
      through.get(nodeId).paths.push(path.path_id);
    }
    for (const block of path.blocks || []) {
      if (block.kind !== 'sampled' || !block.node_id) continue;
      const entry = through.get(block.node_id);
      if (entry && !entry.trains.includes(path.path_id)) entry.trains.push(path.path_id);
    }
  }
  return through;
}

/** How many leading characters every one of these strings shares. */
function commonPrefix(texts) {
  if (!texts.length) return 0;
  const first = texts[0];
  let shared = 0;
  while (shared < first.length && texts.every((text) => text[shared] === first[shared])) shared += 1;
  return shared;
}

/** What a node is, in badges: size, sampled span, and who trains it. */
function badges(node, through) {
  const entry = through.get(node.node_id) || { paths: [], trains: [] };
  return [
    h('span', { class: 'range' }, short(node.node_id, 10)),
    h(
      'span',
      { class: 'range' },
      `${num(node.token_count ?? node.char_count ?? 0)} ${node.token_count ? 'tok' : 'ch'}`,
      node.sampled_token_count ? ` · ${num(node.sampled_token_count)} sampled` : ''
    ),
    node.sampled_token_count
      ? entry.trains.length
        ? h('span', { class: 'badge kind-sampled' }, `trains in ${entry.trains.length}`)
        : h('span', { class: 'badge' }, 'not trained')
      : null,
  ];
}

/** One node's text, with an optional shared prefix dimmed. */
function textBox(body, shared) {
  if (body === undefined) {
    return h('div', { class: 'note' }, 'No decoded text for this node.');
  }
  if (!shared) return h('div', { class: 'text' }, body);
  return h(
    'div',
    { class: 'text' },
    h('span', { class: 'agreed' }, body.slice(0, shared)),
    h('span', { class: 'diverged' }, body.slice(shared) || '(nothing after the shared text)')
  );
}

/** The panel under an opened node: what it is, and what comes next. */
function nodePanel(node, kids, { text, through }) {
  const texts = kids.map((kid) => text.get(kid.node_id) ?? '');
  const many = kids.length > 1;
  // Only worth dimming when there is something to compare against.
  const shared = many && texts.every((value) => value) ? commonPrefix(texts) : 0;
  const identical = many && shared > 0 && texts.every((value) => value.length === shared);

  return h(
    'div',
    { class: 'node-panel' },
    h('div', { class: 'panel-head' }, h('span', { class: 'what' }, 'this node'), ...badges(node, through)),
    textBox(text.get(node.node_id), 0),
    kids.length === 0
      ? h('div', { class: 'note leaf' }, 'Nothing follows: this is a leaf.')
      : h(
          'div',
          { class: 'panel-head' },
          h(
            'span',
            { class: 'what' },
            many ? `branches into ${num(kids.length)}` : 'then'
          ),
          many
            ? h(
                'span',
                { class: 'range' },
                shared
                  ? `they agree for ${num(shared)} characters, then part`
                  : 'they differ from their first character'
              )
            : null
        ),
    kids.length
      ? h(
          'div',
          { class: `continues${many ? ' side-by-side' : ''}` },
          kids.map((kid, index) =>
            h(
              'div',
              { class: 'continuation' },
              h(
                'div',
                { class: 'panel-head' },
                many ? h('span', { class: 'badge kind-sampled' }, `branch ${index + 1}`) : null,
                ...badges(kid, through)
              ),
              textBox(text.get(kid.node_id), many ? shared : 0)
            )
          )
        )
      : null,
    identical
      ? h('div', { class: 'note' }, 'These branches are textually identical; they differ in tokens alone.')
      : null
  );
}

export function renderTree(container, { graph, paths, state, onState }) {
  const nodes = graph.nodes || [];
  if (!nodes.length) {
    mount(container, h('div', { class: 'empty-state' }, 'No nodes were committed for this trajectory.'));
    return;
  }

  const children = new Map();
  const roots = [];
  for (const node of nodes) {
    if (!node.parent_node_id) roots.push(node);
    else {
      if (!children.has(node.parent_node_id)) children.set(node.parent_node_id, []);
      children.get(node.parent_node_id).push(node);
    }
  }

  // `branch_points` is a list of objects, not of ids. Reading it as ids made
  // this set silently empty, so no fork was ever marked in the tree.
  const forks = new Set((graph.branch_points || []).map((point) => point.node_id));
  const leaves = new Set(graph.leaf_node_ids || []);
  const pathList = paths?.paths || [];
  const text = textByNode(pathList);
  const through = pathsByNode(pathList);
  const open = state?.openNode ?? null;

  // A sampled node that more than one path trains is in the loss twice.
  const overTrained = [...through.entries()].filter(([, entry]) => entry.trains.length > 1);

  const rows = [];
  const walk = (node, rail) => {
    const kids = children.get(node.node_id) || [];
    const isFork = kids.length > 1 || forks.has(node.node_id);
    const isOpen = open === node.node_id;
    rows.push(
      h(
        'div',
        {
          class: `node${isFork ? ' fork' : ''}${isOpen ? ' open' : ''}`,
          onclick: () => onState({ openNode: isOpen ? null : node.node_id }),
          title: [
            node.node_id,
            `author ${node.author}`,
            node.sampled_start === null || node.sampled_start === undefined
              ? 'no sampled span'
              : `sampled from ${node.sampled_start}`,
            node.has_logprobs ? 'logprobs present' : 'no logprobs',
            'click to read this node and what follows it',
          ].join('\n'),
        },
        h('span', { class: 'rail' }, rail),
        h('span', { class: 'caret' }, isOpen ? '▾' : '▸'),
        h(
          'span',
          { class: `badge ${node.author === 'model' ? 'kind-sampled' : 'kind-given'}` },
          node.role || '?'
        ),
        h('span', { class: 'tag' }, node.author),
        h(
          'span',
          { class: 'count' },
          `${num(node.token_count ?? node.char_count)} ${node.token_count ? 'tok' : 'ch'}`,
          node.sampled_token_count ? ` · ${num(node.sampled_token_count)} sampled` : '',
          leaves.has(node.node_id) ? ' · leaf' : ''
        ),
        isFork ? h('span', { class: 'badge warn fork-badge' }, `${kids.length} branches`) : null,
        h('span', { class: 'count' }, short(node.node_id, 8))
      )
    );
    if (isOpen) rows.push(nodePanel(node, kids, { text, through }));
    kids.forEach((kid, index) => walk(kid, `${rail}${index === kids.length - 1 ? '  ' : ' |'}  `));
  };
  roots.forEach((root) => walk(root, ''));

  const forkCount = forks.size;
  mount(
    container,
    h(
      'div',
      { class: 'note' },
      `${num(nodes.length)} nodes, ${num(leaves.size)} leaves, ${num(forkCount)} branch point` +
        `${forkCount === 1 ? '' : 's'}. ` +
        'Click any node to read it and what follows it; ' +
        (forkCount
          ? 'a branch point shows its branches side by side.'
          : 'this trace never branched.')
    ),
    // Said either way once a trace has branched. A check that only speaks up
    // when it fails is indistinguishable from a check that never ran, and this
    // is the only view that can run it -- it is a statement about paths in the
    // plural.
    overTrained.length
      ? h(
          'div',
          { class: 'split', style: { borderColor: 'var(--error)', color: 'var(--error)' } },
          `Train-once violated: ${overTrained
            .map(([nodeId, entry]) => `${short(nodeId, 10)} is trained in ${entry.trains.length} paths`)
            .join('; ')}. A sampled node reachable from several branches must be in the loss exactly once.`
        )
      : forks.size && pathList.length > 1
        ? h('div', { class: 'note' }, 'Train-once holds: no sampled node is trained in more than one path.')
        : null,
    h('div', { class: 'tree' }, rows)
  );
}
