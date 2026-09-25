/** A root-to-leaf path, decoded and blocked. The heart of the viewer.
 *
 * The unit here is the path, not the node, because a path is what an
 * export row is -- so what is read on screen and what reaches the trainer are
 * the same object. Blocks are cut on the mask *and* on the turn boundary,
 * which is what makes the generation scaffold fall out as its own block
 * instead of hiding inside an assistant turn.
 *
 * Text and token offsets were captured with the IDs. The browser owns no
 * tokenizer and must not start: a second tokenizer would be a second answer,
 * and the point of TITO mode is that there is one.
 */

import { h, mount, copy } from '../lib/dom.mjs';
import { num, pct, visible, KIND_TITLE } from '../lib/format.mjs';
import { strip, legend } from './strip.mjs';

const COLLAPSE_OVER = 420;

/** One block's text, its logprob ramp, and the link between them.
 *
 * The ramp is one column per token and the text is one span per token, so the
 * two share an index and hovering either highlights the other. That link is
 * the point: the ramp says *where* the model was unsure -- which prose cannot
 * show -- and the text says what it was unsure about.
 *
 * A canvas for the ramp rather than elements, because a span can be four
 * thousand tokens long and four thousand divs is a page that fights back.
 */
function rampFor(block, logprobs) {
  const count = block.token_count;
  const values = [];
  for (let index = 0; index < count; index += 1) values.push(logprobs[block.start + index]);

  // Scale to what is actually here. On an absolute 0..1 probability ramp a
  // confident model puts everything at the top -- 83% above 0.9 on a real
  // trajectory -- and the shading says nothing. Normalising within the block
  // makes its least certain tokens visible.
  const real = values.filter((value) => value !== undefined && value !== null);
  const low = real.length ? Math.min(...real) : -1;
  const span = real.length ? Math.max(...real) - low : 1;

  const canvas = h('canvas', {
    class: 'ribbon',
    height: 10,
    width: Math.max(1, count),
    style: { width: '100%', height: '10px', display: 'block', borderRadius: '2px', imageRendering: 'pixelated' },
    title: 'per-token logprob, scaled to this block. Hover to find the token.',
  });
  const paint = (marked) => {
    const context = canvas.getContext('2d');
    if (!context) return;
    context.clearRect(0, 0, Math.max(1, count), 10);
    for (let index = 0; index < count; index += 1) {
      const value = values[index];
      if (value === undefined || value === null) {
        context.fillStyle = 'rgba(128,128,128,0.25)';
      } else {
        const level = span > 0 ? (value - low) / span : 1;
        context.fillStyle = `rgba(53, 192, 122, ${0.15 + 0.85 * level})`;
      }
      context.fillRect(index, 0, 1, 10);
    }
    if (marked !== null && marked >= 0 && marked < count) {
      context.fillStyle = 'rgba(240, 90, 90, 0.95)';
      context.fillRect(marked, 0, 1, 10);
    }
  };
  requestAnimationFrame(() => paint(null));
  return { canvas, paint, values, count };
}

/** What the hover says about one token. */
function readoutText(block, index, logprobs) {
  const id = (block.token_ids || [])[index];
  const value = logprobs[block.start + index];
  const offsets = block.token_offsets;
  const piece = (block.text || '').slice(offsets[index], offsets[index + 1]);
  const parts = [
    `#${block.start + index}`,
    id === undefined ? 'id ?' : `id ${id}`,
    block.trainable ? 'in the loss' : 'masked',
  ];
  if (piece === '') {
    // Grouped with an earlier token across one multi-byte character, so the
    // characters are shown against that token and this one holds none.
    parts.push('shares a character with an earlier token');
  }
  if (value !== undefined && value !== null) {
    parts.push(`logprob ${value.toFixed(4)}`, `p ${Math.exp(value).toFixed(3)}`);
  }
  return `${JSON.stringify(piece)}  ·  ${parts.join('  ·  ')}`;
}

function blockRow(block, { path, showLogprobs, showControls }) {
  const hasRange = block.start !== undefined && block.start !== null;
  const text = showControls ? visible(block.text || '') : block.text || '';
  const long = (block.text || '').length > COLLAPSE_OVER;
  const collapsible = long && block.kind === 'given';

  // One span per token when the block says where each one starts, so a hover
  // can name it. Offsets are non-decreasing, not strictly increasing: tokens
  // that share one multi-byte character give their characters to the first of
  // them, and the rest render as a hoverable sliver. Without offsets at all it
  // is one run of text, as before.
  const offsets = block.token_offsets;
  const inspectable = Boolean(offsets) && block.token_count > 0;
  const body = inspectable
    ? h(
        'div',
        { class: 'text tokens' },
        Array.from({ length: block.token_count }, (_, index) => {
          const piece = (block.text || '').slice(offsets[index], offsets[index + 1]);
          const shared = piece === '';
          return h('span', { class: shared ? 'tok joined' : 'tok', dataset: { i: String(index) } },
                   shared ? '' : showControls ? visible(piece) : piece);
        })
      )
    : h('div', { class: 'text' }, text);
  const logprobs = path.logprobs || [];
  const ramp =
    showLogprobs && hasRange && block.kind === 'sampled' && logprobs.length
      ? rampFor(block, logprobs)
      : null;
  const readout = inspectable
    ? h('div', { class: 'token-readout' }, 'hover a token')
    : null;

  const row = h(
    'div',
    { class: `block ${block.kind}${collapsible ? ' collapsed' : ''}` },
    h(
      'div',
      { class: 'gutter' },
      h(
        'span',
        { class: `badge kind-${block.kind}`, title: KIND_TITLE[block.kind] },
        block.kind
      ),
      h('span', { class: 'role' }, block.role || '—'),
      hasRange
        ? h(
            'span',
            {
              class: 'range',
              title: 'the range into input_ids, loss_mask and rollout_logprobs. Click to copy.',
              onclick: (event) => copy(`[${block.start}:${block.end}]`, event.currentTarget),
            },
            `[${block.start}:${block.end}]`
          )
        : null,
      hasRange ? h('span', { class: 'role' }, `${num(block.token_count)} tok`) : null
    ),
    h(
      'div',
      { style: { minWidth: '0' } },
      ramp ? h('div', { style: { marginBottom: '5px' } }, ramp.canvas) : null,
      readout,
      body,
      collapsible
        ? h(
            'span',
            {
              class: 'more',
              onclick: (event) => {
                row.classList.toggle('collapsed');
                event.currentTarget.textContent = row.classList.contains('collapsed')
                  ? 'show all'
                  : 'collapse';
              },
            },
            'show all'
          )
        : null
    )
  );
  // The link, in both directions. Hovering a token marks its column in the
  // ramp; moving along the ramp marks the token. Neither is useful alone: a
  // dip you cannot point at is a colour, and a token whose confidence you
  // cannot see is just text.
  if (inspectable) {
    const mark = (index) => {
      for (const node of body.children || []) node.classList?.remove('on');
      if (index !== null && body.children && body.children[index]) {
        body.children[index].classList.add('on');
      }
      if (ramp) ramp.paint(index);
      readout.textContent =
        index === null ? 'hover a token' : readoutText(block, index, logprobs);
    };
    body.addEventListener('mouseover', (event) => {
      const at = event.target?.dataset?.i;
      if (at !== undefined) mark(Number(at));
    });
    body.addEventListener('mouseleave', () => mark(null));
    if (ramp) {
      ramp.canvas.addEventListener('mousemove', (event) => {
        const box = ramp.canvas.getBoundingClientRect();
        const at = Math.floor(((event.clientX - box.left) / box.width) * ramp.count);
        mark(Math.min(Math.max(at, 0), ramp.count - 1));
      });
      ramp.canvas.addEventListener('mouseleave', () => mark(null));
    }
  }

  return row;
}

function pathPicker(paths, chosen, onPick) {
  if (paths.length < 2) return null;
  return h(
    'div',
    { class: 'path-pick' },
    paths.map((path, index) =>
      h(
        'div',
        {
          class: `opt${index === chosen ? ' on' : ''}`,
          onclick: () => onPick(index),
          title: path.path_id,
        },
        h(
          'span',
          { class: 'name' },
          `path ${index}`,
          path.abandoned ? h('span', { class: 'badge info', style: { marginLeft: '6px' } }, 'abandoned') : null
        ),
        strip(path.blocks),
        h(
          'span',
          { class: 'stat' },
          `${num(path.trainable_count)}/${num(path.token_count)} ${pct(path.trainable_count, path.token_count)}`
        )
      )
    )
  );
}

export function renderPath(container, { data, state, onState }) {
  const paths = data.paths || [];
  if (!paths.length) {
    mount(container, h('div', { class: 'empty-state' }, 'This trajectory has no path to show.'));
    return;
  }

  const chosen = Math.min(state.pathIndex || 0, paths.length - 1);
  const path = paths[chosen];
  const tokens = data.mode === 'tokens';
  const blocks = (path.blocks || []).filter((block) => state.showGiven || block.kind !== 'given');

  const notices = [];
  if (!tokens) {
    notices.push(
      'Text mode: no token ids were captured, so there is no mask, no ranges and nothing to re-feed. Blocks are cut on messages, and their widths are characters.'
    );
  }
  if (path.blocking === 'mask') {
    notices.push(
      'Degraded blocking: this path was cut on the mask alone, so the generation scaffold is not separable from the turn it sits in.'
    );
  }
  if (path.masked_reason) notices.push(`Excluded from training: ${path.masked_reason}`);

  mount(
    container,
    pathPicker(paths, chosen, (index) => onState({ pathIndex: index })),
    h('div', { style: { margin: '6px 0 10px' } }, strip(path.blocks, { tall: true })),
    h(
      'div',
      { class: 'toolbar' },
      h('span', { class: 'mono' }, `${num(path.token_count)} ${tokens ? 'tokens' : 'chars'}`),
      h(
        'span',
        { class: 'mono' },
        `${num(path.trainable_count)} trainable (${pct(path.trainable_count, path.token_count)})`
      ),
      path.stop_reason ? h('span', { class: 'mono' }, `stop: ${path.stop_reason}`) : null,
      h(
        'label',
        { title: 'Hide context to leave only what the model produced or was made to look like it produced.' },
        h('input', {
          type: 'checkbox',
          checked: state.showGiven,
          onchange: (event) => onState({ showGiven: event.target.checked }),
        }),
        'given'
      ),
      h(
        'label',
        { title: 'Render newlines and tabs literally. A missing newline reads as nothing otherwise.' },
        h('input', {
          type: 'checkbox',
          checked: state.showControls,
          onchange: (event) => onState({ showControls: event.target.checked }),
        }),
        'control chars'
      ),
      tokens && (path.logprobs || []).length
        ? h(
            'label',
            { title: 'A confidence ramp above each sampled span, scaled to that span. Hover it to find the token.' },
            h('input', {
              type: 'checkbox',
              checked: state.showLogprobs,
              onchange: (event) => onState({ showLogprobs: event.target.checked }),
            }),
            'logprobs'
          )
        : null,
      h('span', { style: { flex: '1' } }),
      h(
        'button',
        {
          class: 'icon-btn',
          title: 'Copy this path as the JSON an export would give.',
          onclick: (event) => copy(JSON.stringify(path, null, 2), event.currentTarget),
        },
        'copy json'
      )
    ),
    notices.map((notice) => h('div', { class: 'note' }, notice)),
    h('div', { style: { margin: '10px 0 14px' } }, legend()),
    blocks.length
      ? blocks.map((block) =>
          blockRow(block, { path, showLogprobs: state.showLogprobs, showControls: state.showControls })
        )
      : h('div', { class: 'empty-state' }, 'Every block here is context. Tick "given" to see it.')
  );
}
