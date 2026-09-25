/** The mask as a shape, before it is read as text.
 *
 * Mask density is the first thing to sanity-check on a rollout and the last
 * thing anyone wants to compute from a table. One segment per block, coloured
 * by kind, width by token count -- so a run of these down a table column shows
 * the attempt that is not like the others without a number being read.
 *
 * Thin segments get a floor, because the three-token scaffold in a
 * five-thousand-token path is exactly the segment worth seeing.
 */

import { h } from '../lib/dom.mjs';
import { KIND_TITLE } from '../lib/format.mjs';

const weight = (block) => block.token_count || block.text?.length || 0;

export function strip(blocks, { tall = false, onPick = null } = {}) {
  const list = blocks || [];
  const total = list.reduce((sum, block) => sum + weight(block), 0);
  if (!total) {
    return h('div', {
      class: `strip ${tall ? 'tall ' : ''}none`,
      title: 'nothing to show: no tokens and no text',
    });
  }
  return h(
    'div',
    { class: `strip${tall ? ' tall' : ''}` },
    list.map((block, index) =>
      h('div', {
        class: `seg ${block.kind}`,
        style: { flex: `${weight(block)} 0 auto` },
        title: `${block.kind} — ${KIND_TITLE[block.kind] || ''}\n${weight(block)} ${
          block.token_count ? 'tokens' : 'chars'
        }${block.start !== undefined ? ` [${block.start}:${block.end}]` : ''}`,
        onclick: onPick ? (event) => { event.stopPropagation(); onPick(index); } : null,
        style2: null,
      })
    )
  );
}

export function legend() {
  return h(
    'div',
    { class: 'legend' },
    ['sampled', 'replayed', 'scaffold', 'given'].map((kind) =>
      h(
        'span',
        { class: 'item', title: KIND_TITLE[kind] },
        h('span', { class: `sw ${kind}` }),
        kind
      )
    )
  );
}
