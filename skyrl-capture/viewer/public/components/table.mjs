/** The trajectory table, and the health band above it.
 *
 * The columns are chosen for one question: which of these attempts is not like
 * the others? So the widest column is the mask strip, not an id -- an id tells
 * you nothing until you already know which one you want, and the strip tells
 * you which one you want.
 */

import { h, mount } from '../lib/dom.mjs';
import { num, pct, short, ago } from '../lib/format.mjs';
import { flagLevel, flagWhy } from '../lib/diagnose.mjs';
import { strip, legend } from './strip.mjs';

export function renderHealth(container, { health, active, onToggle }) {
  if (!health.counts.length) {
    mount(
      container,
      h(
        'div',
        { class: 'health' },
        h(
          'span',
          { class: 'clean' },
          health.scanned
            ? `${health.scanned} of ${health.total} scanned, nothing flagged.`
            : 'scanning...'
        )
      )
    );
    return;
  }
  mount(
    container,
    h(
      'div',
      { class: 'health' },
      health.counts.map(([flag, count]) =>
        h(
          'button',
          {
            class: `chip ${flagLevel(flag)}${active === flag ? ' on' : ''}`,
            title: `${flagWhy(flag)}\n\nclick to show only these`,
            onclick: () => onToggle(flag),
          },
          h('span', { class: 'bead' }),
          h('span', { class: 'n' }, num(count)),
          h('span', { class: 'label' }, flag)
        )
      ),
      health.scanned < health.total
        ? h('span', { class: 'clean' }, `scanning ${health.scanned}/${health.total}`)
        : null
    )
  );
}

function flagBadges(flags) {
  if (!flags.length) return h('span', { class: 'dim' }, '-');
  return h(
    'span',
    { class: 'badges' },
    flags.map((flag) =>
      h('span', { class: `badge ${flagLevel(flag)}`, title: flagWhy(flag) }, flag)
    )
  );
}

export function renderTable(container, { rows, selected, onOpen }) {
  if (!rows.length) {
    mount(container, h('div', { class: 'empty-state' }, 'No trajectories match.'));
    return;
  }

  const head = h(
    'tr',
    {},
    h('th', {}, 'trajectory'),
    h('th', {}, 'task'),
    h('th', { class: 'n' }, 'step'),
    h('th', {}, 'mask'),
    h('th', { class: 'n' }, 'tokens'),
    h('th', { class: 'n' }, 'trainable'),
    h('th', { class: 'n' }, 'paths'),
    h('th', { class: 'n' }, 'reward'),
    h('th', {}, 'flags'),
    h('th', {}, 'status'),
    h('th', {}, 'when')
  );

  const body = rows.map((row) => {
    const summary = row.summary;
    const blocks = row.paths?.paths?.[0]?.blocks;
    const rewardValue = row.annotations?.reward;
    return h(
      'tr',
      {
        class: row.id === selected ? 'active' : '',
        onclick: () => onOpen(row.id),
      },
      h('td', { class: 'mono', title: row.id }, short(row.id, 10)),
      h('td', { class: 'mono dim' }, row.task_id || '-'),
      h('td', { class: 'n mono' }, row.step ?? '-'),
      h(
        'td',
        { class: 'strip-cell' },
        summary ? strip(blocks || []) : h('span', { class: 'spin' }, '...')
      ),
      h('td', { class: 'n mono' }, summary ? num(summary.tokens) : '-'),
      h(
        'td',
        { class: 'n mono' },
        summary ? `${num(summary.trainable)} ${pct(summary.trainable, summary.tokens)}` : '-'
      ),
      h('td', { class: 'n mono dim' }, summary ? summary.paths : '-'),
      h(
        'td',
        { class: 'n mono' },
        typeof rewardValue === 'number' ? rewardValue.toFixed(2) : '-'
      ),
      h('td', {}, summary ? flagBadges(summary.flags) : h('span', { class: 'spin' }, '...')),
      h('td', { class: 'mono dim' }, row.status),
      h('td', { class: 'mono dim', title: row.created_at }, ago(row.created_at))
    );
  });

  mount(
    container,
    h('table', { class: 'rows' }, h('thead', {}, head), h('tbody', {}, body)),
    h('div', { style: { marginTop: '12px' } }, legend())
  );
}
