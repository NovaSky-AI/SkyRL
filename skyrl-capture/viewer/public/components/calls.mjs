/** The model calls behind the trajectory.
 *
 * Not training data -- this is where you look when the record is fine and the
 * *run* was not: a retry that fired twice, a call that took eight seconds, a
 * provider error the harness swallowed, a stream that stopped early.
 */

import { h, mount } from '../lib/dom.mjs';
import { num, short } from '../lib/format.mjs';

const ms = (value) => (value === null || value === undefined ? '-' : `${Math.round(value)}`);

export function renderCalls(container, { exchanges }) {
  const rows = exchanges.data || [];
  if (!rows.length) {
    mount(container, h('div', { class: 'empty-state' }, 'No model calls were captured.'));
    return;
  }

  const head = h(
    'tr',
    {},
    h('th', { class: 'n' }, 'seq'),
    h('th', {}, 'kind'),
    h('th', {}, 'model'),
    h('th', { class: 'n' }, 'status'),
    h('th', {}, 'stream'),
    h('th', { class: 'n' }, 'dur ms'),
    h('th', { class: 'n' }, 'ttft ms'),
    h('th', { class: 'n' }, 'gap ms'),
    h('th', {}, 'stop'),
    h('th', {}, 'flags')
  );

  const body = rows.map((row) => {
    const flags = [];
    if (row.transport_error) flags.push(['error', `transport: ${row.transport_error}`]);
    if (row.provider_error) flags.push(['error', `provider: ${row.provider_error}`]);
    if (row.retry_attempt) flags.push(['warn', `retry ${row.retry_attempt}`]);
    if (row.overlapping) flags.push(['warn', 'overlapping']);
    if (row.completion_reason === 'length') flags.push(['warn', 'truncated']);
    if (row.http_status >= 400) flags.push(['error', `http ${row.http_status}`]);
    return h(
      'tr',
      { title: row.id },
      h('td', { class: 'n mono' }, row.sequence),
      h('td', { class: 'mono dim' }, row.endpoint_kind),
      h('td', { class: 'mono dim' }, short(row.model || '-', 22)),
      h('td', { class: 'n mono' }, row.http_status ?? '-'),
      h('td', { class: 'mono dim' }, row.streaming ? `yes ${num(row.chunk_count)}` : 'no'),
      h('td', { class: 'n mono' }, ms(row.duration_ms)),
      h('td', { class: 'n mono' }, ms(row.ttft_ms)),
      h('td', { class: 'n mono dim' }, ms(row.gap_ms)),
      h('td', { class: 'mono dim' }, row.completion_reason || '-'),
      h(
        'td',
        {},
        flags.length
          ? h('span', { class: 'badges' }, flags.map(([level, text]) => h('span', { class: `badge ${level}` }, text)))
          : h('span', { class: 'dim' }, '-')
      )
    );
  });

  mount(container, h('table', { class: 'rows' }, h('thead', {}, head), h('tbody', {}, body)));
}
