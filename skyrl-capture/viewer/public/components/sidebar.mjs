/** Projects, then runs. The W&B shape, because it is the right one: a run is
 *  the unit people talk about, and a project is how they find it. */

import { h, mount } from '../lib/dom.mjs';
import { num } from '../lib/format.mjs';

export function renderSidebar(container, { runs, active, source, filter, onFilter, onPick, onTheme }) {
  const matching = runs.filter((run) => {
    if (!filter) return true;
    const needle = filter.toLowerCase();
    return run.id.toLowerCase().includes(needle) || (run.project || '').toLowerCase().includes(needle);
  });

  const byProject = new Map();
  for (const run of matching) {
    const key = run.project || '(no project)';
    if (!byProject.has(key)) byProject.set(key, []);
    byProject.get(key).push(run);
  }

  mount(
    container,
    h(
      'div',
      { class: 'brand' },
      h('span', { class: 'mark' }),
      h('span', { class: 'name' }, 'skyrl-capture'),
      h('span', { class: 'spacer', style: { flex: '1' } }),
      h('button', {
        class: 'icon-btn',
        title: 'light / dark',
        onclick: onTheme,
      }, '◑')
    ),
    h(
      'div',
      { class: 'source' },
      h(
        'div',
        { class: 'row' },
        h('span', { class: `dot ${source.kind}` }),
        h('span', {}, source.label)
      ),
      h('div', { class: 'addr', title: source.api }, source.api)
    ),
    h(
      'div',
      { class: 'search' },
      h('input', {
        placeholder: 'filter runs',
        value: filter || '',
        oninput: (event) => onFilter(event.target.value),
      })
    ),
    h(
      'div',
      { class: 'nav' },
      byProject.size === 0
        ? h('div', { class: 'project' }, runs.length ? 'no match' : 'no runs yet')
        : [...byProject.entries()].map(([project, list]) => [
            h('div', { class: 'project' }, project),
            list.map((run) =>
              h(
                'div',
                {
                  class: `run${run.id === active ? ' active' : ''}`,
                  onclick: () => onPick(run.id),
                  title: `${run.id} — ${num(run.trajectory_count)} trajectories`,
                },
                h('span', { class: 'id' }, run.id),
                h('span', { class: 'n' }, num(run.trajectory_count))
              )
            ),
          ])
    )
  );
}
