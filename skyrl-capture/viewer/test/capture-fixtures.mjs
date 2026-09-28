/** Refresh the test fixtures from a live `/v1`.
 *
 * The fixtures are verbatim responses, not hand-written objects, because a
 * hand-written object is a guess about the API that stops being true quietly.
 * This overwrites them from whatever is serving, so a schema change shows up
 * as a failing render test rather than a broken page.
 *
 *     node test/capture-fixtures.mjs http://127.0.0.1:8601 [http://text-api]
 *
 * Wanted from the tokens API: a forked trajectory (one with a replayed
 * assistant turn) and a plain one. From the text API: any trajectory.
 */

import { mkdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const OUT = join(fileURLToPath(new URL('.', import.meta.url)), 'fixtures');
const tokensApi = (process.argv[2] || 'http://127.0.0.1:8601').replace(/\/+$/, '');
const textApi = (process.argv[3] || '').replace(/\/+$/, '');

const get = async (base, path) => {
  const response = await fetch(`${base}${path}`);
  if (!response.ok) throw new Error(`${response.status} on ${base}${path}`);
  return response.json();
};

const save = async (name, value) => {
  await writeFile(join(OUT, `${name}.json`), `${JSON.stringify(value, null, 1)}\n`);
  process.stdout.write(`  ${name}.json\n`);
};

await mkdir(OUT, { recursive: true });

const listing = await get(tokensApi, '/v1/trajectories?limit=200');
const withPaths = [];
for (const row of listing.data) {
  withPaths.push([row, await get(tokensApi, `/v1/trajectories/${row.id}/paths`)]);
}

const forked = withPaths.find(([, paths]) =>
  paths.paths.some((path) => path.blocks.some((block) => block.kind === 'replayed'))
);
const plain = withPaths.find(([, paths]) => paths.paths.length === 1);
if (!forked) throw new Error('no trajectory with a replayed block: capture one that rewrites history');
if (!plain) throw new Error('no single-path trajectory in this record');

await save('paths-forked', forked[1]);
await save('trajectory', await get(tokensApi, `/v1/trajectories/${forked[0].id}`));
await save('graph', await get(tokensApi, `/v1/trajectories/${forked[0].id}/graph`));
await save('exchanges', await get(tokensApi, `/v1/trajectories/${forked[0].id}/exchanges`));
await save('paths-plain', plain[1]);
await save('trajectory-plain', await get(tokensApi, `/v1/trajectories/${plain[0].id}`));

const runs = await get(tokensApi, '/v1/runs');
await save('runs', runs.data);

if (textApi) {
  const textListing = await get(textApi, '/v1/trajectories?limit=5');
  const row = textListing.data[0];
  if (!row) throw new Error('the text API served no trajectories');
  await save('paths-text', await get(textApi, `/v1/trajectories/${row.id}/paths`));
  await save('trajectory-text', await get(textApi, `/v1/trajectories/${row.id}`));
}

process.stdout.write('fixtures refreshed\n');
