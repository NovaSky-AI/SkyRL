/**
 * The viewer's server: static files, and one proxy hop to the capture API.
 *
 * It holds no state and knows nothing about trajectories. Everything on screen
 * comes from `/v1`, which both `skyrl-capture serve` (live, during training)
 * and `skyrl-capture view` (offline, over a record directory) expose in the
 * same shape -- so the same viewer reads a run in progress and a run that
 * finished last week, with no second code path.
 *
 * The proxy exists so the browser talks to exactly one origin. The capture API
 * sets no CORS headers, and it should not have to: it is a private read API,
 * and teaching it about browsers to satisfy a viewer would be the wrong way
 * round.
 *
 *     node server.mjs --api http://127.0.0.1:8080 --port 8750
 *
 * Zero dependencies, on purpose. There is no build, no lockfile and no
 * node_modules to audit; `skyrl-capture view` runs this file as it sits.
 */

import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { extname, join, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = fileURLToPath(new URL('.', import.meta.url));
const PUBLIC = join(HERE, 'public');

const TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
};

function parseArgs(argv) {
  const out = { api: 'http://127.0.0.1:8080', port: 8750, host: '127.0.0.1' };
  for (let i = 0; i < argv.length; i += 1) {
    const flag = argv[i];
    if (flag === '--api') out.api = argv[++i];
    else if (flag === '--port') out.port = Number(argv[++i]);
    else if (flag === '--host') out.host = argv[++i];
  }
  out.api = out.api.replace(/\/+$/, '');
  return out;
}

const args = parseArgs(process.argv.slice(2));

/** Serve one file from `public/`, refusing anything that climbs out of it. */
async function serveStatic(pathname, response) {
  const relative = normalize(pathname === '/' ? '/index.html' : pathname).replace(/^(\.\.[/\\])+/, '');
  const file = join(PUBLIC, relative);
  if (!file.startsWith(PUBLIC)) {
    response.writeHead(403).end('forbidden');
    return;
  }
  try {
    const body = await readFile(file);
    response.writeHead(200, {
      'content-type': TYPES[extname(file)] || 'application/octet-stream',
      // The viewer is developed by reloading it; a cached module is a lie.
      'cache-control': 'no-cache',
    });
    response.end(body);
  } catch {
    // Unknown paths fall back to the shell so deep links survive a reload:
    // the client owns routing below `/`.
    if (extname(relative)) response.writeHead(404).end('not found');
    else serveStatic('/index.html', response);
  }
}

/** Forward `/api/...` to `<api>/...`, unchanged in both directions. */
async function proxy(pathname, search, response) {
  const target = `${args.api}${pathname.slice('/api'.length)}${search}`;
  try {
    const upstream = await fetch(target, { headers: { accept: 'application/json' } });
    const body = await upstream.arrayBuffer();
    response.writeHead(upstream.status, {
      'content-type': upstream.headers.get('content-type') || 'application/json',
      'cache-control': 'no-store',
    });
    response.end(Buffer.from(body));
  } catch (error) {
    // The capture process being down is the normal case during training, not
    // an exception: say which address failed so the fix is obvious.
    response.writeHead(502, { 'content-type': 'application/json' });
    response.end(JSON.stringify({ detail: `cannot reach the capture API at ${args.api}: ${error.message}` }));
  }
}

const server = createServer((request, response) => {
  const url = new URL(request.url, 'http://localhost');
  if (url.pathname === '/__viewer') {
    response.writeHead(200, { 'content-type': 'application/json' });
    response.end(JSON.stringify({ api: args.api }));
    return;
  }
  if (url.pathname.startsWith('/api/')) {
    proxy(url.pathname, url.search, response);
    return;
  }
  serveStatic(url.pathname, response);
});

server.listen(args.port, args.host, () => {
  process.stdout.write(`viewer on http://${args.host}:${args.port}  api=${args.api}\n`);
});
