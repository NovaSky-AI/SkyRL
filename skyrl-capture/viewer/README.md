# The viewer

A browser over `skyrl-capture`'s `/v1` read API: projects and runs down the
left, the run's task × step grid in the middle, one trajectory in a drawer on
the right.

```bash
skyrl-capture view --api http://127.0.0.1:8080   # a run in progress
skyrl-capture view --record ./traces             # a record, no database
```

Or directly, which is all the CLI does:

```bash
node server.mjs --api http://127.0.0.1:8080 --port 8750
```

## Two rules

**It is only ever an HTTP client of `/v1`.** It holds no state, knows nothing
about where the data came from, and has no private endpoint. That is what lets
the same page read a live capture process and a finished record directory —
`skyrl-capture serve` and `skyrl-capture view` answer the same routes, one from
memory and one from files. It also means anything on screen can be fetched
with `curl`.

**No dependencies and no build.** Node's own HTTP server, ES modules served as
they sit. There is no bundler, no lockfile and no `node_modules`; CI fails if
either appears. The server exists only to serve the files and make one proxy
hop, so the browser talks to a single origin and the capture API never has to
learn about CORS.

## Layout

```
server.mjs              static files + /api proxy  (zero dependencies)
public/
  index.html            the shell
  style.css             the whole visual language
  app.mjs               routing, loading, the run page
  lib/dom.mjs           h() and mount(); not a framework
  lib/api.mjs           the /v1 calls, and a bounded fetch pool
  lib/format.mjs        formatting, and the four kinds
  lib/diagnose.mjs      what is wrong with this record, one word per problem
  components/           strip, sidebar, grid, table, drawer, path, compare, tree, calls
test/                   node --test; renders the real views against real payloads
```

## Tests

```bash
node --test test/*.test.mjs
```

`test/dom-shim.mjs` is the smallest `document` that lets the real render
functions run outside a browser — the views themselves, not mocks of them. The
fixtures in `test/fixtures/` are verbatim `/v1` responses from a record of a
real GPU run, including a trajectory with a fork and a replayed assistant turn.
Refresh them against a running API:

```bash
node test/capture-fixtures.mjs http://127.0.0.1:8601 [http://text-mode-api]
```

## What it shows, and what it deliberately does not

[`docs/viewer.md`](../docs/viewer.md) is the reference. The short version: four block kinds rather than a `trainable` boolean, every
block carrying its token range, the mask drawn as a shape before it is read as
text, forks compared from their divergence point, and no training metrics —
reward curves belong to the trainer, and the viewer's job stops at *is this
record right*.
