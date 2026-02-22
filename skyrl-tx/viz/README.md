# SkyRL Dashboard

Local visualization dashboard for SkyRL training runs.

## Setup

```bash
cd skyrl-tx/viz
bun install
bun dev
```

Dashboard runs at http://localhost:3003

## Architecture

- Next.js App Router frontend
- Reads from `tinker.db` (SQLite)
- Tables: `training_runs`, `training_steps`, `sessions`, `models`, `futures`, `checkpoints`

## Environment Variables

- `TINKER_DB_PATH` - Path to tinker database (default: `./tx/tinker/tinker.db`)
