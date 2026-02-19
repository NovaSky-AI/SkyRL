# SkyRL Documentation

The SkyRL documentation site combines two systems:

- **[Fumadocs](https://fumadocs.dev/) + Next.js** for the main user guide (`/docs`)
- **[MkDocs](https://www.mkdocs.org/) + [mkdocstrings](https://mkdocstrings.github.io/)** for the API reference (`/api-ref`)

Both are served as a single deployment on Vercel.

## Project Structure

```
docs/
├── app/                    # Next.js app routes (fumadocs)
├── content/docs/           # User guide content (.mdx files)
├── lib/                    # Shared layout config
├── mkdocs/                 # MkDocs API reference source
│   ├── mkdocs.yaml         # MkDocs configuration
│   ├── content/            # API reference markdown files
│   │   ├── index.md        # API reference landing page
│   │   ├── data.md         # Data interface API
│   │   ├── generator.md    # Generator API
│   │   ├── trainer.md      # Trainer API
│   │   ├── ...
│   │   └── stylesheets/    # Custom CSS for MkDocs theme
│   └── overrides/          # MkDocs Material theme overrides
├── public/                 # Static assets
│   └── api-ref/            # (generated) MkDocs build output
├── middleware.js            # Routes /api-ref/* to MkDocs static files
├── next.config.mjs          # Next.js configuration
├── package.json
└── vercel.json             # Vercel deployment config
```

## Prerequisites

- **Node.js** >= 18
- **[uv](https://docs.astral.sh/uv/)** (Python package manager) for building API reference
- Python packages are installed automatically via `uv sync`

## Local Development

### Fumadocs (User Guide) Only

```bash
cd docs
npm install
npm run dev
```

This starts the Next.js dev server at `http://localhost:3000`. The API reference at `/api-ref` won't be available unless you build it separately.

### Build API Reference Locally

```bash
cd docs
npm run build:api-ref
```

This runs `uv sync --extra docs` in `skyrl-train/` and then builds the MkDocs API reference into `docs/public/api-ref/`. You can then access it via the dev server.

### Full Production Build

```bash
cd docs
npm install
npm run build
npm start
```

This builds both the API reference (MkDocs) and the main site (Next.js), then starts the production server.

## Deployment

Deployed on Vercel at [docs.skyrl.ai](https://docs.skyrl.ai). The Vercel build command (in `vercel.json`) installs `uv`, then runs `npm run build` which chains:

1. `uv sync --extra docs` (install Python deps in `skyrl-train/`)
2. `mkdocs build` (generate API reference HTML into `public/api-ref/`)
3. `next build` (build the fumadocs site, which includes the static API reference)

## Adding Documentation

### User Guide Pages

1. Create a `.mdx` file in `content/docs/`
2. Add frontmatter:
   ```mdx
   ---
   title: Your Page Title
   description: A brief description
   ---

   Your content here...
   ```
3. Update `content/docs/meta.json` if needed for navigation ordering

### API Reference Pages

1. Create a `.md` file in `mkdocs/content/`
2. Use mkdocstrings directives to auto-generate from docstrings:
   ```markdown
   # Module Name

   ::: skyrl_train.module_name
       options:
         show_root_heading: true
         members_order: source
   ```
3. Add the page to the `nav` section in `mkdocs/mkdocs.yaml`
