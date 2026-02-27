# SkyRL Documentation

Built with [Fumadocs](https://fumadocs.dev/) + Next.js. API reference pages are auto-generated from Python docstrings using [griffe2md](https://github.com/mkdocstrings/griffe2md).

## Project Structure

```
docs/
├── app/                    # Next.js app routes (fumadocs)
├── content/docs/           # User guide content (.mdx files)
├── lib/                    # Shared layout config
├── api-pages.yaml          # API reference page definitions
├── generate-api-docs.py    # Generates API reference .mdx from docstrings
├── vercel.json             # Vercel deployment config
├── package.json
└── public/                 # Static assets
```

## Prerequisites

- **Node.js** >= 18
- **[uv](https://docs.astral.sh/uv/)** (Python package manager) for building API reference
- Python packages are installed automatically via `uv sync`

## Local Development

### User Guide Only

```bash
cd docs
npm install
npm run dev
```

This starts the Next.js dev server at `http://localhost:3000`. API reference pages won't be available unless you generate them first.

### Generate API Reference

```bash
uv run --extra dev python docs/generate-api-docs.py
```

This generates `.mdx` files into `content/docs/api-ref/` from the definitions in `api-pages.yaml`.

### Full Production Build

```bash
cd docs
npm install
npm run build
npm start
```

## Deployment

Deployed on Vercel at [docs.skyrl.ai](https://docs.skyrl.ai). The Vercel build command (in `vercel.json`):

1. Installs `uv`
2. Runs `generate-api-docs.py` (generates API reference mdx files)
3. Runs `next build`

Generated API reference `.mdx` files are gitignored — they're built fresh on each deploy.

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

Edit `api-pages.yaml` to add pages. Each page group has a `path` and a list of pages with sections:

```yaml
- path: skyrl
  pages:
    - slug: my-page
      title: My Page
      description: Description for the page.
      sections:
        - heading: Section Name
          description: Optional section description.
          objects:
            - skyrl.module.ClassName
```

Then regenerate: `uv run --extra dev python docs/generate-api-docs.py`
