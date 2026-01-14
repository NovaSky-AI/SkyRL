# SkyRL Landing Page

This directory contains a Hugo-based landing page for SkyRL.

## Local Development

### Prerequisites
Install Hugo (extended version recommended):
```bash
# macOS
brew install hugo

# Or download from https://github.com/gohugoio/hugo/releases
```

### Running Locally
```bash
hugo server -D
```

Then visit http://localhost:1313/SkyRL/

### Building
```bash
hugo --gc --minify
```

The built site will be in the `public/` directory.

## Deployment

The landing page automatically deploys to GitHub Pages when you push to the main branch.

### Setup GitHub Pages (one-time)
1. Go to repository Settings → Pages
2. Under "Source", select "GitHub Actions"
3. The workflow in `.github/workflows/hugo.yml` will handle deployment

The site will be available at: https://novasky-ai.github.io/SkyRL/

## Structure

- `content/_index.md` - Landing page markdown content
- `layouts/` - Hugo templates and layouts
- `hugo.toml` - Hugo configuration
- `.github/workflows/hugo.yml` - Automated deployment workflow
