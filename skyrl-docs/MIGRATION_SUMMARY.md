# SkyRL Documentation Migration Summary

**Date:** January 24, 2026
**Migration:** Sphinx (RST) → fumadocs (MDX)

## Overview

Successfully migrated SkyRL documentation from Sphinx/ReadTheDocs to fumadocs/Vercel.

## What Was Migrated

### Content (36 files converted)
- ✅ Getting Started (4 pages)
- ✅ Tutorials (5 pages)
- ✅ Examples (11 pages)
- ✅ Platforms (4 pages)
- ✅ Recipes (3 pages)
- ✅ Algorithms (2 pages)
- ✅ Configuration (2 pages)
- ✅ Checkpointing & Logging (1 page)
- ✅ Datasets (1 page)
- ✅ SkyAgent (1 page)
- ✅ Troubleshooting (1 page)
- ✅ Main index page

### Assets
- ✅ 13 images migrated to `/public/images/`
- ✅ Images organized by section (tutorials, examples, getting-started, skyagent)

### Not Migrated
- ❌ API Reference documentation (`/api/` directory)
  - Reason: Using manual narrative docs instead of auto-generated API docs for better UX
  - Can be added later if needed

## Technical Stack

**Before:**
- Sphinx (Python documentation generator)
- RST (reStructuredText) format
- ReadTheDocs hosting
- Auto-generated API docs with autodoc

**After:**
- fumadocs v16.4.8 (React documentation framework)
- Next.js 16.1.4 (React framework)
- MDX format (Markdown + JSX)
- Vercel hosting
- Manual narrative documentation

## Features Included

✅ **Dark mode** - Built-in with fumadocs-ui
✅ **Search** - Full-text search across all docs
✅ **Syntax highlighting** - GitHub light/dark themes
✅ **Mobile responsive** - Works on all screen sizes
✅ **Fast build times** - Static site generation with Next.js
✅ **Auto navigation** - Sidebar generated from file structure
✅ **Type-safe** - TypeScript throughout

## Deployment Setup

### Repository Structure
```
SkyRL/
├── skyrl-train/
│   └── docs/           # Old Sphinx docs (keep for reference)
└── skyrl-docs/         # New fumadocs site (deploy this)
```

### Vercel Configuration
- **Root Directory:** `skyrl-docs`
- **Framework:** Next.js
- **Build Command:** `npm run build` (auto-detected)
- **Install Command:** `npm install` (auto-detected)
- **Output Directory:** `.next` (auto-detected)

### Custom Domain Setup
**Recommended:** `docs.skyrl.ai`

DNS Configuration:
```
Type: CNAME
Name: docs
Value: cname.vercel-dns.com
TTL: Auto
```

Alternative: Point `skyrl.ai` apex domain to Vercel if you want the docs at the root.

## Next Steps

1. **Commit the new docs:**
   ```bash
   git add skyrl-docs/
   git commit -m "Add fumadocs documentation site"
   git push
   ```

2. **Deploy to Vercel:**
   - Go to [vercel.com/new](https://vercel.com/new)
   - Import your GitHub repository
   - Set root directory to `skyrl-docs`
   - Deploy

3. **Configure custom domain:**
   - In Vercel project settings → Domains
   - Add `docs.skyrl.ai`
   - Update DNS with provided CNAME record

4. **Update links:**
   - Update README.md in main repo to point to new docs URL
   - Add redirect from old ReadTheDocs URL (optional)

5. **Clean up (optional):**
   - Can keep old Sphinx docs for reference
   - Remove `.readthedocs.yaml` if fully migrating

## Conversion Notes

### What Converted Automatically
- Headings (RST underlines → Markdown `#`)
- Code blocks with syntax highlighting
- Links (internal and external)
- Inline code
- Lists (ordered and unordered)
- Blockquotes (from RST notes/warnings)

### Manual Fixes Applied
- Escaped HTML-like tags (e.g., `<custom-trainer>` → `\<custom-trainer\>`)
- Fixed image directives (RST `.. image::` → MDX `![]()`)
- Fixed JSX expression parsing errors
- Escaped comparison operators in inline code

### Known Limitations
- Complex RST directives (e.g., `:linenos:`, `:caption:`) were simplified or removed
- Some cross-references may need manual adjustment
- No auto-generated API docs (by design)

## File Locations

- **Old docs:** `/home/tyler/SkyRL/skyrl-train/docs/`
- **New docs:** `/home/tyler/SkyRL/skyrl-docs/`
- **Conversion script:** `/home/tyler/SkyRL/skyrl-docs/convert_rst_to_mdx.py`
- **Images:** `/home/tyler/SkyRL/skyrl-docs/public/images/`

## Testing

Build succeeds with all 38+ pages generated:
```bash
cd skyrl-docs
npm run build
# ✓ Generated successfully
```

Dev server works:
```bash
npm run dev
# Visit http://localhost:3000
```

## Support

For issues or questions:
- fumadocs docs: https://fumadocs.dev
- Next.js docs: https://nextjs.org/docs
- Vercel docs: https://vercel.com/docs
