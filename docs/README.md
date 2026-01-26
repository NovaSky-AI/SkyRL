# SkyRL Documentation

This is the documentation site for SkyRL, built with [fumadocs](https://fumadocs.dev/) and Next.js.

## Development

```bash
# Install dependencies
npm install

# Run dev server
npm run dev

# Build for production
npm run build

# Start production server
npm start
```

## Deployment

This site is deployed on Vercel at [docs.skyrl.ai](https://docs.skyrl.ai).

## Adding New Documentation

1. Create a new `.mdx` file in `content/docs/`
2. Add frontmatter with title and description:
   ```mdx
   ---
   title: Your Page Title
   description: A brief description
   ---

   # Your Page Title

   Your content here...
   ```
3. The page will automatically appear in the navigation