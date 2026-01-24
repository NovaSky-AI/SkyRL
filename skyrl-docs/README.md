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

### Deploying to Vercel

1. **Initial Setup:**
   - Go to [vercel.com](https://vercel.com) and sign in
   - Click "Add New Project"
   - Import your GitHub repository
   - Set the **Root Directory** to `skyrl-docs`
   - Vercel will auto-detect Next.js and configure build settings
   - Click "Deploy"

2. **Configure Custom Domain:**
   - In your Vercel project dashboard, go to "Settings" → "Domains"
   - Add `docs.skyrl.ai` as a custom domain
   - Vercel will provide DNS records to add to your domain registrar:
     - Type: `CNAME`
     - Name: `docs`
     - Value: `cname.vercel-dns.com`
   - Add this CNAME record in your domain registrar's DNS settings
   - Wait for DNS propagation (usually a few minutes)

3. **Automatic Deployments:**
   - Every push to `main` branch will automatically deploy to production
   - Pull requests will get preview deployments
   - You can configure branch deployments in Vercel settings

## Project Structure

```
skyrl-docs/
├── app/                    # Next.js app directory
│   ├── docs/              # Documentation routes
│   ├── layout.tsx         # Root layout
│   └── page.tsx           # Homepage
├── content/
│   └── docs/              # MDX documentation files
│       ├── getting-started/
│       ├── tutorials/
│       ├── examples/
│       ├── platforms/
│       ├── recipes/
│       └── ...
├── lib/
│   └── source.ts          # Documentation source configuration
├── public/
│   └── images/            # Static assets
├── next.config.mjs        # Next.js configuration
├── source.config.ts       # Fumadocs configuration
└── tailwind.config.js     # Tailwind CSS configuration
```

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

## Migrated from Sphinx

This documentation was migrated from Sphinx (RST) to fumadocs (MDX). The original Sphinx docs are in `/home/tyler/SkyRL/skyrl-train/docs/`.
