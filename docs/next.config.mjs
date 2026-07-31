import { createMDX } from 'fumadocs-mdx/next';

const withMDX = createMDX();

/** @type {import('next').NextConfig} */
const config = {
  reactStrictMode: true,
  async redirects() {
    return [
      {
        source: '/',
        destination: '/docs',
        permanent: true,
      },
      {
        // The standalone configuration overview page was removed; the generated
        // API reference for the config dataclasses is now the single source of truth.
        source: '/docs/configuration/config',
        destination: '/docs/api-ref/skyrl/config',
        permanent: true,
      },
    ];
  },
};

export default withMDX(config);
