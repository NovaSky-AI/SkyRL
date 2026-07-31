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
        // Retired page; redirect to the generated config reference.
        source: '/docs/configuration/config',
        destination: '/docs/api-ref/skyrl/config',
        permanent: true,
      },
      {
        source: '/docs/configuration/placement',
        destination: '/docs/tutorials/placement',
        permanent: true,
      },
    ];
  },
};

export default withMDX(config);
