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
      {
        source: '/docs/getting-started/supported_models',
        destination: '/docs/models',
        permanent: true,
      },
      {
        source: '/docs/examples/beam',
        destination: '/docs/models/beam',
        permanent: true,
      },
      {
        source: '/docs/examples/kimi_k2_int4_lora',
        destination: '/docs/models/kimi',
        permanent: true,
      },
    ];
  },
};

export default withMDX(config);
