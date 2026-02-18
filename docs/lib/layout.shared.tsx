import type { BaseLayoutProps } from 'fumadocs-ui/layouts/shared';

export function baseOptions(): BaseLayoutProps {
  return {
    nav: {
      title: 'SkyRL',
    },
    links: [
      {
        text: 'API Reference',
        url: '/api-ref/',
        active: 'nested-url',
      },
    ],
  };
}
