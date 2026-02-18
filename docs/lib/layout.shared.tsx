import type { BaseLayoutProps } from 'fumadocs-ui/layouts/shared';

export function baseOptions(): BaseLayoutProps {
  return {
    nav: {
      title: 'SkyRL',
    },
    githubUrl: 'https://github.com/NovaSky-AI/SkyRL',
    links: [
      {
        text: 'API Reference',
        url: '/api-ref/',
        active: 'nested-url',
      },
    ],
  };
}
