import { defineConfig, defineDocs } from 'fumadocs-mdx/config';
import { rehypeCodeDefaultOptions } from 'fumadocs-core/mdx-plugins';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

export const docs = defineDocs({
  dir: 'content/docs',
});

export default defineConfig({
  mdxOptions: {
    remarkPlugins: [remarkMath],
    // rehypeKatex must run before shiki to process math blocks first
    rehypePlugins: (v) => [rehypeKatex, ...v],
    rehypeCodeOptions: {
      ...rehypeCodeDefaultOptions,
      // Shiki ships no `promql` grammar; alias it to `text` so PromQL code
      // fences render (unhighlighted) instead of failing `next build`.
      langAlias: {
        ...rehypeCodeDefaultOptions.langAlias,
        promql: 'text',
      },
    },
  },
});
