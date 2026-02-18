import { source } from '@/lib/source';
import { DocsLayout } from 'fumadocs-ui/layouts/docs';
import { baseOptions } from '@/lib/layout.shared';
import { sortPageTree } from '@/lib/sort-tree';

export default function Layout({ children }: LayoutProps<'/docs'>) {
  const tree = sortPageTree(source.getPageTree());
  tree.children.push({
    type: 'separator',
  });
  tree.children.push({
    type: 'page',
    name: 'API Reference',
    url: '/api-ref/',
    external: true,
  });
  return (
    <DocsLayout tree={tree} {...baseOptions()}>
      {children}
    </DocsLayout>
  );
}
