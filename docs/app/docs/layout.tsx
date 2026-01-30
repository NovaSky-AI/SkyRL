import { source } from '@/lib/source';
import { DocsLayout } from 'fumadocs-ui/layouts/docs';
import { baseOptions } from '@/lib/layout.shared';
import { sortPageTree } from '@/lib/sort-tree';

export default function Layout({ children }: LayoutProps<'/docs'>) {
  const tree = sortPageTree(source.getPageTree());
  return (
    <DocsLayout tree={tree} {...baseOptions()}>
      {children}
    </DocsLayout>
  );
}
