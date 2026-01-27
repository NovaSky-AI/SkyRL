// Define the order of folders in the sidebar
const folderOrder = [
  'getting-started',
  'datasets',
  'tutorials',
  'examples',
  'platforms',
  'recipes',
  'algorithms',
  'configuration',
  'checkpointing-logging',
  'troubleshooting',
  'skyagent',
];

// Define the order of pages within each folder
const pageOrder: Record<string, string[]> = {
  'getting-started': ['installation', 'quickstart', 'overview', 'development'],
  'datasets': ['dataset-preparation'],
  'tutorials': ['new_env', 'one_step_off_async', 'fully_async', 'tools_guide', 'skyrl_gym_generator'],
  'examples': ['megatron', 'ppo', 'lora', 'llm_as_a_judge', 'remote_server', 'training_backends', 'multi_turn_text2sql', 'search', 'flash_rl', 'mini_swe_agent', 'openenv'],
  'platforms': ['overview', 'anyscale', 'runpod', 'skypilot'],
  'recipes': ['overview', 'skyrl-sql', 'searchr1'],
  'algorithms': ['dapo', 'custom_algorithms'],
  'configuration': ['config', 'placement'],
  'checkpointing-logging': ['checkpointing'],
  'troubleshooting': ['troubleshooting'],
  'skyagent': ['agent-overview'],
};

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function sortPageTree(tree: any): any {
  if (!tree.children) return tree;

  const sortedChildren = [...tree.children].sort((a: any, b: any) => {
    const aName = a.type === 'folder' ? a.name : '';
    const bName = b.type === 'folder' ? b.name : '';
    const aIndex = folderOrder.indexOf(aName);
    const bIndex = folderOrder.indexOf(bName);
    if (aIndex === -1 && bIndex === -1) return 0;
    if (aIndex === -1) return 1;
    if (bIndex === -1) return -1;
    return aIndex - bIndex;
  });

  // Sort pages within folders
  for (const item of sortedChildren) {
    if (item.type === 'folder' && item.children && pageOrder[item.name]) {
      const order = pageOrder[item.name];
      item.children.sort((a: any, b: any) => {
        const aSlug = a.type === 'page' ? a.slug : '';
        const bSlug = b.type === 'page' ? b.slug : '';
        const aIndex = order.indexOf(aSlug);
        const bIndex = order.indexOf(bSlug);
        if (aIndex === -1 && bIndex === -1) return 0;
        if (aIndex === -1) return 1;
        if (bIndex === -1) return -1;
        return aIndex - bIndex;
      });
    }
  }

  return { ...tree, children: sortedChildren };
}
