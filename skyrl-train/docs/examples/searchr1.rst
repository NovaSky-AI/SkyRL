Reproducing SearchR1 with SkyRL
=====================================================

In this example, we walk through how to reproduce results from `Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning <https://arxiv.org/pdf/2503.09516>`_.

Specifically, we show convergence with Qwen2.5-3B-Instruct and GRPO (with VLLM async rollouts).

Task Overview
-------------

Attribution
-------------
We thank the authors of Search-R1 for their work: `paper <https://arxiv.org/pdf/2503.09516>`_, `code <https://github.com/PeterGriffinJin/Search-R1>`_.
Additionally we thank the SGLang + Verl team for their work reproducing Search-R1 in Verl, which we use to validate our results: `doc <https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/multi-turn/tool_examples/verl-multiturn-searchR1-like.md>`_, 
`wandb <https://wandb.ai/lingchang-ustc/search_async_rl/runs/21rubwvs/workspace?nw=nwuserlingchang>`_, and `PR <https://github.com/volcengine/verl/pull/1682>`_.







