Multi-Turn RL for Search with SkyRL
=====================================================

In this example, we walk through an example for training a multi-turn search agent with Qwen2.5-3B-Instruct and GRPO (with VLLM async rollouts), using the dataset and recipe
from `Search-R1 <https://arxiv.org/pdf/2503.09516>`_.

You can find the exact commands to reproduce our results in the :doc:`../recipes/search` recipe.

Task Overview
-------------

In this task, the agent is given a natural language question and the ability to query a search engine. The agent must use the search engine to answer the question.
An example prompt is shown below:

.. code-block:: text

    You are a helpful and harmless assistant.
    
    Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. 
    After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> 
    and it will return the top searched results between <information> and </information>. You can search as many times as you want. 
    If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. 
    For example, <answer> Beijing </answer>. 

    
    Question: In what year was the company that was founded as Sound of Music added to the S&P 500?

The agent is given n turns to output an answer to the question within the <answer> and </answer> tags, meaning the agent has n - 1 turns to query the search engine by outputting a query inside the <search> and </search> tags. 
A reward of 0 is given for incorrect responses, and a reward of 1 is given for correct responses (we do not apply format rewards).

Attribution
-------------
We thank the authors of Search-R1 for their work: `paper <https://arxiv.org/pdf/2503.09516>`_, `code <https://github.com/PeterGriffinJin/Search-R1>`_.
Additionally we thank the SGLang + Verl team for their work reproducing Search-R1 in Verl, which we use to validate our results: `doc <https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/multi-turn/tool_examples/verl-multiturn-searchR1-like.md>`_, 
`wandb <https://wandb.ai/lingchang-ustc/search_async_rl/runs/21rubwvs/workspace?nw=nwuserlingchang>`_, and `PR <https://github.com/volcengine/verl/pull/1682>`_.
