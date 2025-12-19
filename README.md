<div align="center">

# SkyRL: A Modular Full-stack RL Library for LLMs

<p align="center">
| <a href="https://skyrl.readthedocs.io/en/latest/"><b>Documentation</b></a> | <a href="https://x.com/NovaSkyAI"><b>Twitter/X</b></a> | <a href="https://huggingface.co/NovaSky-AI"><b>Huggingface</b></a> | <a href="https://join.slack.com/t/skyrl/shared_invite/zt-3f6ncn5b8-QawzK3uks6ka3KWoLwsi5Q"><b>Slack Workspace</b></a> |
</p>

</div>

---

# Overview

SkyRL is a full-stack RL library that provides the following components:

- [`skyrl-agent`](./skyrl-agent): Our agent layer for training long-horizon, real-world agents. For exact reproduction of [SkyRL-v0](https://novasky-ai.notion.site/skyrl-v0) results, please checkout to commit a0d50c482436af7fac8caffa4533616a78431d66.
- [`skyrl-train`](./skyrl-train): Our modular, performant training framework for RL.
- [`skyrl-gym`](./skyrl-gym): Our gymnasium of tool-use tasks, including a library of math, coding, search and SQL environments implemented in the Gymnasium API.
- (**EXPERIMENTAL**) [`skyrl-tx`](./skyrl-tx): A cross-platform library to enable users to expose a local [Tinker](https://thinkingmachines.ai/tinker/)-like REST API for model post-training.

# Getting Started

For a guide on developing with SkyRL, take at look at our [Development Guide](https://skyrl.readthedocs.io/en/latest/getting-started/development.html) docs.

For model training, checkout [`skyrl-train`](./skyrl-train) to start using, modifying, or building on top of the SkyRL training stack. See our [quickstart docs](https://skyrl.readthedocs.io/en/latest/index.html) to ramp up!

For building environments, checkout [`skyrl-gym`](./skyrl-gym) to integrate your task in the simple gymnasium interface.

For agentic pipelines, check out [`skyrl-agent`](./skyrl-agent) for our work on optimizing and scaling pipelines for multi-turn tool use LLMs on long-horizon, real-environment tasks.


# News
- **[2025/11/26]** 🎉 We released SkyRL-Agent: An agent layer for efficient, multi-turn, long-horizon agent training and evaluation. [[Paper](https://arxiv.org/pdf/2511.16108)]
- **[2025/10/06]** 🎉 We released SkyRL tx: An open implementation of a backend for the Tinker API to run a Tinker-like service on their own hardware. [[Blog](https://novasky-ai.notion.site/skyrl-tx)]
- **[2025/06/26]** 🎉 We released SkyRL-v0.1: A highly-modular, performant RL training framework. [[Blog](https://novasky-ai.notion.site/skyrl-v01)]
- **[2025/06/26]** 🎉 We released SkyRL-Gym: A library of RL environments for LLMs implemented with the Gymnasium API. [[Blog](https://novasky-ai.notion.site/skyrl-v01)]
- **[2025/05/20]** 🎉 We released SkyRL-SQL: a multi-turn RL training pipeline for Text-to-SQL, along with SkyRL-SQL-7B — a model trained on just 653 samples that outperforms both GPT-4o and o4-mini!
- **[2025/05/06]** 🎉 We released SkyRL-v0: our open RL training pipeline for multi-turn tool use LLMs, optimized for long-horizon, real-environment tasks like SWE-Bench!

# Links
- 📜 [SkyRL-Agent Paper](https://arxiv.org/pdf/2511.16108)
- 📜 [On-Policy Distillation on SkyRL Blog Post](https://novasky-ai.notion.site/on-policy-distillation)
- 📜 [Search-R1 on SkyRL Blog Post](https://novasky-ai.notion.site/skyrl-searchr1)
- 📜 [SkyRL-v0.1 Blog Post](https://novasky-ai.notion.site/skyrl-v01)
- 📜 [SkyRL-SQL Blog Post](https://novasky-ai.notion.site/skyrl-sql)
- 📜 [SkyRL-v0 Blog Post](https://novasky-ai.notion.site/skyrl-v0)

# Projects using SkyRL
- [Biomni-R0](https://biomni.stanford.edu/blog/biomni-r0-technical-report/): Using RL to Hill-Climb Biomedical Reasoning Agents to Expert-Level ![GitHub Repo stars](https://img.shields.io/github/stars/snap-stanford/Biomni)
- [How to Train Your Advisor](https://github.com/az1326/advisor-models): Steering Black-Box LLMs with Advisor Models ![GitHub Repo stars](https://img.shields.io/github/stars/az1326/advisor-models)
- [OpenThoughts-Agent](https://github.com/open-thoughts/OpenThoughts-Agent): Data recipes and robust infrastructure for training AI agents ![GitHub Repo stars](https://img.shields.io/github/stars/open-thoughts/OpenThoughts-Agent)

# Acknowledgement

This work is done at [**Berkeley Sky Computing Lab**](https://sky.cs.berkeley.edu/) in collaboration with [**Anyscale**](https://www.anyscale.com/), with generous compute support from [**Anyscale**](https://www.anyscale.com/), [**Databricks**](https://www.databricks.com/), [**NVIDIA**](https://developer.nvidia.com/brev), [**Lambda Labs**](https://lambdalabs.com/service/gpu-cloud?srsltid=AfmBOop5FnmEFTkavVtdZDsLWvHWNg6peXtat-OXJ9MW5GMNsk756PE5), [**AMD**](https://www.amd.com/en.html), [**AWS**](https://aws.amazon.com/), and [**Modal**](https://modal.com/).

We adopt many lessons and code from several great projects such as [veRL](https://github.com/volcengine/verl), [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF), [Search-R1](https://github.com/PeterGriffinJin/Search-R1), [OpenReasonerZero](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero), and [NeMo-RL](https://github.com/NVIDIA-NeMo/RL). We appreciate each of these teams and their contributions to open-source research!


# Citation

If you find the work in this repository helpful, please consider citing:

```bibtex
@misc{cao2025skyrl,
  title     = {SkyRL-v0: Train Real-World Long-Horizon Agents via Reinforcement Learning},
  author    = {Shiyi Cao and Sumanth Hegde and Dacheng Li and Tyler Griggs and Shu Liu and Eric Tang and Jiayi Pan and Xingyao Wang and Akshay Malik and Graham Neubig and Kourosh Hakhamaneshi and Richard Liaw and Philipp Moritz and Matei Zaharia and Joseph E. Gonzalez and Ion Stoica},
  year      = {2025},
}
```

```bibtex
@misc{liu2025skyrlsql,
      title={SkyRL-SQL: Matching GPT-4o and o4-mini on Text2SQL with Multi-Turn RL},
      author={Shu Liu and Sumanth Hegde and Shiyi Cao and Alan Zhu and Dacheng Li and Tyler Griggs and Eric Tang and Akshay Malik and Kourosh Hakhamaneshi and Richard Liaw and Philipp Moritz and Matei Zaharia and Joseph E. Gonzalez and Ion Stoica},
      year={2025},
}
```

```bibtex
@misc{griggs2025skrylv01,
      title={Evolving SkyRL into a Highly-Modular RL Framework},
      author={Tyler Griggs and Sumanth Hegde and Eric Tang and Shu Liu and Shiyi Cao and Dacheng Li and Charlie Ruan and Philipp Moritz and Kourosh Hakhamaneshi and Richard Liaw and Akshay Malik and Matei Zaharia and Joseph E. Gonzalez and Ion Stoica},
      year={2025},
      note={Notion Blog}
}
```
