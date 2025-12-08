<div align="center">

<img alt="SkyRL-Agent" src="./assets/new-cropped.png" width="320">

Training and evaluating modern AI agents with modular tasks, tools, and backends.

[![arXiv](https://img.shields.io/badge/arXiv-2511.16108-b31b1b.svg)](https://arxiv.org/pdf/2511.16108)
[![HF Model](https://img.shields.io/badge/HuggingFace-SA--SWE--32B-orange.svg)](https://huggingface.co/NovaSky-AI/SA-SWE-32B)

<br/>
<img alt="SkyRL Agent Overview" src="./assets/skyrl-agent-svg.svg" width="85%">
<br/>

</div>

## News 📰✨

- 🚀 Initial public release with SWE, MemAgent (step-wise training), and Web Research examples!

## Why SkyRL-Agent

- Unified interface for agentic tasks and training backends
- Pluggable tools (browser, search, code execution, finish, etc.)
- Efficient and flexible async dispatching strategies
- Works with OpenAI-compatible serving (vLLM/others), VERL, SkyRL-Train, Tinker. Switch the backend with one line configuration!

<p align="center">
  <img alt="Dispatcher Flow" src="./assets/skyrl-agent-dispatch-svg.svg" width="80%">
</p>

## Quickstart

```bash
git clone --recurse-submodules https://github.com/NovaSky-AI/SkyRL.git 
# our working directory
cd skyrl-agent
```

Then head to the `examples/` folder to run tasks (training and inference). Each task’s script/YAML documents its own knobs and environment requirements.

## Results & Profiling (glimpses)

<p align="center">
  <img alt="GPU Utilization" src="./assets/gpu_util_plot.png" width="46%">
  &nbsp;&nbsp;
  <img alt="Reward Usage Mix" src="./assets/reward_usage_none_combo_plot.png" width="46%">
</p>

## Roadmap

- [ ] OSWorld Integration
- [ ] Simplify SWE agent training code path
- [ ] More training recipes
- [ ] Evaluation harness unification

## Acknowledgements
Huge thanks to these projects:
- [VERL](https://github.com/volcengine/verl)
- [OpenHands](https://github.com/OpenHands/OpenHands)
- [LocAgent](https://github.com/gersteinlab/LocAgent)
- [R2E-Gym](https://github.com/R2E-Gym/R2E-Gym)
- [Tinker Cookbook](https://github.com/thinking-machines-lab/tinker-cookbook)
- [rLLM](https://github.com/rllm-org/rllm)
- [WebThinker](https://github.com/RUC-NLPIR/WebThinker)
- [WebSailor](https://github.com/Alibaba-NLP/DeepResearch)
- [ARPO](https://github.com/dvlab-research/ARPO)
- [MemAgent](https://github.com/BytedTsinghua-SIA/MemAgent)

## Citation

```bibtex
@article{cao2025skyrl,
  title={SkyRL-Agent: Efficient RL Training for Multi-turn LLM Agent},
  author={Cao, Shiyi and Li, Dacheng and Zhao, Fangzhou and Yuan, Shuo and Hegde, Sumanth R and Chen, Connor and Ruan, Charlie and Griggs, Tyler and Liu, Shu and Tang, Eric and others},
  journal={arXiv preprint arXiv:2511.16108},
  year={2025}
}
```