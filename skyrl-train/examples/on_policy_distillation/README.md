# On-Policy Distillation
This folder contains scripts for running On-Policy Distillation on SkyRL.

<p align="center">
  <img src="https://github.com/user-attachments/assets/f6ecc9da-cb67-4935-b95a-193ba6c4843c" width="45%" />
  <img src="https://github.com/user-attachments/assets/255c46bd-f443-4f36-b8fc-9546c3a01cf1" width="45%" />
</p>

On-Policy Distillation is a technique recently highlighted by [Thinking Machines](https://thinkingmachines.ai/blog/on-policy-distillation), which combines the benefits of on-policy RL style training, with the dense reward signal of distillation. The main idea is to collect on-policy samples from a student model, use a teacher to grade each token from the student samples, update the student policy accordingly, and repeat. On-Policy Distillation has previously been shown to be a more compute efficient approach than RL for efficiently post-training models ([Agarwal et al](https://arxiv.org/abs/2306.13649), [Gu et al](https://arxiv.org/abs/2306.08543), [Qwen3 team](https://arxiv.org/abs/2505.09388)).

In `main_on_policy_distill.py` we provide a simple example for modifying SkyRL to implement On-Policy Distillation by replacing the ref model with a teacher model, and modifying the reward/advantage computation logic to use the reverse KL loss.
<img width="471" height="51" alt="image" src="https://github.com/user-attachments/assets/4d5a9649-832a-4ba9-86af-5de063f2773f" />

## Quickstart
To get started, first set up the dataset from the DAPO example:

```bash
# Run from the `skyrl-train` directory
bash examples/algorithms/dapo/prepare_dapo_data.sh
```

Then, just make sure to set the path to your desired teacher model, and you're ready to kick off training!

```bash
TEACHER_MODEL=<YOUR_MODEL_HERE>
bash examples/on_policy_distillation/run_on_policy_distill_math_qwen3_4b.sh trainer.ref.mode.path=$TEACHER_MODEL
```
