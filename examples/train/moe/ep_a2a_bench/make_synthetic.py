"""Write a synthetic dumped batch shaped like the Qwen3.8 2k/8k DAPO step, to test moe_imbalance.py.

Lengths: prompt ~191 tokens, responses uniform 300-8192 with ~1/3 at the cap, in prompt groups of 8
(like DAPO). Routes: Zipf-skewed expert popularity per layer, so B is non-zero.
"""

import sys

import numpy as np
import torch

from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.backends.skyrl_train.utils.packed_tensor import PackedTensor

rng = np.random.default_rng(0)
prompts, n_per, layers, topk, experts = 32, 8, 4, 10, 512
lengths = []
for _ in range(prompts):
    for _ in range(n_per):
        resp = 8192 if rng.random() < 0.34 else int(rng.integers(300, 8192))
        lengths.append(191 + resp)
lengths = np.array(lengths)
max_len = int(lengths.max())
mask = torch.zeros((len(lengths), max_len), dtype=torch.int64)
for i, L in enumerate(lengths):
    mask[i, :L] = 1

pop = 1.0 / np.arange(1, experts + 1) ** 0.6
segs = []
for L in lengths:
    rows = np.empty((L, layers, topk), dtype=np.int16)
    for l in range(layers):
        perm = np.random.default_rng(l).permutation(experts)  # each layer has its own hot experts
        p = pop[np.argsort(perm)]
        p = p / p.sum()
        for t in range(0, L, 512):  # Gumbel top-k: top-k without replacement, vectorized
            k = min(512, L - t)
            keys = np.log(p)[None, :] + rng.gumbel(size=(k, experts))
            rows[t : t + k, l] = np.argpartition(-keys, topk, axis=1)[:, :topk]
    segs.append(torch.from_numpy(rows))

batch = TrainingInputBatch({"attention_mask": mask, "rollout_expert_indices": PackedTensor.from_segments(segs)})
batch.metadata = {"policy_mini_batch_boundaries": [(0, 128), (128, 256)]}
batch.save(sys.argv[1])
print("wrote", sys.argv[1], "samples", len(lengths), "tokens", int(lengths.sum()))
