"""DeepSeek-V4.1 Engram: n-gram hash lookups gated into the mHC residual stream.

Parity reference: ``inference/engram.py`` (hashing) and ``Engram`` / ``ParallelEngramEmbedding`` in
``inference/model.py`` of the DeepSeek-V4.1-Flash-1017 checkpoint.

The hash tables (``layers.N.engram.embed.{weight,scale}``: ~384M rows x 256 FP8 + ue8m0 scales per 32,
~101 GB per layer) are frozen: they are plain tensors, not parameters, so they get no gradient,
optimizer state, checkpoint or weight sync (vLLM keeps its own copy). Each rank holds a contiguous row
shard over a node-local group and loads it straight from the HF safetensors on first use. Ranks see
different tokens (data parallel), so a lookup all-gathers the group's hash ids, dequantizes the local
rows, and reduce-scatters the rows back (every row is non-zero on exactly one rank, so the sum is exact).

The trainable parts are ``linear_wkv`` (hashed rows -> one key per hc stream plus a shared value) and the
``q_weight`` / ``k_weight`` gate vectors.
"""

import json
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist
from megatron.core.transformer.module import MegatronModule

_ENGRAM_GROUPS = {}
# Where each rank keeps its table shard: "gpu" (default) or "cpu" (pinned host memory, ~25 GiB/GPU saved for the
# two full-model tables; each lookup then gathers its rows on the host, a few hundred MB per microbatch).
ENGRAM_TABLE_DEVICE = os.environ.get("SKYRL_DSV41_ENGRAM_TABLE_DEVICE", "gpu")


def _find_next_prime(start: int, seen: set) -> int:
    from sympy import isprime

    candidate = start + 1
    while not isprime(candidate) or candidate in seen:
        candidate += 1
    return candidate


def build_compressed_token_map(tokenizer_path: str):
    """Reference ``build_compressed_token_map``: token ids that normalize alike share an id."""
    from tokenizers import Regex, Tokenizer, normalizers

    backend = Tokenizer.from_file(os.path.join(tokenizer_path, "tokenizer.json"))
    sentinel = ""
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )
    key_to_new, lookup = {}, []
    for token_id in range(backend.get_vocab_size(with_added_tokens=True)):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "�" in text:
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        lookup.append(key_to_new.setdefault(key, len(key_to_new)))
    return lookup, len(key_to_new)


def compute_hash_multipliers(layer_ids, max_ngram_size: int, vocab_size: int) -> torch.Tensor:
    max_long = np.iinfo(np.int64).max
    bound = max(1, (max_long // vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        values = np.random.default_rng(10007 * layer_id).integers(0, bound, size=(max_ngram_size,), dtype=np.int64)
        rows.append(torch.tensor(values * 2 + 1))
    return torch.stack(rows)


@dataclass(frozen=True)
class EngramSpec:
    layer_ids: tuple
    num_embeddings: tuple
    max_ngram_size: int
    vocab_size: int
    n_heads: int
    head_dim: int
    pad_token_id: int
    compressed_vocab_size: int
    tokenizer_path: str
    hf_path: str

    @property
    def n_hash_cols(self) -> int:
        return (self.max_ngram_size - 1) * self.n_heads


class NgramHasher:
    """Reference ``NgramHashState`` for full sequences (no decode cache): ``[b, s]`` token ids to
    ``[b, s, n_engram_layers, n_hash_cols]`` table rows."""

    def __init__(self, spec: EngramSpec):
        primes, seen = [], set()
        for _ in spec.layer_ids:
            per_ngram = []
            for _ in range(spec.max_ngram_size - 1):
                sizes, current = [], spec.vocab_size - 1
                for _ in range(spec.n_heads):
                    current = _find_next_prime(current, seen)
                    seen.add(current)
                    sizes.append(current)
                per_ngram.append(sizes)
            primes.append(per_ngram)
        token_map, vocab = build_compressed_token_map(spec.tokenizer_path)
        assert vocab == spec.compressed_vocab_size, (vocab, spec.compressed_vocab_size)
        flat = [[p for per_ngram in layer for p in per_ngram] for layer in primes]
        self.max_ngram_size = spec.max_ngram_size
        self.primes = torch.tensor(primes)  # [L, n-1, heads]
        self.offsets = torch.tensor(np.array([np.cumsum([0, *sizes[:-1]]) for sizes in flat]))  # [L, cols]
        self.multipliers = compute_hash_multipliers(spec.layer_ids, spec.max_ngram_size, vocab)  # [L, n]
        self.token_map = torch.tensor(token_map)
        self.pad_id = token_map[spec.pad_token_id]

    def _to(self, device):
        if self.token_map.device != device:
            for name in ("primes", "offsets", "multipliers", "token_map"):
                setattr(self, name, getattr(self, name).to(device))

    @torch.no_grad()
    def __call__(self, input_ids: torch.Tensor) -> torch.Tensor:
        self._to(input_ids.device)
        compressed = self.token_map[input_ids]
        seqlen = input_ids.size(1)
        positions = torch.arange(seqlen, device=input_ids.device)
        tokens = []
        for shift in range(self.max_ngram_size):
            shifted = torch.roll(compressed, shifts=shift, dims=1)
            tokens.append(torch.where(positions < shift, self.pad_id, shifted))
        tokens = torch.stack(tokens, dim=-1)  # [b, s, n]
        products = tokens.unsqueeze(2) * self.multipliers  # [b, s, L, n]
        rolling, hashes = products[..., 0], []
        for i in range(1, self.max_ngram_size):
            rolling = torch.bitwise_xor(rolling, products[..., i])
            hashes.append(rolling.unsqueeze(-1) % self.primes[:, i - 1])
        return torch.cat(hashes, dim=-1) + self.offsets


def engram_group(group_size: int):
    """A node-local process group per ``group_size`` consecutive ranks (built collectively once)."""
    world = dist.get_world_size()
    group_size = min(group_size, world)
    if group_size not in _ENGRAM_GROUPS:
        assert world % group_size == 0, (world, group_size)
        mine = None
        for start in range(0, world, group_size):
            group = dist.new_group(list(range(start, start + group_size)))
            if start <= dist.get_rank() < start + group_size:
                mine = group
        _ENGRAM_GROUPS[group_size] = mine
    return _ENGRAM_GROUPS[group_size]


class DSv41Engram(MegatronModule):
    """``x [s, b, n*C]`` -> ``x + gate * value`` per hc stream (reference ``Engram.forward``)."""

    def __init__(self, config, layer_id: int, spec: EngramSpec, group_size: int = 8):
        super().__init__(config=config)
        from megatron.core.extensions.transformer_engine import TELinear

        self.layer_id = layer_id
        self.hash_index = spec.layer_ids.index(layer_id)
        self.spec = spec
        self.hc_mult = config.mhc_num_residual_streams
        self.dim = config.hidden_size
        self.eps = config.layernorm_epsilon
        self.group_size = group_size
        self.num_embeddings = spec.num_embeddings[self.hash_index]
        self.linear_wkv = TELinear(
            spec.n_hash_cols * spec.head_dim,
            self.dim * (self.hc_mult + 1),
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
            tp_comm_buffer_name=None,
        )
        device = torch.cuda.current_device()
        self.q_weight = torch.nn.Parameter(torch.ones(self.hc_mult, self.dim, device=device, dtype=config.params_dtype))
        self.k_weight = torch.nn.Parameter(torch.ones(self.hc_mult, self.dim, device=device, dtype=config.params_dtype))
        self._table = None  # (fp8 rows [part, head_dim], e8m0 scales [part, head_dim/32], row_start)

    def _load_table(self, device):
        from safetensors import safe_open

        group = engram_group(self.group_size)
        rank, size = dist.get_rank(group), dist.get_world_size(group)
        part = (self.num_embeddings + size - 1) // size
        start, end = rank * part, min((rank + 1) * part, self.num_embeddings)
        with open(os.path.join(self.spec.hf_path, "model.safetensors.index.json")) as f:
            weight_map = json.load(f)["weight_map"]
        prefix = f"layers.{self.layer_id}.engram.embed"
        rows = {}
        for suffix in ("weight", "scale"):
            key = f"{prefix}.{suffix}"
            with safe_open(os.path.join(self.spec.hf_path, weight_map[key]), framework="pt") as f:
                part_rows = f.get_slice(key)[start:end]
                rows[suffix] = part_rows.pin_memory() if ENGRAM_TABLE_DEVICE == "cpu" else part_rows.to(device)
        self._table = (rows["weight"], rows["scale"].view(torch.uint8), start)

    @torch.no_grad()
    def _lookup(self, hash_ids: torch.Tensor) -> torch.Tensor:
        """``hash_ids [T, cols]`` (global rows) -> ``[T, cols, head_dim]`` bf16 rows."""
        if self._table is None:
            self._load_table(hash_ids.device)
        weight, scale, start = self._table
        group = engram_group(self.group_size)
        size = dist.get_world_size(group)
        n_tokens = torch.tensor([hash_ids.size(0)], device=hash_ids.device)
        all_n = [torch.zeros_like(n_tokens) for _ in range(size)]
        dist.all_gather(all_n, n_tokens, group=group)
        max_n = int(max(n.item() for n in all_n))
        padded = torch.zeros(max_n, hash_ids.size(1), dtype=hash_ids.dtype, device=hash_ids.device)
        padded[: hash_ids.size(0)] = hash_ids
        gathered = torch.empty(size * max_n, hash_ids.size(1), dtype=hash_ids.dtype, device=hash_ids.device)
        dist.all_gather_into_tensor(gathered, padded, group=group)
        local = gathered - start
        mask = (local < 0) | (local >= weight.size(0))
        local = local.masked_fill(mask, 0)
        block = weight.size(1) // scale.size(1)
        if weight.device.type == "cpu":
            local_cpu = local.cpu()
            rows = weight[local_cpu].pin_memory().to(hash_ids.device, non_blocking=True)
            row_scales = scale[local_cpu].pin_memory().to(hash_ids.device, non_blocking=True)
        else:
            rows, row_scales = weight[local], scale[local]
        values = rows.float().unflatten(-1, (-1, block))
        exponents = row_scales.to(torch.int32) - 127
        values = torch.ldexp(values, exponents.unsqueeze(-1).float()).flatten(-2).to(torch.bfloat16)
        values = values.masked_fill(mask.unsqueeze(-1), 0)
        out = torch.empty(max_n, *values.shape[1:], dtype=values.dtype, device=values.device)
        dist.reduce_scatter_tensor(out, values, group=group)
        return out[: hash_ids.size(0)]

    def forward(self, hidden_states: torch.Tensor, hash_ids: torch.Tensor) -> torch.Tensor:
        """hidden_states ``[s, b, n*C]``; hash_ids ``[s, b, cols]`` for this engram layer."""
        s, b, _ = hidden_states.shape
        n, dim = self.hc_mult, self.dim
        rows = self._lookup(hash_ids.reshape(s * b, -1)).view(s, b, -1)
        kv, _ = self.linear_wkv(rows)
        key, value = kv.split([n * dim, dim], dim=-1)
        key = key.float().unflatten(-1, (n, dim))
        weight = self.q_weight.float() * self.k_weight.float()
        h = hidden_states.float().view(s, b, n, dim)
        # Normalized per (token, hc stream) over dim, not jointly over the streams.
        rstd = torch.rsqrt(h.square().mean(-1) + self.eps) * torch.rsqrt(key.square().mean(-1) + self.eps)
        dot = (h * weight * key).sum(-1) * rstd * dim**-0.5
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
        out = h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)
        return out.to(hidden_states.dtype).view(s, b, n * dim)
