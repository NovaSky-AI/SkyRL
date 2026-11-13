"""``reload`` weight sync: export HF weights, restart the rollout server on them.

The other backends push tensors into a running vLLM 0.31 engine through vLLM's
weight-transfer API. That needs a matching vLLM on both sides. ``reload`` needs
nothing from the rollout engine except "start on this checkpoint", so it works
with engines that do not speak that API (vLLM-TPU, an HF ``generate`` server,
an older vLLM). The cost is a full checkpoint write and an engine restart per
step, so it suits engines that cannot take weights in place.

One sync:

1. Every policy rank joins the full-state-dict gather (FSDP2 collective).
2. Rank 0 casts floating tensors to the inference dtype and writes HF safetensors
   to ``SKYRL_RELOAD_EXPORT_DIR/<job>/v<version>``, plus the tokenizer and processor
   files of the policy model, so a server can start from the export alone.
   ``<job>`` is unique per trainer process, and a job removes only its own exports,
   so jobs that share a host or an export dir do not touch each other's files.
3. Rank 0 publishes to every reload agent (one per rollout server), concurrently:
   * ``http`` (default): ``PUT /v1/files/<version>/<name>`` for each file, then
     ``POST /v1/reload {"version": v}``;
   * ``path``: ``POST /v1/reload {"version": v, "model_dir": <dir>}`` when the
     agent can read the export dir (same host or shared mount).
   The agent restarts its server on the new weights and answers once it is healthy.

An agent is any HTTP service that implements these two calls and owns the server process.

Configuration (environment, read on the policy workers):

* ``SKYRL_RELOAD_AGENT_URLS``: comma-separated agent URLs, one per entry of
  ``external_server_urls`` and in the same order. Default: each server URL with
  its port + 1.
* ``SKYRL_RELOAD_EXPORT_DIR`` (default ``/tmp/skyrl_reload``), ``SKYRL_RELOAD_TRANSPORT``
  (``http`` | ``path``), ``SKYRL_RELOAD_TIMEOUT_S`` (default 3600).
* ``SKYRL_RELOAD_SKIP_INITIAL`` (default ``1``): skip the sync the trainer runs before
  step 1. The servers were started on the same checkpoint the policy loads, so that
  sync would only restart them on identical weights. Not skipped after the policy is
  restored from a checkpoint (the servers then run older weights).

This module imports no vLLM.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import shutil
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from typing import List, Optional
from urllib.parse import urlsplit, urlunsplit

import torch
import torch.distributed as dist
from loguru import logger

_DTYPES = {"bfloat16": torch.bfloat16, "bf16": torch.bfloat16, "float16": torch.float16, "fp16": torch.float16}


def _default_agent_url(server_url: str) -> str:
    parts = urlsplit(server_url)
    if parts.port is None:
        raise ValueError(f"cannot derive a reload agent URL from {server_url!r}; set SKYRL_RELOAD_AGENT_URLS")
    host = f"[{parts.hostname}]" if ":" in parts.hostname else parts.hostname  # IPv6 literal
    return urlunsplit((parts.scheme, f"{host}:{parts.port + 1}", "", "", ""))


def _request(method: str, url: str, *, body=None, headers=None, timeout: float) -> dict:
    req = urllib.request.Request(url, data=body, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:2000]
        raise RuntimeError(f"{method} {url} -> HTTP {e.code}: {detail}") from e
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        raise RuntimeError(f"{method} {url} failed: {e}; is the reload agent running?") from e
    return json.loads(raw) if raw else {}


# Not tokenizer or processor files: written by ``save_pretrained`` or not needed by a server.
_NOT_PROCESSING_FILES = ("config.json", "generation_config.json")
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf", ".msgpack", ".h5")


def _processing_files(model_path: str) -> List[str]:
    """Paths of the tokenizer / processor files of ``model_path`` (a local dir or a Hub id)."""

    def wanted(name: str) -> bool:
        return (
            name not in _NOT_PROCESSING_FILES
            and not name.endswith(_WEIGHT_SUFFIXES)
            and not name.endswith(".index.json")
        )

    if not os.path.isdir(model_path):
        from huggingface_hub import snapshot_download

        # The policy already loaded this model, so the files are in the local cache.
        model_path = snapshot_download(model_path, ignore_patterns=[f"*{s}" for s in _WEIGHT_SUFFIXES])
    return [
        os.path.join(model_path, n)
        for n in sorted(os.listdir(model_path))
        if os.path.isfile(os.path.join(model_path, n)) and wanted(n)
    ]


@dataclass
class ReloadWeightSync:
    agent_urls: List[str]
    export_dir: str
    transport: str = "http"
    model_dtype: str = "bfloat16"
    timeout_s: float = 3600.0
    skip_initial: bool = True
    source_model_path: Optional[str] = None
    version: int = 0
    calls: int = 0
    last_timings: dict = field(default_factory=dict)
    _exports: List[str] = field(default_factory=list)

    @classmethod
    def from_env(
        cls, *, server_urls: List[str], model_dtype: str, source_model_path: Optional[str] = None
    ) -> "ReloadWeightSync":
        raw = os.environ.get("SKYRL_RELOAD_AGENT_URLS", "").strip()
        agents = [u.strip().rstrip("/") for u in raw.split(",") if u.strip()] if raw else []
        if not agents:
            agents = [_default_agent_url(u) for u in server_urls]
        if len(agents) != len(server_urls):
            raise ValueError(
                f"SKYRL_RELOAD_AGENT_URLS has {len(agents)} entries but there are {len(server_urls)} "
                "external_server_urls; give one agent per server, in the same order."
            )
        transport = os.environ.get("SKYRL_RELOAD_TRANSPORT", "http")
        if transport not in ("http", "path"):
            raise ValueError(f"SKYRL_RELOAD_TRANSPORT must be 'http' or 'path', got {transport!r}")
        return cls(
            agent_urls=agents,
            # A subdirectory per trainer process: concurrent jobs on one host never share exports.
            export_dir=os.path.join(
                os.environ.get("SKYRL_RELOAD_EXPORT_DIR", "/tmp/skyrl_reload"), f"job-{uuid.uuid4().hex[:12]}"
            ),
            transport=transport,
            model_dtype=model_dtype,
            timeout_s=float(os.environ.get("SKYRL_RELOAD_TIMEOUT_S", "3600")),
            skip_initial=os.environ.get("SKYRL_RELOAD_SKIP_INITIAL", "1") == "1",
            source_model_path=source_model_path,
        )

    # ------------------------------------------------------------------ export
    def _export(self, strategy, model, out_dir: str) -> None:
        """Collective: gather the full state dict; rank 0 writes HF safetensors."""
        from skyrl.backends.skyrl_train.distributed.fsdp_utils import (
            fsdp2_get_full_state_dict,
        )
        from skyrl.backends.skyrl_train.workers.model_wrapper import HFModelWrapper

        fsdp_model = model.model if isinstance(model, HFModelWrapper) else model
        state_dict = fsdp2_get_full_state_dict(fsdp_model, cpu_offload=True, rank0_only=True)
        if dist.get_rank() != 0:
            return

        dtype = _DTYPES.get(str(self.model_dtype).lower(), torch.bfloat16)
        for name, tensor in state_dict.items():
            if torch.is_floating_point(tensor) and tensor.dtype != dtype:
                state_dict[name] = tensor.to(dtype)

        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=True)
        model_to_save = strategy._unwrap_model(model)
        model_to_save.save_pretrained(out_dir, state_dict=state_dict, safe_serialization=True, max_shard_size="4GB")
        strategy._fix_fsdp_config(model_to_save.config).save_pretrained(out_dir)
        del state_dict
        # Tokenizer / processor files (e.g. Qwen3.5's preprocessor_config.json), as the
        # trainer's own HF export has, so a server can start from this directory alone.
        if self.source_model_path:
            for path in _processing_files(self.source_model_path):
                dst = os.path.join(out_dir, os.path.basename(path))
                if not os.path.exists(dst):
                    shutil.copy2(path, dst)

    # ----------------------------------------------------------------- publish
    def _publish_one(self, agent: str, out_dir: str) -> dict:
        if self.transport == "path":
            body = json.dumps({"version": self.version, "model_dir": out_dir}).encode()
            return _request(
                "POST",
                f"{agent}/v1/reload",
                body=body,
                headers={"Content-Type": "application/json"},
                timeout=self.timeout_s,
            )

        names = sorted(os.listdir(out_dir))
        for name in names:
            path = os.path.join(out_dir, name)
            size = os.path.getsize(path)
            with open(path, "rb") as f:
                # urllib streams a file object when Content-Length is set.
                _request(
                    "PUT",
                    f"{agent}/v1/files/{self.version}/{name}",
                    body=f,
                    headers={"Content-Type": "application/octet-stream", "Content-Length": str(size)},
                    timeout=self.timeout_s,
                )
        body = json.dumps({"version": self.version, "files": names}).encode()
        return _request(
            "POST",
            f"{agent}/v1/reload",
            body=body,
            headers={"Content-Type": "application/json"},
            timeout=self.timeout_s,
        )

    def send(self, strategy, model) -> None:
        """Run one sync. Must be called on every policy rank."""
        self.calls += 1
        if self.calls == 1 and self.skip_initial:
            logger.info("[reload] skipping the pre-training sync: servers already run the initial checkpoint")
            return
        self.version += 1
        out_dir = os.path.join(self.export_dir, f"v{self.version:06d}")
        t0 = time.perf_counter()
        self._export(strategy, model, out_dir)
        t_export = time.perf_counter() - t0

        if dist.get_rank() == 0:
            t1 = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.agent_urls)) as pool:
                results = list(pool.map(lambda a: self._publish_one(a, out_dir), self.agent_urls))
            t_publish = time.perf_counter() - t1
            self.last_timings = {
                "reload_export_s": t_export,
                "reload_publish_and_restart_s": t_publish,
            }
            logger.info(
                f"[reload] weight version {self.version}: export {t_export:.1f}s, "
                f"publish+restart {t_publish:.1f}s on {len(self.agent_urls)} server(s): {results}"
            )
            # Keep only this job's latest export (the agents hold their own copies); never
            # touch anything this job did not write.
            for path in self._exports:
                shutil.rmtree(path, ignore_errors=True)
            self._exports = [out_dir]
