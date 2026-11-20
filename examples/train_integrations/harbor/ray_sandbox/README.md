# Harbor on Ray Sandbox

Run `run_harbor_gen.sh`'s generation with Harbor's `modal` environment served by
[Ray Sandbox](https://github.com/ray-project/ray/tree/master/python/ray/experimental/sandbox)'s
Modal-compatible gRPC API instead of Modal's cloud. Harbor and the Modal SDK
(1.5.5, SkyRL's lock) are unchanged; only `MODAL_SERVER_URL` and the token differ.

Experimental: the images overlay `ray.experimental.sandbox` from Ray master,
whose gRPC facade (ray-project/ray#66820) isn't in a Ray release yet.

## Layouts

| | Colocated | Service |
| --- | --- | --- |
| Sandboxes run | In the job's own Ray cluster, next to vLLM | In a separate Anyscale service |
| `MODAL_SERVER_URL` | `http://127.0.0.1:50051`, a facade on every node's loopback | `https://<service URL>` |
| `MODAL_TOKEN_SECRET` | Any value | The service's `RAY_SANDBOX_API_TOKEN` |
| The GPU cluster needs | gVisor's pod spec on nodes that host sandboxes | Nothing extra |

Pick colocated when one job owns its sandboxes and its nodes can run gVisor.
Pick the service to share sandboxes across jobs, scale them apart from GPUs, or
keep gVisor's privileges off the GPU cluster.

## Run

From the SkyRL checkout root:

```bash
# 1. Images: the job image (SkyRL's image + gVisor + the sandbox package) and,
#    for the service layout, the service image. Put the printed name:version
#    in image_uri.
anyscale image build -f examples/train_integrations/harbor/ray_sandbox/Containerfile \
  -n skyrl-harbor-ray-sandbox -r 2.57.0
anyscale image build -f examples/train_integrations/harbor/ray_sandbox/service.Containerfile \
  -n ray-sandbox-modal-envoy -r nightly

# 2. Tasks: CodeContests ships Dockerfiles, and the facade builds no images, so
#    point each task at a registry image. Writing them inside the checkout ships
#    them to every node with the job's working_dir.
curl -LO https://huggingface.co/datasets/open-thoughts/CodeContests/resolve/main/tasks.parquet
uv run --no-project --with pyarrow python examples/train_integrations/harbor/ray_sandbox/prepare_tasks.py \
  --parquet tasks.parquet --out examples/train_integrations/harbor/ray_sandbox/tasks --count 10

# 3a. Colocated.
anyscale job submit -f examples/train_integrations/harbor/ray_sandbox/job-colocated.yaml

# 3b. Service: deploy it, then point a GPU job at it.
anyscale service deploy -f examples/train_integrations/harbor/ray_sandbox/service.yaml \
  --env RAY_SANDBOX_API_TOKEN="$RAY_SANDBOX_API_TOKEN"
anyscale job submit -f examples/train_integrations/harbor/ray_sandbox/job-service.yaml \
  --env MODAL_SERVER_URL=https://<service URL> --env MODAL_TOKEN_SECRET="$RAY_SANDBOX_API_TOKEN"
```

Fill in `cloud` and `cloud_resource` in the YAMLs; the instance types are those
of the CoreWeave resource the runs below used. Outside Anyscale, run
`run_harbor_gen_ray_sandbox.sh` with `MODE=colocated` or `MODE=service` on a
cluster with the same image and pod spec.

- **Colocated, multi-node:** add worker groups with the same pod spec, set the
  head to `resources: {CPU: 0}`, and pass `--env EXPECTED_NODES=<nodes>` so
  `node_facades.py` starts a facade on each. Facades start only on nodes alive
  at launch, so keep the cluster fixed-size.
- **Service edge:** it must be Envoy (Envoy Gateway). The SDK's command-router
  connection does TLS without ALPN, which ingress-nginx refuses. Before
  deploying, apply this policy to the Gateway in the cloud resource's namespace;
  otherwise Envoy's 15 s default cuts off commands that stay silent longer, such
  as verifier runs:

  ```yaml
  apiVersion: gateway.envoyproxy.io/v1alpha1
  kind: BackendTrafficPolicy
  metadata:
    name: streaming-no-request-timeout
    namespace: <the cloud resource's namespace>
  spec:
    targetRefs:
    - {group: gateway.networking.k8s.io, kind: Gateway, name: gateway}
    timeout:
      http: {requestTimeout: 0s, maxStreamDuration: 0s, connectionIdleTimeout: 1h}
  ```

## Results (2026-10-11)

SkyRL `b4b10f48`, Harbor 0.13.1, Qwen3-8B on 4 vLLM engines (RTX PRO 6000),
with #66821's facade (the benchmark below found master equivalent for Harbor).
Steps of N concurrent trials (`NUM_SAMPLES_TO_TEST` raised past 10), capped at
8 turns past the first step. 975 Harbor trials in total; **none failed on a
sandbox**. Most ended with `ContextLengthExceededError`: `run_harbor_gen.sh`'s
8192-token context against 4096-token turns, the same in every layout.

| Run | Sandbox capacity | Trials per step | Peak live sandboxes | Sandbox start p50 / p90 at the largest step | Agent p50 | Step time |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Colocated, 1 node | GPU node, 60 CPU | 10, 16 | 16 | 0.7 / 0.8 s | 85, 90 s | 219, 222 s |
| Colocated, 3 nodes | GPU + 2 CPU workers, 180 CPU | 16, 64, 128 | 128 | 0.9 / 1.4 s | 89, 110, 146 s | 203, 272, 368 s |
| Colocated, 1 node, `cpu_enforcement_policy=limit` | GPU node, 0.125 CPU per sandbox | 128 | 128 | 1.4 / 2.6 s | 146 s | 383 s |
| Service, 1 node | 24 CPU | 10, 16 | 16 | 1.5 / 1.6 s | 88, 97 s | 216, 226 s |
| Service, 9 workers (`service.yaml`) | 144 CPU, 2 facade replicas | 16, 64, 128 | 128 | 1.6 / 1.8 s | 90, 109, 144 s | 218, 281, 372 s |

- **The service matches colocated** step for step. It adds about 0.7 s per
  sandbox start for TLS and the extra hops. During the 9-worker run its two
  replicas answered 39,203 calls without an error.
- **At 128 trials the vLLM engines are the bottleneck.** Median agent time rose
  from about 88 s to 145 s in every layout, while sandbox start and verifier
  time (5.5–7 s) held steady.
- **A node's first sandboxes wait for the image pull** (4–5 s); later ones
  start in about a second.

End to end at 32k context (2026-10-13, Ray master, `MAX_MODEL_LEN=32768`, 3
nodes, 128 trials of 8 turns): 127 trials reached the verifier and 35–37
passed, with no sandbox errors. Sandbox start was p50 0.7 s, p90 2.3–3.3 s;
the agent took p50 about 290 s, nearly all of it the LLM.

### Sandbox benchmark (2026-10-13)

Same Harbor trials with no LLM: an agent that drives the sandbox exactly as
terminus-2 does (tmux install, then send-keys and pane captures each turn) with
a fixed 8 s think time per turn, then the real verifier. Colocated on 3 × 60-CPU
workers; the service on 5 × 16-CPU workers, one facade replica. Every row ran
on Ray master; "spread" rows add a one-line patch that creates sandbox actors
with `scheduling_strategy="SPREAD"`.

| Setup | Trials | Trials/min | Sandbox start p50 / p90 | Agent sandbox overhead p50 |
| --- | ---: | ---: | ---: | ---: |
| Colocated, 1 CPU per sandbox | 128 | 74 | 0.7 / 2.1 s | 4.2 s |
| Colocated, 0.125 CPU (`limit`) | 512 | 193 | 4.8 / 27.5 s | 5.4 s |
| Colocated, 0.125 CPU, sandboxes spread over nodes | 512 | 277 | 1.0 / 1.0 s | 4.6 s |
| Service, 0.125 CPU | 128 | 40 | 4.9 / 6.3 s | 8.4 s |
| Service, 0.125 CPU, spread + `GOMAXPROCS=2` | 128 | 69 | 1.5 / 3.5 s | 5.1 s |

- **Placement decides throughput at high density.** Ray's default placement
  packs a node to about half its CPU before using the next, so 0.125-CPU
  sandboxes pile up: 240 on each of two nodes at 512, which saturated them.
  Spreading sandboxes evenly took 512 trials from 193 to 277 a minute. Ray
  Sandbox doesn't spread named sandboxes yet; until it does, keep
  `override_cpus` realistic (1 CPU placed evenly here) or add nodes.
- **The service is close to colocated once its nodes aren't overpacked**
  (69 vs 74 trials a minute). Each sandbox command costs about 110 ms through
  the service edge against about 20 ms on loopback.
- **#66821 adds nothing for Harbor.** Its burst work targets unnamed
  sandboxes; at 512 named ones it was slower to start them (p50 10.3 s vs
  4.8 s). `GOMAXPROCS=2` for runsc, which gVisor's own shim uses, cut sandbox
  start p90 by 19% at 512 and was neutral or slightly worse elsewhere; the
  configs set it.
- **Per-trial package installs are the main sandbox load.** terminus-2's tmux
  install took 3 s per trial idle and up to 47 s on saturated nodes, and the
  CodeContests verifier installs curl, uv and pytest every run. Baking them
  into the task image removes that work. Two of the benchmark's 6,672 trials
  failed: verifiers that hung until Harbor's 720 s timeout during those
  downloads.

## Concurrency limits

- **CPU sets capacity.** Each sandbox reserves its CPU request in Ray, and
  Harbor's `override_cpus` must be an integer (default 1). So capacity is the
  CPUs available to sandboxes ÷ `override_cpus`; past that, sandboxes wait for
  room until Harbor's 600 s start timeout. `cpu_enforcement_policy=limit`
  requests 0.125 CPU instead: 8× the sandboxes per CPU.
- **Placement fills the creating node first.** Ray keeps sandboxes on the
  facade's node until it's about half busy, then spreads them. Colocated, that's
  the generator's node. For terminus-2 on CodeContests, more than about 2
  sandboxes per CPU on a node slows installs and verifiers sharply.
- **SkyRL drives every trial from one process**, starting at most
  `trajectories_per_second` (5) a second. Keep `max_concurrency` at or below
  sandbox capacity.
- **Limits aren't enforced as cgroups** on these CoreWeave pods
  (`RAY_SANDBOX_IGNORE_CGROUPS=1`); Ray's reservation is the only control.
- **Sandboxes are detached** and end at their TTL, so the script sets
  `sandbox_timeout_secs=3600`. Sandboxes with network access (terminus-2 and
  the verifier both install packages) can reach cluster-internal endpoints.
