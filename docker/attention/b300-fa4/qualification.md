# B300 FA2/FA4 wheel profile

This profile targets CPython 3.12, Linux x86_64, Torch 2.11.0, CUDA 13, and B300
(SM103). It is based on SkyRL commit
`b58b96e81992c9cfe9b08b1c5ed6202ce8c6432e`. The later Torch 2.13 dependency update
requires different extension binaries and is outside this profile.

The combined `flash-attn` distribution owns FA2 and every FA4 runtime file. The
`flash-attn-4` distribution contains only metadata and requires the exact combined
version. TE/Megatron detection continues to use that metadata. Never install the
original FA4 distribution into the combined environment.

## Build and check

Run from the SkyRL root. Keep artifacts outside the checkout.

```bash
uv run --isolated --no-project --python 3.12 docker/attention/b300-fa4/build_wheels.py \
  --download --cache /tmp/fa4-inputs --output /tmp/fa4-release
uv run --isolated --no-project --python 3.12 --with pytest --with tomlkit==0.13.3 --with packaging==25.0 \
  pytest tests/packaging/test_b300_fa4.py tests/test_compat.py
uv run --isolated --no-project --python 3.12 docker/attention/b300-fa4/install_smoke.py \
  --artifacts /tmp/fa4-release --report /tmp/fa4-release/linux-installation.json
```

The installation test requires Linux. On other hosts, `--cross-platform` checks
pip/uv target installations and uv target removals without loading native code;
it does not replace the native Linux test. The dedicated GitHub Actions workflow
builds twice, checks byte reproducibility within its build environment, and runs
native Linux installation tests.

## Generate the image manifest and lock

```bash
uv run --isolated --no-project docker/attention/b300-fa4/image_profile.py generate \
  --artifacts /tmp/fa4-release
uv run --isolated --no-project docker/attention/b300-fa4/image_profile.py check
```

Generation resolves against the exact local release artifacts through a temporary
loopback server, then replaces the staging URLs with the release URLs and keeps
their hashes. This permits preparing a reviewable lock before publication. The
committed lock contains no local server or file paths. The source manifest's hash
guards against applying a stale profile to a newer SkyRL dependency stack.

The generated manifest preserves all extras and relative sources, restricts the
supported platform/Python range, and removes source alternatives outside that
range. Apply it only in a disposable image checkout; `apply` replaces that
checkout's root manifest and lock. Default source installations remain unchanged.

## Publish

Publish both wheel assets, `SHA256SUMS`, and `release-manifest.json` together under
the tag specified by `inputs.json`. Attach the builder, input manifest, license
material, and validation report. Publish as a prerelease while GPU qualification
is pending. Use the recipe commit for an official release tag. A maintainer with
write access to NovaSky-AI/SkyRL must perform official publication.

```bash
uv run --isolated --no-project docker/attention/b300-fa4/image_profile.py check-release
```

This checks published bytes against the manifest. Never replace assets under an
existing release tag. Any payload or dependency change requires a coordinated
revision of both wheels. A preview mirror may host byte-identical artifacts; its
release manifest must contain its actual download URLs. Generate the preview
profile in a separate qualification branch, retaining canonical NovaSky URLs in
the review branch.

## Build and qualify with Trajectory

Use Trajectory's `build-skyrl-image --attention-profile b300-fa4` with a full SkyRL
commit SHA and unique candidate tag, or its existing remote image workflow with
`attention_profile=b300-fa4`. Candidate builds never move `latest`. Both B300
workloads run the installed-payload, dependency, CUDA, and SM103 checks at startup.

Recheck `uv run tcli get --cluster r2z1 nodes --free` and the cluster table, then
deploy one currently free eight-GPU node with `--reserved --image <digest>
--shutdown-after-seconds 7200`. Run Qwen and Nemotron sequentially using their
existing registered B300 deployment configurations. Only manage resources created
for this qualification.

Prepare the upstream reference in a separate environment inside the owned pod.
Export the candidate's frozen dependency closure without FA2, FA4, or local
projects, install it with `--no-deps`, and add the original FA4 input wheel:

```bash
uv export --frozen --extra tinker --extra megatron --no-dev --no-emit-local \
  --no-emit-package flash-attn --no-emit-package flash-attn-4 \
  --output-file /tmp/fa4-reference-requirements.txt
uv venv --python .venv/bin/python /tmp/fa4-reference
uv pip install --python /tmp/fa4-reference/bin/python --no-deps \
  --require-hashes -r /tmp/fa4-reference-requirements.txt
uv pip install --python /tmp/fa4-reference/bin/python --no-deps \
  /tmp/fa4-inputs/flash_attn_4-4.0.0b28-py3-none-any.whl
uv run --active --no-sync docker/attention/b300-fa4/gpu_smoke.py \
  --reference-python /tmp/fa4-reference/bin/python --output /tmp/fa4-results
```

The harness runs each case in a fresh subprocess with a 600-second timeout. It
checks FA2/FA4 dense and variable-length attention, causal/noncausal masks,
BF16 GQA head shapes `(24,4,256)` and `(32,2,128)`, packed boundaries
`[0,128,384]`, forward/backward numerical error, and finite optimizer updates.
It uses the error envelope in upstream `tests/cute/test_flash_attn.py`: twice
the low-precision reference error plus its arithmetic-rounding allowance.
For identical upstream FA4 code and deterministic kernels, candidate/reference
outputs and gradients must also match bit-for-bit. Cold and warm timings are
reported independently.

Run the existing two-step training experiments through XM:

- `binary_choice_training_qwen_3p6_27b_tcli_b300_attention_2step`
- `gsm8k_training_nemotron_3p5_lightning_30b_tcli_gspo_attention_2step`

Set `TCLI_SCHEDULER_SERVICE_IDS` to the owned service and launch with
`uv run xm krun --async --cpu-cluster gke --tcli-gpu-cluster r2z1 <experiment>`.
Require both optimizer steps and evaluations to pass. Preserve each model's
existing runtime configuration and Mamba compatibility; investigate integration
failures before changing image digests.

Capture source SHAs, wheel hashes, image digest, service/node identifiers, XM IDs,
backend detection, loaded CUTLASS version, numerical reports, and training logs.
Stop owned XM jobs before shutting down owned services, even after a failed test.
Update B300 deployment digests only after qualification passes. Roll back by
restoring the prior image digest. Keep default and H200 deployments unchanged.
