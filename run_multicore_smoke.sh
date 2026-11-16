#!/bin/bash
# Multi-core (4 NeuronCore) FSDP GRPO smoke on Trainium2.
#
# The DLAMI's `ray.init()` auto-detects only 1 neuron_cores on trn2.48xlarge
# (Ray's raylet misses the 16-device x 4-core = 64 logical-core topology), so we
# start a Ray head with the correct `neuron_cores` count explicitly before the run.
# `initialize_ray` then attaches to this head via bare `ray.init()`.
set -euo pipefail
source /opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin/activate

# Clean slate: stop any Ray cluster, then start a head with the true core count.
ray stop --force >/dev/null 2>&1 || true
ray start --head --disable-usage-stats --num-cpus=192 --resources='{"neuron_cores": 64}' >/dev/null 2>&1

# The driver's in-process engine pins the last logical NeuronCore itself (inside
# NeuronInferenceEngine), and each FSDP worker pins core == rank in its own process, so
# no core is exported from this shell.
export NKI_ENABLE_TRACE_CACHE=0 USE_HUB_KERNELS=NO
cd ~/SkyRL
MODEL_PATH=~/cholec/base_model DATA_DIR=~/cholec/skyrl_parquet \
  bash examples/train/cholec/run_cholec_neuron.sh \
    trainer.placement.policy_num_gpus_per_node=4 \
    trainer.max_training_steps=2 \
    trainer.train_batch_size=4 \
    trainer.policy_mini_batch_size=4 \
    trainer.eval_batch_size=8 \
    generator.n_samples_per_prompt=2 \
    generator.sampling_params.max_generate_length=64 \
    trainer.eval_interval=0
echo "MULTICORE_SMOKE_DONE"
