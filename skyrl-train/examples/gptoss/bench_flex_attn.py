from transformers.models.gpt_oss.modeling_gpt_oss import GptOssAttention, GptOssRotaryEmbedding
from skyrl_train.patches.gptoss.patch_transformers import patch_GptOssAttention
from transformers import AutoConfig
from skyrl_train.utils.profiler import CudaTimer
import ray
import torch
import random
import numpy as np
from einops import repeat
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--batch_size", type=int, default=4, help="Batch size for processing")
parser.add_argument("--sequence_length", type=int, default=512, help="Prefix length")
parser.add_argument("--layer_idx", type=int, default=5, help="Layer index")
parser.add_argument("--num_warmup_trials", type=int, default=10, help="Number of warmup trials that are ignored")
parser.add_argument("--num_trials", type=int, default=20, help="Number of trials total")
parser.add_argument("--torch_dtype", type=lambda x: getattr(torch, x), default=torch.bfloat16, help="Torch dtype")
parser.add_argument("--with-attention", help="Whether to add attention mask", action="store_true")
args = parser.parse_args()


@ray.remote(num_gpus=1)
def run_bench(args):

    BATCH_SIZE = args.batch_size
    SEQUENCE_LENGTH = args.sequence_length
    MODEL_NAME = "unsloth/gpt-oss-20b-BF16"
    LAYER_IDX = args.layer_idx
    DEVICE = "cuda:0"
    NUM_WARMUP_TRIALS = args.num_warmup_trials
    NUM_TRIALS = args.num_trials
    TORCH_DTYPE = args.torch_dtype
    torch.cuda.set_device(DEVICE)
    torch.cuda.init()

    config = AutoConfig.from_pretrained(MODEL_NAME)
    config.torch_dtype = TORCH_DTYPE
    config._attn_implementation = "eager"

    def generate_inputs(batch_size, sequence_length, hidden_size, device, torch_dtype, with_attention: bool = False):
        prompt = torch.randn(batch_size, sequence_length, hidden_size).to(device, dtype=torch_dtype)
        attention_mask = None
        if with_attention:
            # zero out 30%
            num_zeros = int(0.3 * sequence_length)
            left_padding = random.randint(0, num_zeros)
            right_padding = num_zeros - left_padding
            attention_mask = torch.cat(
                (
                    torch.zeros(batch_size, left_padding),
                    torch.ones(batch_size, sequence_length - num_zeros),
                    torch.zeros(batch_size, right_padding),
                ),
                dim=1,
            ).to(device, dtype=torch.long)
        return prompt, attention_mask

    # use 256 MB tensor to clear L2 cache (same as triton timing approach)
    x = torch.empty(int(256 * (1024**2)), dtype=torch.int8, device="cuda")

    def flush_cache():
        x.zero_()

    ## eager
    eager_gptoss = GptOssAttention(config, LAYER_IDX)
    eager_gptoss.to(DEVICE, dtype=TORCH_DTYPE)

    rotary_emb = GptOssRotaryEmbedding(config).to(DEVICE, dtype=TORCH_DTYPE)

    fwd_times = []
    bwd_times = []
    try:
        for i in range(NUM_TRIALS):

            prompt, attention_mask = generate_inputs(
                BATCH_SIZE, SEQUENCE_LENGTH, config.hidden_size, DEVICE, TORCH_DTYPE, args.with_attention
            )
            full_batch = prompt.contiguous()
            position_ids = (
                repeat(torch.arange(SEQUENCE_LENGTH, dtype=TORCH_DTYPE), "L -> B L", B=BATCH_SIZE)
                .to(DEVICE, dtype=torch.long)
                .contiguous()
            )
            position_embeddings = rotary_emb(prompt, position_ids)
            grad_out = torch.randn_like(full_batch)

            with CudaTimer(DEVICE) as results:
                output, _ = eager_gptoss(
                    full_batch, position_embeddings=position_embeddings, attention_mask=attention_mask
                )
            with CudaTimer(DEVICE) as results_bwd:
                output.backward(grad_out)
            eager_gptoss.zero_grad()
            del output
            flush_cache()
            if i >= NUM_WARMUP_TRIALS:
                fwd_times.append(results.elapsed_time)
                bwd_times.append(results_bwd.elapsed_time)
        eager_time = np.mean(fwd_times) + np.mean(bwd_times)
        print(f"Eager Time Fwd: {np.mean(fwd_times)} ± {np.std(fwd_times)}")
        print(f"Eager Time Bwd: {np.mean(bwd_times)} ± {np.std(bwd_times)}")
        print(f"Overall mean eager time: {eager_time}")
    except torch.OutOfMemoryError as e:
        print(f"Eager attention OOM'ed with the following traceback: {e}. Skipping...")
    finally:
        eager_gptoss.zero_grad()
        eager_gptoss.to("cpu")
        flush_cache()

    # FLEX ATTENTION
    patch_GptOssAttention()
    flex_gptoss = GptOssAttention(config, LAYER_IDX)
    flex_gptoss.to(DEVICE, dtype=TORCH_DTYPE)

    fwd_times = []
    bwd_times = []
    try:
        for i in range(NUM_TRIALS):

            prompt, attention_mask = generate_inputs(
                BATCH_SIZE, SEQUENCE_LENGTH, config.hidden_size, DEVICE, TORCH_DTYPE, args.with_attention
            )
            full_batch = prompt.contiguous()
            # #full_batch = torch.cat((chosen_full, rejected_full), dim=0).contiguous()
            position_ids = None
            position_ids = (
                repeat(torch.arange(SEQUENCE_LENGTH, dtype=TORCH_DTYPE), "L -> B L", B=BATCH_SIZE)
                .to(DEVICE, dtype=TORCH_DTYPE)
                .contiguous()
            )
            position_embeddings = rotary_emb(prompt, position_ids)
            grad_out = torch.randn_like(full_batch)

            with CudaTimer(DEVICE) as results:
                output, _ = flex_gptoss(
                    full_batch, position_embeddings=position_embeddings, attention_mask=attention_mask
                )
            with CudaTimer(DEVICE) as results_bwd:
                output.backward(grad_out)
            flex_gptoss.zero_grad()
            del output
            flush_cache()
            if i >= NUM_WARMUP_TRIALS:
                fwd_times.append(results.elapsed_time)
                bwd_times.append(results_bwd.elapsed_time)

        flex_attn_time = np.mean(fwd_times) + np.mean(bwd_times)
        print(f"Flex Attn Time Fwd: {np.mean(fwd_times)} ± {np.std(fwd_times)}")
        print(f"Flex Attn Time Bwd: {np.mean(bwd_times)} ± {np.std(bwd_times)}")
        print(f"Flex Attn overall time: {flex_attn_time}")
    except torch.OutOfMemoryError as e:
        print(f"Flex attention OOM'ed with the following traceback: {e}. Skipping...")
    finally:
        flex_gptoss.zero_grad()
        flex_gptoss.to("cpu")
        flush_cache()
        # need to ensure that any future torch.compile calls doesn't get messed up by the previous compile (which has differently shaped inputs)
        torch.compiler.reset()


ray.get(run_bench.remote(args))
