import ray


@ray.remote(num_gpus=1)
def run_task():

    from skyrl_train.patches.gptoss.patch_transformers import custom_attention_mask, custom_attention
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from skyrl_train.model_wrapper import logprobs_from_logits

    from transformers import AttentionInterface, AttentionMaskInterface

    AttentionInterface.register("custom_flex", custom_attention)
    AttentionMaskInterface.register("custom_flex", custom_attention_mask)

    MODEL = "unsloth/gpt-oss-20b-BF16"
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model.eval()

    model.gradient_checkpointing_enable()

    input_text = "Hello, how are you?"

    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    input_ids = torch.cat(
        [
            torch.tensor(
                [
                    [
                        tokenizer.pad_token_id,
                    ]
                ],
                dtype=input_ids.dtype,
            ),
            input_ids,
        ],
        dim=1,
    )
    attention_mask = torch.cat(
        [torch.tensor([0], dtype=torch.long), torch.tensor([1] * (input_ids.shape[1] - 1), dtype=torch.long)]
    ).unsqueeze(0)

    input_ids = input_ids.to("cuda")
    position_ids = attention_mask.long().cumsum(-1) - 1
    attention_mask = attention_mask.to("cuda")
    position_ids = position_ids.to("cuda")

    with torch.no_grad():
        print(type(input_ids))
        output1 = model(input_ids, attention_mask=attention_mask, position_ids=position_ids)
        logprobs1 = logprobs_from_logits(output1["logits"], input_ids)

    model.set_attn_implementation("custom_flex")

    with torch.no_grad():
        output2 = model(input_ids, attention_mask=attention_mask, position_ids=position_ids)
        logprobs2 = logprobs_from_logits(output2["logits"], input_ids)
        # NOTE: this is actually a high error in logprobs.
        # TODO: revisit flex attention patches to see if this can be improved
        torch.testing.assert_close(logprobs1, logprobs2, atol=1e-2, rtol=0.02)


ray.init()
ray.get(run_task.remote())
