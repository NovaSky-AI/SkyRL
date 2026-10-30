"""SDPO prompt conditioning, paired inputs, response scoring, and losses."""

import math
import random
import re
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass

import torch
from torch.utils.checkpoint import checkpoint

from skyrl.backends.skyrl_train.utils.torch_utils import (
    chunked_entropy_from_logits,
    logprobs_from_logits,
    logprobs_from_logits_v2,
    safe_exp_delta,
)
from skyrl.train.dataset.preprocess import convert_prompts_responses_to_batch_tensors
from skyrl.backends.skyrl_train.utils.ppo_utils import ppo_policy_loss


@torch.no_grad()
def response_entropy_sum(logits, mask, temperature=1.0, chunk_size=128):
    """Full-vocabulary entropy over loss tokens, without a gradient or large FP32 copy."""
    total = torch.zeros((), device=logits.device, dtype=torch.float32)
    for start in range(0, logits.shape[1], chunk_size):
        stop = start + chunk_size
        entropy = chunked_entropy_from_logits(
            logits[:, start:stop].float() / temperature,
            requires_grad=False,
            attention_mask=mask[:, start:stop],
            chunk_size=chunk_size,
        )
        total += entropy.sum()
    return total


def strip_thinking(text, thinking_prefilled=False):
    """Remove reasoning, including a block opened by the generation prompt."""
    if thinking_prefilled and not text.lstrip().startswith("<think>"):
        text = "<think>" + text
    # A closing tag without an opening tag denotes a prefilled reasoning block.
    closing = text.find("</think>")
    opening = text.find("<think>")
    if closing >= 0 and (opening < 0 or closing < opening):
        text = text[closing + len("</think>") :]
    return re.sub(r"<think>.*?(?:</think>|$)", "", text, flags=re.DOTALL).strip()


def student_answer_loss_mask(tokenizer, response_ids, prompt_ids, loss_mask):
    """Mask <think> spans in token space without changing sequence alignment.

    Handles reasoning opened in the prompt and unfinished reasoning. Existing
    zero masks (observations, padding, invalid responses) remain zero. Both
    reasoning delimiters are excluded from supervision.
    """
    if len(response_ids) != len(loss_mask):
        raise ValueError("Response IDs and loss mask must have equal lengths")
    opening = tokenizer.encode("<think>", add_special_tokens=False)
    closing = tokenizer.encode("</think>", add_special_tokens=False)
    if not opening or not closing or opening == closing:
        raise ValueError("Tokenizer must distinguish thinking delimiters")
    suffix = tokenizer.decode(prompt_ids[-32:], skip_special_tokens=False)
    in_thinking = suffix.rfind("<think>") > suffix.rfind("</think>")

    # Some renderers omit the opening delimiter from the returned prompt IDs.
    # A closing delimiter before any opening delimiter still ends prefilled reasoning.
    def matches(index, marker):
        return response_ids[index : index + len(marker)] == marker

    if not in_thinking:
        for i in range(len(response_ids)):
            if matches(i, opening):
                break
            if matches(i, closing):
                in_thinking = True
                break
    result = list(loss_mask)
    i = 0
    while i < len(response_ids):
        marker = opening if matches(i, opening) else closing if matches(i, closing) else None
        if marker is not None:
            result[i : i + len(marker)] = [0] * len(marker)
            in_thinking = marker == opening
            i += len(marker)
        else:
            if in_thinking:
                result[i] = 0
            i += 1
    return result


def sibling_text(tokenizer, response_ids, prompt_ids, remove_thinking):
    text = tokenizer.decode(response_ids, skip_special_tokens=True)
    if remove_thinking:
        suffix = tokenizer.decode(prompt_ids[-32:], skip_special_tokens=False)
        text = strip_thinking(text, thinking_prefilled=suffix.rstrip().endswith("<think>"))
    return text


def extract_expert_traces(input_batch, config):
    """Dataset columns arrive in env_extras, already repeated per rollout.

    Remove the trace even when expert teaching is disabled, so environments
    cannot accidentally expose it to the student. Do not mutate dataset rows.
    """
    batch = dict(input_batch)
    count = len(input_batch["prompts"])
    extras = input_batch.get("env_extras")
    if extras is None:
        return batch, [None] * count
    if len(extras) != count:
        raise ValueError("env_extras must have one entry per rollout prompt")
    cleaned, traces = [], []
    for extra in extras:
        extra = deepcopy(extra)
        trace = extra.pop(config.expert_trace_key, None)
        if config.demonstration_source in {"sibling", "feedback_only"}:
            trace = None
        elif trace is not None and not isinstance(trace, str):
            raise ValueError("Expert traces must be text strings or null")
        traces.append(trace if trace and trace.strip() else None)
        cleaned.append(extra)
    batch["env_extras"] = cleaned
    return batch, traces


@dataclass
class TeacherPrompts:
    token_ids: list[list[int]]
    eligible: list[bool]
    metrics: dict[str, float]


def build_teacher_prompts(
    tokenizer,
    prompts,
    original_prompt_ids,
    responses,
    rewards,
    uids,
    feedback,
    config,
    demonstration_allowed=None,
    expert_traces=None,
    rng=None,
    multimodal_encoder=None,
):
    n = len(responses)
    if any(len(x) != n for x in (prompts, original_prompt_ids, rewards, uids, feedback)):
        raise ValueError("All rollout fields must have the same number of rows")
    allowed = [True] * n if demonstration_allowed is None else demonstration_allowed
    if len(allowed) != n:
        raise ValueError("demonstration_allowed must have one entry per response")
    experts = [None] * n if expert_traces is None else expert_traces
    if len(experts) != n:
        raise ValueError("expert_traces must have one entry per response")
    successes = {}
    for i, (uid, reward) in enumerate(zip(uids, rewards)):
        if (
            allowed[i]
            and responses[i]
            and math.isfinite(reward)
            and (config.sibling_selection == "best_reward" or reward >= config.success_reward_threshold)
        ):
            successes.setdefault(uid, []).append(i)
    if config.sibling_selection == "best_reward":
        # Choose once per group, before self-exclusion: never substitute a
        # runner-up for the winning row. Ignore unusable/empty demonstrations.
        rng = rng if rng is not None else random
        for uid, candidates in successes.items():
            usable = []
            for j in candidates:
                text = sibling_text(
                    tokenizer,
                    responses[j],
                    original_prompt_ids[j],
                    config.remove_thinking,
                )
                if text.strip():
                    usable.append(j)
            if usable:
                highest = max(rewards[j] for j in usable)
                successes[uid] = [rng.choice([j for j in usable if rewards[j] == highest])]
            else:
                successes[uid] = []
    group_sizes = Counter(uids)
    correct_singletons_skipped = 0
    tokens, eligible = [], []
    solutions_used = feedback_used = overflow_count = 0
    experts_used = siblings_used = 0
    for i in range(n):
        if (
            config.skip_correct_singletons
            and group_sizes[uids[i]] == 1
            and rewards[i] >= config.success_reward_threshold
        ):
            tokens.append(list(original_prompt_ids[i]))
            eligible.append(False)
            correct_singletons_skipped += 1
            continue
        solution = None
        for j in successes.get(uids[i], []):
            if config.exclude_self_demonstration and j == i:
                continue
            solution = sibling_text(tokenizer, responses[j], original_prompt_ids[j], config.remove_thinking)
            if solution.strip():
                break
            solution = None
        expert = experts[i]
        if expert is not None and not isinstance(expert, str):
            raise ValueError("Expert traces must be text strings or null")
        if expert is not None and not config.expert_trace_include_thinking:
            # An unfinished thinking block extends to the end of the trace.
            # Filter before selection/budgeting so empty references fall back.
            expert = strip_thinking(expert)
        expert = expert if expert and expert.strip() else None
        source = config.demonstration_source
        use_expert = expert is not None and (
            source in {"expert_only", "expert_then_sibling"} or (source == "sibling_then_expert" and solution is None)
        )
        if use_expert:
            solution = expert
        elif source in {"expert_only", "feedback_only"}:
            solution = None
        raw_feedback = feedback[i]
        has_feedback = (
            config.include_environment_feedback
            and isinstance(raw_feedback, str)
            and bool(raw_feedback.strip())
            and (solution is None or not config.feedback_only_without_solution)
        )
        use = bool(solution is not None or has_feedback)
        ids = original_prompt_ids[i]
        if use:
            messages = deepcopy(prompts[i])
            multimodal = config.vision_language and isinstance(messages[-1]["content"], list) if messages else False
            if (
                not messages
                or messages[-1]["role"] != "user"
                or (not multimodal and not isinstance(messages[-1]["content"], str))
            ):
                raise ValueError("SDPO requires text conversations ending with a user message")
            enriched = config.reprompt_template.format(
                prompt="" if multimodal else messages[-1]["content"],
                student_response=(
                    config.student_response_template.format(
                        student_response=sibling_text(
                            tokenizer, responses[i], original_prompt_ids[i], config.remove_thinking
                        )
                    )
                    if config.include_student_response
                    else ""
                ),
                solution=(
                    (
                        config.expert_trace_template.format(expert_trace=solution)
                        if use_expert
                        else config.solution_template.format(successful_previous_attempt=solution)
                    )
                    if solution is not None
                    else ""
                ),
                feedback=config.feedback_template.format(feedback_raw=raw_feedback) if has_feedback else "",
            )
            if multimodal and multimodal_encoder is not None:
                ids = multimodal_encoder(i, messages, enriched, has_feedback)
            elif multimodal:
                ids = append_teacher_hint(tokenizer, list(ids), enriched, config.chat_template_kwargs)
            else:
                messages[-1]["content"] = enriched
                ids = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    return_dict=False,
                    add_generation_prompt=True,
                    **config.chat_template_kwargs,
                )
            if (
                len(ids) > config.max_teacher_prompt_length
                or len(ids) + len(responses[i]) > config.max_teacher_sequence_length
            ):
                if config.overflow == "error":
                    raise ValueError(f"Teacher context exceeds token budget at row {i}")
                if config.overflow == "truncate_right":
                    budget = min(
                        config.max_teacher_prompt_length,
                        config.max_teacher_sequence_length - len(responses[i]),
                    )
                    if budget < 1:
                        raise ValueError("Teacher budget must leave room for a prompt and the original response")
                    ids = ids[:budget]
                else:
                    ids, use = original_prompt_ids[i], False
                overflow_count += 1
        tokens.append(list(ids))
        eligible.append(use)
        solutions_used += int(use and solution is not None)
        experts_used += int(use and use_expert)
        siblings_used += int(use and solution is not None and not use_expert)
        feedback_used += int(use and has_feedback)
    denominator = max(n, 1)
    return TeacherPrompts(
        tokens,
        eligible,
        {
            "eligible_fraction": sum(eligible) / denominator,
            "correct_singleton_skipped_fraction": correct_singletons_skipped / denominator,
            "correct_eligible_fraction": sum(
                use and reward >= config.success_reward_threshold for use, reward in zip(eligible, rewards)
            )
            / denominator,
            "solution_fraction": solutions_used / denominator,
            "expert_trace_fraction": experts_used / denominator,
            "sibling_fraction": siblings_used / denominator,
            "feedback_fraction": feedback_used / denominator,
            "context_overflow_fraction": overflow_count / denominator,
        },
    )


def collate_teacher_inputs(prompt_ids, response_ids, pad_token_id):
    if len(prompt_ids) != len(response_ids) or not response_ids:
        raise ValueError("Expected nonempty paired prompt/response lists")
    if any(not p or not r for p, r in zip(prompt_ids, response_ids)):
        raise ValueError("Every sample needs a nonempty prompt and response")
    # Use the same token alignment as the student's SkyRL batch. Reward and
    # response masks are required by the shared converter but unused here.
    tensors = convert_prompts_responses_to_batch_tensors(
        pad_token_id,
        prompt_ids,
        response_ids,
        rewards=[[0.0] for _ in response_ids],
        loss_masks=[[1] * len(response) for response in response_ids],
    )
    return tensors[0], tensors[1]


def response_logits(model, sequences, attention_mask, response_length, **model_inputs):
    if response_length < 1 or sequences.shape[1] <= response_length:
        raise ValueError("Response scoring requires at least one preceding prompt token")
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)
    if model_inputs.get("image_grid_thw") is not None or model_inputs.get("video_grid_thw") is not None:
        position_ids = None  # let the VLM construct multimodal rotary positions
        model_inputs["mm_token_type_ids"] = (sequences == model.config.image_token_id).long()
        if model_inputs.get("video_grid_thw") is not None:
            model_inputs["mm_token_type_ids"][sequences == model.config.video_token_id] = 2
    output = model(
        input_ids=sequences,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
        **model_inputs,
    )
    # Clone to release logits for the much longer teacher prompt. No tokenizer
    # round trip: response positions correspond to the original generated IDs.
    return output.logits[:, -response_length - 1 : -1].clone()


def _action_logprobs(logits, actions, temperature):
    logits = logits.float() / temperature
    # The Flash-Attention implementation selected by SkyRL is CUDA-only.
    if logits.is_cuda:
        return logprobs_from_logits(logits, actions, inplace_backward=False)
    return logprobs_from_logits_v2(logits, actions)


def chunked_action_logprobs(logits, actions, temperature, chunk_size=128):
    """Score actions with bounded FP32 intermediates during forward/backward."""
    flat = logits.reshape(-1, logits.shape[-1])
    ids = actions.reshape(-1)
    chunks = []
    for start in range(0, ids.numel(), chunk_size):
        args = (flat[start : start + chunk_size], ids[start : start + chunk_size], temperature)
        if torch.is_grad_enabled() and logits.requires_grad:
            chunk = checkpoint(_action_logprobs, *args, use_reentrant=False)
        else:
            chunk = _action_logprobs(*args)
        chunks.append(chunk)
    return torch.cat(chunks).reshape(actions.shape)


def _distribution(logits, temperature, indices=None, add_tail=True):
    logp = (logits.float() / temperature).log_softmax(-1)
    if indices is None:
        return logp
    selected = logp.gather(-1, indices)
    if add_tail:
        # Sum the complement in log space: 1 - selected_mass loses the tail
        # when a peaked distribution rounds selected_mass to one in FP32.
        tail = logp.scatter(-1, indices, float("-inf")).logsumexp(-1, keepdim=True)
        return torch.cat((selected, tail), dim=-1)
    return selected - selected.logsumexp(-1, keepdim=True)


def divergence(student, teacher, alpha):
    """Per-position divergence between normalized log distributions."""
    teacher = teacher.detach()
    if alpha == 0:
        return (teacher.exp() * (teacher - student)).sum(-1)
    if alpha == 1:
        return (student.exp() * (student - teacher)).sum(-1)
    mixture = torch.logaddexp(student + math.log1p(-alpha), teacher + math.log(alpha))
    return (1 - alpha) * (student.exp() * (student - mixture)).sum(-1) + alpha * (
        teacher.exp() * (teacher - mixture)
    ).sum(-1)


def sdpo_loss(
    student_logits,
    teacher_logits,
    response_ids,
    loss_mask,
    config,
    temperature=1.0,
    rollout_logprobs=None,
    advantage_clip=None,
    return_advantage_stats=False,
):
    """Return a masked SUM; the caller applies the configured normalization.

    Float32 vocabulary intermediates are chunked and recomputed in backward.
    Teacher logits may remain on CPU and transfer one chunk at a time.
    """
    if temperature <= 0 or not math.isfinite(temperature):
        raise ValueError("temperature must be finite and positive")
    if student_logits.shape != teacher_logits.shape or student_logits.shape[:2] != response_ids.shape:
        raise ValueError("Student/teacher logits and response positions must align")
    if loss_mask.shape != response_ids.shape:
        raise ValueError("Loss mask must align with response IDs")
    if config.rollout_is_clip is not None and rollout_logprobs is None:
        raise ValueError("rollout_is_clip requires generator rollout_logprobs")
    if advantage_clip is not None and config.alpha != 1:
        raise ValueError("Advantage clipping requires reverse KL (alpha=1)")
    teacher_logits = teacher_logits.detach()
    vocab = student_logits.shape[-1]
    s = student_logits.reshape(-1, vocab)
    t = teacher_logits.reshape(-1, vocab)
    ids = response_ids.reshape(-1)
    mask = loss_mask.reshape(-1)
    old = rollout_logprobs.reshape(-1) if rollout_logprobs is not None else None

    advantage_sum = student_logits.new_zeros((), dtype=torch.float32)
    advantage_count = student_logits.new_zeros((), dtype=torch.float32)
    advantage_clipped_count = student_logits.new_zeros((), dtype=torch.float32)
    advantage_min = student_logits.new_full((), float("inf"), dtype=torch.float32)
    advantage_max = student_logits.new_full((), float("-inf"), dtype=torch.float32)

    def chunk_loss(student, teacher, actions, weights, behavior):
        teacher = teacher.to(device=student.device)
        chunk_advantage_sum = student_logits.new_zeros((), dtype=torch.float32)
        chunk_advantage_count = student_logits.new_zeros((), dtype=torch.float32)
        chunk_clipped_count = student_logits.new_zeros((), dtype=torch.float32)
        chunk_advantage_min = student_logits.new_full((), float("inf"), dtype=torch.float32)
        chunk_advantage_max = student_logits.new_full((), float("-inf"), dtype=torch.float32)

        action_logp = None
        if config.loss == "sampled_reverse_kl" or config.rollout_is_clip is not None:
            action_logp = _action_logprobs(student, actions, temperature)
        if config.loss == "sampled_reverse_kl":
            teacher_action = _action_logprobs(teacher, actions, temperature)
            advantage = (action_logp - teacher_action).detach()
            # SRPO: mask out any tokens that are not used for SDPO loss using the SDPO loss mask (weights)
            valid = weights > 0
            # Get sum and count of advantages for tokens only used for SDPO
            chunk_advantage_sum = advantage.abs().masked_fill(~valid, 0).sum()
            chunk_advantage_count = valid.sum()
            # Signed, unclipped extrema; ignored tokens must not affect the range.
            chunk_advantage_min = advantage.masked_fill(~valid, float("inf")).amin()
            chunk_advantage_max = advantage.masked_fill(~valid, float("-inf")).amax()

            if advantage_clip is not None:
                chunk_clipped_count = ((advantage.abs() > advantage_clip) & valid).sum()
                advantage = advantage.clamp(min=-advantage_clip, max=advantage_clip)
            per_token = advantage * action_logp

        else:
            indices = None
            if config.topk is not None and config.topk < vocab:
                indices = student.detach().topk(config.topk, dim=-1).indices
            student_dist = _distribution(student, temperature, indices, config.add_tail)
            teacher_dist = _distribution(teacher, temperature, indices, config.add_tail)
            per_token = divergence(student_dist, teacher_dist, config.alpha)
            if config.alpha == 1 and (return_advantage_stats or advantage_clip is not None):
                # Each retained candidate and the tail is one category; statistics
                # weight its detached log-ratio by the student's probability.
                delta = (student_dist - teacher_dist).detach()
                probability = student_dist.exp()
                valid = weights > 0
                mass = probability.detach() * valid.unsqueeze(-1)
                chunk_advantage_sum = (mass * delta.abs()).sum()
                chunk_advantage_count = valid.sum()
                chunk_advantage_min = delta.masked_fill(~valid.unsqueeze(-1), float("inf")).amin()
                chunk_advantage_max = delta.masked_fill(~valid.unsqueeze(-1), float("-inf")).amax()
                if advantage_clip is not None:
                    chunk_clipped_count = (mass * (delta.abs() > advantage_clip)).sum()
                    surrogate = (probability * delta.clamp(-advantage_clip, advantage_clip)).sum(-1)
                    # Report the KL value; take the gradient of the clipped surrogate.
                    per_token = per_token.detach() + surrogate - surrogate.detach()
        if config.rollout_is_clip is not None:
            ratio = safe_exp_delta(action_logp.detach() - behavior).clamp(max=config.rollout_is_clip)
            per_token = per_token * ratio
        return (
            (per_token * weights).sum(),
            chunk_advantage_sum,
            chunk_advantage_count,
            chunk_advantage_min,
            chunk_advantage_max,
            chunk_clipped_count,
        )

    total = student_logits.new_zeros((), dtype=torch.float32)
    for start in range(0, len(s), config.loss_chunk_size):
        sl = slice(start, start + config.loss_chunk_size)
        args = (s[sl], t[sl], ids[sl], mask[sl], old[sl] if old is not None else None)
        if torch.is_grad_enabled() and s.requires_grad:
            chunk_total, chunk_sum, chunk_count, chunk_min, chunk_max, chunk_clipped = checkpoint(
                chunk_loss, *args, use_reentrant=False
            )
        else:
            chunk_total, chunk_sum, chunk_count, chunk_min, chunk_max, chunk_clipped = chunk_loss(*args)

        total = total + chunk_total
        advantage_sum += chunk_sum
        advantage_count += chunk_count
        advantage_clipped_count += chunk_clipped
        advantage_min = torch.minimum(advantage_min, chunk_min)
        advantage_max = torch.maximum(advantage_max, chunk_max)
    if return_advantage_stats:
        return total, advantage_sum, advantage_count, advantage_min, advantage_max, advantage_clipped_count
    return total


def pad_bool_mask(mask, batch_size):
    mask = torch.as_tensor(mask, dtype=torch.bool).reshape(-1)
    pad = batch_size - mask.numel()
    if pad < 0:
        raise ValueError("routing mask is longer than the training batch")
    if pad:
        mask = torch.cat((mask, torch.zeros(pad, dtype=torch.bool)))
    return mask


def srpo_route_masks(rewards, teacher_available, threshold, *, positive_rewards=False, success_only=False):
    """SDPO iff the rollout is incorrect and a teacher exists.
       z_SDPO = (1 - c) * m, z_GRPO = 1 - z_SDPO where:
        - c is sequence reward (1 or 0 for correct or incorrect)
        - m is teacher availability (1 or 0 for available or not)
    For some batch of rollouts, we keep rewards for correct rollouts and teacher availability for incorrect rollouts.
    With success_only, missing teacher feedback never routes a failure to GRPO.
    """
    rewards = torch.as_tensor(rewards, dtype=torch.float32)
    teacher = torch.as_tensor(teacher_available, dtype=torch.bool)
    if positive_rewards:
        return rewards.eq(0) & teacher, rewards.gt(0) & rewards.isfinite()
    correct = (rewards >= threshold).to(dtype=torch.bool)
    if success_only:
        finite = rewards.isfinite()
        return (~correct) & teacher & finite, correct & finite
    sdpo_mask = (~correct) & teacher
    grpo_mask = ~sdpo_mask
    return sdpo_mask, grpo_mask


@torch.no_grad()
def dw_sdpo_weights(teacher_logits, sdpo_mask, beta, temperature=1.0, chunk_size=128):
    """w' = exp(-beta H) from teacher entropy.

    H is the temperature-scaled teacher entropy. Caller divides by the mean
    over Omega_sdpo and multiplies into the SDPO loss mask:
    loss_mask_dw = w'/mean(w') * loss_mask_sdpo
    """
    if beta < 0:
        raise ValueError("entropy_beta must be >= 0")
    # Cast and scale only a token chunk. Casting the full [B, T, V] tensor
    # before calling the chunked helper defeats its memory bound at long T.
    weights = torch.empty(teacher_logits.shape[:2], device=teacher_logits.device, dtype=torch.float32)
    for start in range(0, teacher_logits.shape[1], chunk_size):
        stop = start + chunk_size
        entropy = chunked_entropy_from_logits(
            teacher_logits[:, start:stop].float() / temperature,
            requires_grad=False,
            attention_mask=sdpo_mask[:, start:stop],
            chunk_size=chunk_size,
        )
        weights[:, start:stop] = torch.exp(-beta * entropy)
    return weights


def append_teacher_hint(tokenizer, original_ids, hint, chat_template_kwargs):
    """Insert text before the final user-message terminator, retaining image IDs.

    Re-tokenizing image placeholders would lose the renderer's expanded image
    token counts. Require an exact chat-suffix match instead of guessing.
    """
    marker = "SDPO_USER_CONTENT_SENTINEL_7f381"
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": marker}],
        tokenize=False,
        add_generation_prompt=True,
        **chat_template_kwargs,
    )
    if not isinstance(rendered, str) or rendered.count(marker) != 1:
        raise ValueError("Cannot identify the multimodal chat-template suffix")
    suffix = tokenizer.encode(rendered.split(marker)[1], add_special_tokens=False)
    if not suffix or original_ids[-len(suffix) :] != suffix:
        raise ValueError("Rendered VLM prompt does not match teacher chat-template suffix")
    return original_ids[: -len(suffix)] + tokenizer.encode(hint, add_special_tokens=False) + suffix


def vision_inputs(microbatch, prefix=""):
    result = {}
    for key in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"):
        value = microbatch.get(prefix + key)
        if value is not None:
            tensor = torch.cat(value.tensors, dim=0) if hasattr(value, "tensors") else value
            if tensor.numel():
                result[key] = tensor
    return result


def encode_visual_teacher_prompt(processor, messages, hint, images, chat_template_kwargs, video=None):
    """Process original observations and teacher-only feedback images/video together."""
    messages = deepcopy(messages)
    # HF processors accept image blocks; dataset prompts use OpenAI image_url blocks.
    for message in messages:
        if isinstance(message["content"], list):
            message["content"] = [
                {"type": "image", "image": item["image_url"]["url"]} if item.get("type") == "image_url" else item
                for item in message["content"]
            ]
    label = "EXECUTION VIDEO of the attempted plan:" if video else "FINAL scene after the attempted plan:"
    messages[-1]["content"].append({"type": "text", "text": hint + "\n" + label})
    messages[-1]["content"].extend({"type": "image", "image": url} for url in images)
    video_kwargs = {}
    if video:
        import base64
        import io
        import numpy as np
        from PIL import Image

        frames = video["frames"]
        indices = video["frame_indices"]
        if not frames or len(frames) != len(indices) or video["fps"] <= 0:
            raise ValueError("Video frames require matching simulation indices and positive fps")
        if any(b <= a for a, b in zip(indices, indices[1:])):
            raise ValueError("Video frame indices must be strictly increasing")
        pixels = np.stack(
            [
                np.array(Image.open(io.BytesIO(base64.b64decode(frame.split(",", 1)[1], validate=True))).convert("RGB"))
                for frame in frames
            ]
        )
        messages[-1]["content"].append({"type": "video", "video": pixels})
        video_kwargs = {
            "videos_kwargs": {
                "do_sample_frames": False,
                "video_metadata": [
                    {
                        "fps": video["fps"],
                        "total_num_frames": video["total_num_frames"],
                        "frames_indices": indices,
                    }
                ],
            }
        }
    encoded = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        **chat_template_kwargs,
        **video_kwargs,
    )
    if "pixel_values" not in encoded or "image_grid_thw" not in encoded:
        raise ValueError("Teacher visual feedback requires image pixels and image_grid_thw")
    if video and ("pixel_values_videos" not in encoded or "video_grid_thw" not in encoded):
        raise ValueError("Processor did not produce native video inputs")
    return encoded
