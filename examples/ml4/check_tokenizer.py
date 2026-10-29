"""Trainer tokenizer vs mistral-common (what vLLM uses with ``tokenizer_mode=mistral``).

The trainer loads its tokenizer the way ``main_base`` does (``skyrl.utils.tok.get_tokenizer``).
With ``tokenizer_config.json = {"tokenizer_class": "MistralCommonBackend"}`` in the model dir
that is transformers' mistral-common backend over the checkpoint's ``tekken.json``.

Checks, each a hard failure:
  1. plain text (spaces, newlines, tabs, unicode): same ids as ``mistral_common``'s Tekkenizer,
     ``decode(encode(x)) == x``, and spaces / newlines change the ids (catches a byte-level vocab
     rebuilt as Metaspace, which drops them silently);
  2. DAPO prompts through ``apply_chat_template`` exactly as SkyRL calls it (dataset length filter,
     batched generator; token-in-token-out, so these ids are what vLLM receives) equal
     ``encode_chat_completion`` of the same messages (vLLM's chat path in mistral mode).
Reported, not failed: ``chat_template.jinja`` rendered on the same messages vs the decoded
mistral-common prompt (the jinja file is not what either side uses).

  python examples/ml4/check_tokenizer.py <model_dir> [--parquet dapo-math-17k-cleaned.parquet] [-n 32]
"""

import argparse
import os
import sys

DAPO_PROMPT = (
    "Solve the following math problem step by step. The last line of your response should be of the form "
    "Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n{}\n\n"
    'Remember to put your answer on its own line after "Answer:".'
)
PROBLEMS = [
    "Let $x$ and $y$ be real numbers such that $x^2 + y^2 = 1$. Find the maximum of $x + y$.",
    "  Leading spaces,\ttabs,\n\nblank lines and trailing spaces   ",
    "Unicode: é, ü, ß, 中文, ∑_{k=1}^n k², 🙂, a nbsp.",
]
TEXTS = [
    "hello world",
    "hello  world",
    " leading space",
    "trailing space ",
    "line one\nline two\n\n\nline five",
    "\tindent\n    four spaces\r\nCRLF",
    "naïve café — “quotes” 中文 日本語 한국어 🙂👍🏽",
    "x = 1;\nif (x > 0) {\n  return x * 2;\n}\n",
    DAPO_PROMPT.format(PROBLEMS[0]),
]


def load_prompts(parquet: str | None, n: int) -> list[list[dict]]:
    if parquet is None:
        return [[{"role": "user", "content": DAPO_PROMPT.format(p)}] for p in PROBLEMS]
    import pandas as pd

    rows = pd.read_parquet(parquet, columns=["prompt"])["prompt"].head(n)
    return [[{"role": m["role"], "content": m["content"]} for m in prompt] for prompt in rows]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--parquet", help="DAPO parquet (SkyRL format); default: built-in DAPO-style prompts")
    ap.add_argument("-n", type=int, default=32, help="prompts to read from --parquet")
    args = ap.parse_args()

    from mistral_common.protocol.instruct.request import ChatCompletionRequest
    from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

    from skyrl.utils.tok import get_tokenizer

    failures = []

    def check(ok: bool, what: str) -> None:
        print(("ok    " if ok else "FAIL  ") + what)
        if not ok:
            failures.append(what)

    # As main_base builds it.
    tok = get_tokenizer(args.model_dir, trust_remote_code=True, use_fast=True, padding_side="left")
    ref = MistralTokenizer.from_file(os.path.join(args.model_dir, "tekken.json"))
    raw = ref.instruct_tokenizer.tokenizer
    print(f"trainer tokenizer: {type(tok).__name__}; reference: {type(raw).__name__} {raw.version}")
    print(
        f"bos/eos/pad: trainer {tok.bos_token_id}/{tok.eos_token_id}/{tok.pad_token_id}, ref {raw.bos_id}/{raw.eos_id}"
    )
    check(tok.eos_token_id == raw.eos_id and tok.bos_token_id == raw.bos_id, "bos/eos ids match")

    # 1. plain text
    for x in TEXTS:
        ids = tok.encode(x, add_special_tokens=False)
        check(ids == raw.encode(x, bos=False, eos=False), f"ids == mistral_common: {x[:40]!r}")
        check(tok.decode(ids) == x, f"decode(encode(x)) == x: {x[:40]!r}")
    for a, b in [("a b", "ab"), ("a\nb", "ab"), ("a  b", "a b")]:
        check(tok.encode(a, add_special_tokens=False) != tok.encode(b, add_special_tokens=False), f"{a!r} != {b!r}")

    # 2. chat: SkyRL's calls vs mistral-common's request encoding
    prompts = load_prompts(args.parquet, args.n)
    want = [ref.encode_chat_completion(ChatCompletionRequest(messages=p)).tokens for p in prompts]
    single = [tok.apply_chat_template(p, add_generation_prompt=True, return_dict=False, tokenize=True) for p in prompts]
    batched = tok.apply_chat_template(prompts, add_generation_prompt=True, tokenize=True, return_dict=False)
    check(single == want, f"apply_chat_template == encode_chat_completion ({len(prompts)} prompts)")
    check([list(b) for b in batched] == want, "batched apply_chat_template (generator) == encode_chat_completion")
    for p, ids in zip(prompts, want):
        text = ref.decode(ids, special_token_policy=SpecialTokenPolicy.IGNORE)
        check(p[-1]["content"] in text, "prompt text survives the chat encoding")
    print("chat prompt (special tokens kept):")
    print(repr(ref.decode(want[0], special_token_policy=SpecialTokenPolicy.KEEP)))

    # chat_template.jinja: reported only
    jinja_path = os.path.join(args.model_dir, "chat_template.jinja")
    if os.path.exists(jinja_path):
        from transformers.utils.chat_template_utils import render_jinja_template

        template = open(jinja_path, encoding="utf-8").read()
        same = 0
        for p, ids in zip(prompts, want):
            rendered = render_jinja_template(
                [p], chat_template=template, add_generation_prompt=True, bos_token="<s>", eos_token="</s>"
            )[0][0]
            expected = ref.decode(ids, special_token_policy=SpecialTokenPolicy.KEEP)
            if rendered == expected:
                same += 1
            elif same == 0:
                print(f"jinja differs:\n  jinja:          {rendered[:400]!r}\n  mistral_common: {expected[:400]!r}")
        print(f"chat_template.jinja == mistral_common prompt text: {same}/{len(prompts)} (not used by either side)")
    else:
        print("no chat_template.jinja in the model dir")

    print(f"\n{len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
