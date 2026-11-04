"""Keep the KeyChain prompts the base model answers correctly in some, but not all, samples
(0 < correct < n, as in LoongRL): those are the groups that carry a GRPO advantage.

    uv run --isolated --extra megatron python examples/train/qwen3_8_flash_next/keychain/filter_keychain.py \\
        $HOME/data/keychain/pool_train.parquet $HOME/data/keychain/keychain_64k_train_filtered.parquet \\
        $HOME/data/keychain/calib_train_s*.jsonl
"""

import argparse
import collections
import json

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pool", help="pool_<split>.parquet from make_keychain.py")
    parser.add_argument("out", help="filtered .parquet")
    parser.add_argument("calibration", nargs="+", help="calibrate_keychain.py .jsonl shards")
    args = parser.parse_args()

    df = pd.read_parquet(args.pool)
    rec = {}
    for path in args.calibration:
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                rec[r["row"]] = r
    samples = sum(r["n"] for r in rec.values())
    print(
        f"scored {len(rec)}/{len(df)} prompts: pass {sum(r['correct'] for r in rec.values()) / samples:.3f}, "
        f"truncated {sum(r['trunc'] for r in rec.values()) / samples:.3f}, "
        f"mean response {sum(sum(r['resp_tokens']) for r in rec.values()) / samples:.0f} tokens"
    )
    print("correct-of-n histogram:", dict(sorted(collections.Counter(r["correct"] for r in rec.values()).items())))

    keep = sorted(i for i, r in rec.items() if 0 < r["correct"] < r["n"])
    kept = df.iloc[keep].reset_index(drop=True)
    if keep:
        kept_pass = sum(rec[i]["correct"] for i in keep) / sum(rec[i]["n"] for i in keep)
        print(
            f"kept {len(keep)} prompts, pass {kept_pass:.3f}, by source {dict(collections.Counter(kept['data_source']))}"
        )
    kept.to_parquet(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
