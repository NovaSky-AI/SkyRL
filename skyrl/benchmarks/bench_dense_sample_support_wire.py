"""Measure dense sampler-support list versus packed JSON transport."""

import statistics
import time

import numpy as np
import orjson

from skyrl.backends.skyrl_train.inference_servers.sample_support_set_wire import (
    decode_sample_support_set,
    encode_sample_support_set,
)


def _median_ms(fn, iterations: int) -> float:
    durations = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        durations.append((time.perf_counter() - start) * 1000)
    return statistics.median(durations)


def _report(label: str, support: np.ndarray, iterations: int) -> None:
    list_json = orjson.dumps(support.tolist())
    packed_json = orjson.dumps(encode_sample_support_set(support))

    print(f"scenario={label} shape={support.shape} iterations={iterations}")
    print(f"list_json_bytes={len(list_json)} packed_json_bytes={len(packed_json)}")
    print(f"size_ratio={len(packed_json) / len(list_json):.4f}")
    print(
        "list_encode_ms=%.3f packed_encode_ms=%.3f"
        % (
            _median_ms(lambda: orjson.dumps(support.tolist()), iterations),
            _median_ms(lambda: orjson.dumps(encode_sample_support_set(support)), iterations),
        )
    )
    print(
        "list_decode_ms=%.3f packed_decode_ms=%.3f"
        % (
            _median_ms(lambda: orjson.loads(list_json), iterations),
            _median_ms(lambda: decode_sample_support_set(orjson.loads(packed_json)).tolist(), iterations),
        )
    )


def main() -> None:
    rng = np.random.default_rng(17)
    tokens, iterations = 4096, 100
    for top_k in (8, 64):
        full = rng.integers(0, 152_064, size=(tokens, top_k), dtype=np.int32)
        variable = full.copy()
        widths = rng.integers(1, top_k + 1, size=tokens)
        variable[np.arange(top_k)[None, :] >= widths[:, None]] = -1

        _report(f"top_k_{top_k}_full", full, iterations)
        _report(f"top_k_{top_k}_top_p_variable", variable, iterations)


if __name__ == "__main__":
    main()
