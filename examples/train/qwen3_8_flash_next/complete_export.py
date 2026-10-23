"""Make a full-FT HF export of Qwen3.8-Flash-Next (qwen4_exp) loadable by adding the tensors it omits.

    uv run --isolated --no-project python examples/train/qwen3_8_flash_next/complete_export.py \
        <export_dir> <base_dir>

``trainer.hf_save_interval`` exports only what the Megatron model owns. For this model that leaves
out the frozen PLE n-gram tables (a plain tensor, not a Megatron parameter), the vision tower
(``language_model_only=true``) and the MTP head (not built with ``mtp_num_layers=0``). vLLM needs
the PLE tables even for text, so the export does not load on its own.

This copies every tensor that is in ``<base_dir>``'s index but not in the export's:

- base shards whose tensors are all missing (the PLE-only shards) are copied whole;
- the rest are packed into ``model-base-extra.safetensors``;
- ``model.safetensors.index.json`` is rewritten (the export's original is kept as
  ``model.safetensors.index.export_orig.json``),

then checks every base tensor's name, dtype and shape against the result. Tensors are copied as
raw safetensors bytes (no torch needed). The merged MTP head is the base's, paired with a
fine-tuned trunk, so MTP speculative decoding may accept fewer tokens.
"""

import argparse
import json
import os
import shutil
import struct

INDEX = "model.safetensors.index.json"
EXTRA = "model-base-extra.safetensors"


def read_header(path):
    with open(path, "rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(size)), 8 + size


def read_tensor(path, name):
    header, data_start = read_header(path)
    start, end = header[name]["data_offsets"]
    with open(path, "rb") as f:
        f.seek(data_start + start)
        return header[name], f.read(end - start)


def write_safetensors(path, tensors):
    """tensors: list of (name, header_entry, raw_bytes)."""
    meta, offset = {}, 0
    for name, entry, data in tensors:
        meta[name] = {"dtype": entry["dtype"], "shape": entry["shape"], "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
    header = json.dumps(meta, separators=(",", ":")).encode()
    header += b" " * (-len(header) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header)))
        f.write(header)
        for _, _, data in tensors:
            f.write(data)
    return offset


def complete(export_dir, base_dir):
    base_map = json.load(open(os.path.join(base_dir, INDEX)))["weight_map"]
    index_path = os.path.join(export_dir, INDEX)
    export_index = json.load(open(index_path))
    backup = os.path.join(export_dir, "model.safetensors.index.export_orig.json")
    if not os.path.exists(backup):
        shutil.copyfile(index_path, backup)
    export_map = export_index["weight_map"]

    missing_by_file = {}
    for name, filename in base_map.items():
        if name not in export_map:
            missing_by_file.setdefault(filename, []).append(name)
    file_sizes = {}
    for filename in base_map.values():
        file_sizes[filename] = file_sizes.get(filename, 0) + 1

    extra = []
    for filename, names in missing_by_file.items():
        if len(names) == file_sizes[filename] and not os.path.exists(os.path.join(export_dir, filename)):
            shutil.copyfile(os.path.join(base_dir, filename), os.path.join(export_dir, filename))
            export_map.update({name: filename for name in names})
        else:
            extra += [(name, *read_tensor(os.path.join(base_dir, filename), name)) for name in names]
    whole = sum(len(n) for f, n in missing_by_file.items() if export_map.get(n[0]) == f)
    print(f"copied {whole} tensors as whole base shards")
    if extra:
        nbytes = write_safetensors(os.path.join(export_dir, EXTRA), extra)
        export_map.update({name: EXTRA for name, _, _ in extra})
        print(f"packed {len(extra)} tensors ({nbytes / 2**30:.2f} GiB) into {EXTRA}")

    export_index["weight_map"] = dict(sorted(export_map.items()))
    with open(index_path, "w") as f:
        json.dump(export_index, f, indent=2)
    return verify(export_dir, base_dir, base_map, export_map)


def verify(export_dir, base_dir, base_map, export_map):
    headers, problems = {}, []

    def header(directory, filename):
        key = (directory, filename)
        if key not in headers:
            headers[key] = read_header(os.path.join(directory, filename))[0]
        return headers[key]

    for name, filename in base_map.items():
        if name not in export_map:
            problems.append(f"missing {name}")
            continue
        got = header(export_dir, export_map[name]).get(name)
        want = header(base_dir, filename)[name]
        if got is None:
            problems.append(f"{name} not in {export_map[name]}")
        elif (got["dtype"], got["shape"]) != (want["dtype"], want["shape"]):
            problems.append(f"{name}: {got['dtype']}{got['shape']} != base {want['dtype']}{want['shape']}")
    print(f"verified {len(base_map)} tensors against the base: {len(problems)} problems")
    for p in problems[:20]:
        print("  " + p)
    return not problems


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("export_dir", help="HF export dir, e.g. <export_path>/global_step_10/policy")
    parser.add_argument("base_dir", help="the base HF checkpoint the run started from")
    args = parser.parse_args()
    raise SystemExit(0 if complete(args.export_dir, args.base_dir) else 1)
