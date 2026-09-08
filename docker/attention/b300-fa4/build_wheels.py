"""Repackage pinned FA2/FA4 wheels with exclusive ownership of flash_attn.cute."""

import argparse
import base64
import csv
import hashlib
import io
import json
import re
import shutil
import urllib.request
import zipfile
from email import policy
from email.parser import BytesParser
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent
CUTE = "flash_attn/cute/"
PROVENANCE = "skyrl-provenance.json"


def canonical_name(value):
    return re.sub(r"[-_.]+", "-", value).lower()


def requirement_name(value):
    return canonical_name(re.match(r"[A-Za-z0-9_.-]+", value).group())


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def read_wheel(path):
    with zipfile.ZipFile(path) as archive:
        names = [entry.filename for entry in archive.infolist() if not entry.is_dir()]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate wheel entries: {path}")
        for name in names:
            if PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts or "\\" in name:
                raise ValueError(f"Unsafe wheel path: {name}")
        files = {name: archive.read(name) for name in names}
    validate_record(files)
    return files


def dist_info(files):
    roots = {name.split("/", 1)[0] for name in files if ".dist-info/" in name}
    if len(roots) != 1:
        raise ValueError(f"Expected one dist-info directory, got {sorted(roots)}")
    return roots.pop()


def record_hash(data):
    return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()


def validate_record(files):
    record = f"{dist_info(files)}/RECORD"
    rows = list(csv.reader(io.StringIO(files[record].decode())))
    if len(rows) != len(files) or {row[0] for row in rows} != set(files):
        raise ValueError("RECORD does not cover the wheel exactly once")
    for name, digest, size in rows:
        if name == record:
            if digest or size:
                raise ValueError("RECORD must not hash itself")
        elif digest != record_hash(files[name]) or size != str(len(files[name])):
            raise ValueError(f"RECORD mismatch: {name}")


def write_wheel(path, files):
    files = dict(files)
    record = f"{dist_info(files)}/RECORD"
    rows = [(name, record_hash(data), str(len(data))) for name, data in sorted(files.items())]
    stream = io.StringIO(newline="")
    csv.writer(stream, lineterminator="\n").writerows(rows + [(record, "", "")])
    files[record] = stream.getvalue().encode()
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data, compresslevel=9)
    validate_record(files)


def read_metadata(files):
    return BytesParser(policy=policy.compat32).parsebytes(files[f"{dist_info(files)}/METADATA"])


def build_pair(manifest, cache, output):
    cache, output = Path(cache), Path(output)
    inputs = {}
    for name, spec in manifest["inputs"].items():
        path = cache / spec["filename"]
        if sha256(path) != spec["sha256"]:
            raise ValueError(f"Input SHA-256 mismatch: {path.name}")
        inputs[name] = read_wheel(path)
        if read_metadata(inputs[name])["Version"] != spec["version"]:
            raise ValueError(f"Input version mismatch: {path.name}")
    fa2, fa4 = inputs["fa2"], inputs["fa4"]
    old2, old4 = dist_info(fa2), dist_info(fa4)
    root2 = f"flash_attn-{manifest['combined_version']}.dist-info"
    root4 = f"flash_attn_4-{manifest['companion_version']}.dist-info"
    unexpected = [name for name in fa4 if not name.startswith((CUTE, old4 + "/"))]
    if unexpected or not any(name.startswith(CUTE) for name in fa4):
        raise ValueError(f"Unexpected FA4 payload: {unexpected}")

    payload = {name: data for name, data in fa2.items() if not name.startswith((CUTE, old2 + "/"))}
    payload.update({name: data for name, data in fa4.items() if name.startswith(CUTE)})
    inventory = {name: hashlib.sha256(data).hexdigest() for name, data in payload.items()}
    combined = dict(payload)
    for name, data in fa2.items():
        if name.startswith(old2 + "/") and name.rsplit("/", 1)[-1] not in {"RECORD", "METADATA", "WHEEL"}:
            combined[name.replace(old2, root2, 1)] = data
    licenses = {name: data for name, data in fa4.items() if name.startswith(old4 + "/") and "license" in name.lower()}
    if not licenses:
        raise ValueError("The FA4 wheel must include its license")
    for name, data in licenses.items():
        combined[f"{root2}/licenses/FA4-{PurePosixPath(name).name}"] = data
    combined[f"{root2}/WHEEL"] = fa2[f"{old2}/WHEEL"]

    metadata = read_metadata(fa2)
    metadata.replace_header("Version", manifest["combined_version"])
    if "Requires-Python" in metadata:
        del metadata["Requires-Python"]
    metadata["Requires-Python"] = manifest["requires_python"]
    requirements = metadata.get_all("Requires-Dist", []) + read_metadata(fa4).get_all("Requires-Dist", [])
    del metadata["Requires-Dist"]
    pins = manifest["runtime_pins"]
    requirements = {requirement for requirement in requirements if requirement_name(requirement) not in pins}
    requirements.update(pins.values())
    for requirement in sorted(requirements):
        metadata["Requires-Dist"] = requirement
    combined[f"{root2}/METADATA"] = metadata.as_bytes(policy=policy.compat32.clone(max_line_length=0))
    combined[f"{root2}/{PROVENANCE}"] = json_bytes({"build": manifest, "payload_sha256": inventory})

    companion = {
        f"{root4}/METADATA": (
            "Metadata-Version: 2.4\nName: flash-attn-4\n"
            f"Version: {manifest['companion_version']}\n"
            "Summary: FA4 metadata companion for SkyRL's combined FA2/FA4 wheel\n"
            f"Requires-Python: {manifest['requires_python']}\n"
            f"Requires-Dist: flash-attn=={manifest['combined_version']}\n"
            "Provides-Extra: cu13\nLicense-Expression: BSD-3-Clause\n\n"
        ).encode(),
        f"{root4}/WHEEL": b"Wheel-Version: 1.0\nGenerator: skyrl-fa4-repack\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
    }
    for name, data in licenses.items():
        companion[f"{root4}/licenses/{PurePosixPath(name).name}"] = data
    tag = manifest["inputs"]["fa2"]["filename"].split("-", 2)[2]
    output.mkdir(parents=True, exist_ok=True)
    paths = [
        output / f"flash_attn-{manifest['combined_version']}-{tag}",
        output / f"flash_attn_4-{manifest['companion_version']}-py3-none-any.whl",
    ]
    write_wheel(paths[0], combined)
    write_wheel(paths[1], companion)
    release = {
        "build": manifest,
        "artifacts": [
            {
                "filename": path.name,
                "sha256": sha256(path),
                "size": path.stat().st_size,
                "url": f"https://github.com/{manifest['release_repository']}/releases/download/{manifest['release_tag']}/{path.name.replace('+', '%2B')}",
            }
            for path in paths
        ],
    }
    (output / "release-manifest.json").write_bytes(json_bytes(release))
    (output / "SHA256SUMS").write_text(
        "".join(f"{item['sha256']}  {item['filename']}\n" for item in release["artifacts"])
    )
    return release


def download_inputs(manifest, cache):
    cache.mkdir(parents=True, exist_ok=True)
    for spec in manifest["inputs"].values():
        path = cache / spec["filename"]
        if not path.exists():
            temporary = path.with_suffix(".download")
            request = urllib.request.Request(spec["url"], headers={"User-Agent": "SkyRL-wheel-builder/skyrl1"})
            with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as stream:
                shutil.copyfileobj(response, stream)
            if sha256(temporary) != spec["sha256"]:
                temporary.unlink()
                raise ValueError(f"Input SHA-256 mismatch: {spec['filename']}")
            temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=HERE / "inputs.json")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if args.download:
        download_inputs(manifest, args.cache)
    print(json.dumps(build_pair(manifest, args.cache, args.output), indent=2))


if __name__ == "__main__":
    main()
