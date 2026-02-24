#!/usr/bin/env python3
"""Generate API reference docs for fumadocs using griffe2md.

Uses a curated approach: each page defines sections with hand-written descriptions,
and specific classes/functions are rendered individually (like mkdocstrings :::
directives). This matches the style of the mkdocs API reference.

Usage:
    python scripts/generate-api-docs.py

Requires: pip install griffe2md
"""

import os
import re

from griffe import GriffeLoader
from griffe2md import render_object_docs

# ---------------------------------------------------------------------------
# Page definitions — curated like the mkdocs API reference
# Each page has sections, each section has a description and list of objects
# to render. This mirrors the mkdocs `:::` directive approach.
# ---------------------------------------------------------------------------

PAGES = [
    {
        "slug": "trainer",
        "title": "Trainer",
        "description": "Trainer API - The Trainer drives the training loop.",
        "sections": [
            {
                "heading": "Trainer Class",
                "description": "",
                "objects": ["skyrl_train.trainer.RayPPOTrainer"],
            },
            {
                "heading": "Dispatch APIs",
                "description": "",
                "objects": [
                    "skyrl_train.distributed.dispatch.WorkerDispatch",
                ],
            },
            {
                "heading": "Actor APIs",
                "description": "The base worker abstraction in SkyRL.",
                "objects": [
                    "skyrl_train.trainer.PPORayActorGroup",
                ],
            },
        ],
    },
    {
        "slug": "data",
        "title": "Data Interface",
        "description": "Data Interface - Training data types and structures.",
        "sections": [
            {
                "heading": "Generator APIs",
                "description": "",
                "objects": [
                    "skyrl_train.generators.base.GeneratorInput",
                    "skyrl_train.generators.base.GeneratorOutput",
                    "skyrl_train.generators.base.MetricsOutput",
                ],
            },
        ],
    },
    {
        "slug": "generators",
        "title": "Generator",
        "description": "Generator API - The Generator generates trajectories for training.",
        "sections": [
            {
                "heading": "Core APIs",
                "description": "",
                "objects": [
                    "skyrl_train.generators.base.GeneratorInterface",
                    "skyrl_train.inference_engines.base.InferenceEngineInterface",
                    "skyrl_train.inference_engines.inference_engine_client.InferenceEngineClient",
                ],
            },
        ],
    },
    {
        "slug": "entrypoints",
        "title": "Entrypoint",
        "description": "Entrypoint API - Training and evaluation entrypoints.",
        "sections": [
            {
                "heading": "Training Entrypoint",
                "description": "The main entrypoint is the `BasePPOExp` class which runs the main training loop.",
                "objects": [
                    "skyrl_train.entrypoints.main_base.BasePPOExp",
                ],
            },
            {
                "heading": "Evaluation Entrypoint",
                "description": "The evaluation-only entrypoint is the `EvalOnlyEntrypoint` class which runs evaluation without training.",
                "objects": [
                    "skyrl_train.entrypoints.main_generate.EvalOnlyEntrypoint",
                ],
            },
        ],
    },
    {
        "slug": "env-vars",
        "title": "Environment Variables",
        "description": "Configuration via environment variables.",
        "sections": [
            {
                "heading": None,  # No section heading, render directly
                "description": "",
                "objects": ["skyrl_train.env_vars"],
            },
        ],
    },
    {
        "slug": "registry",
        "title": "Algorithm Registry",
        "description": "Algorithm Registry API - Register custom algorithm functions.",
        "sections": [
            {
                "heading": "Base Registry Classes",
                "description": "The registry system provides a way to register and manage custom algorithm functions across distributed Ray environments.",
                "objects": [
                    "skyrl_train.utils.ppo_utils.BaseFunctionRegistry",
                    "skyrl_train.utils.ppo_utils.RegistryActor",
                    "skyrl_train.utils.ppo_utils.sync_registries",
                ],
            },
            {
                "heading": "Advantage Estimator Registry",
                "description": "The advantage estimator registry manages functions that compute advantages and returns.",
                "objects": [
                    "skyrl_train.utils.ppo_utils.AdvantageEstimatorRegistry",
                    "skyrl_train.utils.ppo_utils.AdvantageEstimator",
                    "skyrl_train.utils.ppo_utils.register_advantage_estimator",
                ],
            },
            {
                "heading": "Policy Loss Registry",
                "description": "The policy loss registry manages functions that compute policy losses for PPO.",
                "objects": [
                    "skyrl_train.utils.ppo_utils.PolicyLossRegistry",
                    "skyrl_train.utils.ppo_utils.PolicyLossType",
                    "skyrl_train.utils.ppo_utils.register_policy_loss",
                ],
            },
        ],
    },
]

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "content", "docs", "api-ref", "skyrl", "skyrl-train")

SEARCH_PATHS = [os.path.join(os.path.dirname(__file__), "..", "skyrl-train")]

# Config for rendering individual objects
RENDER_CONFIG = {
    "heading_level": 3,
    "show_if_no_docstring": True,
    "members_order": "source",
    "show_signature_annotations": True,
    "separate_signature": True,
    "docstring_section_style": "table",
    "show_object_full_path": False,
    "show_root_full_path": False,
    "show_root_members_full_path": False,
    "show_bases": True,
}

# Kind label styles: (display_text, tailwind_classes)
LABEL_STYLES = {
    "module": ("module", "bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200"),
    "class": ("class", "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200"),
    "function": ("method", "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200"),
    "attribute": ("attr", "bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200"),
}

# Property labels shown as additional badges (from griffe obj.labels)
PROPERTY_LABELS = {
    "async",
    "classmethod",
    "staticmethod",
    "property",
    "abstractmethod",
}
PROPERTY_LABEL_STYLE = "bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300"

BADGE_CLASSES = "inline-block px-1.5 py-0.5 rounded text-xs font-medium align-middle"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_global_maps(loader, pages):
    """Build kind_map and labels_map for all objects across all pages."""
    kind_map = {}
    labels_map = {}

    def walk(obj):
        try:
            kind = obj.kind.value
        except Exception:
            return
        kind_map[obj.path] = kind
        try:
            labels_map[obj.path] = obj.labels
        except Exception:
            labels_map[obj.path] = set()
        try:
            for member in obj.members.values():
                walk(member)
        except Exception:
            pass

    for page in pages:
        for section in page["sections"]:
            for obj_path in section["objects"]:
                try:
                    parts = obj_path.split(".")
                    for i in range(len(parts), 0, -1):
                        try:
                            mod = loader.load(".".join(parts[:i]))
                            break
                        except Exception:
                            continue
                    walk(mod)
                except Exception:
                    pass
    return kind_map, labels_map


def build_page_slug_map(pages):
    """Map object path prefixes to page slugs for cross-references."""
    mapping = {}
    for page in pages:
        for section in page["sections"]:
            for obj_path in section["objects"]:
                mapping[obj_path] = page["slug"]
                # Also map the parent module path, but skip top-level packages
                # (e.g. "skyrl_train") to avoid overly broad prefix matching
                parts = obj_path.rsplit(".", 1)
                if len(parts) > 1 and "." in parts[0]:
                    mapping[parts[0]] = page["slug"]
    return mapping


def slugify_anchor(name: str, kind: str = None, prop_labels: set = None) -> str:
    """Convert heading text to fumadocs anchor format.

    Labels appear before the name in the heading, so the anchor is:
      <kind>-<props>-<name>
    e.g. `class RayPPOTrainer` -> `class-rayppotrainer`
         `method async eval`  -> `method-async-eval`
    """
    parts = []
    if kind and kind in LABEL_STYLES:
        parts.append(LABEL_STYLES[kind][0])
    if prop_labels:
        for lbl in sorted(prop_labels & PROPERTY_LABELS):
            parts.append(lbl)
    slug = re.sub(r"[^a-zA-Z0-9_\- ]", "", name).lower()
    parts.append(slug)
    return "-".join(parts)


def make_kind_badge(kind: str) -> str:
    """Create an inline HTML badge for an object kind."""
    if kind not in LABEL_STYLES:
        return ""
    text, classes = LABEL_STYLES[kind]
    return f'<span className="{classes} {BADGE_CLASSES}">{text}</span>'


def make_prop_badge(label: str) -> str:
    """Create an inline HTML badge for a property label (async, classmethod, etc.)."""
    return f'<span className="{PROPERTY_LABEL_STYLE} {BADGE_CLASSES}">{label}</span>'


def add_type_labels(md: str, kind_map: dict, labels_map: dict) -> str:
    """Add colored type labels to headings (before the object name)."""

    def replace_heading(match):
        hashes = match.group(1)
        name = match.group(2)

        kind = None
        obj_path = None
        if name in kind_map:
            kind = kind_map[name]
            obj_path = name
        else:
            for path, k in kind_map.items():
                if path.endswith("." + name) or path == name:
                    kind = k
                    obj_path = path
                    break

        if not kind:
            return f"{hashes} `{name}`"

        # Build badges: kind first, then property labels
        badges = [make_kind_badge(kind)]
        prop_labels = labels_map.get(obj_path, set()) if obj_path else set()
        for lbl in sorted(prop_labels & PROPERTY_LABELS):
            badges.append(make_prop_badge(lbl))

        badge_str = " ".join(badges)
        return f"{hashes} {badge_str} `{name}`"

    md = re.sub(r"(#{2,6}) `([^`]+)`", replace_heading, md)
    return md


def fix_cross_references(md: str, obj_path: str, obj_slug_map: dict, kind_map: dict, labels_map: dict) -> str:
    """Fix griffe2md anchor links to work across fumadocs pages."""

    def replace_link(match):
        link_text = match.group(1)
        anchor = match.group(2)

        for prefix, slug in sorted(obj_slug_map.items(), key=lambda x: -len(x[0])):
            if anchor.startswith(prefix):
                kind = kind_map.get(anchor)
                prop_labels = labels_map.get(anchor, set())
                name = anchor.split(".")[-1]
                fumadocs_anchor = slugify_anchor(name, kind, prop_labels)

                current_slug = obj_slug_map.get(obj_path, "")
                if slug == current_slug:
                    return f"[{link_text}](#{fumadocs_anchor})"
                else:
                    return f"[{link_text}](/docs/api-ref/skyrl/skyrl-train/{slug}#{fumadocs_anchor})"

        # External reference — strip the link, keep text
        return link_text

    md = re.sub(r"\[([^\]]*)\]\(#([^)]+)\)", replace_link, md)
    return md


def sanitize_for_mdx(md: str) -> str:
    """Make griffe2md output compatible with MDX."""

    def replace_details(match):
        content = match.group(0)
        summary_match = re.search(r"<summary>(.*?)</summary>", content)
        summary = summary_match.group(1) if summary_match else "Details"
        body_match = re.search(r"</summary>\s*(.*?)\s*</details>", content, re.DOTALL)
        body = body_match.group(1).strip() if body_match else ""
        return f"**{summary}:**\n\n{body}"

    md = re.sub(r"<details[^>]*>.*?</details>", replace_details, md, flags=re.DOTALL)

    def replace_code_tag(match):
        inner = match.group(1)
        if re.search(r"\[.*?\]\(.*?\)", inner):
            return inner
        return f"`{inner}`"

    md = re.sub(r"<code>(.*?)</code>", replace_code_tag, md)

    # Convert HTML comments to MDX comments
    md = re.sub(r"<!--(.*?)-->", r"{/*\1*/}", md, flags=re.DOTALL)

    # Escape curly braces outside of code blocks, inline code, and JSX
    lines = md.split("\n")
    in_code_block = False
    result_lines = []
    for line in lines:
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
        if not in_code_block and "className=" not in line and "{/*" not in line:
            # Escape { and } and angle brackets that aren't inside inline code
            parts = re.split(r"(`[^`]+`)", line)
            new_parts = []
            for i, part in enumerate(parts):
                if i % 2 == 0:
                    part = part.replace("{", "\\{").replace("}", "\\}")
                    # Escape angle brackets that look like JSX tags but aren't
                    # real HTML (e.g. <request-body>, <headers-dict>)
                    # Skip actual HTML tags (span, div, details, summary, etc)
                    part = re.sub(
                        r"<(?!/?(?:span|div|p|a|br|hr|img|table|tr|td|th|ul|ol|li|strong|em|code|pre|h[1-6]|details|summary)\b)([a-zA-Z][a-zA-Z0-9_-]*(?:-[a-zA-Z0-9_]+)*)>",
                        r"&lt;\1&gt;",
                        part,
                    )
                new_parts.append(part)
            line = "".join(new_parts)
        result_lines.append(line)
    md = "\n".join(result_lines)

    return md


def add_source_blocks(md: str, obj, search_path: str) -> str:
    """Append collapsible source code blocks after each member heading + body.

    Inserts a <details> block with the source code after each rendered
    member (identified by headings deeper than the root heading level).
    """
    if not hasattr(obj, "members"):
        return md

    # Build map of member name -> source info (members only, not root)
    source_info = {}
    for name, member in obj.members.items():
        try:
            fp = member.filepath
            start = member.lineno
            end = member.endlineno
            if fp and start and end:
                rel_path = os.path.relpath(str(fp), search_path)
                source_info[name] = (rel_path, start, end)
        except Exception:
            continue

    if not source_info:
        return md

    # Find root heading level (first heading in the markdown)
    root_level = None
    for line in md.split("\n"):
        m = re.match(r"^(#{2,6})\s", line)
        if m:
            root_level = len(m.group(1))
            break

    if root_level is None:
        return md

    # Insert source blocks after each member section (deeper than root level)
    lines = md.split("\n")
    result = []
    i = 0
    while i < len(lines):
        line = lines[i]
        heading_match = re.match(r"^(#{2,6})\s.*`([^`]+)`", line)
        heading_level = len(heading_match.group(1)) if heading_match else 0

        # Only add source for member headings (deeper than root)
        if heading_match and heading_level > root_level and heading_match.group(2) in source_info:
            member_name = heading_match.group(2)
            rel_path, start, end = source_info[member_name]

            result.append(line)
            i += 1

            # Collect all lines until the next heading of same or higher level
            while i < len(lines):
                next_heading = re.match(r"^(#{2,6})\s", lines[i])
                if next_heading and len(next_heading.group(1)) <= heading_level:
                    break
                result.append(lines[i])
                i += 1

            # Insert source block
            result.append("")
            result.append("<details>")
            result.append(f"<summary>Source code in `{rel_path}:{start}-{end}`</summary>")
            result.append("")
            try:
                with open(os.path.join(search_path, rel_path)) as sf:
                    src_lines = sf.readlines()
                src = "".join(src_lines[start - 1 : end])
                result.append("```python")
                result.append(src.rstrip())
                result.append("```")
            except Exception:
                result.append(f"*Could not read source from `{rel_path}`*")
            result.append("")
            result.append("</details>")
            result.append("")
        else:
            result.append(line)
            i += 1

    return "\n".join(result)


def render_object(loader, obj_path, kind_map, labels_map, obj_slug_map):
    """Render a single object (class/function/module) to processed MDX."""
    parts = obj_path.split(".")

    # Navigate to the object: load the top module, then walk to the member
    obj = None
    for i in range(len(parts), 0, -1):
        try:
            mod = loader.load(".".join(parts[:i]))
            obj = mod
            for part in parts[i:]:
                obj = obj.members[part]
            break
        except Exception:
            continue

    if obj is None:
        return f"\n{{/* Could not load {obj_path} */}}\n"

    try:
        config = dict(RENDER_CONFIG)
        # For regular classes, filter out attribute members to keep docs
        # focused on the API. Keep attributes for data classes (TypedDict,
        # dataclass, etc.) where fields ARE the API.
        if obj.kind.value == "class":
            bases = [b.name if hasattr(b, "name") else str(b) for b in getattr(obj, "bases", [])]
            is_data_class = any(b in ("TypedDict", "NamedTuple") or "dataclass" in b.lower() for b in bases)
            if not is_data_class:
                config["members"] = [name for name, member in obj.members.items() if member.kind.value != "attribute"]
        md = render_object_docs(obj, config=config)
    except Exception as e:
        return f"\n{{/* Error rendering {obj_path}: {e} */}}\n"

    md = sanitize_for_mdx(md)
    md = fix_cross_references(md, obj_path, obj_slug_map, kind_map, labels_map)
    md = add_type_labels(md, kind_map, labels_map)
    md = add_source_blocks(md, obj, SEARCH_PATHS[0])

    return md


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    loader = GriffeLoader(search_paths=SEARCH_PATHS, docstring_parser="google")

    # Preload stdlib modules so griffe can resolve aliases like `import os`
    for stdlib_mod in ["os", "dataclasses", "typing", "enum", "abc", "pathlib", "collections"]:
        try:
            loader.load(stdlib_mod)
        except Exception:
            pass

    kind_map, labels_map = build_global_maps(loader, PAGES)
    obj_slug_map = build_page_slug_map(PAGES)

    for page in PAGES:
        slug = page["slug"]
        outpath = os.path.join(OUTPUT_DIR, f"{slug}.mdx")
        parts = []

        # Frontmatter
        parts.append(
            f"""---
title: "{page['title']}"
description: "{page['description']}"
---
"""
        )

        for section in page["sections"]:
            # Section heading
            if section["heading"]:
                parts.append(f"\n## {section['heading']}\n")
            if section["description"]:
                parts.append(f"\n{section['description']}\n")

            # Render each object
            for obj_path in section["objects"]:
                obj_md = render_object(loader, obj_path, kind_map, labels_map, obj_slug_map)
                if obj_md:
                    parts.append(obj_md)

        content = "\n".join(parts)

        with open(outpath, "w") as f:
            f.write(content)
        print(f"OK: {slug}.mdx ({len(content)} bytes)")


if __name__ == "__main__":
    main()
