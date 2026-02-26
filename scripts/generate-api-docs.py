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

import yaml
from griffe import GriffeLoader
from griffe2md import render_object_docs

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
DOCS_API_REF = os.path.join(REPO_ROOT, "docs", "content", "docs", "api-ref")
SEARCH_PATHS = [REPO_ROOT, os.path.join(REPO_ROOT, "skyrl-gym")]

# ---------------------------------------------------------------------------
# Page definitions — loaded from api-pages.yaml
# Each group has a `path` that derives `output_dir` and `url_prefix`.
# ---------------------------------------------------------------------------

PAGES_YAML = os.path.join(os.path.dirname(__file__), "api-pages.yaml")


def load_pages(path=PAGES_YAML):
    """Load page definitions from YAML config."""
    with open(path) as f:
        raw_groups = yaml.safe_load(f)
    pages = []
    for group in raw_groups:
        group_path = group["path"]
        for page in group["pages"]:
            page["output_dir"] = group_path
            page["url_prefix"] = f"/docs/api-ref/{group_path}"
            pages.append(page)
    return pages


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
    "module": (
        "module",
        "bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200",
    ),
    "class": (
        "class",
        "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200",
    ),
    "function": (
        "method",
        "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200",
    ),
    "attribute": (
        "attr",
        "bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200",
    ),
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
    """Map object path prefixes to (url_prefix, slug) for cross-references."""
    mapping = {}
    for page in pages:
        val = (page["url_prefix"], page["slug"])
        for section in page["sections"]:
            for obj_path in section["objects"]:
                mapping[obj_path] = val
                # Also map the parent module path, but skip top-level packages
                parts = obj_path.rsplit(".", 1)
                if len(parts) > 1 and "." in parts[0]:
                    mapping[parts[0]] = val
    return mapping


def slugify_anchor(name: str, kind: str = None, prop_labels: set = None) -> str:
    """Convert heading text to fumadocs anchor format."""
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
    """Create an inline HTML badge for a property label."""
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

        badges = [make_kind_badge(kind)]
        prop_labels = labels_map.get(obj_path, set()) if obj_path else set()
        for lbl in sorted(prop_labels & PROPERTY_LABELS):
            badges.append(make_prop_badge(lbl))

        badge_str = " ".join(badges)
        return f"{hashes} {badge_str} `{name}`"

    md = re.sub(r"(#{2,6}) `([^`]+)`", replace_heading, md)
    return md


def fix_cross_references(
    md: str,
    obj_path: str,
    obj_slug_map: dict,
    kind_map: dict,
    labels_map: dict,
    current_url_prefix: str,
    current_slug: str,
) -> str:
    """Fix griffe2md anchor links to work across fumadocs pages."""

    def replace_link(match):
        link_text = match.group(1)
        anchor = match.group(2)

        for prefix, (url_prefix, slug) in sorted(obj_slug_map.items(), key=lambda x: -len(x[0])):
            if anchor.startswith(prefix):
                kind = kind_map.get(anchor)
                prop_labels = labels_map.get(anchor, set())
                name = anchor.split(".")[-1]
                fumadocs_anchor = slugify_anchor(name, kind, prop_labels)

                if url_prefix == current_url_prefix and slug == current_slug:
                    return f"[{link_text}](#{fumadocs_anchor})"
                else:
                    return f"[{link_text}]({url_prefix}/{slug}#{fumadocs_anchor})"

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
            parts = re.split(r"(`[^`]+`)", line)
            new_parts = []
            for i, part in enumerate(parts):
                if i % 2 == 0:
                    part = part.replace("{", "\\{").replace("}", "\\}")
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


def add_source_blocks(md: str, obj, search_paths: list) -> str:
    """Add a single collapsible source code block for the top-level object (class/function)."""
    try:
        fp = obj.filepath
        start = obj.lineno
        end = obj.endlineno
        if not (fp and start and end):
            return md
    except Exception:
        return md

    # Find the relative path
    rel_path = None
    sp_found = None
    for sp in search_paths:
        abs_sp = os.path.abspath(sp)
        abs_fp = os.path.abspath(str(fp))
        if abs_fp.startswith(abs_sp):
            rel_path = os.path.relpath(abs_fp, abs_sp)
            sp_found = sp
            break

    if not rel_path:
        return md

    # Insert the source block just before the first member heading
    # (one level deeper than root), i.e. after the parameters table.
    lines = md.split("\n")
    root_level = None
    insert_idx = None
    for idx, line in enumerate(lines):
        m = re.match(r"^(#{2,6})\s", line)
        if m:
            level = len(m.group(1))
            if root_level is None:
                root_level = level
            elif level > root_level:
                # First member heading — insert just before it
                insert_idx = idx
                break

    if insert_idx is None:
        return md

    # Build the source block
    try:
        with open(os.path.join(sp_found, rel_path)) as sf:
            src_lines = sf.readlines()
        src = "".join(src_lines[start - 1 : end])
        source_block = [
            "",
            "<details>",
            f"<summary>Source code in `{rel_path}:{start}-{end}`</summary>",
            "",
            "```python",
            src.rstrip(),
            "```",
            "",
            "</details>",
            "",
        ]
    except Exception:
        return md

    return "\n".join(lines[:insert_idx] + source_block + lines[insert_idx:])


def render_object(loader, obj_path, kind_map, labels_map, obj_slug_map, page):
    """Render a single object (class/function/module) to processed MDX."""
    parts = obj_path.split(".")

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
        md = render_object_docs(obj, config=config)
    except Exception as e:
        return f"\n{{/* Error rendering {obj_path}: {e} */}}\n"

    md = sanitize_for_mdx(md)
    md = fix_cross_references(
        md,
        obj_path,
        obj_slug_map,
        kind_map,
        labels_map,
        page["url_prefix"],
        page["slug"],
    )
    md = add_type_labels(md, kind_map, labels_map)
    md = add_source_blocks(md, obj, SEARCH_PATHS)

    return md


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    all_pages = load_pages()
    loader = GriffeLoader(search_paths=SEARCH_PATHS, docstring_parser="google")

    # Preload stdlib modules so griffe can resolve aliases
    for stdlib_mod in [
        "__future__",
        "os",
        "dataclasses",
        "typing",
        "enum",
        "abc",
        "pathlib",
        "collections",
        "pydantic",
    ]:
        try:
            loader.load(stdlib_mod)
        except Exception:
            pass

    kind_map, labels_map = build_global_maps(loader, all_pages)
    obj_slug_map = build_page_slug_map(all_pages)

    for page in all_pages:
        output_dir = os.path.join(DOCS_API_REF, page["output_dir"])
        os.makedirs(output_dir, exist_ok=True)

        slug = page["slug"]
        outpath = os.path.join(output_dir, f"{slug}.mdx")
        parts = []

        parts.append(
            f"""---
title: "{page['title']}"
description: "{page['description']}"
---
"""
        )

        for section in page["sections"]:
            if section["heading"]:
                parts.append(f"\n## {section['heading']}\n")
            if section["description"]:
                parts.append(f"\n{section['description']}\n")

            for obj_path in section["objects"]:
                obj_md = render_object(
                    loader,
                    obj_path,
                    kind_map,
                    labels_map,
                    obj_slug_map,
                    page,
                )
                if obj_md:
                    parts.append(obj_md)

        content = "\n".join(parts)

        with open(outpath, "w") as f:
            f.write(content)
        print(f"OK: {page['output_dir']}/{slug}.mdx ({len(content)} bytes)")


if __name__ == "__main__":
    main()
