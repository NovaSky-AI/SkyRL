#!/usr/bin/env python3
"""
Convert RST documentation files to MDX format.
"""

import os
import re
import sys
from pathlib import Path

def extract_title(content):
    """Extract the first heading from RST content."""
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if i > 0 and len(line) > 0:
            # Check if previous line is a title (next line is all = or -)
            if all(c in '=-~`' for c in line) and len(line) >= 3:
                title = lines[i-1].strip()
                if title:
                    return title
    return "Untitled"

def convert_headings(content):
    """Convert RST headings to Markdown format."""
    lines = content.split('\n')
    result = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Check if next line is an underline
        if i + 1 < len(lines):
            next_line = lines[i + 1]
            if len(next_line) > 0 and all(c in '=-~`^' for c in next_line) and len(next_line) >= 3:
                title = line.strip()
                if title:
                    # Determine heading level based on character
                    char = next_line[0]
                    if char == '=':
                        heading = f"# {title}"
                    elif char == '-':
                        heading = f"## {title}"
                    elif char == '~':
                        heading = f"### {title}"
                    elif char == '^':
                        heading = f"#### {title}"
                    else:
                        heading = f"### {title}"

                    result.append(heading)
                    result.append('')  # Add blank line after heading
                    i += 2  # Skip both the title and underline
                    continue

        result.append(line)
        i += 1

    return '\n'.join(result)

def convert_code_blocks(content):
    """Convert RST code blocks to Markdown."""
    lines = content.split('\n')
    result = []
    in_code_block = False
    code_language = ""
    code_lines = []
    code_indent = 0

    i = 0
    while i < len(lines):
        line = lines[i]

        # Check for code block directive
        code_block_match = re.match(r'\.\.\s+code-block::\s*(\w*)', line)
        if code_block_match:
            code_language = code_block_match.group(1) or ""
            in_code_block = True
            code_lines = []
            code_indent = 0
            i += 1
            # Skip blank lines after directive and detect indentation from first non-empty line
            while i < len(lines) and lines[i].strip() == '':
                i += 1
            # Detect indentation level from first code line
            if i < len(lines) and lines[i].strip():
                code_indent = len(lines[i]) - len(lines[i].lstrip())
            continue

        if in_code_block:
            # Check if we're still in the code block (indented or blank)
            if line.strip() == '':
                code_lines.append('')
                i += 1
                continue
            elif len(line) - len(line.lstrip()) >= code_indent and code_indent > 0:
                # Remove the detected indentation
                code_lines.append(line[code_indent:])
                i += 1
                continue
            else:
                # End of code block - use placeholder to avoid backtick conversion
                result.append(f'<<<CODEBLOCK_START:{code_language}>>>')
                result.extend(code_lines)
                result.append('<<<CODEBLOCK_END>>>')
                result.append('')
                in_code_block = False
                code_language = ""
                code_lines = []
                code_indent = 0
                # Process current line normally
                result.append(line)
                i += 1
                continue

        result.append(line)
        i += 1

    # Close any open code block at end of file
    if in_code_block:
        result.append(f'<<<CODEBLOCK_START:{code_language}>>>')
        result.extend(code_lines)
        result.append('<<<CODEBLOCK_END>>>')

    return '\n'.join(result)

def convert_inline_code(content):
    """Convert RST inline code (double backticks) to Markdown (single backticks)."""
    # Convert ``code`` to `code`
    content = re.sub(r'``([^`]+)``', r'`\1`', content)
    return content

def convert_links(content):
    """Convert RST links to Markdown."""
    # External links: `text <url>`_
    content = re.sub(r'`([^<`]+)\s+<([^>]+)>`_', r'[\1](\2)', content)

    # Internal doc links: :doc:`text <path>`
    content = re.sub(r':doc:`([^<`]+)\s+<([^>]+)>`', r'[\1](\2)', content)
    # Internal doc links: :doc:`path`
    content = re.sub(r':doc:`([^`]+)`', r'[\1](\1)', content)

    # Ref links: :ref:`label` - convert to anchor links
    content = re.sub(r':ref:`([^`]+)`', r'[\1](#\1)', content)

    # Remove other role markers like :code_link:
    content = re.sub(r':code_link:`([^`]+)`', r'`\1`', content)

    return content

def convert_directives(content):
    """Convert or remove RST directives."""
    # Remove .. _label: style reference markers
    content = re.sub(r'\.\.\s+_[^:]+:', '', content)

    # Remove .. toctree:: directives and their content
    content = re.sub(r'\.\. toctree::[^\n]*\n(\s+:[^\n]+\n)*(\s+[^\n]+\n)*', '', content)

    # Convert .. note:: to markdown blockquote
    content = re.sub(r'\.\. note::\s*\n\s*\n', '\n> **Note:** ', content)

    # Convert .. warning:: to markdown blockquote
    content = re.sub(r'\.\. warning::\s*\n\s*\n', '\n> **Warning:** ', content)

    # Convert .. tip:: to markdown blockquote
    content = re.sub(r'\.\. tip::\s*\n\s*\n', '\n> **Tip:** ', content)

    # Remove other directives, but EXCLUDE code-block which is handled separately
    content = re.sub(r'\.\. (?!code-block)[a-zA-Z0-9_-]+::[^\n]*\n', '', content)

    return content

def convert_image_paths(content):
    """Update image paths to /images/."""
    content = re.sub(r'\.\. image::\s+([^\n]+)', r'![Image](/images/\1)', content)
    content = re.sub(r'\.\. figure::\s+([^\n]+)', r'![Figure](/images/\1)', content)
    return content

def clean_content(content):
    """Clean up the converted content."""
    # Replace code block placeholders with actual markdown
    content = re.sub(r'<<<CODEBLOCK_START:([^>]*)>>>', r'```\1', content)
    content = re.sub(r'<<<CODEBLOCK_END>>>', r'```', content)

    # Remove multiple blank lines
    content = re.sub(r'\n\n\n+', '\n\n', content)

    # Remove leading/trailing whitespace
    content = content.strip()

    return content

def convert_rst_to_mdx(rst_content, title=None):
    """Main conversion function."""
    # Extract title if not provided
    if title is None:
        title = extract_title(rst_content)

    # Apply conversions in order
    # CRITICAL: convert_code_blocks MUST come before convert_directives
    # because directives removal would otherwise catch code-block directives
    content = rst_content
    content = convert_headings(content)
    content = convert_code_blocks(content)  # MUST be before convert_directives
    content = convert_links(content)
    content = convert_directives(content)
    content = convert_image_paths(content)
    content = convert_inline_code(content)  # Must come after code blocks
    content = clean_content(content)

    # Create frontmatter
    frontmatter = f"""---
title: "{title}"
description: "{title}"
---

"""

    return frontmatter + content

def process_file(src_path, dst_path):
    """Convert a single RST file to MDX."""
    print(f"Converting: {src_path} -> {dst_path}")

    # Read source file
    with open(src_path, 'r', encoding='utf-8') as f:
        rst_content = f.read()

    # Convert to MDX
    mdx_content = convert_rst_to_mdx(rst_content)

    # Create destination directory
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    # Write MDX file
    with open(dst_path, 'w', encoding='utf-8') as f:
        f.write(mdx_content)

    print(f"  ✓ Created {dst_path}")

def main():
    source_dir = Path("/home/tyler/SkyRL/skyrl-train/docs")
    target_dir = Path("/home/tyler/SkyRL/skyrl-docs/content/docs")

    # Find all RST files excluding api/
    rst_files = []
    for rst_file in source_dir.rglob("*.rst"):
        # Skip api directory
        if '/api/' in str(rst_file):
            continue
        rst_files.append(rst_file)

    print(f"Found {len(rst_files)} RST files to convert\n")

    for rst_file in sorted(rst_files):
        # Calculate relative path from source_dir
        rel_path = rst_file.relative_to(source_dir)

        # Create target path with .mdx extension
        dst_path = target_dir / rel_path.with_suffix('.mdx')

        try:
            process_file(rst_file, dst_path)
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n✓ Conversion complete! Converted {len(rst_files)} files.")

if __name__ == "__main__":
    main()
