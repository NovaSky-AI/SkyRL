import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "SkyRL-Train"
copyright = "2025, NovaSkyAI"
author = "NovaSkyAI"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx.ext.extlinks",
    "sphinx_autodoc_typehints",
    "myst_parser",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# External links configuration
extlinks = {
    'example_script': ('https://github.com/NovaSky-AI/skyrl/blob/main/skyrl-train/examples/%s', '%s'),
    'example_file': ('https://github.com/NovaSky-AI/skyrl/blob/main/skyrl-train/examples/%s', None),
}

def linkcode_resolve(domain, info):
    """
    Determine the URL corresponding to Python object
    """
    if domain != 'py':
        return None
    if not info['module']:
        return None
    
    # For example files, create direct GitHub links
    filename = info['module'].replace('.', '/')
    
    # Check if it's an example script
    if filename.startswith('examples/'):
        return f"https://github.com/NovaSky-AI/skyrl/blob/main/skyrl-train/{filename}"
    
    return f"https://github.com/NovaSky-AI/skyrl/blob/main/skyrl-train/{filename}.py"
