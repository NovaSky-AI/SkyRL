"""RLM training example.

Importing this package registers task-specific env subclasses with skyrl_gym.
The import is at package level (rather than inside ``main_rlm.py``) so that
env-id registration also fires inside Ray worker processes, which only
deserialize submodules (e.g. ``rlm_generator``) and never re-execute the
training entry point.
"""

from . import multi_paper_env  # noqa: F401  -- triggers env registration as a side effect
