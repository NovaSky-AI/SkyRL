"""RLM training example.

Importing this package registers task-specific env subclasses with skyrl_gym.
"""

from . import envs  # noqa: F401  — triggers env registration as a side effect
