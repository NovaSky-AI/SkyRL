"""Arctic RL integration for SkyRL.

Lives at ``integrations/arctic_rl/`` (PEP 420 namespace under ``integrations/``);
not pip-installed. Invoked via core dispatch::

    python -m skyrl.train.entrypoints.main_base \\
        trainer.override_entrypoint=integrations.arctic_rl.entrypoint <flags>

Provides ``ArcticPPOTrainer`` and ``ArcticGenerator`` that route all GPU work
to an Arctic RL server; depends on ``arctic_platform.rl`` for the client.
"""

from . import envs as _envs  # noqa: F401  side-effect: register `bird` env
from .generator import ArcticGenerator
from .trainer import ArcticPPOTrainer

__all__ = ["ArcticPPOTrainer", "ArcticGenerator"]
