"""Upstream kinds contributed from outside this package.

The registry is open by design: a provider is a class and a ``register()``
call. Importing the module is what registers it, which is all an in-process
caller has to do.

A separate ``skyrl-capture serve`` imports only this package, so it has to be told:

    skyrl-capture serve --upstream-module my_project.capture_upstream

The names travel in ``Config.upstream_modules`` rather than the environment, so
an embedded caller that builds a `Config` in Python configures them the same
way a command line does. One list, loaded in ``build_runtime``.

Why an upstream kind belongs to the caller and not here: the wire is a property
of the engine, and capture is framework neutral. Shipping a wire for every
trainer would mean this package tracking each of their HTTP shapes, which is
the coupling the registry exists to avoid.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from importlib import import_module

logger = logging.getLogger(__name__)


def load_modules(names: Iterable[str], *, source: str = "--upstream-module") -> tuple[str, ...]:
    """Import each module, which is what registers what it contributes.

    Names may be comma- or space-separated within an entry, so one environment
    variable and repeated flags behave the same.

    Raises rather than warning. A module that did not load surfaces much later
    as a target refused for an unknown type, which is a confusing way to learn
    that a path has a typo.
    """
    imported: list[str] = []
    for entry in names:
        for name in (part for chunk in entry.split(",") for part in chunk.split()):
            try:
                import_module(name)
            except Exception as error:
                raise RuntimeError(
                    f"{source} names {name!r}, which could not be imported: {error}"
                ) from error
            imported.append(name)
    return tuple(imported)
