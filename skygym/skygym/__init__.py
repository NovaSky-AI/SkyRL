"""Root `__init__` of the gym module setting the `__all__` of skyrl modules."""

from skygym.core import Env
from skygym import error

from skygym.envs.registration import (
    make,
    spec,
    register,
    registry,
    pprint_registry,
)
from skygym import tools, envs

# Define __all__ to control what's exposed
__all__ = [
    # core classes
    "Env",
    # registration
    "make",
    "spec",
    "register",
    "registry",
    "pprint_registry",
    # module folders
    "envs",
    "tools",
    "error",
]
__version__ = "0.0.0"
