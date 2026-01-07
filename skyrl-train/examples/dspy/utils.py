"""
Utility module for DSPy programs with a mapping from program names to program classes
and reward function names to reward functions.
"""

from typing import Dict, Type, Callable, Any
import dspy

# Import programs from different modules
from .lcb.programs import (
    NaiveCodeGenerator,
    CodeGeneratorWithRanker,
    CodeGeneratorWithRanker_prog,
    CodeGeneratorWithRanker_test,
)
from .lcb.lcb_utils import (
    CodeGeneratorWithIteratedRanker,
    CodeGeneratorWithSelfDebug,
)
from .hover.programs import Hover, Hover_query_gen
from .papillon.programs import PAPILLON, PAPILLON_request_gen

# Import reward functions from different modules
from .lcb.utils import (
    local_reward_fn,
    final_reward_fn,
)
from .papillon.utils import (
    compute_overall_score,
    compute_query_leakage,
)
from .hover.utils import (
    hover_final_reward_fn,
    hover_query_reward_fn,
)

# Import data functions from different modules
from .lcb.data import lcb_data
from .papillon.data import papillon_data
from .hover.data import hover_data

# Mapping from program name (string) to program class
DSPY_PROGRAM_MAP: Dict[str, Type[dspy.Module]] = {
    "NaiveCodeGenerator": NaiveCodeGenerator,
    "CodeGeneratorWithRanker": CodeGeneratorWithRanker,
    "CodeGeneratorWithRanker_prog": CodeGeneratorWithRanker_prog,
    "CodeGeneratorWithRanker_test": CodeGeneratorWithRanker_test,
    "CodeGeneratorWithIteratedRanker": CodeGeneratorWithIteratedRanker,
    "CodeGeneratorWithSelfDebug": CodeGeneratorWithSelfDebug,
    "Hover_query_gen": Hover_query_gen,
    "papillon_request_gen": PAPILLON_request_gen,
}

# Mapping from reward function name (string) to reward function
REWARD_FUNCTION_MAP: Dict[str, Callable[..., Any]] = {
    "lcb_assert_test_gen": local_reward_fn,
    "lcb_final_reward_fn": final_reward_fn,
    "papillon_final_reward_fn": compute_overall_score,
    "papillon_query_leakage": compute_query_leakage,
    "hover_final_reward_fn": hover_final_reward_fn,
    "hover_query_reward_fn": hover_query_reward_fn,
}

# Mapping from benchmark name (string) to data function
BENCHMARK_DATA_MAP: Dict[str, Callable[[], tuple]] = {
    "lcb": lcb_data,
    "papillon": papillon_data,
    "hover": hover_data,
}


def get_program(program_name: str) -> Type[dspy.Module]:
    """
    Get a DSPy program class by name.
    
    Args:
        program_name: Name of the program as a string
        
    Returns:
        The program class
        
    Raises:
        KeyError: If the program name is not found in the mapping
    """
    if program_name not in DSPY_PROGRAM_MAP:
        raise KeyError(
            f"Program '{program_name}' not found. Available programs: {list(DSPY_PROGRAM_MAP.keys())}"
        )
    return DSPY_PROGRAM_MAP[program_name]


def get_reward_function(reward_function_name: str) -> Callable[..., Any]:
    """
    Get a reward function by name.
    
    Args:
        reward_function_name: Name of the reward function as a string
        
    Returns:
        The reward function
        
    Raises:
        KeyError: If the reward function name is not found in the mapping
    """
    if reward_function_name not in REWARD_FUNCTION_MAP:
        raise KeyError(
            f"Reward function '{reward_function_name}' not found. Available reward functions: {list(REWARD_FUNCTION_MAP.keys())}"
        )
    return REWARD_FUNCTION_MAP[reward_function_name]


def get_benchmark_data(benchmark_name: str) -> Callable[[], tuple]:
    """
    Get a benchmark data function by name.
    
    Args:
        benchmark_name: Name of the benchmark as a string (e.g., "lcb", "papillon")
        
    Returns:
        The data function that returns (trainset, testset) tuple
        
    Raises:
        KeyError: If the benchmark name is not found in the mapping
    """
    if benchmark_name not in BENCHMARK_DATA_MAP:
        raise KeyError(
            f"Benchmark '{benchmark_name}' not found. Available benchmarks: {list(BENCHMARK_DATA_MAP.keys())}"
        )
    return BENCHMARK_DATA_MAP[benchmark_name]()

