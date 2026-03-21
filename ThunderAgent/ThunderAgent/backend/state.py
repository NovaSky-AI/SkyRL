"""Backend state management."""
import logging
import time
from typing import Optional, Dict, TYPE_CHECKING

from .metrics_base import MetricsClient
from .vllm_metrics import VLLMMetricsClient

if TYPE_CHECKING:
    from ..program.state import Program, ProgramState

from ..program.state import ProgramStatus

logger = logging.getLogger(__name__)

# Buffer tokens reserved per program for decode phase
DECODE_BUFFER = 512

# Default coefficient for acting tokens (can be overridden per backend)
DEFAULT_TOOL_COEFFICIENT = 1.0

# Buffer tokens reserved per active program (for decode headroom)
BUFFER_PER_PROGRAM = 100


class BackendState:
    """State of a single backend with metrics monitoring.
    
    Uses a MetricsClient for backend communication and parsing.
    """
    
    def __init__(
        self,
        url: str,
        tool_coefficient: float = DEFAULT_TOOL_COEFFICIENT,
        metrics_client: Optional[MetricsClient] = None,
        use_acting_token_decay: bool = False,
    ):
        self.url = url
        self.tool_coefficient = tool_coefficient
        self.use_acting_token_decay = use_acting_token_decay
        
        # Metrics client (handles backend communication and parsing)
        if metrics_client is None:
            metrics_client = VLLMMetricsClient(url)
        self.metrics_client = metrics_client
        
        # Program tracking - all token stats are computed from this dict
        # Key is program_id (str), value is Program object
        self._programs: Dict[str, "Program"] = {}
        
        # Shared tokens (prefix cache savings), updated only during thrashing check
        # = reasoning_program_tokens - vllm_actual_used_tokens
        self.shared_tokens: int = 0
        
        # Future paused tokens: sum of tokens from REASONING programs marked for pause
        # These will be released when they transition to ACTING
        self.future_paused_tokens: int = 0
        
        # Flag to skip concurrent scheduling (non-blocking, allows temporary overflow)
        self.scheduling_in_progress: bool = False

    @property
    def completions_url(self) -> str:
        """Chat completions API endpoint."""
        return f"{self.url}/v1/chat/completions"
    
    # Delegate to metrics_client
    @property
    def healthy(self) -> bool:
        """Whether the backend is healthy (from metrics client)."""
        return self.metrics_client.healthy
    
    @property
    def cache_config(self):
        """Cache config from metrics client."""
        return self.metrics_client.cache_config
    
    @property
    def latest_metrics(self):
        """Latest metrics from metrics client."""
        return self.metrics_client.latest_metrics
    
    @property
    def metrics_history(self):
        """Metrics history from metrics client."""
        return self.metrics_client.metrics_history
    
    @property
    def reasoning_program_tokens(self) -> int:
        """Sum of tokens from all REASONING programs."""
        return sum(p.total_tokens for p in self._programs.values() 
                   if p.status == ProgramStatus.REASONING)
    
    @property
    def acting_program_tokens(self) -> int:
        """Sum of tokens from all ACTING programs."""
        return sum(p.total_tokens for p in self._programs.values() 
                   if p.status == ProgramStatus.ACTING)
    
    @property
    def active_program_tokens(self) -> int:
        """Computed: reasoning_tokens + tool_coefficient * acting_tokens."""
        return int(self.reasoning_program_tokens + self.tool_coefficient * self.acting_program_tokens)
    
    @property
    def active_program_count(self) -> int:
        """Number of active (REASONING + ACTING) programs."""
        return len(self._programs)
    
    @property
    def reasoning_program_count(self) -> int:
        """Number of REASONING programs."""
        return sum(1 for p in self._programs.values() if p.status == ProgramStatus.REASONING)
    
    @property
    def acting_program_count(self) -> int:
        """Number of ACTING programs."""
        return sum(1 for p in self._programs.values() if p.status == ProgramStatus.ACTING)
    
    @property
    def total_program_tokens(self) -> int:
        """Sum of tokens from all programs on this backend."""
        return sum(p.total_tokens for p in self._programs.values())
    
    def update_shared_tokens(self) -> None:
        """Update shared_tokens from latest vLLM metrics.
        
        Call this during thrashing check to get fresh prefix cache savings.
        """
        self.shared_tokens = self.metrics_client.calculate_shared_tokens(
            self.reasoning_program_tokens
        )
    
    @property
    def active_program_tokens_ratio(self) -> float:
        """Ratio of active program tokens to total capacity."""
        if not self.cache_config or self.cache_config.total_tokens_capacity == 0:
            return 0.0
        return self.active_program_tokens / self.cache_config.total_tokens_capacity
    
    # -------------------------------------------------------------------------
    # Capacity Check
    # -------------------------------------------------------------------------
    
    def has_capacity(self, extra_tokens: int = 0, extra_count: int = 0) -> bool:
        """Check if adding extra tokens/programs would exceed capacity.
        
        Uses cached shared_tokens (updated during thrashing check).
        Constraint: active_tokens - shared_tokens + buffer <= total_capacity
        where buffer = (active_count + extra_count) * BUFFER_PER_PROGRAM
        """
        if not self.cache_config:
            return True  # No config, assume ok
        
        tokens = self.active_program_tokens + extra_tokens
        count = self.active_program_count + extra_count
        buffer = count * BUFFER_PER_PROGRAM
        required = tokens - self.shared_tokens + buffer
        return required <= self.cache_config.total_tokens_capacity
    
    def capacity_overflow(self, include_future_release: bool = False) -> int:
        """Return how many tokens we're over capacity (0 if within capacity).
        
        Uses cached shared_tokens (updated during thrashing check).
        Formula: active_tokens - shared_tokens + buffer - capacity
        where buffer = active_count * BUFFER_PER_PROGRAM
        
        Args:
            include_future_release: If True, subtract future_paused_tokens from the calculation.
                                   Use this when checking if more programs need to be paused.
        """
        if not self.cache_config:
            return 0
        buffer = self.active_program_count * BUFFER_PER_PROGRAM
        required = self.active_program_tokens - self.shared_tokens + buffer
        if include_future_release:
            required -= self.future_paused_tokens
        overflow = required - self.cache_config.total_tokens_capacity
        return max(0, overflow)
    
    def remaining_capacity(self) -> int:
        """Return remaining capacity for new programs (can be negative if over capacity).
        
        Uses cached shared_tokens (updated during thrashing check).
        """
        if not self.cache_config:
            return float('inf')
        buffer = self.active_program_count * BUFFER_PER_PROGRAM
        used = self.active_program_tokens - self.shared_tokens + buffer
        return self.cache_config.total_tokens_capacity - used

    def remaining_capacity_with_decay(self) -> int:
        """Remaining capacity with exponential decay on acting tokens.

        Each ACTING program's tokens are weighted by 2^(-t), where t is
        seconds since entering ACTING. Used by resume logic to optimistically
        estimate available capacity.
        """
        if not self.cache_config:
            return float('inf')
        now = time.time()
        acting_decayed = 0.0
        for p in self._programs.values():
            if p.status == ProgramStatus.ACTING and p.acting_since is not None:
                t = now - p.acting_since
                acting_decayed += p.total_tokens * (2.0 ** -t)
        effective_tokens = int(self.reasoning_program_tokens + acting_decayed)
        buffer = self.active_program_count * BUFFER_PER_PROGRAM
        used = effective_tokens - self.shared_tokens + buffer
        return self.cache_config.total_tokens_capacity - used

    # -------------------------------------------------------------------------
    # Program Registration
    # -------------------------------------------------------------------------
    
    def register_program(self, program_id: str, program: "Program") -> None:
        """Register a program with this backend.
        
        All token stats (reasoning_program_tokens, acting_program_tokens, etc.)
        are computed from the registered programs.
        """
        self._programs[program_id] = program
    
    def unregister_program(self, program_id: str) -> None:
        """Unregister a program from this backend."""
        self._programs.pop(program_id, None)
    
    # -------------------------------------------------------------------------
    # Metrics Monitoring (delegated to metrics_client)
    # -------------------------------------------------------------------------
    
    async def start_monitoring(self, interval: float = 5.0):
        """Start background metrics monitoring for this backend."""
        await self.metrics_client.start_monitoring(interval)
    
    async def stop_monitoring(self):
        """Stop background metrics monitoring."""
        await self.metrics_client.stop_monitoring()
    
    async def fetch_metrics(self) -> bool:
        """Fetch metrics from vLLM."""
        return await self.metrics_client.fetch_metrics()
    
    async def fetch_cache_config(self) -> bool:
        """Fetch cache config from vLLM."""
        return await self.metrics_client.fetch_cache_config()
    
    def to_dict(self, *, paused_program_count: Optional[int] = None) -> dict:
        """Convert to dict for API response."""
        if paused_program_count is None:
            paused_program_count = 0
        
        # Program state info
        result = {
            "url": self.url,
            "active_program_tokens": self.active_program_tokens,
            "reasoning_program_tokens": self.reasoning_program_tokens,
            "acting_program_tokens": self.acting_program_tokens,
            "active_program_count": self.active_program_count,
            "reasoning_program_count": self.reasoning_program_count,
            "acting_program_count": self.acting_program_count,
            "active_program_tokens_ratio": round(self.active_program_tokens_ratio, 4),
            "total_program_tokens": self.total_program_tokens,
            "paused_program_count": paused_program_count,
            "future_paused_tokens": self.future_paused_tokens,
            "shared_tokens": self.shared_tokens,
            "buffer_per_program": BUFFER_PER_PROGRAM,
            "capacity_overflow": self.capacity_overflow(),
        }
        
        # Merge metrics client info
        result.update(self.metrics_client.to_dict())
        
        return result
