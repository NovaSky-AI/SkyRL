"""Router with program state tracking - supports multiple backends."""
import asyncio
import json
import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable, Awaitable, Tuple

import httpx
from fastapi.responses import Response

from ..backend import BackendState, MetricsClient, SGLangMetricsClient, VLLMMetricsClient
from ..backend.skyrl_metrics import SkyRLMetricsClient
from ..program import Program, ProgramStatus, ProgramState
from ..profile.state import ProfileState
from ..config import get_config
from .vllm_request_processor import (
    forward_streaming_request,
    forward_non_streaming_request,
    forward_get_request,
)

logger = logging.getLogger(__name__)


@dataclass
class PausedInfo:
    """Metadata for a paused program stored in the global paused pool."""
    program_id: str
    total_tokens: int
    paused_at: float
    origin_backend: Optional[str]  # None for new programs that haven't been assigned yet
    step_count: int


class MultiBackendRouter:
    """Router with program state tracking, supports multiple backends."""

    @staticmethod
    def _create_metrics_client(backend_type: str, url: str) -> MetricsClient:
        """Create a metrics client for the given backend type.
        
        Args:
            backend_type: Backend type ("vllm" or "sglang")
            url: Backend base URL
        
        Returns:
            MetricsClient implementation for the backend type
        """
        if backend_type == "vllm":
            return VLLMMetricsClient(url)
        if backend_type == "sglang":
            return SGLangMetricsClient(url)
        if backend_type == "skyrl":
            return SkyRLMetricsClient(url)
        raise ValueError(f"Unsupported backend_type: {backend_type}")

    def __init__(
        self, 
        backend_urls: str | List[str], 
        *, 
        profile_enabled: bool = False,
        scheduling_enabled: bool = True,
        scheduler_interval: float = 5.0,
        backend_type: str = "vllm",
        acting_token_weight: float = 1.0,
        use_acting_token_decay: bool = False,
    ) -> None:
        # Support single URL string or list of URLs
        if isinstance(backend_urls, str):
            backend_urls = [url.strip() for url in backend_urls.split(",") if url.strip()]
        
        # Weight for acting tokens in capacity calculation
        self.acting_token_weight = acting_token_weight
        
        # All backends (pass acting_token_weight as tool_coefficient)
        self.backends: Dict[str, BackendState] = {}
        for url in backend_urls:
            metrics_client = self._create_metrics_client(backend_type, url)
            self.backends[url] = BackendState(
                url=url,
                tool_coefficient=acting_token_weight,
                metrics_client=metrics_client,
                use_acting_token_decay=use_acting_token_decay,
            )
        
        # All programs (single source of truth)
        # Key: program_id, Value: Program (which includes backend_url)
        self.programs: Dict[str, Program] = {}

        # Global paused pool shared across backends (program_id -> PausedInfo).
        self.global_waiting_queue: Dict[str, PausedInfo] = {}

        # Lock for atomic claim (select + pop) from global_waiting_queue.
        self.pause_resume_lock = asyncio.Lock()
        
        # Profile configuration
        self.profile_enabled = profile_enabled
        
        # Scheduling mode: True = "tr" (capacity scheduling), False = "default" (pure proxy)
        self.scheduling_enabled = scheduling_enabled

        self.client = httpx.AsyncClient(
            timeout=900.0,
            limits=httpx.Limits(max_connections=None, max_keepalive_connections=None),
        )
        
        # Scheduler task for periodic capacity check
        self._scheduler_task: Optional[asyncio.Task] = None
        self._scheduler_stop = False
        self._scheduler_interval = scheduler_interval
        
        # Global char-to-token ratio for token estimation
        # Initial value is 5.0 (1 token ≈ 5 chars), updated with momentum after each request
        self.char_to_token_ratio: float = 5.0
        self._ratio_initialized: bool = False

    async def start(self):
        """Start the router."""
        logger.info(f"Started router with {len(self.backends)} backend(s): {list(self.backends.keys())}")
        
        # Always fetch cache config (needed for active_program_tokens_ratio)
        for backend in self.backends.values():
            await backend.fetch_cache_config()
        
        # Start metrics monitoring on each backend if enabled
        config = get_config()
        if config.metrics_enabled:
            for backend in self.backends.values():
                await backend.start_monitoring(config.metrics_interval)
        
        # Start the periodic scheduler if scheduling is enabled
        if self.scheduling_enabled:
            self._scheduler_stop = False
            self._scheduler_task = asyncio.create_task(self._scheduler_loop())
            logger.info(f"Started scheduler loop (interval={self._scheduler_interval}s)")

    async def stop(self):
        """Stop the router."""
        # Stop the scheduler
        if self._scheduler_task:
            self._scheduler_stop = True
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
            self._scheduler_task = None
            logger.info("Stopped scheduler loop")
        
        # Stop metrics monitoring on each backend
        for backend in self.backends.values():
            await backend.stop_monitoring()
        
        await self.client.aclose()
        logger.info("Router stopped")

    # -------------------------------------------------------------------------
    # Backend Selection
    # -------------------------------------------------------------------------

    def get_backend(self, url: str) -> Optional[BackendState]:
        """Get a backend by URL."""
        return self.backends.get(url)

    def get_default_backend(self) -> BackendState:
        """Get the first backend (for simple single-backend usage)."""
        return next(iter(self.backends.values()))

    def select_backend_for_new_program_default(self) -> BackendState:
        """Select the least loaded backend for a new program."""
        # Count programs per backend
        #### TODO change backend assign logistics
        backend_load: Dict[str, int] = {url: 0 for url in self.backends}
        for state in self.programs.values():
            if state.backend_url in backend_load:
                backend_load[state.backend_url] += 1
        
        # Find the backend with least programs (only consider healthy ones)
        min_load = float('inf')
        best_backend = None
        for url, load in backend_load.items():
            backend = self.backends[url]
            if backend.healthy and load < min_load:
                min_load = load
                best_backend = backend
        
        # Fallback to first backend if all unhealthy
        return best_backend or self.get_default_backend()

    # -------------------------------------------------------------------------
    # Program State Management
    # -------------------------------------------------------------------------

    @staticmethod
    def _estimate_system_prompt_tokens(payload: Dict[str, Any]) -> int:
        """Estimate system prompt tokens from the first request payload."""
        messages = payload.get("messages")
        if not isinstance(messages, list):
            return 0
        
        parts: List[str] = []
        for msg in messages:
            if not isinstance(msg, dict) or msg.get("role") != "system":
                continue
            content = msg.get("content")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    text = item.get("text") or item.get("input_text")
                    if isinstance(text, str):
                        parts.append(text)
        
        if not parts:
            return 0
        
        text = "\n".join(parts)
        return max(0, len(text) // 5)

    def get_or_create_program(self, program_id: str) -> Program:
        """Get existing program or create new one.
        
        Only creates the program and estimates token count.
        Backend assignment is deferred to update_program_before_request.
        
        Args:
            program_id: Unique identifier for the program
            payload: Request payload, used to estimate token count for new programs
        """
        if program_id not in self.programs:
            profile = ProfileState(program_id=program_id) if self.profile_enabled else None
            state = Program(
                program_id=program_id,
                backend_url=None,
                status=ProgramStatus.REASONING,  # Request arrived, program is reasoning
                state=ProgramState.ACTIVE,
                profile=profile,
            )
            # Token estimation is done in update_program_before_request
            self.programs[program_id] = state
            logger.debug(f"Created program {program_id}")
        
        return self.programs[program_id]
    
    def _select_backend_for_new_program(self, estimated_tokens: int = 0) -> Optional[str]:
        """Select a backend for a new program (TR mode only).
        
        This function is only called when scheduling_enabled=True.
        
        Args:
            estimated_tokens: Estimated token count for the new program
        
        Returns:
            backend_url if can assign directly (queue empty + has capacity)
            None if should wait in queue
        """
        from ..backend.state import BUFFER_PER_PROGRAM
        
        # Only assign if queue is empty and backend has capacity
        if len(self.global_waiting_queue) > 0:
            return None  # Queue not empty, must wait for fairness
        
        # Find backend with least active tokens that has capacity for this program
        best_backend = None
        min_tokens = float('inf')
        required_capacity = estimated_tokens + BUFFER_PER_PROGRAM
        
        for backend in self.backends.values():
            if not backend.healthy:
                continue
            if backend.remaining_capacity() < required_capacity:
                continue  # Not enough capacity for this program
            if backend.active_program_tokens < min_tokens:
                min_tokens = backend.active_program_tokens
                best_backend = backend
        
        return best_backend.url if best_backend else None

    def get_backend_for_program(self, program_id: str) -> BackendState:
        """Get the backend assigned to a program."""
        state = self.programs.get(program_id)
        if state and state.backend_url in self.backends:
            return self.backends[state.backend_url]
        return self.get_default_backend()

    async def update_program_before_request(self, program_id: str, state: Program, payload: Dict[str, Any]) -> bool:
        """Update program state before sending request to vLLM.
        
        If scheduling_enabled=False (default mode): pure proxy, no capacity checks.
        If scheduling_enabled=True (tr mode): wait for scheduler to resume if PAUSED.
        
        Returns: True if can proceed
        """
        state.step_count += 1
        is_new_program = state.step_count == 1
        
        # Update context_len and estimate total_tokens using char_to_token_ratio
        state.context_len = len(json.dumps(payload, ensure_ascii=False))
        state.total_tokens = int(state.context_len / self.char_to_token_ratio)
        
        # ---------------------------------------------------------------------
        # Default mode: pure proxy, no scheduling
        # ---------------------------------------------------------------------
        if not self.scheduling_enabled:
            backend = self.backends.get(state.backend_url)
            if not backend:
                # Assign to least loaded backend
                backend = self.select_backend_for_new_program_default()
                state.backend_url = backend.url
            
            if is_new_program:
                backend.register_program(program_id, state)
            # Status change is enough - token stats are computed from program status
            state.status = ProgramStatus.REASONING
            state.acting_since = None
            return True
        
        # ---------------------------------------------------------------------
        # TR mode: scheduler-based capacity management
        # ---------------------------------------------------------------------
        # Step 1: Handle PAUSED programs (wait for resume)
        # Note: waiting_event is created in _pause_program and cleared in _resume_program.
        # If waiting_event is not None, the program is still PAUSED and we must wait.
        # Set status to REASONING before waiting so the scheduler sees the correct
        # priority: REASONING programs (pending request) > ACTING programs (idle).
        if state.waiting_event is not None:
            state.status = ProgramStatus.REASONING
            state.acting_since = None
            await self._wait_for_resume(program_id, state)
            # After _wait_for_resume returns, _resume_program has been called which:
            # - Registered program with target backend
            # - Set state.state = ACTIVE
            # - Cleared waiting_event = None
            # Fall through to Step 3
        
        # Step 2: Handle new programs (assign backend or queue)
        if is_new_program and state.backend_url is None:
            backend_url = self._select_backend_for_new_program(state.total_tokens)
            if backend_url:
                # Direct assignment: register program with backend
                state.backend_url = backend_url
                backend = self.backends[backend_url]
                backend.register_program(program_id, state)
                logger.debug(f"Assigned new program {program_id} to {backend_url}")
                state.status = ProgramStatus.REASONING
                state.acting_since = None
                return True
            else:
                # Queue and wait — set REASONING before waiting for priority
                state.status = ProgramStatus.REASONING
                state.acting_since = None
                state.waiting_event = asyncio.Event()
                state.state = ProgramState.PAUSED
                self._add_to_global_waiting_queue_sync(program_id, state, backend=None)
                logger.debug(f"Queued new program {program_id} (tokens={state.total_tokens})")
                await self._wait_for_resume(program_id, state)
                # After resume: program registered with backend by _resume_program
                return True
        
        # Step 3: Normal case - existing ACTIVE program with backend
        backend = self.backends.get(state.backend_url)
        if not backend:
            logger.error(f"Program {program_id} has no valid backend")
            return False
        
        state.status = ProgramStatus.REASONING
        state.acting_since = None
        return True

    def update_program_after_request(
        self, program_id: str, state: Program, total_tokens: int, prompt_tokens: int = 0
    ) -> None:
        """Update program state after receiving response from vLLM.
        
        Transitions to ACTING (off GPU, executing tool).
        Updates token counts. If marked for pause, pause immediately.
        Also updates the global char_to_token_ratio for future token estimation.
        
        Args:
            program_id: The program ID
            state: The program state
            total_tokens: Total tokens from vLLM response (prompt + completion)
            prompt_tokens: Prompt/prefill tokens from vLLM response
        """
        # Transition to ACTING
        state.status = ProgramStatus.ACTING
        state.acting_since = time.time()
        
        # Update global char_to_token_ratio based on actual prefill
        # ratio = context_len / prompt_tokens (chars per token)
        if prompt_tokens > 0 and state.context_len > 0:
            current_ratio = state.context_len / prompt_tokens
            if not self._ratio_initialized:
                # First request: directly assign
                self.char_to_token_ratio = current_ratio
                self._ratio_initialized = True
                logger.debug(f"Initialized char_to_token_ratio={self.char_to_token_ratio:.2f}")
            else:
                # Subsequent requests: momentum update (0.2 new + 0.8 old)
                self.char_to_token_ratio = 0.2 * current_ratio + 0.8 * self.char_to_token_ratio
                logger.debug(f"Updated char_to_token_ratio={self.char_to_token_ratio:.2f} (sample={current_ratio:.2f})")
        
        # Update total_tokens - token stats are computed from program state
        state.total_tokens = total_tokens
        
        # If marked for pause, pause now (while in ACTING state)
        if state.marked_for_pause:
            self._clear_mark_and_pause(program_id, state)

    def update_program_tokens_streaming(self, state: Program, delta_tokens: int) -> None:
        """Update program tokens incrementally during streaming.
        
        Called periodically (e.g., every 20 tokens) during streaming to 
        update the token counts in real-time. The program is in REASONING
        state during streaming.
        
        Args:
            state: The program state
            delta_tokens: Number of new tokens since last update
        """
        # Just update program's total_tokens - backend stats are computed from program state
        state.total_tokens += delta_tokens

    async def release_program(self, program_id: str) -> bool:
        """Stop a program and release its resources.
        
        Removes tokens from tracking. Resume of waiting programs is handled by the scheduler.
        """
        if program_id not in self.programs:
            return False
        
        state = self.programs[program_id]
        backend = self.backends.get(state.backend_url) if state.backend_url else None
        
        # Clean up based on current lifecycle state
        if state.state == ProgramState.PAUSED:
            # Remove from waiting queue
            await self._remove_from_global_waiting_queue(program_id)
            if state.waiting_event:
                state.waiting_event.set()  # Unblock any waiting coroutine
        elif backend and state.state == ProgramState.ACTIVE:
            backend.unregister_program(program_id)
        
        # Clear mark if was marked
        if backend and state.marked_for_pause:
            backend.future_paused_tokens -= state.total_tokens
            if backend.future_paused_tokens < 0:
                backend.future_paused_tokens = 0
            state.marked_for_pause = False
        
        state.state = ProgramState.TERMINATED
        
        # Remove from programs dict
        del self.programs[program_id]
        
        logger.info(f"Released and removed program: {program_id}")
        return True

    def get_programs_on_backend(self, backend_url: str) -> Dict[str, Program]:
        """Get all programs assigned to a specific backend."""
        return {
            pid: state for pid, state in self.programs.items()
            if state.backend_url == backend_url
        }

    # -------------------------------------------------------------------------
    # Global Paused Pool
    # -------------------------------------------------------------------------

    def _add_to_global_waiting_queue_sync(
        self, program_id: str, state: Program, backend: Optional[BackendState] = None,
    ) -> None:
        """Add a program to the global waiting queue (synchronous version)."""
        paused_info = PausedInfo(
            program_id=program_id,
            total_tokens=state.total_tokens,
            paused_at=time.time(),
            origin_backend=backend.url if backend else None,
            step_count=state.step_count,
        )
        self.global_waiting_queue[program_id] = paused_info

    async def _remove_from_global_waiting_queue(self, program_id: str) -> Optional[PausedInfo]:
        """Remove a program from the global paused pool."""
        async with self.pause_resume_lock:
            return self.global_waiting_queue.pop(program_id, None)

    def _get_paused_programs_sorted(
        self, ascending: bool = True
    ) -> List[Tuple[str, Program, PausedInfo]]:
        """Get paused programs from the global pool, sorted by total_tokens.

        Callers that require consistency should hold pause_resume_lock.
        """
        programs: List[Tuple[str, Program, PausedInfo]] = []
        for pid, info in self.global_waiting_queue.items():
            state = self.programs.get(pid)
            if not state:
                continue
            programs.append((pid, state, info))
        return sorted(programs, key=lambda x: x[2].total_tokens, reverse=not ascending)

    def get_paused_counts_by_backend(self) -> Dict[str, int]:
        """Count paused programs per backend using paused pool metadata."""
        counts = {url: 0 for url in self.backends}
        for info in self.global_waiting_queue.values():
            if info.origin_backend in counts:
                counts[info.origin_backend] += 1
        return counts

    # -------------------------------------------------------------------------
    # Capacity-based Scheduling (pause/resume)
    # -------------------------------------------------------------------------

    def _get_acting_programs_sorted(self, backend_url: str, ascending: bool = True) -> List[Tuple[str, Program]]:
        """Get ACTING programs on a backend, sorted by total_tokens.
        
        Args:
            ascending: If True, smallest first. If False, largest first.
        """
        programs = [
            (pid, state) for pid, state in self.programs.items()
            if state.backend_url == backend_url and state.status == ProgramStatus.ACTING
        ]
        return sorted(programs, key=lambda x: x[1].total_tokens, reverse=not ascending)

    def _get_reasoning_programs_sorted(self, backend_url: str, ascending: bool = True) -> List[Tuple[str, Program]]:
        """Get REASONING programs on a backend, sorted by total_tokens.
        
        Args:
            ascending: If True, smallest first. If False, largest first.
        """
        programs = [
            (pid, state) for pid, state in self.programs.items()
            if state.backend_url == backend_url 
            and state.status == ProgramStatus.REASONING
            and not state.marked_for_pause  # Exclude already marked
        ]
        return sorted(programs, key=lambda x: x[1].total_tokens, reverse=not ascending)

    def _pause_program(self, program_id: str, state: Program) -> None:
        """Pause a program: remove from active and total, add to global paused pool.
        
        Only call this for ACTING programs. REASONING programs should be marked instead.
        Sets backend_url to None and saves origin_backend for resume.
        """
        backend = self.backends.get(state.backend_url)
        if not backend:
            return
        
        # Unregister from backend
        backend.unregister_program(program_id)
        
        # Add to global paused pool
        self._add_to_global_waiting_queue_sync(program_id, state, backend)
        
        # Save origin backend and clear current backend
        state.origin_backend = state.backend_url
        state.backend_url = None
        state.state = ProgramState.PAUSED
        
        # Create waiting event if needed
        if state.waiting_event is None:
            state.waiting_event = asyncio.Event()
        else:
            state.waiting_event.clear()
        
        logger.info(f"Paused program {program_id} from {state.origin_backend} (tokens={state.total_tokens})")

    def _mark_program_for_pause(self, program_id: str, state: Program) -> None:
        """Mark a REASONING program for pause. It will be paused on next request.
        
        Adds the program's tokens to future_paused_tokens for capacity calculation.
        """
        backend = self.backends.get(state.backend_url)
        if not backend:
            return
        
        state.marked_for_pause = True
        backend.future_paused_tokens += state.total_tokens
        
        logger.info(f"Marked program {program_id} for pause (tokens={state.total_tokens}, future_paused={backend.future_paused_tokens})")

    def _clear_mark_and_pause(self, program_id: str, state: Program) -> None:
        """Clear the mark from a program and pause it.
        
        Called when a marked program's next request arrives.
        """
        backend = self.backends.get(state.backend_url)
        if not backend:
            return
        
        # Clear the mark and subtract from future_paused_tokens
        state.marked_for_pause = False
        backend.future_paused_tokens -= state.total_tokens
        if backend.future_paused_tokens < 0:
            backend.future_paused_tokens = 0
        
        # Now actually pause the program
        self._pause_program(program_id, state)

    def _resume_program(
        self,
        state: Program,
        target_backend: Optional[BackendState] = None,
    ) -> None:
        """Resume a paused program after it has been claimed from the pool.
        
        Args:
            state: The program state (contains program_id, origin_backend, etc.)
            target_backend: Backend to resume to (may differ from origin for migration)
        
        Uses state.origin_backend as fallback if target_backend is not provided.
        """
        # Use origin_backend as fallback (backend_url is None when paused)
        origin_backend = self.backends.get(state.origin_backend) if state.origin_backend else None
        backend = target_backend or origin_backend
        if not backend:
            return

        if state.state != ProgramState.PAUSED:
            return

        # Register with target backend
        backend.register_program(state.program_id, state)
        state.backend_url = backend.url
        state.origin_backend = None  # Clear origin_backend after resume
        state.state = ProgramState.ACTIVE
        
        # Signal waiting event and clear it (resume completes the pause-resume cycle)
        if state.waiting_event:
            state.waiting_event.set()
            state.waiting_event = None
        
        logger.info(f"Resumed program {state.program_id} to {backend.url} (status={state.status.value}, tokens={state.total_tokens}, active={backend.active_program_tokens})")

    async def _claim_specific_paused(
        self, program_id: str
    ) -> Optional[Program]:
        """Claim a specific paused program in a lock-protected way."""
        async with self.pause_resume_lock:
            info = self.global_waiting_queue.get(program_id)
            if info is None:
                return None
            state = self.programs.get(program_id)
            if not state or state.state != ProgramState.PAUSED:
                self.global_waiting_queue.pop(program_id, None)
                return None
            self.global_waiting_queue.pop(program_id, None)
            return state

    # -------------------------------------------------------------------------
    # Periodic Scheduler
    # -------------------------------------------------------------------------

    async def _scheduler_loop(self):
        """Periodic scheduler loop: check thrashing, update shared_tokens, resume."""
        while not self._scheduler_stop:
            try:
                await asyncio.sleep(self._scheduler_interval)
                await self._scheduled_check()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduler error: {e}", exc_info=True)

    async def _scheduled_check(self):
        """Periodic check: resume waiting programs, enforce capacity, update metrics."""
        # Step 1: Fetch fresh metrics
        for backend in self.backends.values():
            await backend.fetch_metrics()
        
        # Step 2: Resume waiting programs (uses decay-adjusted capacity if enabled)
        await self._greedy_resume()
        
        # Step 3: Check thrashing and pause if needed (uses original capacity, no decay)
        for url, backend in self.backends.items():
            if backend.cache_config and backend.remaining_capacity() < 0:
                await self._pause_until_safe(backend)

    async def _pause_until_safe(self, backend: BackendState):
        """Pause programs until backend is within capacity.
        
        Priority: ACTING first (smallest tokens), then REASONING (smallest tokens).
        """
        paused_count = 0
        
        while backend.remaining_capacity() < 0:
            # Priority 1: Pause ACTING programs (smallest first)
            acting_programs = self._get_acting_programs_sorted(backend.url, ascending=True)
            if acting_programs:
                program_id, state = acting_programs[0]
                self._pause_program(program_id, state)
                paused_count += 1
                logger.info(f"Scheduler paused ACTING program {program_id} (tokens={state.total_tokens})")
                continue
            
            # Priority 2: Pause REASONING programs (smallest first)
            # Note: REASONING programs are on GPU, we just mark them for pause
            reasoning_programs = self._get_reasoning_programs_sorted(backend.url, ascending=True)
            if reasoning_programs:
                program_id, state = reasoning_programs[0]
                self._mark_program_for_pause(program_id, state)
                paused_count += 1
                logger.info(f"Scheduler marked REASONING program {program_id} for pause (tokens={state.total_tokens})")
                # After marking, we've accounted for future_paused_tokens, continue checking
                continue
            
            # No more programs to pause
            break
        
        if paused_count > 0:
            logger.info(f"Scheduler paused/marked {paused_count} programs on {backend.url}")

    async def _greedy_resume(self):
        """Resume waiting programs using Best Fit Decreasing (BFD) bin-packing.
        
        Algorithm:
        1. Compute total backend capacity across all healthy backends.
        2. Select programs in priority order whose cumulative required tokens
           ≤ total capacity — the maximum set we can possibly resume.
        3. BFD placement: sort selected programs descending by required tokens,
           place the largest on the backend with the highest remaining capacity,
           re-sort backends after each placement.
        4. Terminate when all selected programs are placed or the smallest
           remaining program exceeds every backend's remaining capacity.
        
        Priority order for selection:
        1. REASONING programs (step > 1) - highest priority
        2. NEW programs (step = 1) - medium priority
        3. ACTING programs - lowest priority
        Within each group, sorted by token count ascending.
        """
        from ..backend.state import BUFFER_PER_PROGRAM
        
        # --- Collect backend capacities ---
        # Use decay-adjusted capacity when enabled (optimistic for resume)
        backend_caps: list[tuple[BackendState, int]] = []
        total_capacity = 0
        for url, backend in self.backends.items():
            if not backend.cache_config or not backend.healthy:
                continue
            remaining = (backend.remaining_capacity_with_decay()
                         if backend.use_acting_token_decay
                         else backend.remaining_capacity())
            if remaining > BUFFER_PER_PROGRAM:
                backend_caps.append((backend, remaining))
                total_capacity += remaining
        
        if not backend_caps or total_capacity <= 0:
            return
        
        async with self.pause_resume_lock:
            paused_programs = self._get_paused_programs_sorted(ascending=True)
            
            if not paused_programs:
                return
            
            # --- Priority grouping (each group ascending by tokens) ---
            # Programs with a pending request have status=REASONING (set before waiting).
            # Programs paused without a pending request have status=ACTING.
            reasoning_group: list[Program] = []   # REASONING with step > 1 (highest priority)
            new_program_group: list[Program] = []  # step = 1 (new programs)
            acting_group: list[Program] = []       # ACTING (lowest priority)
            
            for pid, state, info in paused_programs:
                if state.step_count == 1:
                    new_program_group.append(state)
                elif state.status == ProgramStatus.REASONING:
                    reasoning_group.append(state)
                else:
                    acting_group.append(state)
            
            candidates_by_priority = reasoning_group + new_program_group + acting_group
            
            # --- Step 1: Select max programs fitting within total capacity ---
            # Walk through priority order; include a program if cumulative fits.
            resumable_programs: list[Program] = []
            cumulative_tokens = 0
            for state in candidates_by_priority:
                required_tokens = state.total_tokens + BUFFER_PER_PROGRAM
                if cumulative_tokens + required_tokens <= total_capacity:
                    resumable_programs.append(state)
                    cumulative_tokens += required_tokens
            
            if not resumable_programs:
                return
            
            # --- Step 2: BFD placement (largest first → highest capacity backend) ---
            resumable_programs.sort(key=lambda s: -s.total_tokens)  # descending by tokens
            backend_caps.sort(key=lambda x: -x[1])                  # descending by capacity
            
            resumed_count = 0
            reasoning_resumed = new_resumed = acting_resumed = 0
            min_required_tokens = resumable_programs[-1].total_tokens + BUFFER_PER_PROGRAM
            
            for state in resumable_programs:
                if not backend_caps:
                    break
                
                required_tokens = state.total_tokens + BUFFER_PER_PROGRAM
                max_backend_capacity = backend_caps[0][1]
                
                # Early termination: if even the smallest program
                # can't fit the largest backend, nothing more can be placed.
                if min_required_tokens > max_backend_capacity:
                    break
                
                # Current program too large for the largest backend — skip it,
                # smaller programs following may still fit.
                if required_tokens > max_backend_capacity:
                    continue
                
                # Place on the backend with the most remaining capacity
                target_backend, target_remaining = backend_caps[0]
                self.global_waiting_queue.pop(state.program_id, None)
                self._resume_program(state, target_backend=target_backend)
                resumed_count += 1
                
                # Track which priority group
                if state.step_count == 1:
                    new_resumed += 1
                elif state.status == ProgramStatus.REASONING:
                    reasoning_resumed += 1
                else:
                    acting_resumed += 1
                
                # Update capacity and re-sort backends
                updated_remaining = target_remaining - required_tokens
                if updated_remaining > BUFFER_PER_PROGRAM:
                    backend_caps[0] = (target_backend, updated_remaining)
                    backend_caps.sort(key=lambda x: -x[1])  # re-sort after update
                else:
                    backend_caps.pop(0)  # backend full, remove from candidates
            
            if resumed_count > 0:
                logger.info(
                    f"Scheduler resumed {resumed_count} programs "
                    f"(reasoning={reasoning_resumed}, new={new_resumed}, acting={acting_resumed})"
                )

    async def _wait_for_resume(self, program_id: str, state: Program, timeout: float = 1800.0) -> None:
        """Wait for a paused program to be resumed.
        
        If timeout (30 min), force resume the program regardless of capacity.
        """
        if state.waiting_event is None:
            return
        
        try:
            await asyncio.wait_for(state.waiting_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"Program {program_id} wait timeout after {timeout}s, forcing resume")
            claimed_state = await self._claim_specific_paused(program_id)
            if claimed_state is None:
                return
            
            # For new programs without origin backend, force assign to least loaded backend
            target_backend = None
            if claimed_state.origin_backend is None:
                target_backend = self.select_backend_for_new_program_default()
                logger.info(f"Force assigning new program {program_id} to {target_backend.url}")
            
            self._resume_program(claimed_state, target_backend=target_backend)

    def get_program_stats(self) -> Dict[str, Any]:
        """Get statistics about all programs."""
        reasoning = sum(1 for p in self.programs.values() if p.status == ProgramStatus.REASONING)
        acting = sum(1 for p in self.programs.values() if p.status == ProgramStatus.ACTING)
        paused = len(self.global_waiting_queue)
        marked = sum(1 for p in self.programs.values() if p.marked_for_pause)
        
        # Per-backend stats
        paused_counts = self.get_paused_counts_by_backend()
        per_backend = {}
        for url, backend in self.backends.items():
            progs = self.get_programs_on_backend(url)
            per_backend[url] = {
                "total": len(progs),
                "reasoning": sum(1 for p in progs.values() if p.status == ProgramStatus.REASONING),
                "acting": sum(1 for p in progs.values() if p.status == ProgramStatus.ACTING),
                "paused": paused_counts.get(url, 0),
                "marked_for_pause": sum(1 for p in progs.values() if p.marked_for_pause),
                "future_paused_tokens": backend.future_paused_tokens,
            }
        
        return {
            "total": len(self.programs),
            "reasoning": reasoning,
            "acting": acting,
            "paused": paused,
            "marked_for_pause": marked,
            "per_backend": per_backend,
        }

    # -------------------------------------------------------------------------
    # Request Proxying
    # -------------------------------------------------------------------------

    @staticmethod
    def extract_total_tokens(payload: Any) -> Optional[int]:
        """Extract total_tokens from the response."""
        if not isinstance(payload, dict):
            return None
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            return None
        if "total_tokens" in usage:
            val = usage.get("total_tokens")
            if isinstance(val, (int, float)) and math.isfinite(val):
                return int(val)
        return None

    async def proxy_request(
        self,
        backend: BackendState,
        payload: Dict[str, Any],
        *,
        on_usage: Callable[[int, Optional[int], Optional[int], Optional[int]], Awaitable[None]] | None = None,
        on_first_token: Callable[[], None] | None = None,
        on_token: Callable[[], None] | None = None,
        on_token_progress: Callable[[int], None] | None = None,
    ) -> Response:
        """Proxy request to a specific backend.
        
        Args:
            backend: Target backend
            payload: Request payload
            on_usage: Callback with (total_tokens, prompt_tokens, completion_tokens, cached_tokens)
            on_first_token: Callback when first token is received (streaming only)
            on_token: Callback for each token (streaming only)
            on_token_progress: Callback with delta token count at intervals (streaming only)
        """
        url = backend.completions_url
        
        if payload.get("stream"):
            return await forward_streaming_request(
                self.client,
                url,
                payload,
                on_usage=on_usage,
                on_first_token=on_first_token,
                on_token=on_token,
                on_token_progress=on_token_progress,
            )
        else:
            return await forward_non_streaming_request(
                self.client,
                url,
                payload,
                on_usage=on_usage,
            )

    async def proxy_get(self, backend_url: str, path: str) -> Response:
        """Proxy a GET request to a backend."""
        url = f"{backend_url.rstrip('/')}{path}"
        return await forward_get_request(self.client, url)
