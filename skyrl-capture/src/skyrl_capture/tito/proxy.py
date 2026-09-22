"""The TITO route: one exact turn, from request bytes to an observed exchange.

Flow for one turn:

1. Parse the OpenAI request.
2. Under the trajectory's session lock, prepare the turn against the trace:
   match in message space and find a bridge transition.
3. Bridge from the previous exact prompt+completion if the renderer can, else
   full render.
4. Call token-in/token-out inference with the exact prompt IDs.
5. Parse the sampled IDs back into an assistant message.
6. Commit to the trace -- which verifies token-space fidelity -- **before**
   responding.
7. Derive the exchange and queue its journal append.
8. Send the synthesized response while that append is running.
9. **Wait for the append to be durable before closing the response.**
10. Close, then record that the response was delivered.

Steps 6 and 9 are the deliberate divergence from text-mode fail-open capture,
discussed in docs/design/token-capture-parity.md: here the proxy *produced*
the tokens, so a response whose tokens could not be attributed exactly, or
could not be written down, would be a silent training-data corruption. A
cleanly closed response therefore always has its exact exchange on disk, and a
process that dies before that leaves a failed connection -- which the client's
ordinary retry handles, because the request was never anything special.

Overlapping the append with the send is what keeps this cheap: TTFT and chunk
cadence are unchanged, and all that is added before the close is whatever
durability still owes.

This class knows the engine, the session manager, the commit coordinator and
the upstream configuration, and nothing else. It does not know what a data
plane is, and it cannot reach one.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from typing import Any

import orjson

from skyrl_capture import upstream as text_protocols
from skyrl_capture.config import TitoUpstream
from skyrl_capture.domain.records import ActiveTrajectory, now
from skyrl_capture.ids import exchange_id as new_exchange_id
from skyrl_capture.persistence.journal import ExchangeDeliveryConfirmed, TrajectoryPoisoned
from skyrl_capture.tito import protocol
from skyrl_capture.tito.engine import (
    TokenEngine,
    TokenUpstreamError,
    build_sampling_params,
    canonical_sampling_params,
)
from skyrl_capture.tito.protocol import ProtocolError
from skyrl_capture.tito.sessions import TokenSessionManager
from skyrl_capture.tito.trace import convert_routed_experts
from skyrl_capture.tito.types import ModelTurnResult, TokenError
from skyrl_capture.writer.commits import CommitCoordinator
from skyrl_capture.writer.derive import derive_exchange
from skyrl_capture.writer.exchange import Exchange

logger = logging.getLogger(__name__)

DEFAULT_MAX_MODEL_LEN = 32768


class TitoProxy:
    def __init__(
        self,
        *,
        engine: TokenEngine,
        sessions: TokenSessionManager,
        upstream: TitoUpstream,
        commits: CommitCoordinator,
        header_allowlist: tuple[str, ...],
        clock_epoch: str,
        commit_timeout: float = 30.0,
    ) -> None:
        self._engine = engine
        # Public: it belongs to this proxy, and `finish` releases a
        # trajectory's trace through it. Text capture has no equivalent
        # because it holds nothing between turns.
        self.sessions = sessions
        self._upstream = upstream
        # A token route speaks OpenAI chat completions to the workload, whatever
        # the engine underneath looks like -- so the client-facing errors and
        # the reading of the request and the response it synthesized are that
        # protocol's, and the engine's wire is a separate thing entirely.
        self._protocol = text_protocols.get("openai")
        self._commits = commits
        self._header_allowlist = header_allowlist
        self._clock_epoch = clock_epoch
        # How long a response may wait for its exchange to become durable
        # before the connection is failed instead of closed cleanly.
        self._commit_timeout = commit_timeout
        self.turns_served = 0
        self.close_wait_ns = 0
        self.delivery_unconfirmed = 0
        self.trajectories_poisoned = 0
        self.render_full = 0
        self.render_bridged = 0
        self.commit_failures = 0
        # Where a turn's wall time goes, summed in nanoseconds. Two
        # perf_counter calls per phase against phases measured in milliseconds,
        # so it is always on: knowing which phase grew is most of diagnosing a
        # slow turn, and inferring it from end-to-end latency is guesswork.
        self.phase_ns: dict[str, int] = dict.fromkeys(
            ("warmup", "trace", "prepare", "render", "upstream", "parse", "commit", "respond"), 0
        )
        # How late the event loop is running its own callbacks. A turn's CPU
        # work blocks every other turn on this process, so this is the
        # difference between "the proxy is busy" and "the proxy is blocked".
        self.loop_lag_ms_max = 0.0
        self.loop_lag_ms_total = 0.0
        self.loop_lag_samples = 0

    # -- route -------------------------------------------------------------
    async def handle(
        self,
        *,
        scope: Any,
        send: Any,
        active: ActiveTrajectory,
        body: bytes,
        request_start_wall: int,
        request_start_mono: int,
        suffix: str,
    ) -> None:
        await self._serve_turn(
            scope=scope,
            send=send,
            active=active,
            body=body,
            request_start_wall=request_start_wall,
            request_start_mono=request_start_mono,
            suffix=suffix,
        )

    async def _serve_turn(
        self,
        *,
        scope: Any,
        send: Any,
        active: ActiveTrajectory,
        body: bytes,
        request_start_wall: int,
        request_start_mono: int,
        suffix: str,
    ) -> None:
        if not suffix.endswith("/chat/completions"):
            # There is no text proxy to fall back to: this process renders the
            # tokens, and an endpoint it cannot render is a configuration
            # mistake rather than something to forward blind.
            await _send_json(
                send,
                404,
                {
                    "error": {
                        "message": (
                            f"{suffix} is not available in token capture: this process "
                            "renders chat completions and forwards nothing else"
                        ),
                        "type": "invalid_request_error",
                        "code": "unsupported_endpoint",
                    }
                },
            )
            return
        try:
            request = protocol.parse_chat_request(body, registered_model=self._upstream.model)
        except ProtocolError as error:
            await _send_json(send, error.status, error.body)
            return

        # The session lock is held across the response, not only across
        # inference: the next turn of this trajectory may not start until this
        # one's exchange is durable, or it would render this assistant message
        # from the client's history and record model tokens as client-authored.
        try:
            async with self.sessions.lock(active.id):
                outcome = await self._execute_turn(active, request)
                await self._deliver(
                    scope=scope,
                    send=send,
                    active=active,
                    request=request,
                    outcome=outcome,
                    body=body,
                    suffix=suffix,
                    request_start_wall=request_start_wall,
                    request_start_mono=request_start_mono,
                )
        except ProtocolError as error:
            await _send_json(send, error.status, error.body)
            return
        except TokenUpstreamError as error:
            status = error.status or 502
            await _send_json(
                send,
                status,
                {"error": {"message": str(error), "type": "api_error", "code": "tokens_upstream"}},
            )
            return
        except TokenError as error:
            # Attribution failures are handled at the commit site, which poisons
            # the trajectory and still returns the completion. Anything reaching
            # here is a trace-level failure before inference ran, so there is no
            # completion to hand back.
            self.commit_failures += 1
            logger.error("token capture failed for %s: %s", active.id, error)
            await _send_json(
                send,
                500,
                {
                    "error": {
                        "message": f"token capture could not prepare this turn: {error}",
                        "type": "api_error",
                        "code": "tokens_capture_failed",
                    }
                },
            )
            return

    # -- one response --------------------------------------------------------
    async def _deliver(
        self,
        *,
        scope: Any,
        send: Any,
        active: ActiveTrajectory,
        request: protocol.ChatRequest,
        outcome: dict[str, Any],
        body: bytes,
        suffix: str,
        request_start_wall: int,
        request_start_mono: int,
    ) -> None:
        """Queue the append, send the response, and close only once it is durable."""
        response = outcome["response"]
        if request.stream:
            frames = protocol.build_stream_frames(
                response,
                decode_token=outcome["renderer"].decode_token,
                completion_ids=outcome["completion_ids"],
            )
            response_bytes = b"".join(frames)
        else:
            frames = [orjson.dumps(response)]
            response_bytes = frames[0]

        # The outcome is complete here, so this is when the client could first
        # have had a byte of it. Measuring at the send instead would time this
        # process's own writes rather than the turn.
        outcome_ready_mono = time.perf_counter_ns()
        commit = await self._queue_commit(
            active=active,
            scope=scope,
            request=request,
            outcome=outcome,
            body=body,
            suffix=suffix,
            response_bytes=response_bytes,
            frame_count=len(frames),
            request_start_wall=request_start_wall,
            request_start_mono=request_start_mono,
            outcome_ready_mono=outcome_ready_mono,
        )

        headers: list[tuple[bytes, bytes]] = (
            [(b"content-type", b"text/event-stream"), (b"cache-control", b"no-cache")]
            if request.stream
            else [(b"content-type", b"application/json")]
        )
        if outcome["poisoned"]:
            # The caller gets its completion either way; the header is how it
            # learns this trajectory will not take another turn.
            headers.append((b"x-capture-status", b"poisoned"))
        await send({"type": "http.response.start", "status": 200, "headers": headers})
        for frame in frames:
            await send({"type": "http.response.body", "body": frame, "more_body": True})

        # Everything but the final empty frame has gone. This is the wait the
        # design pays for exactness, and it is only whatever durability still
        # owes after inference, rendering and the send.
        waited = time.perf_counter_ns()
        await self._await_durable(active, commit, outcome)
        self.close_wait_ns += time.perf_counter_ns() - waited

        await send({"type": "http.response.body", "body": b"", "more_body": False})
        self.turns_served += 1

        # The send completed, so this exchange was handed to the transport.
        # Queued rather than awaited: the response is already closed, and a
        # confirmation that never lands only makes a recovery cautious.
        exchange_id = outcome["exchange_id"]
        active.confirm_delivery(exchange_id)
        if self._commits.submit(active, ExchangeDeliveryConfirmed(exchange_id, now())) is None:
            self.delivery_unconfirmed += 1

    async def _queue_commit(
        self,
        *,
        active: ActiveTrajectory,
        scope: Any,
        request: protocol.ChatRequest,
        outcome: dict[str, Any],
        body: bytes,
        suffix: str,
        response_bytes: bytes,
        frame_count: int,
        request_start_wall: int,
        request_start_mono: int,
        outcome_ready_mono: int,
    ) -> Any:
        observed = Exchange(
            exchange_id=outcome["exchange_id"],
            trajectory_id=active.id,
            project=active.header.project,
            provider="tokens",
            endpoint_kind="chat_completions",
            method=scope["method"],
            path=suffix,
            query=scope.get("query_string", b"").decode("latin-1"),
            request_start_wall_ns=request_start_wall,
            request_start_mono_ns=request_start_mono,
            first_byte_mono_ns=outcome_ready_mono,
            response_end_wall_ns=time.time_ns(),
            response_end_mono_ns=outcome_ready_mono,
            http_status=200,
            streaming=request.stream,
            request_headers=[
                (name.decode("latin-1"), value.decode("latin-1")) for name, value in scope["headers"]
            ],
            response_headers=[
                ("content-type", "text/event-stream" if request.stream else "application/json")
            ],
            request_body=body,
            response_body=response_bytes,
            request_byte_count=len(body),
            response_byte_count=len(response_bytes),
            chunk_count=frame_count,
            source_metadata={
                "clock_epoch": self._clock_epoch,
                "stream_synthesized": request.stream,
            },
            tokens=outcome["tokens_payload"],
        )
        record = derive_exchange(
            active,
            observed,
            protocol=self._protocol,
            header_allowlist=self._header_allowlist,
            # The response has not been sent yet. What confirms delivery is a
            # second record, appended after the send completes.
            delivery_confirmed=False,
        )
        active.add_exchange(record.exchange, record.graph, record.at)
        return await self._commits.submit_when_ready(
            active, record, timeout=self._commit_timeout
        )

    async def _await_durable(self, active: ActiveTrajectory, commit: Any, outcome: dict[str, Any]) -> None:
        """Block the close on durability. Raises rather than closing cleanly.

        A raise here leaves the response unterminated, so the client sees a
        failed connection and retries the unchanged request -- which is exactly
        what should happen, because the alternative is a completion the record
        does not contain.
        """
        if commit is None:
            self.commit_failures += 1
            raise TokenError(
                "no capture capacity for this turn; the response cannot be completed"
            )
        try:
            await asyncio.wait_for(commit, self._commit_timeout)
        except Exception as error:
            self.commit_failures += 1
            raise TokenError(f"this turn could not be made durable: {error}") from error
        if outcome["poisoned"]:
            # The poison has to outlive this process too: a replacement that
            # recovered the journal without it would happily extend a graph
            # this turn proved cannot be extended.
            active.poison(outcome["poison_reason"], now())
            poison = self._commits.submit(
                active, TrajectoryPoisoned(outcome["poison_reason"], now())
            )
            if poison is not None:
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(poison, self._commit_timeout)

    def _confirm_from_history(self, active: ActiveTrajectory, matched_node_ids: Any) -> None:
        """Resolve a delivery this process could not vouch for.

        A recovered exchange whose response may never have reached the client
        is settled by the client itself: if this request's message history
        contains that exchange's assistant output, the client had it. The
        confirmation is appended, the exchange stops being uncertain, and its
        exact token path is reused like any other -- which is the whole point,
        because the alternative is retokenizing model output as client-authored.

        The converse is deliberately *not* inferred. A request that does not
        continue from an uncertain output is not evidence the client never got
        it: repeated identical prompts are valid resampling, and treating one
        as a retry would delete a sample the trainer is entitled to.
        """
        if not active.integrity.delivery_uncertain_exchange_ids:
            return
        matched = set(matched_node_ids)
        for exchange in list(active.exchanges):
            if not exchange.delivery_uncertain:
                continue
            if exchange.row.get("output_node_id") not in matched:
                continue
            active.confirm_delivery(exchange.id)
            self._commits.submit(active, ExchangeDeliveryConfirmed(exchange.id, now()))

    def _poison(self, trajectory_id: str, reason: str) -> None:
        """Count a trajectory whose graph can no longer be extended correctly.

        The caller is mid-turn and is owed its completion, so this must not
        turn an attribution failure into a failed inference request. The
        in-memory seal already stops this process; what closes the route for a
        *replacement* process is the journal record, and that is appended
        beside the exchange, before the response is allowed to close.
        """
        self.trajectories_poisoned += 1
        logger.error("poisoning trajectory %s: %s", trajectory_id, reason)

    def error_body(self, status: int, message: str) -> bytes:
        """An error in the shape this proxy's clients expect: OpenAI's."""
        return self._protocol.error_body(status, message)

    async def sample_loop_lag(self, interval: float = 0.05) -> None:
        """How late the event loop runs a timer it promised to run.

        A turn's CPU work -- rendering, committing, serialising -- blocks every
        other turn on this process, and that shows up here rather than in any
        per-phase total. It is the difference between a proxy that is busy and
        one that is blocked, and it is the number to look at when throughput is
        fine but latency is not.
        """
        loop = asyncio.get_running_loop()
        while True:
            before = loop.time()
            await asyncio.sleep(interval)
            lag = (loop.time() - before - interval) * 1000
            if lag > 0:
                self.loop_lag_ms_max = max(self.loop_lag_ms_max, lag)
                self.loop_lag_ms_total += lag
            self.loop_lag_samples += 1

    # -- one turn ----------------------------------------------------------
    async def _execute_turn(
        self, active: ActiveTrajectory, request: protocol.ChatRequest
    ) -> dict[str, Any]:
        clock = time.perf_counter_ns
        # Building the renderer is a once-per-process cost that happens to be
        # paid by whichever turns arrive first, and it is seconds. Counted
        # apart from `trace`, because a startup cost inside a per-turn phase
        # reads as a per-turn cost -- see `warmup` in `stats()`.
        mark = clock()
        renderer = await self.sessions.renderer()
        self.phase_ns["warmup"] += clock() - mark

        mark = clock()
        trace = await self.sessions.trace_for(active.id)
        self.sessions.attach_text_renderer(trace, renderer)
        self.phase_ns["trace"] += clock() - mark

        mark = clock()
        pending = trace.prepare_turn(request.messages, tools=request.tools)
        self._confirm_from_history(active, pending.matched_node_ids)
        self.phase_ns["prepare"] += clock() - mark

        mark = clock()
        rendered = None
        matched_count = 0
        if pending.bridge_transition_id is not None:
            transition = trace.transition(pending.bridge_transition_id)
            matched_count = len(trace.path_to(transition.assistant_node_id))
            new_messages = pending.messages[matched_count:]
            if new_messages:
                rendered = await asyncio.to_thread(
                    renderer.bridge,
                    trace.transition_prompt_ids(pending.bridge_transition_id),
                    trace.transition_completion_ids(pending.bridge_transition_id),
                    new_messages,
                    tools=pending.tools,
                )
        if rendered is None:
            # Bridging was impossible, or its boundary did not line up. A full
            # render is always valid, so this costs a slow turn and never
            # correctness.
            rendered = await asyncio.to_thread(renderer.render, pending.messages, tools=pending.tools)
            self.render_full += 1
            message_indices = list(rendered.message_indices)
        else:
            self.render_bridged += 1
            # Indices already cover the tail alone, so shifting them into
            # full-message index space is proportional to the turn rather than
            # to the conversation.
            message_indices = [
                index + matched_count if index >= 0 else -1
                for index in rendered.message_indices
            ]

        self.phase_ns["render"] += clock() - mark

        upstream = self._upstream
        config = upstream.config
        max_model_len = int(config.get("max_model_len") or DEFAULT_MAX_MODEL_LEN)
        sampling = build_sampling_params(
            request,
            prompt_token_count=len(rendered.token_ids),
            max_model_len=max_model_len,
            stop_token_ids=renderer.get_stop_token_ids(),
            default_max_tokens=config.get("default_max_tokens"),
        )
        if sampling.get("_context_exhausted"):
            raise ProtocolError(
                400,
                f"rendered prompt of {len(rendered.token_ids)} tokens leaves no room within "
                f"max_model_len={max_model_len}",
                code="context_length_exceeded",
            )

        mark = clock()
        engine_output = await self._engine.generate(
            protocol=upstream.engine,
            url=upstream.url,
            credential=upstream.api_key,
            prompt_token_ids=rendered.token_ids,
            sampling_params=sampling,
            model=upstream.model or request.model,
            session_id=active.id,
        )
        self.phase_ns["upstream"] += clock() - mark

        mark = clock()
        completion_ids = tuple(engine_output["completion_ids"])
        assistant_message = await asyncio.to_thread(
            renderer.parse_response, completion_ids, tools=pending.tools
        )
        routed = convert_routed_experts(engine_output["routed_experts"])
        if routed is not None and len(routed) != len(rendered.token_ids) + len(completion_ids):
            raise TokenUpstreamError(
                "routed-expert data must cover the full prompt and completion sequence"
            )
        self.phase_ns["parse"] += clock() - mark

        mark = clock()
        exchange_id = new_exchange_id()
        result = ModelTurnResult(
            prompt_token_ids=tuple(rendered.token_ids),
            # Tail-only, matching what the renderer returned.
            prompt_message_indices=tuple(message_indices),
            reused_prefix_length=rendered.reused_prefix_length,
            completion_ids=completion_ids,
            completion_logprobs=tuple(engine_output["completion_logprobs"]),
            assistant_message=assistant_message,
            stop_reason=engine_output["stop_reason"],
            routed_experts=routed,
            model=request.model,
            sampling_params=canonical_sampling_params(sampling),
        )
        # Verify and commit before the response is built. A failure here is an
        # attribution failure, not an inference failure: the engine produced
        # this completion and the caller is owed it. What must not happen is a
        # *next* turn, which would render this assistant message from the
        # client's history and record model-generated tokens as client-authored.
        # So the completion is returned and the trajectory is stopped.
        poison_reason: str | None = None
        try:
            commit = trace.commit(pending, result, exchange_id=exchange_id)
        except TokenError as error:
            self.commit_failures += 1
            trace.seal()
            poison_reason = str(error)
            self._poison(active.id, poison_reason)
            commit = None
        self.phase_ns["commit"] += clock() - mark

        mark = clock()
        response = protocol.build_chat_response(
            request=request,
            assistant_message=assistant_message,
            prompt_token_ids=rendered.token_ids,
            completion_ids=completion_ids,
            completion_logprobs=result.completion_logprobs,
            stop_reason=result.stop_reason,
            decode_token=renderer.decode_token,
            response_id=f"chatcmpl-{exchange_id}",
        )

        nodes = (
            [
                {**trace.node(node_id).index_row(), "payload": trace.node(node_id).payload()}
                for node_id in commit.created_node_ids
            ]
            if commit is not None
            else []
        )
        # A poisoned turn is still captured: the tokens are what inference saw,
        # and a record of the gap is worth more than silence about it. It
        # carries no node structure, because there is none to carry.
        graph = (
            {
                "transition_id": commit.transition_id,
                "matched_message_count": commit.matched_message_count,
                "assistant_node_id": commit.assistant_node_id,
                "input_leaf_node_id": commit.input_leaf_node_id,
                "parent_output_node_id": commit.parent_output_node_id,
                "created_node_ids": list(commit.created_node_ids),
                "reused_node_ids": list(commit.reused_node_ids),
                "branched": commit.branched,
            }
            if commit is not None
            else {
                "transition_id": None,
                "matched_message_count": 0,
                "assistant_node_id": None,
                "input_leaf_node_id": None,
                "parent_output_node_id": None,
                "created_node_ids": [],
                "reused_node_ids": [],
                "branched": False,
                "poisoned": True,
            }
        )
        tokens_payload = {
            "renderer": renderer.name,
            "tokenizer": upstream.tokenizer,
            "model": request.model,
            "prompt_token_ids": list(rendered.token_ids),
            "prompt_message_indices": list(message_indices),
            "reused_prefix_length": rendered.reused_prefix_length,
            "prompt_token_count": len(rendered.token_ids),
            "completion_ids": list(completion_ids),
            "completion_logprobs": list(result.completion_logprobs),
            "stop_reason": result.stop_reason,
            "sampling_params": result.sampling_params,
            "tools": list(pending.tools) if pending.tools else None,
            "tools_hash": pending.tools_hash,
            "bridge_transition_id": pending.bridge_transition_id,
            "has_routed_experts": routed is not None,
            "nodes": nodes,
            **graph,
        }
        self.phase_ns["respond"] += clock() - mark
        return {
            "response": response,
            "exchange_id": exchange_id,
            "tokens_payload": tokens_payload,
            "renderer": renderer,
            "completion_ids": completion_ids,
            "poisoned": commit is None,
            "poison_reason": poison_reason or "",
        }

    def stats(self) -> dict[str, Any]:
        return {
            "turns_served": self.turns_served,
            "render_full": self.render_full,
            "render_bridged": self.render_bridged,
            "commit_failures": self.commit_failures,
            "trajectories_poisoned": self.trajectories_poisoned,
            # What exactness costs at the end of a turn: the time a response
            # spends waiting for its own exchange to reach the disk.
            "close_wait_ms_per_turn": round(
                self.close_wait_ns / 1e6 / max(1, self.turns_served), 3
            ),
            "delivery_unconfirmed": self.delivery_unconfirmed,
            # Both, because one of them lies on its own. `phase_ms_per_turn`
            # divides a lifetime sum by a lifetime turn count, so anything paid
            # once decays through it like 1/n and reads as a per-turn cost that
            # is merely improving. `warmup` is exactly that -- seconds of
            # tokenizer build, charged to the first turns -- and it read as
            # 130 ms/turn at 139 turns and 8 ms/turn at 2348, having cost the
            # same 18 s throughout.
            #
            # The totals are what to diff between two scrapes to get a rate
            # that means something. Keep both: the average is the right thing
            # to glance at once a process is warm.
            "phase_ms_per_turn": {
                name: round(total / 1e6 / max(1, self.turns_served), 3)
                for name, total in self.phase_ns.items()
            },
            "phase_ms_total": {
                name: round(total / 1e6, 3) for name, total in self.phase_ns.items()
            },
            "loop_lag_ms": {
                "max": round(self.loop_lag_ms_max, 2),
                "mean": round(self.loop_lag_ms_total / max(1, self.loop_lag_samples), 3),
                "samples": self.loop_lag_samples,
            },
            **self.sessions.stats(),
        }


async def _send_json(send: Any, status: int, payload: dict[str, Any]) -> None:
    await _send_raw(send, status, orjson.dumps(payload))


async def _send_raw(
    send: Any, status: int, body: bytes, *, extra_headers: list[tuple[bytes, bytes]] | None = None
) -> None:
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [(b"content-type", b"application/json"), *(extra_headers or [])],
        }
    )
    await send({"type": "http.response.body", "body": body, "more_body": False})
