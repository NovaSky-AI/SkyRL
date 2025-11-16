import httpx
from datetime import datetime, timezone
from sqlmodel.ext.asyncio.session import AsyncSession

from tx.tinker import types
from tx.tinker.config import EngineConfig
from tx.tinker.db_models import FutureDB, RequestStatus
from tx.utils.log import logger
from tx.utils.storage import download_and_unpack


class ExternalInferenceClient:
    """Client for calling external inference engines (e.g., vLLM)."""

    def __init__(self, engine_config: EngineConfig, db_engine):
        self.base_url = f"{engine_config.external_inference_url}/v1"
        self.api_key = engine_config.external_inference_api_key
        self.checkpoints_base = engine_config.checkpoints_base
        self.lora_base_dir = engine_config.external_inference_lora_base
        self.db_engine = db_engine

    async def call_and_store_result(
        self,
        request_id: int,
        sample_req,
        model_id: str,
        checkpoint_id: str,
    ):
        """Background task to call external engine and store result in database."""
        try:
            async with httpx.AsyncClient(
                base_url=self.base_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=httpx.Timeout(300.0, connect=10.0),  # 5 minutes for inference, 10s for connect
            ) as http_client:
                result = await self._forward_to_engine(sample_req, model_id, checkpoint_id, http_client)
            result_data = result.model_dump()
            status = RequestStatus.COMPLETED
        except Exception as e:
            logger.exception("External engine error")
            result_data = {"error": str(e), "status": "failed"}
            status = RequestStatus.FAILED

        async with AsyncSession(self.db_engine) as session:
            future = await session.get(FutureDB, request_id)
            future.result_data = result_data
            future.status = status
            future.completed_at = datetime.now(timezone.utc)
            await session.commit()

    async def _forward_to_engine(
        self, request, model_id: str, checkpoint_id: str, http_client: httpx.AsyncClient
    ) -> types.SampleOutput:
        """Forward request to vLLM with dynamic LoRA loading.

        Extracts the checkpoint to the configured external_inference_lora_base and references it by a model name
        that vLLM can dynamically load via the lora_filesystem_resolver plugin.
        """
        prompt_tokens = [token for chunk in request.prompt.chunks for token in chunk.tokens]
        checkpoint_path = self.checkpoints_base / model_id / "sampler_weights" / f"{checkpoint_id}.tar.gz"
        model_name = f"{model_id}_{checkpoint_id}"
        target_dir = self.lora_base_dir / model_name
        target_dir.parent.mkdir(parents=True, exist_ok=True)

        # Extract the checkpoint if it doesn't already exist
        if not target_dir.exists():
            try:
                with download_and_unpack(checkpoint_path) as extracted_path:
                    extracted_path.rename(target_dir)
            except FileExistsError:
                # This could happen if two processes try to download the file.
                # In that case the other process won the race and created target_dir.
                pass

        payload = {
            "model": model_name,
            "prompt": prompt_tokens,
            "max_tokens": request.sampling_params.max_tokens,
            "temperature": request.sampling_params.temperature,
            "logprobs": True,
            "stream": False,
            "return_token_ids": True,
        }

        response = await http_client.post("/completions", json=payload)
        response.raise_for_status()
        result = response.json()

        sequences = []
        for choice in result["choices"]:
            lp = choice["logprobs"]
            sequences.append(
                types.GeneratedSequence(
                    tokens=choice["token_ids"],
                    logprobs=lp["token_logprobs"],
                    stop_reason=choice["finish_reason"],
                )
            )

        return types.SampleOutput(sequences=sequences, prompt_logprobs=[])
