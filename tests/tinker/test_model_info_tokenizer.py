"""Preserve tokenizer identity across the native API and Tinker SDK."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from tinker import types as sdk_types
from tinker.lib.public_interfaces.sampling_client import _load_tokenizer_from_model_info

from skyrl.tinker import api


@pytest.mark.asyncio
@pytest.mark.parametrize("model_path", ["/scratch/model", "/shared/models/pinned", "Qwen/Qwen3-0.6B"])
@pytest.mark.parametrize("rank", [0, 32])
async def test_model_info_preserves_tokenizer_and_adapter_identity(model_path, rank):
    model = SimpleNamespace(
        model_id="model-test", status="ready", base_model=model_path, lora_config={"rank": rank, "alpha": 32, "seed": 0}
    )
    with patch.object(api, "get_model", new=AsyncMock(return_value=model)):
        response = await api.get_model_info(api.GetInfoRequest(model_id=model.model_id), session=object())
    parsed = sdk_types.GetInfoResponse.model_validate(response.model_dump())
    assert parsed.is_lora is (rank > 0)
    assert parsed.lora_rank == rank
    _load_tokenizer_from_model_info.cache_clear()
    try:
        with patch("transformers.models.auto.tokenization_auto.AutoTokenizer.from_pretrained") as load:
            _load_tokenizer_from_model_info(parsed.model_data.model_name, parsed.model_data.tokenizer_id)
        assert load.call_args.args[0] == model_path
    finally:
        _load_tokenizer_from_model_info.cache_clear()
