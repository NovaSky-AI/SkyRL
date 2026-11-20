"""``patch_hybrid_output_processor``: upstream #7054's ``output_processor`` hook in the pinned HybridModel.forward."""

import ast
import functools
import inspect
import textwrap
import types

import pytest

pytestmark = pytest.mark.megatron

hybrid_model = pytest.importorskip("megatron.core.models.hybrid.hybrid_model", reason="requires the megatron extra")

from skyrl.backends.skyrl_train.patches.megatron import (  # noqa: E402
    patch_hybrid_output_processor as patch,
)


@pytest.fixture
def mcore_forward(monkeypatch):
    """A fresh copy of megatron-core's own ``HybridModel.forward``, installed for the test.

    The patch rewrites the function in place (importing megatron_worker already may have), so the copy is built from
    its unpatched code and the test never patches the real one.
    """
    forward = inspect.unwrap(hybrid_model.HybridModel.forward)
    code, defaults, kwdefaults, annotations = patch._ORIGINALS.get(
        forward, (forward.__code__, forward.__defaults__, forward.__kwdefaults__, forward.__annotations__)
    )
    copy = types.FunctionType(code, forward.__globals__, forward.__name__, defaults, forward.__closure__)
    copy.__kwdefaults__ = dict(kwdefaults) if kwdefaults is not None else None
    copy.__annotations__ = dict(annotations)
    copy.__qualname__ = forward.__qualname__
    copy.__doc__ = forward.__doc__
    monkeypatch.setattr(hybrid_model.HybridModel, "forward", copy)
    return copy


def _calls(source: str, name: str) -> list[ast.Call]:
    """The ``name(...)`` / ``x.name(...)`` calls in ``source``, in source order."""
    return sorted(
        (
            n
            for n in ast.walk(ast.parse(textwrap.dedent(source)))
            if isinstance(n, ast.Call) and name in (getattr(n.func, "id", None), getattr(n.func, "attr", None))
        ),
        key=lambda n: n.lineno,
    )


def test_the_hook_sits_where_gptmodel_has_it_and_passes_its_arguments(mcore_forward):
    from megatron.core.models.gpt import gpt_model

    assert patch.apply_hybrid_output_processor_patch()
    params = inspect.signature(hybrid_model.HybridModel.forward).parameters
    assert params["output_processor"].default is None and params["output_processor_context"].default is None

    # The source the patch compiles: upstream's hunks applied to megatron-core's own forward.
    source = patch.apply_patch(inspect.getsource(mcore_forward), patch._UPSTREAM_7054_HUNKS)
    (hook,) = _calls(source, "output_processor")
    # After the MTP block (process_mtp_loss), instead of the output projection.
    assert _calls(source, "process_mtp_loss")[-1].lineno < hook.lineno < _calls(source, "output_layer")[0].lineno
    # The fused head's processor is written against GPTModel's call.
    (gpt_hook,) = _calls(inspect.getsource(gpt_model.GPTModel._postprocess), "output_processor")
    assert not hook.args and [kw.arg for kw in hook.keywords] == [kw.arg for kw in gpt_hook.keywords]


def test_hunks_that_do_not_apply_leave_forward_alone_and_the_fused_head_refuses_mtp(mcore_forward, monkeypatch):
    """Never the fused head's post_process-off fallback, which would skip the MTP block."""
    import torch

    from skyrl.backends.skyrl_train.distributed.megatron.fused_lm_head import (
        call_model_with_fused_lm_head,
    )

    drifted = patch._UPSTREAM_7054_HUNKS.replace("-    ) -> Tensor:", "-    ) -> DriftedTensor:")
    monkeypatch.setattr(patch, "_UPSTREAM_7054_HUNKS", drifted)
    assert not patch.apply_hybrid_output_processor_patch()
    assert hybrid_model.HybridModel.forward is mcore_forward

    model = hybrid_model.HybridModel.__new__(hybrid_model.HybridModel)
    torch.nn.Module.__init__(model)
    model.config = types.SimpleNamespace(mtp_num_layers=1, use_mup=False, fine_grained_activation_offloading=False)
    model.post_process = True
    with pytest.raises(NotImplementedError, match="HybridModel with MTP"):
        call_model_with_fused_lm_head(model, output_processor=lambda **_: None, output_processor_context={})
    assert model.post_process


def test_a_wrapper_installed_first_stays_installed_and_wraps_the_hooked_forward(mcore_forward, monkeypatch):
    """DeepSeek-V4.1 wraps HybridModel.forward when model_bridges is imported, before megatron_worker applies this
    patch (deepseek_v41/model.py). The hook goes into megatron-core's function under the wrapper, which stays."""
    original_code = mcore_forward.__code__

    @functools.wraps(mcore_forward)
    def wrapper(self, *args, **kwargs):
        return mcore_forward(self, *args, **kwargs)

    monkeypatch.setattr(hybrid_model.HybridModel, "forward", wrapper)
    assert patch.apply_hybrid_output_processor_patch()

    assert hybrid_model.HybridModel.forward is wrapper
    assert inspect.unwrap(wrapper) is mcore_forward
    assert "output_processor" in mcore_forward.__code__.co_varnames
    params = inspect.signature(wrapper).parameters
    assert params["output_processor"].default is None and params["output_processor_context"].default is None
    # Tracebacks still point into megatron-core's hybrid_model.py, at the method's own name.
    assert mcore_forward.__code__.co_filename == original_code.co_filename
    assert mcore_forward.__code__.co_firstlineno == original_code.co_firstlineno
    assert mcore_forward.__code__.co_qualname == "HybridModel.forward"
    # Applying again through the wrapper is a no-op.
    patched_code = mcore_forward.__code__
    assert patch.apply_hybrid_output_processor_patch()
    assert hybrid_model.HybridModel.forward is wrapper and mcore_forward.__code__ is patched_code


def test_a_noop_that_logs_delete_me_once_megatron_core_has_the_hook(monkeypatch):
    from loguru import logger

    def forward(self, input_ids, output_processor=None, output_processor_context=None):
        return input_ids

    monkeypatch.setattr(hybrid_model.HybridModel, "forward", forward)
    messages = []
    handle = logger.add(messages.append, level="INFO")
    try:
        assert not patch.apply_hybrid_output_processor_patch()
    finally:
        logger.remove(handle)
    assert hybrid_model.HybridModel.forward is forward
    assert any("delete patch_hybrid_output_processor.py" in m for m in messages)
