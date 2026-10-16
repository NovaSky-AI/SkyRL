# Must precede the first transformer_engine import in the process, and every SkyRL
# module that imports TE or Megatron lives under this package. See the module docstring.
from skyrl.backends.skyrl_train.patches.te.pin_nvrtc import pin_te_nvrtc_to_pip_cuda

pin_te_nvrtc_to_pip_cuda()

# transformers has no Beam config yet; register a vendored one so `AutoConfig` resolves
# `model_type="beam"` in every SkyRL process (driver, workers, inference servers).
from skyrl.backends.skyrl_train.patches.transformers.configuration_beam import (  # noqa: E402
    register_beam_config,
)

register_beam_config()
