# Must precede the first transformer_engine import in the process, and every SkyRL
# module that imports TE or Megatron lives under this package. See the module docstring.
from skyrl.backends.skyrl_train.patches.te.pin_nvrtc import pin_te_nvrtc_to_pip_cuda

pin_te_nvrtc_to_pip_cuda()
