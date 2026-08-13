"""Self-retiring LIBFABRIC shim for Ray RDT on EFA fabrics.

Ray <= 2.56 hardcodes NIXL's UCX backend (``nixl_agent_config(backends=["UCX"])``),
which has no working EFA data path. Upstream fixed this with auto-selection
(ray-project/ray#62339, post-2.56); until SkyRL's pin includes it, this shim
patches ``NixlTensorTransport.get_nixl_agent`` to select LIBFABRIC when EFA
devices are present. It is a no-op when the installed Ray already has the
upstream fix (``_is_efa_available`` on the module) or on non-EFA nodes.

Call ``ensure_ray_rdt_libfabric()`` in every process type that creates a NIXL
agent: the RDT producer server and the vLLM inference workers.
"""

import glob
import logging
import os

logger = logging.getLogger(__name__)


def _efa_present() -> bool:
    if glob.glob("/sys/class/net/efa*"):
        return True
    for ib_dev in glob.glob("/sys/class/infiniband/*"):
        try:
            driver = os.path.realpath(os.path.join(ib_dev, "device", "driver"))
        except OSError:
            continue
        if os.path.basename(driver) == "efa":
            return True
    return False


def ensure_ray_rdt_libfabric() -> None:
    try:
        import ray.experimental.rdt.nixl_tensor_transport as t
    except Exception:  # noqa: BLE001 - no ray/rdt in this process
        return
    if hasattr(t, "_is_efa_available"):
        return  # upstream #62339 present (or file already patched)
    cls = t.NixlTensorTransport
    if getattr(cls, "_skyrl_libfabric_shim", False):
        return
    if not _efa_present():
        return

    def get_nixl_agent(self):  # mirrors Ray 2.56's method, backend swapped
        if self._nixl_agent is not None:
            return self._nixl_agent
        import uuid

        import ray
        from nixl._api import nixl_agent, nixl_agent_config

        actor_id = ray.get_runtime_context().get_actor_id() or f"RAY-DRIVER-{uuid.uuid4()}"
        self._nixl_agent = nixl_agent(actor_id, nixl_agent_config(backends=["LIBFABRIC"]))
        print("[rdt-libfabric-shim] NIXL agent created with LIBFABRIC backend", flush=True)
        return self._nixl_agent

    cls.get_nixl_agent = get_nixl_agent
    cls._skyrl_libfabric_shim = True
    logger.info("[rdt-libfabric-shim] patched Ray RDT NIXL backend to LIBFABRIC (EFA)")
