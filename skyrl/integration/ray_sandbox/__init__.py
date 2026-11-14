"""Ray Sandbox (``ray.experimental.sandbox``) behind Harbor's ``modal`` environment.

Ray Sandbox serves a Modal-compatible gRPC API (ray-project/ray#66820), so Harbor runs its trials
on Ray with ``environment.type=modal``, the stock Modal SDK, and ``MODAL_SERVER_URL`` pointing at a
Ray Sandbox server. ``node_facades`` starts one such server on every node's loopback.

SkyRL's own environment stays on its Ray release: only the cluster image carries the sandbox
package, from Ray master (examples/train_integrations/harbor/ray_sandbox/Containerfile). The
harbor and harbor_skycap integrations both use it.
"""
