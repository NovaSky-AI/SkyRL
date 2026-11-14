"""Ray sandboxes (``ray.experimental.sandbox``) as an agent environment.

* ``actor.HarborSandbox``: a Ray actor that owns one gVisor sandbox. It needs only Ray.
* ``harbor_env.RaySandboxEnvironment``: a Harbor environment on that actor. Select it in a
  Harbor trial config with ``environment.import_path``; it needs the ``harbor`` extra.

The harbor and harbor_skycap integrations both use it. It needs Ray with the sandbox fixes of
October 2026 (Ray master @ 78e4b1f713; the first release with them is expected to be 2.61) in the
job's environment and on every node. SkyRL's pinned Ray (2.57.0) doesn't have them yet. Nodes that
host sandboxes also need the tools in docker/Dockerfile.ray-sandbox.
"""
