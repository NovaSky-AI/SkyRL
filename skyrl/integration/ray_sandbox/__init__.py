"""Ray sandboxes (``ray.experimental.sandbox``) as an agent environment.

* ``actor.HarborSandbox``: a Ray actor that owns one gVisor sandbox. It needs only Ray.
* ``harbor_env.RaySandboxEnvironment``: a Harbor environment on that actor. Select it in a
  Harbor trial config with ``environment.import_path``; it needs the ``harbor`` extra.

The harbor and harbor_skycap integrations both use it. Nodes that host sandboxes need the
tools in docker/Dockerfile.ray-sandbox, and Ray at the commit that pyproject.toml pins.
"""
