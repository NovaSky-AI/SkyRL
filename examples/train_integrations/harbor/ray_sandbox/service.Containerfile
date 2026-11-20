# syntax=docker/dockerfile:1.7
# Service image for the non-colocated layout (service.yaml, README.md): Ray
# Sandbox's Modal-compatible gRPC facade as a Ray Serve app, from Ray master
# (ray-project/ray#66820). Clients use the stock Modal SDK.
#
#   anyscale image build -f examples/train_integrations/harbor/ray_sandbox/service.Containerfile \
#     -n ray-sandbox-modal-envoy -r nightly
FROM anyscale/ray:nightly-py312

# gVisor (runsc release-20260817 + hostinet MSG_CMSG_CLOEXEC patch) and the
# user-mode network bridge for network="public" sandboxes.
RUN ARCH=$(uname -m) && test "$ARCH" = x86_64 \
 && curl -fsSLo /tmp/runsc https://github.com/xyuzh/gvisor-runsc-patched/releases/download/runsc-cmsg-cloexec-v1/runsc \
 && echo "f2677cf7706ef5b5db507b71a6af7f4932628c9353b095e3909276fee1ee0782  /tmp/runsc" | sha256sum -c - \
 && sudo install -m 755 /tmp/runsc /usr/local/bin/runsc && rm /tmp/runsc \
 && curl -fsSLo /tmp/slirp4netns https://github.com/rootless-containers/slirp4netns/releases/download/v1.3.5/slirp4netns-$ARCH \
 && sudo install -m 755 /tmp/slirp4netns /usr/local/bin/slirp4netns && rm /tmp/slirp4netns \
 && runsc --version | head -1 && slirp4netns --version | head -1

# erofs-utils 1.9.4 (mkfs.erofs --tar) for the sandbox image cache.
RUN sudo apt-get update -qq && sudo apt-get install -y -qq --no-install-recommends \
      build-essential autoconf automake libtool pkg-config liblz4-dev libzstd-dev uuid-dev git procps \
 && git clone -q --depth 1 --branch v1.9.4 https://github.com/erofs/erofs-utils.git /tmp/erofs-utils \
 && cd /tmp/erofs-utils && ./autogen.sh >/dev/null && ./configure --disable-fuse >/dev/null && make -j8 >/dev/null \
 && sudo make install >/dev/null && cd / && rm -rf /tmp/erofs-utils \
 && mkfs.erofs --version 2>&1 | head -1

# ray.experimental.sandbox at Ray master (#66820's merge commit), routed
# ahead of the platform-synced ray copy by a meta-path
# finder from a .pth, so every Python process sees it. That includes Serve's
# gRPC proxies, which import grpc_facade.add_servicers_to_server at startup and
# never get a working_dir.
RUN sudo mkdir -p /opt/ray-sandbox-overlay/pylib && sudo chown -R ray /opt/ray-sandbox-overlay \
 && cat > /opt/ray-sandbox-overlay/pylib/ray_sandbox_overlay_graft.py <<'EOF'
"""Route ``ray.experimental.sandbox`` to the overlay copy under /opt.

Anyscale re-syncs its managed ``ray`` package into the image's Python env at
cluster start, so files placed inside ``site-packages/ray`` do not survive.
This finder is installed from a ``.pth`` file and wins over the shipped copy
in every Python process (drivers, workers, actors).
"""

import importlib.abc
import importlib.util
import os
import sys

_ROOT = "/opt/ray-sandbox-overlay/pkg/sandbox"
_PREFIX = "ray.experimental.sandbox"


class _OverlayFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != _PREFIX and not fullname.startswith(_PREFIX + "."):
            return None
        rel = fullname[len(_PREFIX):].lstrip(".").replace(".", os.sep)
        base = os.path.join(_ROOT, rel) if rel else _ROOT
        if os.path.isdir(base):
            init = os.path.join(base, "__init__.py")
            if not os.path.exists(init):
                return None
            return importlib.util.spec_from_file_location(
                fullname, init, submodule_search_locations=[base]
            )
        module = base + ".py"
        if os.path.exists(module):
            return importlib.util.spec_from_file_location(fullname, module)
        return None


if not any(isinstance(f, _OverlayFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _OverlayFinder())
EOF
RUN pip install -q grpclib \
 && git init -q /tmp/rayclone && cd /tmp/rayclone \
 && git remote add origin https://github.com/ray-project/ray.git \
 && git sparse-checkout init --cone && git sparse-checkout set python/ray/experimental/sandbox \
 && git fetch -q --depth 1 --filter=blob:none origin 68174559b27a4eba8db6b3c7ef7aaa2678ec20c8 && git checkout -q FETCH_HEAD \
 && sudo mkdir -p /opt/ray-sandbox-overlay/pkg && sudo cp -r python/ray/experimental/sandbox /opt/ray-sandbox-overlay/pkg/sandbox \
 && sudo find /opt/ray-sandbox-overlay/pkg/sandbox -type d -name tests -prune -exec rm -rf {} + \
 && sudo chown -R ray /opt/ray-sandbox-overlay && cd / && rm -rf /tmp/rayclone \
 && SITE=$(python -c "import site; print(site.getsitepackages()[0])") \
 && printf '/opt/ray-sandbox-overlay/pylib\nimport ray_sandbox_overlay_graft\n' > "$SITE/zz_ray_sandbox_overlay.pth" \
 && python -c "from ray.experimental.sandbox.http import grpc_app; print('grpc_app from', grpc_app.__file__)"
