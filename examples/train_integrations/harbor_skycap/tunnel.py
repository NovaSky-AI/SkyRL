"""A Cloudflare quick tunnel: a random public ``https://*.trycloudflare.com`` URL to a local one.

It needs no account and is gone when it stops. ``exposure.CloudflareQuickTunnel``
opens one per skycap server, to the server's harness gateway. The cloudflared
binary is taken from ``PATH``, or downloaded once from Cloudflare's releases.

cloudflared is tied to the process that started it (``spawn_tied``): it is
stopped when that process exits however it exits, a ``kill -9`` or Ray's
``ray.kill`` of a server actor included, so a tunnel never outlives its server.
"""

import contextlib
import os
import platform
import re
import shutil
import signal
import stat
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path
from typing import Deque, List, Optional

from loguru import logger

#: A quick tunnel's own URL in cloudflared's output. api.trycloudflare.com is where tunnels are
#: requested from, and shows up in cloudflared's error lines when that request fails.
TUNNEL_URL = re.compile(r"https://(?!api\.)[-a-z0-9]+\.trycloudflare\.com")

#: Runs "$@" and kills it once stdin reaches EOF: when ``stop`` closes the pipe, or when the process
#: holding its other end dies, which the kernel does however that process exits. The watcher's output
#: goes to /dev/null so that the command's exit closes stdout, which is how a reader sees it exit.
_TIED = 'exec 3<&0; "$@" 0<&- 3<&- & child=$!; ( cat <&3 >/dev/null; kill "$child" ) >/dev/null 2>&1 & wait "$child"'


def spawn_tied(args: List[str]) -> subprocess.Popen:
    """Start ``args`` so that it stops when this process exits, however it exits.

    The process returned is a ``sh`` wrapper in a session of its own; its stdout carries the
    command's stdout and stderr. ``stop_tied`` stops it.
    """
    return subprocess.Popen(
        ["sh", "-c", _TIED, "sh", *args],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )


def stop_tied(process: subprocess.Popen, timeout: float) -> None:
    """Stop what ``spawn_tied`` started: close its stdin, then kill its process group if it lingers."""
    try:
        if process.stdin is not None:
            process.stdin.close()
        process.wait(timeout)
    except (OSError, subprocess.TimeoutExpired):
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()


class CloudflareTunnel:
    """A Cloudflare quick tunnel to a local URL: a random public ``https://*.trycloudflare.com`` URL."""

    #: Consecutive successful probes before the tunnel counts as up.
    STABLE_PROBES = 5

    def __init__(self, local_url: str) -> None:
        self.local_url = local_url
        self.url: Optional[str] = None
        self._process: Optional[subprocess.Popen] = None

    def start(self, timeout: float = 120.0, attempts: int = 3) -> str:
        """Open the tunnel and return its URL once it reaches the local URL.

        Creating a quick tunnel sometimes fails on Cloudflare's side; each attempt starts
        cloudflared afresh, and the last one's error names what it printed.
        """
        for attempt in range(1, attempts + 1):
            try:
                return self._start_once(timeout)
            except TimeoutError as error:
                if attempt == attempts:
                    raise
                logger.warning(f"quick tunnel attempt {attempt}/{attempts} failed, retrying: {error}")
                time.sleep(5 * attempt)
        raise AssertionError("unreachable")

    def _start_once(self, timeout: float) -> str:
        self.url = None
        self._process = spawn_tied([_cloudflared(), "tunnel", "--no-autoupdate", "--url", self.local_url])
        found = threading.Event()
        recent: Deque[str] = deque(maxlen=20)

        def read_output() -> None:
            # Drained for the tunnel's life, so cloudflared never blocks on a full pipe.
            assert self._process is not None and self._process.stdout is not None
            for line in self._process.stdout:
                recent.append(line.rstrip())
                match = TUNNEL_URL.search(line)
                if match and not found.is_set():
                    self.url = match.group(0)
                    found.set()
            found.set()  # cloudflared exited

        threading.Thread(target=read_output, name="cloudflared", daemon=True).start()
        deadline = time.monotonic() + timeout
        if not found.wait(timeout) or self.url is None:
            self.stop()
            raise TimeoutError(f"cloudflared gave no tunnel URL within {timeout}s; it printed: {list(recent)[-5:]}")
        # The URL is printed before it resolves, and its DNS can flap for a while after, so wait
        # until several requests in a row reach the gateway. A made-up trajectory gets skycap's
        # own 404, where a tunnel not yet up gets an error page or no address.
        probe = f"{self.url}/t/tr_probe/v1/models"
        streak = 0
        while time.monotonic() < deadline:
            streak = streak + 1 if _reaches_skycap(probe) else 0
            if streak >= self.STABLE_PROBES:
                logger.info(f"tunnel {self.url} -> {self.local_url}")
                return self.url
            time.sleep(2)
        self.stop()
        raise TimeoutError(
            f"tunnel {self.url} did not reach {self.local_url} within {timeout}s; "
            f"cloudflared printed: {list(recent)[-5:]}"
        )

    def stop(self, timeout: float = 10.0) -> None:
        process, self._process = self._process, None
        if process is not None:
            stop_tied(process, timeout)


def _reaches_skycap(url: str) -> bool:
    request = urllib.request.Request(url, headers={"User-Agent": "skycap-tunnel-probe"})
    try:
        with urllib.request.urlopen(request, timeout=10):
            return True
    except urllib.error.HTTPError as error:
        return error.code == 404 and "unknown trajectory" in error.read().decode(errors="replace")
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _cloudflared() -> str:
    """The cloudflared binary: on PATH, or downloaded once from Cloudflare's releases."""
    found = shutil.which("cloudflared")
    if found:
        return found
    arch = {"x86_64": "amd64", "amd64": "amd64", "aarch64": "arm64", "arm64": "arm64"}[platform.machine().lower()]
    path = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "skyrl" / f"cloudflared-linux-{arch}"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-{arch}"
        logger.info(f"downloading cloudflared from {url}")
        partial = path.with_suffix(".partial")
        urllib.request.urlretrieve(url, partial)
        partial.chmod(partial.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        partial.rename(path)
    return str(path)
