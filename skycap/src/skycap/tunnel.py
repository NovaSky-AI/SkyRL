"""A Cloudflare quick tunnel: a random public ``https://*.trycloudflare.com`` URL to a local one.

It needs no account and is gone when it stops. ``skycap.exposure.CloudflareQuickTunnel``
opens one per server, to the server's harness listener. The cloudflared binary
is taken from ``PATH``, or downloaded once from Cloudflare's releases (Linux).

cloudflared is tied to the process that started it (``spawn_tied``): it stops
when that process exits however it exits, a ``kill -9`` included, so a tunnel
never outlives its server.
"""

from __future__ import annotations

import contextlib
import logging
import os
import platform
import re
import shutil
import signal
import stat
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path

logger = logging.getLogger(__name__)

#: A quick tunnel's own URL in cloudflared's output. api.trycloudflare.com is where tunnels are
#: requested from, and shows up in cloudflared's error lines when that request fails.
TUNNEL_URL = re.compile(r"https://(?!api\.)[-a-z0-9]+\.trycloudflare\.com")

#: Runs "$@" and kills it once stdin reaches EOF: when ``stop_tied`` closes the pipe, or when the process
#: holding its other end dies, which the kernel does however that process exits. The watcher's output
#: goes to /dev/null so that the command's exit closes stdout, which is how a reader sees it exit.
_TIED = 'exec 3<&0; "$@" 0<&- 3<&- & child=$!; ( cat <&3 >/dev/null; kill "$child" ) >/dev/null 2>&1 & wait "$child"'


def spawn_tied(args: list[str]) -> subprocess.Popen[str]:
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


def stop_tied(process: subprocess.Popen[str], timeout: float) -> None:
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
    """A Cloudflare quick tunnel to a local URL that serves skycap's harness routes."""

    #: Consecutive successful probes before the tunnel counts as up.
    STABLE_PROBES = 5

    def __init__(self, local_url: str) -> None:
        self.local_url = local_url
        self.url: str | None = None
        self._process: subprocess.Popen[str] | None = None

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
                logger.warning("quick tunnel attempt %d/%d failed, retrying: %s", attempt, attempts, error)
                time.sleep(5 * attempt)
        raise AssertionError("unreachable")

    def _start_once(self, timeout: float) -> str:
        process = spawn_tied([_cloudflared(), "tunnel", "--no-autoupdate", "--url", self.local_url])
        self._process = process
        found: dict[str, str] = {}
        printed = threading.Event()
        recent: deque[str] = deque(maxlen=20)

        def read_output() -> None:
            # Drained for the tunnel's life, so cloudflared never blocks on a full pipe.
            assert process.stdout is not None
            for line in process.stdout:
                recent.append(line.rstrip())
                match = TUNNEL_URL.search(line)
                if match and not printed.is_set():
                    found["url"] = match.group(0)
                    printed.set()
            printed.set()  # cloudflared exited

        threading.Thread(target=read_output, name="cloudflared", daemon=True).start()
        deadline = time.monotonic() + timeout
        if not printed.wait(timeout) or "url" not in found:
            self.stop()
            raise TimeoutError(f"cloudflared gave no tunnel URL within {timeout}s; it printed: {list(recent)[-5:]}")
        url = found["url"]
        # The URL is printed before it resolves, and its DNS can flap for a while after, so wait
        # until several requests in a row reach skycap. A made-up trajectory gets skycap's own 404,
        # where a tunnel not yet up gets an error page or no address.
        probe = f"{url}/t/tr_probe/v1/models"
        streak = 0
        while time.monotonic() < deadline:
            streak = streak + 1 if _reaches_skycap(probe) else 0
            if streak >= self.STABLE_PROBES:
                logger.info("tunnel %s -> %s", url, self.local_url)
                self.url = url
                return url
            time.sleep(2)
        self.stop()
        raise TimeoutError(
            f"tunnel {url} did not reach {self.local_url} within {timeout}s; cloudflared printed: {list(recent)[-5:]}"
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
    """The cloudflared binary: on PATH, or downloaded once from Cloudflare's releases (Linux only)."""
    found = shutil.which("cloudflared")
    if found:
        return found
    machine = platform.machine().lower()
    arch = {"x86_64": "amd64", "amd64": "amd64", "aarch64": "arm64", "arm64": "arm64"}.get(machine)
    if platform.system() != "Linux" or arch is None:
        raise RuntimeError(f"cloudflared is not on PATH; install it for {platform.system()} {machine}")
    path = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "skycap" / f"cloudflared-linux-{arch}"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-{arch}"
        logger.info("downloading cloudflared from %s", url)
        # A temporary file of this process's own, renamed into place: servers that start together on one
        # node each download, and the last rename wins with a complete binary.
        fd, partial = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.", suffix=".partial")
        os.close(fd)
        try:
            urllib.request.urlretrieve(url, partial)
            os.chmod(partial, os.stat(partial).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            os.replace(partial, path)
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(partial)
    return str(path)
