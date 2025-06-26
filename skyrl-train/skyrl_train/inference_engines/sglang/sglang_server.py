import os
import sys

from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.server_args import prepare_server_args, ServerArgs
from sglang.srt.utils import kill_process_tree
import sglang
from pathlib import Path
import subprocess

PATCH_FILE_PATH = Path(__file__).parent / "sglang.patch"

def apply_patch():
    try:
        sglang_path = Path(sglang.__file__).parent
        subprocess.run(["patch", "-p1", "-d", str(sglang_path), "-i", str(PATCH_FILE_PATH), "--batch", "--forward"], check=True)
    except Exception as e:
        print(f"Failed to patch sglang: {e}", file=sys.stderr)
        sys.exit(1)

class SGLangServer:
    def __init__(self, server_args: ServerArgs):
        self.server_args = server_args

    def run_server(self) -> None:
        try:
            launch_server(self.server_args)
        finally:
            kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    apply_patch()
    server_args = prepare_server_args(sys.argv[1:])
    sglang_server = SGLangServer(server_args)
    sglang_server.run_server()
