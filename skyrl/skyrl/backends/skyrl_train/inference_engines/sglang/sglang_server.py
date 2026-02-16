import os
import sys

from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.server_args import prepare_server_args, ServerArgs
from sglang.srt.utils import kill_process_tree


class SGLangServer:
    def __init__(self, server_args: ServerArgs):
        self.server_args = server_args

    def run_server(self) -> None:
        try:
            launch_server(self.server_args)
        finally:
            kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    args = sys.argv[1:]
    # SGLang requires `skip-tokenizer-init` to do token-in-token-out with `/generate` endpoint
    if "--skip-tokenizer-init" not in args:
        args.append("--skip-tokenizer-init")
    server_args = prepare_server_args(args)
    sglang_server = SGLangServer(server_args)
    sglang_server.run_server()
