import os
import signal
import uvloop
from vllm import AsyncLLMEngine
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.system_utils import set_ulimit
from vllm.entrypoints.openai.cli_args import (
    make_arg_parser,
    validate_parsed_serve_args,
)
from vllm.entrypoints.launcher import serve_http
from vllm.entrypoints.openai.api_server import (
    create_server_socket,
    build_app,
    init_app_state,
)
import vllm.envs as envs
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.usage.usage_lib import UsageContext
from fastapi import Request


# TODO(tgriggs): Handle errors and use best practices for vLLM server
# TODO(tgriggs): Return correct status codes.
class VllmServer:
    def __init__(self, args):
        self.server_args = args

    async def run_server(self, **uvicorn_kwargs) -> None:
        sock_addr = (self.server_args.host or "", self.server_args.port)
        sock = create_server_socket(sock_addr)

        set_ulimit()

        def signal_handler(*_) -> None:
            # Interrupt server on sigterm while initializing
            raise KeyboardInterrupt("terminated")

        signal.signal(signal.SIGTERM, signal_handler)

        # TODO(tgriggs): Move this elsewhere, make configurable.
        os.environ["VLLM_USE_V1"] = "1"
        engine_args = AsyncEngineArgs.from_cli_args(self.server_args)
        engine = AsyncLLMEngine.from_engine_args(
            engine_args=engine_args,
            usage_context=UsageContext.OPENAI_API_SERVER,
        )

        sock_addr = (self.server_args.host or "", self.server_args.port)
        sock = create_server_socket(sock_addr)
        app = build_app(self.server_args)

        @app.post("/init_weight_update_communicator")
        async def _init_weight_update_communicator(request: Request):
            import pickle
            from skyrl_train.weight_sync import BroadcastInitInfo

            data = await request.json()
            init_info = BroadcastInitInfo(**data)

            # Pickle to preserve type through collective_rpc
            pickled_init_info = pickle.dumps(init_info)

            await engine.collective_rpc(
                "init_weight_update_communicator",
                args=(pickled_init_info,),
            )
            return {"status": "ok"}

        @app.post("/sleep")
        async def _sleep(request: Request):
            data = await request.json()
            level = data.get("level")

            # TODO(team): remove once vllm fixes this
            # otherwise waking it up will output gibberish: https://github.com/vllm-project/vllm/issues/17103
            await engine.reset_prefix_cache()

            await engine.sleep(level)
            return {"status": "ok"}

        @app.post("/wake_up")
        async def _wake_up(request: Request):
            data = await request.json()
            tags = data.get("tags")
            await engine.wake_up(tags)
            return {"status": "ok"}

        @app.post("/reset_prefix_cache")
        async def _reset_prefix_cache(request: Request):
            await engine.reset_prefix_cache()
            return {"status": "ok"}

        @app.post("/update_weights")
        async def _update_weights(request: Request):
            import pickle
            from skyrl_train.weight_sync import BroadcastWeightUpdateRequest

            # Convert the HTTP request to a BroadcastWeightUpdateRequest
            # TODO(haochen): only the broadcast strategy is currently supported
            # for the remote inference engine path.
            # To support other strategies, we'll need to add a "strategy=xxx"
            # parameter in the HTTP request.
            data = await request.json()
            weight_request = BroadcastWeightUpdateRequest(**data)

            # Pickle to preserve type through collective_rpc
            pickled_request = pickle.dumps(weight_request)

            await engine.collective_rpc(
                "load_weights",
                args=(pickled_request,),
            )
            return {"status": "ok"}

        @app.post("/destroy_weights_update_group")
        async def _destroy_weights_update_group(request: Request):
            data = await request.json()  # noqa: F841
            await engine.collective_rpc(
                "teardown_weight_receiver",
                args=(),
            )
            return {"status": "ok"}

        await init_app_state(engine, app.state, args)

        shutdown_task = await serve_http(
            app,
            sock,
            host=self.server_args.host,
            port=self.server_args.port,
            log_level=self.server_args.uvicorn_log_level,
            timeout_keep_alive=envs.VLLM_HTTP_TIMEOUT_KEEP_ALIVE,
            ssl_keyfile=self.server_args.ssl_keyfile,
            ssl_certfile=self.server_args.ssl_certfile,
            ssl_ca_certs=self.server_args.ssl_ca_certs,
            ssl_cert_reqs=self.server_args.ssl_cert_reqs,
            **uvicorn_kwargs,
        )

        await shutdown_task

        sock.close()

    def run_server_uvloop(self, **uvicorn_kwargs) -> None:
        uvloop.run(self.run_server(**uvicorn_kwargs))


if __name__ == "__main__":
    parser = FlexibleArgumentParser(description="vLLM OpenAI-Compatible RESTful API server.")
    parser = make_arg_parser(parser)
    args = parser.parse_args()
    validate_parsed_serve_args(args)

    vllm_server = VllmServer(args)
    vllm_server.run_server_uvloop()
