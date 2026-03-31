import socket
import ray
from cloudpathlib import AnyPath

from skyrl.backends.backend import AbstractBackend
from skyrl.backends.jax import JaxBackendConfig, JaxBackendImpl
from skyrl.tinker import types
from skyrl.utils.log import logger

def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@ray.remote
class RayJaxActor:
    """Ray Actor wrapper for JaxBackendImpl.
    
    Each actor runs JaxBackendImpl and communicates with other actors
    via JAX distributed (NCCL) for data parallel operations.
    """
    def __init__(self, base_model: str, config: JaxBackendConfig, process_id: int):
        self.base_model = base_model
        self.config = config.model_copy()
        self.process_id = process_id
        self.backend = None

        if process_id == 0:
            self.node_ip = ray.util.get_node_ip_address()
            self.port = get_free_port()
            self.coordinator_address = f"{self.node_ip}:{self.port}"
        else:
            self.coordinator_address = None

    def get_coordinator_address(self) -> str:
        return self.coordinator_address

    def setup(self, coordinator_address: str | None = None):
        """Initializes JAX distributed and creates JaxBackendImpl."""
        import jax # Import here to avoid issues or ensure it's loaded in the actor

        addr = coordinator_address or self.coordinator_address
        logger.info(f"Worker {self.process_id} initializing JAX distributed with coordinator {addr}")

        jax.distributed.initialize(
            coordinator_address=addr,
            num_processes=self.config.num_processes,
            process_id=self.process_id,
        )
        self.backend = JaxBackendImpl(self.base_model, self.config, self.process_id)
        logger.info(f"Worker {self.process_id} JaxBackendImpl initialized.")

    # =========================================================================
    # Proxied Backend Methods
    # =========================================================================

    def create_model(self, model_id: str, lora_config: types.LoraConfig) -> None:
        self.backend.create_model(model_id, lora_config)

    def forward_backward(self, prepared_batch: types.PreparedModelPassBatch):
        return self.backend.forward_backward(prepared_batch)

    def forward(self, prepared_batch: types.PreparedModelPassBatch):
        return self.backend.forward(prepared_batch)

    def optim_step(self, model_id: str, request_data: types.OptimStepInput):
        return self.backend.optim_step(model_id, request_data)

    def sample(self, prepared_batch: types.PreparedSampleBatch):
        return self.backend.sample(prepared_batch)

    def save_checkpoint(self, output_path: AnyPath, model_id: str) -> None:
        self.backend.save_checkpoint(output_path, model_id)

    def load_checkpoint(self, checkpoint_path: AnyPath, model_id: str) -> None:
        self.backend.load_checkpoint(checkpoint_path, model_id)

    def save_sampler_checkpoint(self, output_path: AnyPath, model_id: str, persist: bool = True) -> None:
        self.backend.save_sampler_checkpoint(output_path, model_id, persist)

    def has_model(self, model_id: str) -> bool:
        return self.backend.has_model(model_id)

    def delete_model(self, model_id: str) -> None:
        self.backend.delete_model(model_id)

    def get_metrics(self) -> types.EngineMetrics:
        return self.backend.metrics


class RayJaxBackend(AbstractBackend):
    """Proxy Backend that orchestrates Ray actors for multi-node JAX execution.
    
    This class runs in the driver program (Tinker Engine process) and proxies
    commands to all JAX workers running as Ray actors.
    """
    def __init__(self, base_model: str, config: JaxBackendConfig):
        self.base_model = base_model
        self.config = config.model_copy()
        
        num_processes = self.config.num_processes or 1
        self.config.num_processes = num_processes
        
        logger.info(f"Initializing RayJaxBackend with num_processes={num_processes}")

        # Instantiate a Ray placement group
        from ray.util.placement_group import placement_group
        logger.info("Instantiating Ray placement group for JAX workers...")
        bundles = self.config.ray_placement_group_bundles
        if not bundles:
            bundles = [{"CPU": 1}] * num_processes
        self.pg = placement_group(bundles, strategy="SPREAD")
        ray.get(self.pg.ready())

        self.workers = []
        
        # Create worker 0 (coordinator)
        w0_options = self.config.ray_actor_options.copy()
        w0_options.update({
            "placement_group": self.pg,
            "placement_group_bundle_index": 0,
        })
        w0 = RayJaxActor.options(**w0_options).remote(self.base_model, self.config, 0)
        self.workers.append(w0)

        # Retrieve dynamically allocated coordinator address from actor 0
        coordinator_address = ray.get(w0.get_coordinator_address.remote())
        
        # Create other workers
        for i in range(1, num_processes):
            wi_options = self.config.ray_actor_options.copy()
            wi_options.update({
                "placement_group": self.pg,
                "placement_group_bundle_index": i,
            })
            w = RayJaxActor.options(**wi_options).remote(self.base_model, self.config, i)
            self.workers.append(w)

        # Trigger setup on all workers
        # This will block until JAX distributed is initialized on all workers
        setup_refs = [w0.setup.remote()]
        for w in self.workers[1:]:
            setup_refs.append(w.setup.remote(coordinator_address))

        ray.get(setup_refs)
        logger.info("RayJaxBackend is fully initialized and distributed cluster is ready.")

    @property
    def metrics(self) -> types.EngineMetrics:
        return ray.get(self.workers[0].get_metrics.remote())

    def create_model(self, model_id: str, lora_config: types.LoraConfig) -> None:
        ray.get([w.create_model.remote(model_id, lora_config) for w in self.workers])

    def delete_model(self, model_id: str) -> None:
        ray.get([w.delete_model.remote(model_id) for w in self.workers])

    def has_model(self, model_id: str) -> bool:
        return ray.get(self.workers[0].has_model.remote(model_id))

    def forward_backward(self, prepared_batch: types.PreparedModelPassBatch):
        results = ray.get([w.forward_backward.remote(prepared_batch) for w in self.workers])
        return results[0]

    def forward(self, prepared_batch: types.PreparedModelPassBatch):
        results = ray.get([w.forward.remote(prepared_batch) for w in self.workers])
        return results[0]

    def optim_step(self, model_id: str, request_data: types.OptimStepInput) -> types.OptimStepOutput | types.ErrorResponse:
        results = ray.get([w.optim_step.remote(model_id, request_data) for w in self.workers])
        return results[0]

    def sample(self, prepared_batch: types.PreparedSampleBatch) -> dict[str, types.SampleOutput | types.ErrorResponse]:
        results = ray.get([w.sample.remote(prepared_batch) for w in self.workers])
        return results[0]

    def load_checkpoint(self, checkpoint_path: AnyPath, model_id: str) -> None:
        ray.get([w.load_checkpoint.remote(checkpoint_path, model_id) for w in self.workers])

    def save_checkpoint(self, output_path: AnyPath, model_id: str) -> None:
        ray.get([w.save_checkpoint.remote(output_path, model_id) for w in self.workers])

    def save_sampler_checkpoint(self, output_path: AnyPath, model_id: str, persist: bool = True) -> None:
        ray.get([w.save_sampler_checkpoint.remote(output_path, model_id, persist) for w in self.workers])
