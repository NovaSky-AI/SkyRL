import socket
import ray
from cloudpathlib import AnyPath

from skyrl.backends.backend import AbstractBackend
from skyrl.backends.jax import JaxBackend, JaxBackendConfig, run_worker
from skyrl.tinker import types
from skyrl.utils.log import logger

def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@ray.remote
class RayJaxCoordinatorActor:
    """Ray Actor wrapper for the JaxBackend coordinator (process_id = 0).
    
    This actor dynamically allocates a port, provides its coordinator_address
    to workers, and then blocks initializing jax.distributed until workers join.
    """
    def __init__(self, base_model: str, config: JaxBackendConfig):
        self.base_model = base_model
        # Use model_copy so we don't accidentally mutate unintended shared state
        self.config = config.model_copy()
        
        # Determine coordinator address
        self.node_ip = ray.util.get_node_ip_address()
        self.port = get_free_port()
        self.coordinator_address = f"{self.node_ip}:{self.port}"
        
        # Update config with dynamically found address
        self.config.coordinator_address = self.coordinator_address
        self.backend = None

    def get_coordinator_address(self) -> str:
        return self.coordinator_address

    def setup(self):
        """Initializes the backend. Blocks until all workers connect to the coordinator."""
        self.backend = JaxBackend(self.base_model, self.config)

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


@ray.remote
class RayJaxWorkerActor:
    """Ray Actor wrapper for JaxBackend workers (process_id > 0)."""
    def __init__(self, coordinator_address: str, num_processes: int, process_id: int):
        self.coordinator_address = coordinator_address
        self.num_processes = num_processes
        self.process_id = process_id

    def run(self):
        """Run the worker loop infinitely."""
        run_worker(self.coordinator_address, self.num_processes, self.process_id)


class RayJaxBackend(AbstractBackend):
    """Proxy Backend that orchestrates Ray actors for multi-node JAX execution.
    
    Locally, this class acts like a normal AbstractBackend. Internally, it creates 
    a RayJaxCoordinatorActor (which internally wraps JaxBackend) to execute work,
    and dynamically provisions RayJaxWorkerActors matching `num_processes`.
    """
    def __init__(self, base_model: str, config: JaxBackendConfig):
        self.base_model = base_model
        self.config = config.model_copy()
        
        num_processes = self.config.num_processes or 1
        self.config.num_processes = num_processes
        
        logger.info(f"Initializing RayJaxBackend with num_processes={num_processes}")

        # Instantiate the coordinator but do not run setup yet (to avoid blocking)
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        current_node_id = ray.get_runtime_context().get_node_id()
        logger.info(f"Scheduling RayJaxCoordinatorActor on node {current_node_id} to co-locate with TinkerEngine")
        self.actor = RayJaxCoordinatorActor.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=current_node_id,
                soft=False,
            )
        ).remote(self.base_model, self.config)
        
        # Retrieve dynamically allocated coordinator address from actor
        coordinator_address = ray.get(self.actor.get_coordinator_address.remote())
        
        self.worker_tasks = []
        if num_processes > 1:
            for i in range(1, num_processes):
                worker = RayJaxWorkerActor.remote(coordinator_address, num_processes, i)
                self.worker_tasks.append(worker.run.remote())
                
        # Trigger the coordinator setup, initializing JAX distributed. 
        # This will block until the workers connect successfully.
        ray.get(self.actor.setup.remote())
        
        logger.info("RayJaxBackend is fully initialized and distributed cluster is ready.")

    @property
    def metrics(self) -> types.EngineMetrics:
        return ray.get(self.actor.get_metrics.remote())

    def create_model(self, model_id: str, lora_config: types.LoraConfig) -> None:
        ray.get(self.actor.create_model.remote(model_id, lora_config))

    def delete_model(self, model_id: str) -> None:
        ray.get(self.actor.delete_model.remote(model_id))

    def has_model(self, model_id: str) -> bool:
        return ray.get(self.actor.has_model.remote(model_id))

    def forward_backward(self, prepared_batch: types.PreparedModelPassBatch):
        return ray.get(self.actor.forward_backward.remote(prepared_batch))

    def forward(self, prepared_batch: types.PreparedModelPassBatch):
        return ray.get(self.actor.forward.remote(prepared_batch))

    def optim_step(self, model_id: str, request_data: types.OptimStepInput) -> types.OptimStepOutput | types.ErrorResponse:
        return ray.get(self.actor.optim_step.remote(model_id, request_data))

    def sample(self, prepared_batch: types.PreparedSampleBatch) -> dict[str, types.SampleOutput | types.ErrorResponse]:
        return ray.get(self.actor.sample.remote(prepared_batch))

    def load_checkpoint(self, checkpoint_path: AnyPath, model_id: str) -> None:
        ray.get(self.actor.load_checkpoint.remote(checkpoint_path, model_id))

    def save_checkpoint(self, output_path: AnyPath, model_id: str) -> None:
        ray.get(self.actor.save_checkpoint.remote(output_path, model_id))

    def save_sampler_checkpoint(self, output_path: AnyPath, model_id: str, persist: bool = True) -> None:
        ray.get(self.actor.save_sampler_checkpoint.remote(output_path, model_id, persist))
