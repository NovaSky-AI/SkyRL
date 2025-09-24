import ray
import os


@ray.remote(num_cpus=1)
def test_deps():
    print(os.environ.get("PYTHONPATH"))
    import transformer_engine
    print(transformer_engine.__version__)

print(os.environ.get("PYTHONPATH"))
ray.get(test_deps.remote())