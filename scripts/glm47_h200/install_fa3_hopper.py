"""
Install flash-attn v3 Hopper kernels on GPU worker nodes.

TransformerEngine 2.10 looks for `flash_attn_3` package for native FA3 support.
Without it, TE falls back to cuDNN FA3 which is slower.

This script:
1. Clones flash-attention repo
2. Builds the Hopper kernels from flash-attention/hopper/
3. Installs flash_attn_3 module in site-packages

Usage:
    RAY_ADDRESS=auto python scripts/glm47_h200/install_fa3_hopper.py
"""
import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy


BUILD_SCRIPT = r"""
import subprocess, os, sys, shutil

VENV_PIP = '/home/ray/SkyRL/.venv/bin/pip'
VENV_PYTHON = '/home/ray/SkyRL/.venv/bin/python'
CUDA_HOME = '/usr/local/cuda'
REPO_DIR = '/home/ray/flash-attention'
HOPPER_DIR = os.path.join(REPO_DIR, 'hopper')

env = {**os.environ, 'CUDA_HOME': CUDA_HOME, 'MAX_JOBS': '8'}

def run(cmd, **kwargs):
    print(f'>>> {" ".join(cmd) if isinstance(cmd, list) else cmd}', flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, **kwargs)
    if r.stdout:
        # Print last part of stdout
        lines = r.stdout.strip().split('\n')
        for line in lines[-20:]:
            print(line, flush=True)
    return r

# Step 1: Clone
print('=== Step 1: Clone flash-attention ===', flush=True)
if not os.path.exists(REPO_DIR):
    r = run(['git', 'clone', '--depth=1', 'https://github.com/Dao-AILab/flash-attention.git', REPO_DIR],
            timeout=120)
    if r.returncode != 0:
        print(f'Clone failed: {r.stderr[-500:]}', flush=True)
        sys.exit(1)
else:
    print('Already cloned', flush=True)

# Step 2: Check hopper dir
if not os.path.exists(HOPPER_DIR):
    print(f'ERROR: {HOPPER_DIR} not found', flush=True)
    r = run(['ls', '-la', REPO_DIR])
    sys.exit(1)
print(f'Hopper dir contents: {os.listdir(HOPPER_DIR)[:15]}', flush=True)

# Step 3: Ensure ninja is installed
print('\n=== Step 3: Build deps ===', flush=True)
r = run([VENV_PYTHON, '-c', 'import ninja; print(f"ninja={ninja.__version__}")'])
if r.returncode != 0:
    print('Installing ninja...', flush=True)
    run([VENV_PIP, 'install', 'ninja'], timeout=60)

# Step 4: Build
print('\n=== Step 4: Build FA3 Hopper kernels ===', flush=True)
print('This may take 10-20 minutes...', flush=True)
r = run([VENV_PYTHON, 'setup.py', 'install'], cwd=HOPPER_DIR, timeout=1800)
if r.returncode != 0:
    print(f'\nBuild FAILED (rc={r.returncode})', flush=True)
    print('STDERR:', flush=True)
    for line in r.stderr.strip().split('\n')[-30:]:
        print(line, flush=True)
    sys.exit(1)
print('\nBuild succeeded!', flush=True)

# Step 5: Install flash_attn_3 module
print('\n=== Step 5: Install flash_attn_3 module ===', flush=True)
r = run([VENV_PYTHON, '-c', 'import site; print(site.getsitepackages()[0])'])
site_packages = r.stdout.strip().split('\n')[-1]

fa3_dir = os.path.join(site_packages, 'flash_attn_3')
os.makedirs(fa3_dir, exist_ok=True)

src = os.path.join(HOPPER_DIR, 'flash_attn_interface.py')
dst = os.path.join(fa3_dir, 'flash_attn_interface.py')
if os.path.exists(src):
    shutil.copy2(src, dst)
    with open(os.path.join(fa3_dir, '__init__.py'), 'w') as f:
        f.write('from .flash_attn_interface import *\n')
    print(f'Installed: {dst}', flush=True)
else:
    print(f'WARNING: {src} not found, listing built files...', flush=True)
    for root, dirs, files in os.walk(HOPPER_DIR):
        for fn in files:
            if fn.endswith('.py') and 'interface' in fn:
                print(f'  Found: {os.path.join(root, fn)}', flush=True)

# Step 6: Verify
print('\n=== Step 6: Verify ===', flush=True)
r = run([VENV_PYTHON, '-c', 'from flash_attn_3.flash_attn_interface import flash_attn_func; print("SUCCESS: flash_attn_3 imported!")'])
if 'SUCCESS' in r.stdout:
    print('\nFA3 Hopper kernels installed successfully!', flush=True)
else:
    print(f'\nVerification failed: {r.stdout} {r.stderr[-300:]}', flush=True)
    sys.exit(1)
"""


@ray.remote(num_cpus=2)
def build_on_node(node_ip):
    import subprocess, os, tempfile

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(BUILD_SCRIPT)
        script_path = f.name

    env = {**os.environ, 'CUDA_HOME': '/usr/local/cuda'}
    env.pop('RAY_ADDRESS', None)

    print(f'[{os.uname().nodename}] Starting FA3 Hopper build...', flush=True)
    proc = subprocess.run(
        ['/home/ray/SkyRL/.venv/bin/python', script_path],
        capture_output=True, text=True, env=env,
        timeout=1800)

    os.unlink(script_path)
    output = proc.stdout + proc.stderr
    print(output[-5000:] if len(output) > 5000 else output, flush=True)

    return proc.returncode == 0


def main():
    ray.init(address="auto")

    gpu_nodes = [n for n in ray.nodes() if n['Alive'] and n['Resources'].get('GPU', 0) > 0]
    print(f"Building FA3 Hopper kernels on {len(gpu_nodes)} GPU nodes")

    # Build on first node first to catch errors
    node0 = gpu_nodes[0]
    sched0 = NodeAffinitySchedulingStrategy(node_id=node0['NodeID'], soft=False)
    print(f"\n--- Building on node 0: {node0['NodeManagerAddress']} ---")

    result = ray.get(
        build_on_node.options(scheduling_strategy=sched0).remote(node0['NodeManagerAddress']),
        timeout=2400
    )

    if not result:
        print("Build FAILED on first node. Aborting.")
        return

    print(f"\nBuild succeeded on node 0! Building on remaining {len(gpu_nodes)-1} nodes in parallel...")

    # Build on remaining nodes in parallel
    refs = []
    for node in gpu_nodes[1:]:
        sched = NodeAffinitySchedulingStrategy(node_id=node['NodeID'], soft=False)
        refs.append(
            build_on_node.options(scheduling_strategy=sched).remote(node['NodeManagerAddress'])
        )

    if refs:
        results = ray.get(refs, timeout=2400)
        for i, (node, ok) in enumerate(zip(gpu_nodes[1:], results)):
            status = "OK" if ok else "FAILED"
            print(f"  Node {node['NodeManagerAddress']}: {status}")

    print("\nDone!")


if __name__ == "__main__":
    main()
