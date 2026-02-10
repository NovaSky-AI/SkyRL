from skyrl_agent.tasks.osworld.desktop_env.desktop_env import DesktopEnv
import ray
import socket
import psutil

@ray.remote(num_cpus=1, num_gpus=0)
class DesktopEnvRay:
    def __init__(self, *args, **kwargs):
        hostname = socket.gethostname()
        print(f"DesktopEnvRay actor started on {hostname}")
        
        # Increase Docker port allocation timeout for Ray distributed execution
        import os
        os.environ['DESKTOP_ENV_LOCK_TIMEOUT'] = '120'
        
        # This needs to be the path to the VM image on the worker/CPU node
        kwargs['path_to_vm'] = "/home/ubuntu/shuo-vir/OSWorld_llm_agentsynth/docker_vm_data/Ubuntu.qcow2"
        
        self.desktop_env = DesktopEnv(*args, **kwargs)

    # Explicitly define the methods that Ray needs to expose
    async def _start_emulator_async(self):
        return await self.desktop_env._start_emulator_async()
    
    def _start_emulator(self):
        return self.desktop_env._start_emulator()
    
    def step(self, action, pause):
        return self.desktop_env.step(action, pause=pause)
    
    async def step_async(self, action, pause):
        return await self.desktop_env.step_async(action, pause=pause)
    
    async def reset(self, *args, **kwargs):
        return await self.desktop_env.reset(*args, **kwargs)
    
    def _get_obs(self):
        return self.desktop_env._get_obs()
    
    async def _get_obs_async(self):
        return await self.desktop_env._get_obs_async()
    
    async def evaluate(self, *args, **kwargs):
        return await self.desktop_env.evaluate(*args, **kwargs)
    
    def close(self):
        return self.desktop_env.close()
    
    
class DesktopEnvInterface:
    def __init__(self, desktop_env, cpu_node: bool = False):
        self.desktop_env = desktop_env
        self.cpu_node = cpu_node
    
    async def _start_emulator_async(self):
        """Start emulator async - call .remote() if using CPU, otherwise call normally"""
        if self.cpu_node:
            # For Ray actors, we need to get the result of the remote call
            result = self.desktop_env._start_emulator_async.remote()
            return await result
        else:
            return await self.desktop_env._start_emulator_async()
    
    def _start_emulator(self):
        """Start emulator - call .remote() if using CPU, otherwise call normally"""
        if self.cpu_node:
            return ray.get(self.desktop_env._start_emulator.remote())
        else:
            return self.desktop_env._start_emulator()
    
    def step(self, action, pause):
        """Step action - call .remote() if using CPU, otherwise call normally"""
        if self.cpu_node:
            return ray.get(self.desktop_env.step.remote(action, pause))
        else:
            return self.desktop_env.step(action, pause=pause)

    async def step_async(self, action, pause):
        """Step action async - call .remote() if using CPU, otherwise call normally"""
        if self.cpu_node:
            result = self.desktop_env.step_async.remote(action, pause=pause)
            return await result
        else:
            return await self.desktop_env.step_async(action, pause=pause)
    
    async def reset(self, *args, **kwargs):
        """Reset environment - call .remote() if using CPU, otherwise call normally"""
        if self.cpu_node:
            result = self.desktop_env.reset.remote(*args, **kwargs)
            return await result
        else:
            return await self.desktop_env.reset(*args, **kwargs)
    
    def _get_obs(self):
        """Get observation - call .remote() if using CPU, otherwise call normally"""
        if self.cpu_node:
            return ray.get(self.desktop_env._get_obs.remote())
        else:
            return self.desktop_env._get_obs()
    
    async def _get_obs_async(self):
        """Get observation async - call .remote() if using CPU, otherwise call normally"""
        if self.cpu_node:
            result = self.desktop_env._get_obs_async.remote()
            return await result
        else:
            return await self.desktop_env._get_obs_async()
    
    async def evaluate(self, *args, **kwargs):
        """Evaluate - call .remote() if using CPU, otherwise call normally"""
        if self.cpu_node:
            result = self.desktop_env.evaluate.remote(*args, **kwargs)
            return await result
        else:
            return await self.desktop_env.evaluate(*args, **kwargs)
    
    def close(self):
        """Close the desktop environment - call .remote() if using CPU, otherwise call normally"""
        if self.cpu_node:
            return ray.get(self.desktop_env.close.remote())
        else:
            return self.desktop_env.close()
    