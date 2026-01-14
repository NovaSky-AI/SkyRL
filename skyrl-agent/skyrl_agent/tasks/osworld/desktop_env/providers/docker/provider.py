import logging
import os
import platform
import time
import asyncio
import docker
import psutil
import requests
import aiohttp
from filelock import FileLock
from pathlib import Path

from skyrl_agent.tasks.osworld.desktop_env.providers.base import Provider

logger = logging.getLogger("desktopenv.providers.docker.DockerProvider")
logger.setLevel(logging.INFO)

WAIT_TIME = 3
RETRY_INTERVAL = 1
LOCK_TIMEOUT = 120  # Increased timeout for concurrent Ray actors


class PortAllocationError(Exception):
    pass


class DockerProvider(Provider):
    # Class-level async lock to coordinate file lock access across instances
    _async_lock = None
    # Class-level set to track ports that are reserved but not yet bound
    _reserved_ports = set()
    
    @classmethod
    def _get_async_lock(cls):
        """Lazy initialization of async lock to avoid event loop issues."""
        try:
            # Check if we have a lock and if it's still valid for the current event loop
            if cls._async_lock is not None:
                # Try to access the lock's loop - this will raise RuntimeError if bound to dead loop
                cls._async_lock._get_loop()
                return cls._async_lock
        except RuntimeError:
            # Lock is bound to a dead event loop, create a new one
            cls._async_lock = None
        
        # Create new lock if we don't have one or the old one was invalid
        if cls._async_lock is None:
            cls._async_lock = asyncio.Lock()
        return cls._async_lock
    
    @classmethod
    def cleanup_all_reserved_ports(cls):
        """Clean up all reserved ports. Useful between training steps."""
        cls._reserved_ports.clear()
        logger.info("Cleared all reserved ports")
    
    def __init__(self, region: str, env_id: int):
        self.client = docker.from_env()
        self.server_port = None
        self.vnc_port = None
        self.chromium_port = None
        self.vlc_port = None
        self.container = None
        self.environment = {"DISK_SIZE": "32G", "RAM_SIZE": "4G", "CPU_CORES": "4"}  # Modify if needed
        self.env_id = env_id
        temp_dir = Path(os.getenv('TEMP') if platform.system() == 'Windows' else '/tmp')
        self.lock_file = temp_dir / "docker_port_allocation.lck"
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)

    def _get_deterministic_ports(self):
        """Get deterministic ports based on environment ID."""
        # Base ports for each service
        base_ports = {
            'vnc': 8006,
            'server': 5000, 
            'chromium': 9222,
            'vlc': 8081
        }
        
        # Calculate ports by adding env_id * 100 to base ports
        # This gives us ranges like: 8006, 8106, 8206, ... for VNC
        port_offset = self.env_id
        
        ports = {}
        for service, base_port in base_ports.items():
            ports[service] = base_port + port_offset
            
        logger.info(f"Environment {self.env_id} allocated ports: {ports}")
        return ports

    def _wait_for_vm_ready(self, timeout: int = 300):
        """Wait for VM to be ready by checking screenshot endpoint."""
        start_time = time.time()
        def check_screenshot():
            try:
                response = requests.get(
                    f"http://localhost:{self.server_port}/screenshot",
                    timeout=(10, 10)
                )
                return response.status_code == 200
            except Exception:
                return False

        while time.time() - start_time < timeout:
            if check_screenshot():
                return True
            logger.info("Checking if virtual machine is ready...")
            time.sleep(RETRY_INTERVAL)
        
        raise TimeoutError("VM failed to become ready within timeout period")

    async def _wait_for_vm_ready_async(self, timeout: int = 300):
        """Async version: Wait for VM to be ready by checking screenshot endpoint."""
        start_time = asyncio.get_event_loop().time()
        
        async def check_screenshot():
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"http://localhost:{self.server_port}/screenshot",
                        timeout=aiohttp.ClientTimeout(total=10)
                    ) as response:
                        return response.status == 200
            except Exception:
                return False

        while (asyncio.get_event_loop().time() - start_time) < timeout:
            if await check_screenshot():
                return True
            logger.info("Checking if virtual machine is ready...")
            await asyncio.sleep(RETRY_INTERVAL)
        
        raise TimeoutError("VM failed to become ready within timeout period")

    async def start_emulator_async(self, path_to_vm: str, headless: bool, os_type: str):
        """
        Async version of start_emulator with deterministic port allocation.
        No locks needed since ports are deterministically assigned based on env_id.
        """
        # Step 1: Allocate ports deterministically (no lock needed)
        ports = self._get_deterministic_ports()
        self.vnc_port = ports['vnc']
        self.server_port = ports['server']
        self.chromium_port = ports['chromium']
        self.vlc_port = ports['vlc']
        
        # Step 2: Start container
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._start_container_sync, path_to_vm, headless, os_type)
        
        # Step 3: Wait for VM to be ready
        await self._wait_for_vm_ready_async()

    def _allocate_ports_sync(self):
        """Allocate ports deterministically - no lock needed."""
        ports = self._get_deterministic_ports()
        self.vnc_port = ports['vnc']
        self.server_port = ports['server']
        self.chromium_port = ports['chromium']
        self.vlc_port = ports['vlc']

    def _start_container_sync(self, path_to_vm: str, headless: bool, os_type: str):
        """Start the Docker container - can run concurrently since ports are pre-allocated."""
        allocated_ports = [self.vnc_port, self.server_port, self.chromium_port, self.vlc_port]
        try:
            # Check if KVM is available
            devices = []
            if os.path.exists("/dev/kvm"):
                devices.append("/dev/kvm")
                logger.info("KVM device found, using hardware acceleration")
            else:
                self.environment["KVM"] = "N"
                logger.warning("KVM device not found, running without hardware acceleration (will be slower)")

            self.container = self.client.containers.run(
                "happysixd/osworld-docker",
                environment=self.environment,
                cap_add=["NET_ADMIN"],
                devices=devices,
                volumes={
                    os.path.abspath(path_to_vm): {
                        "bind": "/System.qcow2",
                        "mode": "ro"
                    }
                },
                ports={
                    8006: self.vnc_port,
                    5000: self.server_port,
                    9222: self.chromium_port,
                    8080: self.vlc_port
                },
                detach=True
            )

            logger.info(f"Started container with ports - VNC: {self.vnc_port}, "
                       f"Server: {self.server_port}, Chrome: {self.chromium_port}, VLC: {self.vlc_port}")

        except Exception as e:
            # Clean up if anything goes wrong
            if self.container:
                try:
                    self.container.stop()
                    self.container.remove()
                except:
                    pass
            raise e


    def start_emulator(self, path_to_vm: str, headless: bool, os_type: str):
        # Allocate ports deterministically (no lock needed)
        ports = self._get_deterministic_ports()
        self.vnc_port = ports['vnc']
        self.server_port = ports['server']
        self.chromium_port = ports['chromium']
        self.vlc_port = ports['vlc']
        
        try:
            # Check if KVM is available
            devices = []
            if os.path.exists("/dev/kvm"):
                devices.append("/dev/kvm")
                logger.info("KVM device found, using hardware acceleration")
            else:
                self.environment["KVM"] = "N"
                logger.warning("KVM device not found, running without hardware acceleration (will be slower)")

            self.container = self.client.containers.run(
                "happysixd/osworld-docker",
                environment=self.environment,
                cap_add=["NET_ADMIN"],
                devices=devices,
                volumes={
                    os.path.abspath(path_to_vm): {
                        "bind": "/System.qcow2",
                        "mode": "ro"
                    }
                },
                ports={
                    8006: self.vnc_port,
                    5000: self.server_port,
                    9222: self.chromium_port,
                    8080: self.vlc_port
                },
                detach=True
            )

            logger.info(f"Started container with ports - VNC: {self.vnc_port}, "
                       f"Server: {self.server_port}, Chrome: {self.chromium_port}, VLC: {self.vlc_port}")

            # Wait for VM to be ready
            self._wait_for_vm_ready()

        except Exception as e:
            # Clean up if anything goes wrong
            if self.container:
                try:
                    self.container.stop()
                    self.container.remove()
                except:
                    pass
            raise e

    def get_ip_address(self, path_to_vm: str) -> str:
        if not all([self.server_port, self.chromium_port, self.vnc_port, self.vlc_port]):
            raise RuntimeError("VM not started - ports not allocated")
        return f"localhost:{self.server_port}:{self.chromium_port}:{self.vnc_port}:{self.vlc_port}"

    def save_state(self, path_to_vm: str, snapshot_name: str):
        raise NotImplementedError("Snapshots not available for Docker provider")

    def revert_to_snapshot(self, path_to_vm: str, snapshot_name: str):
        self.stop_emulator(path_to_vm)

    def stop_emulator(self, path_to_vm: str, region=None, *args, **kwargs):
        # Note: region parameter is ignored for Docker provider
        # but kept for interface consistency with other providers
        if self.container:
            logger.info("Stopping VM...")
            # Store ports for cleanup before clearing them
            ports_to_cleanup = [self.vnc_port, self.server_port, self.chromium_port, self.vlc_port]
            try:
                self.container.stop()
                self.container.remove()
                time.sleep(WAIT_TIME)
            except Exception as e:
                logger.error(f"Error stopping container: {e}")
            finally:
                self.container = None
                self.server_port = None
                self.vnc_port = None
                self.chromium_port = None
                self.vlc_port = None
                
                # No need to clean up reserved ports since we use deterministic allocation
