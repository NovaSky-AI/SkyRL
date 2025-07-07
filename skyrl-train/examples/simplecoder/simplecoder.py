from dataclasses import dataclass
import subprocess
import tempfile
from typing import Optional, Dict, Any, List

from openai import OpenAI

@dataclass
class ExecutionResult:
    """Result of a command execution."""
    output: str
    error: Optional[str] = None
    return_code: int = 0


class GuixExecutor:
    """Guix-based executor that runs commands in a sandboxed Guix shell environment."""
    
    def __init__(self, working_dir: str, manifest_file: Optional[str] = None):
        """Initialize the Guix executor.
        
        Args:
            working_dir: Working directory of the execution
            manifest_file: Path to a Guix manifest file specifying packages
        """
        self.working_dir = working_dir
        self.manifest_file = manifest_file
        self.current_env = ""
    
    def execute(
        self, 
        command: str, 
        timeout: int = 30,
    ) -> ExecutionResult:
        """Execute a command in a sandboxed Guix shell."""

        guix_cmd = ["guix", "shell"]

        if self.manifest_file:
            guix_cmd.extend(["-m", self.manifest_file])

        # Add container options for sandboxing
        guix_cmd.extend([
            "--container",          # Run in container for isolation
            "--network",            # Allow network access
        ])

        with tempfile.NamedTemporaryFile(mode="w", suffix="_env.sh", delete=False) as env_file:
            env_file.write(self.current_env)

        container_env_path = "/tmp/env.sh"
        with tempfile.NamedTemporaryFile(mode="w", suffix="_script.sh", delete=False) as script_file:
            script_file.write(f"source {container_env_path}\n")
            script_file.write(f"{command}\n")
            script_file.write(f"export -p > {container_env_path}\n")
        
        container_script_path = "/tmp/script.sh"
        guix_cmd.extend([f"--share={script_file.name}={container_script_path}"])
        guix_cmd.extend([f"--share={env_file.name}={container_env_path}"])

        guix_cmd.extend(["--", "sh", container_script_path])

        try:
            result = subprocess.run(
                guix_cmd,
                shell=False,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.working_dir,
            )
            return ExecutionResult(
                output=result.stdout or "",
                error=result.stderr if result.stderr else None,
                return_code=result.returncode,
                timeout=False
            )
        except Exception as e:
            return ExecutionResult(
                output="",
                error=f"Execution failed: {str(e)}",
                return_code=-1,
            )


# executor = GuixExecutor()
# executor.execute("ls")

