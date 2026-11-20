from skyrl_gym.tools.core import tool, ToolGroup
import subprocess


class PythonCodeExecutorToolGroup(ToolGroup):
    def __init__(self, timeout: float = 10.0):
        self.timeout = timeout
        super().__init__(name="PythonCodeExecutorToolGroup")

    @tool
    def python(self, code: str) -> str:
        try:
            result = subprocess.run(
                ["python", "-c", code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=self.timeout, text=True
            )
            if result.returncode != 0:
                error = result.stderr.strip() or result.stdout.strip() or f"exited with code {result.returncode}"
                return f"Error executing Python code: {error}"
            return result.stdout.strip() if result.stdout else ""
        except subprocess.TimeoutExpired:
            return f"Error executing Python code: Execution timed out after {self.timeout} seconds."
