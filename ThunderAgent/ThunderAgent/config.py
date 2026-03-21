"""ThunderAgent configuration."""
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    """ThunderAgent configuration (set via command line args)."""
    # Backend configuration
    backends: List[str] = field(default_factory=lambda: ["http://localhost:8000"])
    
    # Router mode: "default" (pure proxy) or "tr" (capacity scheduling)
    router_mode: str = "tr"

    # Backend type: "vllm", "sglang", or "skyrl"
    backend_type: str = "vllm"
    
    # Profile configuration
    profile_enabled: bool = False
    profile_dir: str = "/tmp/thunderagent_profiles"
    
    # Metrics monitoring configuration
    metrics_enabled: bool = False
    metrics_interval: float = 5.0  # seconds between metrics fetch
    
    # Scheduler configuration
    scheduler_interval: float = 5.0  # seconds between scheduler checks
    acting_token_weight: float = 1.0  # weight for acting tokens in capacity calculation
    use_acting_token_decay: bool = False  # use 2^(-t) decay for acting tokens in resume logic


# Global config instance (set by __main__.py before app starts)
_config: Config = Config()


def get_config() -> Config:
    """Get the global config instance."""
    return _config


def set_config(config: Config) -> None:
    """Set the global config instance."""
    global _config
    _config = config
