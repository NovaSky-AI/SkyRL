from .base import VMManager, Provider


def create_vm_manager_and_provider(provider_name: str, region: str, use_proxy: bool = False, env_id: int = 0):
    """
    Factory function to get the Virtual Machine Manager and Provider instances based on the provided provider name.
    
    Args:
        provider_name (str): The name of the provider (e.g., "aws", "vmware", etc.)
        region (str): The region for the provider
        use_proxy (bool): Whether to use proxy-enabled providers (currently only supported for AWS)
        env_id (int): Environment ID for deterministic port allocation (used by Docker provider)
    """
    provider_name = provider_name.lower().strip()
    if provider_name == "vmware":
        from .vmware.manager import VMwareVMManager
        from .vmware.provider import VMwareProvider
        return VMwareVMManager(), VMwareProvider(region)
    elif provider_name == "virtualbox":
        from .virtualbox.manager import VirtualBoxVMManager
        from .virtualbox.provider import VirtualBoxProvider
        return VirtualBoxVMManager(), VirtualBoxProvider(region)
    elif provider_name in ["aws", "amazon web services"]:
        from .aws.manager import AWSVMManager
        from .aws.provider import AWSProvider
        return AWSVMManager(), AWSProvider(region)
    elif provider_name == "azure":
        from .azure.manager import AzureVMManager
        from .azure.provider import AzureProvider
        return AzureVMManager(), AzureProvider(region)
    elif provider_name == "docker":
        from .docker.manager import DockerVMManager
        from .docker.provider import DockerProvider
        return DockerVMManager(), DockerProvider(region, env_id)
    else:
        raise NotImplementedError(f"{provider_name} not implemented!")
