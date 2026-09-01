async def replace_lora_adapter(models, lora_name, lora_path, lora_request_type):
    """Load an adapter under a fresh id, replacing the prior registration."""
    async with models.lora_resolver_lock[lora_name]:
        old_request = models.lora_requests.get(lora_name)
        if old_request is not None:
            await models.engine_client.remove_lora(old_request.lora_int_id)
            del models.lora_requests[lora_name]

        lora_int_id = models.lora_id_counter.inc(1)
        lora_request = lora_request_type(
            lora_name=lora_name,
            lora_int_id=lora_int_id,
            lora_path=lora_path,
        )
        await models.engine_client.add_lora(lora_request)
        models.lora_requests[lora_name] = lora_request
        return lora_int_id
