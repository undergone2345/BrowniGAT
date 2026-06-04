def build_resource_state(resource_cfg):
    return {
        "cpu_slots_total": int(resource_cfg.get("cpu_slots", 1)),
        "gpu_slots_total": int(resource_cfg.get("gpu_slots", 0)),
        "max_concurrent_runs": int(resource_cfg.get("max_concurrent_runs", 1)),
        "cpu_slots_available": int(resource_cfg.get("cpu_slots", 1)),
        "gpu_slots_available": int(resource_cfg.get("gpu_slots", 0)),
    }


def can_allocate_resources(resource_state, resource_request):
    return (
        int(resource_request.get("cpu_slots", 0)) <= int(resource_state["cpu_slots_available"])
        and int(resource_request.get("gpu_slots", 0)) <= int(resource_state["gpu_slots_available"])
    )


def allocate_resources(resource_state, resource_request):
    resource_state["cpu_slots_available"] -= int(resource_request.get("cpu_slots", 0))
    resource_state["gpu_slots_available"] -= int(resource_request.get("gpu_slots", 0))
    return {
        "cpu_slots": int(resource_request.get("cpu_slots", 0)),
        "gpu_slots": int(resource_request.get("gpu_slots", 0)),
    }


def release_resources(resource_state, allocation):
    resource_state["cpu_slots_available"] += int(allocation.get("cpu_slots", 0))
    resource_state["gpu_slots_available"] += int(allocation.get("gpu_slots", 0))


def prioritize_queue_items(queue_items):
    return sorted(
        queue_items,
        key=lambda item: (
            -int(item.get("priority", 0)),
            -int(item.get("partition_budget", 0)),
            item.get("queue_id", ""),
        ),
    )
