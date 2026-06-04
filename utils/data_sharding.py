def build_data_shard(items, world_size=1, global_rank=0):
    items = list(items)
    world_size = max(1, int(world_size))
    global_rank = max(0, int(global_rank))
    shard_items = [item for idx, item in enumerate(items) if idx % world_size == global_rank]
    return {
        "world_size": world_size,
        "global_rank": global_rank,
        "num_items_total": len(items),
        "num_items_local": len(shard_items),
        "items": shard_items,
    }


def shard_task_samples(samples_by_task, runtime_topology):
    sharded = {}
    shard_summary = {}
    for task_name, samples in samples_by_task.items():
        shard = build_data_shard(
            samples,
            world_size=runtime_topology.get("world_size", 1),
            global_rank=runtime_topology.get("global_rank", 0),
        )
        sharded[task_name] = shard["items"]
        shard_summary[task_name] = {
            "num_items_total": shard["num_items_total"],
            "num_items_local": shard["num_items_local"],
        }
    return sharded, shard_summary
