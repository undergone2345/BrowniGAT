import os


def build_runtime_topology(distributed_cfg):
    env_world_size = os.getenv("WORLD_SIZE")
    env_rank = os.getenv("RANK")
    env_local_rank = os.getenv("LOCAL_RANK")

    world_size = int(env_world_size or distributed_cfg.get("world_size", 1))
    global_rank = int(env_rank or distributed_cfg.get("global_rank", 0))
    local_rank = int(env_local_rank or distributed_cfg.get("local_rank", global_rank))
    backend = distributed_cfg.get("backend", "single_process")

    return {
        "world_size": world_size,
        "global_rank": global_rank,
        "local_rank": local_rank,
        "backend": backend,
        "gradient_sync": bool(distributed_cfg.get("gradient_sync", world_size > 1)),
        "is_distributed": bool(world_size > 1),
    }
