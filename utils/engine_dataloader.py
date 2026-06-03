class SimpleBatchLoader:
    def __init__(self, batches):
        self._batches = list(batches)

    def __iter__(self):
        for batch in self._batches:
            yield batch

    def __len__(self):
        return len(self._batches)


def build_engine_dataloader(epoch_batches, dataloader_cfg):
    shuffle = bool(dataloader_cfg.get("shuffle", False))
    if shuffle:
        # Shuffling is handled at the epoch-batch planning stage to keep the loader deterministic.
        pass

    try:
        from torch.utils.data import DataLoader  # noqa: F401
    except ModuleNotFoundError:
        return SimpleBatchLoader(epoch_batches)

    return SimpleBatchLoader(epoch_batches)
