class BaseTaskHead:
    def __init__(self, task_spec):
        self.task_spec = task_spec

    def forward(self, batch):
        raise NotImplementedError


class MaskedNodeModelingHead(BaseTaskHead):
    def forward(self, batch):
        node_diversity = len({item["node_type"] for item in batch})
        loss = max(0.05, 1.0 / (1 + len(batch) + node_diversity))
        return {"loss": loss, "accuracy_proxy": min(1.0, 0.3 + 0.05 * node_diversity)}


class MaskedEdgeModelingHead(BaseTaskHead):
    def forward(self, batch):
        mean_weight = sum(float(item["weight"]) for item in batch) / max(len(batch), 1)
        loss = max(0.04, 0.8 - 0.5 * mean_weight)
        return {"loss": loss, "weight_recovery_proxy": mean_weight}


class CrossModalAlignmentHead(BaseTaskHead):
    def forward(self, batch):
        modalities = len({item["modality"] for item in batch})
        loss = max(0.03, 0.7 - 0.08 * modalities)
        return {"loss": loss, "alignment_proxy": min(1.0, 0.2 + 0.15 * modalities)}


class PerturbationConditioningHead(BaseTaskHead):
    def forward(self, batch):
        avg_support = sum(float(item["disease_support"]) for item in batch) / max(len(batch), 1)
        loss = max(0.03, 0.75 - 0.4 * avg_support)
        return {"loss": loss, "condition_response_proxy": avg_support}


class SpatialContextPredictionHead(BaseTaskHead):
    def forward(self, batch):
        avg_region = sum(float(item["region_score"]) for item in batch) / max(len(batch), 1)
        avg_cell = sum(float(item["cell_type_score"]) for item in batch) / max(len(batch), 1)
        confidence = (avg_region + avg_cell) / 2.0
        loss = max(0.03, 0.7 - 0.45 * confidence)
        return {"loss": loss, "context_accuracy_proxy": confidence}


def build_task_head(task_spec):
    name = task_spec["name"]
    if name == "masked_node_modeling":
        return MaskedNodeModelingHead(task_spec)
    if name == "masked_edge_modeling":
        return MaskedEdgeModelingHead(task_spec)
    if name == "cross_modal_alignment":
        return CrossModalAlignmentHead(task_spec)
    if name == "perturbation_conditioning":
        return PerturbationConditioningHead(task_spec)
    if name == "spatial_context_prediction":
        return SpatialContextPredictionHead(task_spec)
    raise ValueError(f"Unsupported task head: {name}")
