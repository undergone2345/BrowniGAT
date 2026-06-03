from dataclasses import asdict, dataclass


@dataclass
class PretrainingTaskSpec:
    name: str
    description: str
    required_modalities: list
    supervision_type: str
    prediction_target: str


TASK_REGISTRY = {
    "masked_node_modeling": PretrainingTaskSpec(
        name="masked_node_modeling",
        description="Recover masked biological entities from multimodal graph context.",
        required_modalities=["drug_target", "disease_gene", "pathway", "spatial"],
        supervision_type="self_supervised",
        prediction_target="node_identity",
    ),
    "masked_edge_modeling": PretrainingTaskSpec(
        name="masked_edge_modeling",
        description="Predict missing relations and edge strengths.",
        required_modalities=["drug_target", "disease_gene", "pathway", "spatial"],
        supervision_type="self_supervised",
        prediction_target="edge_type_and_weight",
    ),
    "cross_modal_alignment": PretrainingTaskSpec(
        name="cross_modal_alignment",
        description="Align gene-centric structure across pharmacology, disease, and spatial modalities.",
        required_modalities=["drug_target", "disease_gene", "spatial"],
        supervision_type="contrastive",
        prediction_target="shared_embedding_alignment",
    ),
    "perturbation_conditioning": PretrainingTaskSpec(
        name="perturbation_conditioning",
        description="Condition on intervention-like signals for response prediction tasks.",
        required_modalities=["drug_target", "disease_gene", "pathway"],
        supervision_type="conditional_prediction",
        prediction_target="molecular_response_profile",
    ),
    "spatial_context_prediction": PretrainingTaskSpec(
        name="spatial_context_prediction",
        description="Predict tissue region and cell-context labels from graph context.",
        required_modalities=["spatial", "disease_gene"],
        supervision_type="auxiliary_supervised",
        prediction_target="region_and_cell_context",
    ),
}


def get_task_spec(task_name):
    if task_name not in TASK_REGISTRY:
        raise ValueError(f"Unknown task: {task_name}")
    return TASK_REGISTRY[task_name]


def resolve_enabled_tasks(task_names, available_modalities):
    available_modalities = set(available_modalities)
    resolved = []
    for task_name in task_names:
        task_spec = get_task_spec(task_name)
        if set(task_spec.required_modalities).issubset(available_modalities):
            resolved.append(asdict(task_spec))
    return resolved
