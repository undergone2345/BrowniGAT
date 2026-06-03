from dataclasses import asdict, dataclass


@dataclass
class ModalitySpec:
    name: str
    description: str
    required_files: list
    key_columns: list
    entity_types: list


MODALITY_REGISTRY = {
    "drug_target": ModalitySpec(
        name="drug_target",
        description="Drug-target supervision edges for pharmacology-aware pretraining.",
        required_files=["drug_target.tsv"],
        key_columns=["drug", "target", "action", "score"],
        entity_types=["drug", "gene"],
    ),
    "disease_gene": ModalitySpec(
        name="disease_gene",
        description="Disease-gene association edges for phenotype grounding.",
        required_files=["disease_gene.tsv"],
        key_columns=["disease", "gene", "score", "direction"],
        entity_types=["disease", "gene"],
    ),
    "pathway": ModalitySpec(
        name="pathway",
        description="Pathway membership edges for mechanistic context.",
        required_files=["pathway.tsv"],
        key_columns=["pathway", "gene", "score"],
        entity_types=["pathway", "gene"],
    ),
    "spatial": ModalitySpec(
        name="spatial",
        description="Spatial and cell-type context for tissue-aware representation learning.",
        required_files=["spatial.tsv"],
        key_columns=["region", "cell_type", "gene", "region_score", "cell_type_score"],
        entity_types=["region", "cell_type", "gene"],
    ),
}


def get_modality_spec(modality_name):
    if modality_name not in MODALITY_REGISTRY:
        raise ValueError(f"Unknown modality: {modality_name}")
    return MODALITY_REGISTRY[modality_name]


def describe_modalities(modality_names):
    return [asdict(get_modality_spec(name)) for name in modality_names]
