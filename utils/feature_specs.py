from dataclasses import dataclass


@dataclass
class FeatureSpec:
    name: str
    feature_type: str
    source_column: str
    normalization: str
    vocabulary: str | None = None


FEATURE_SPECS = {
    "gene_token": FeatureSpec("gene_token", "categorical", "gene", "uppercase", "gene_vocab"),
    "drug_token": FeatureSpec("drug_token", "categorical", "drug", "identity", "drug_vocab"),
    "disease_token": FeatureSpec("disease_token", "categorical", "disease", "identity", "disease_vocab"),
    "pathway_token": FeatureSpec("pathway_token", "categorical", "pathway", "identity", "pathway_vocab"),
    "region_token": FeatureSpec("region_token", "categorical", "region", "identity", "region_vocab"),
    "cell_type_token": FeatureSpec("cell_type_token", "categorical", "cell_type", "identity", "celltype_vocab"),
    "edge_score": FeatureSpec("edge_score", "continuous", "score", "minmax"),
    "region_score": FeatureSpec("region_score", "continuous", "region_score", "minmax"),
    "cell_type_score": FeatureSpec("cell_type_score", "continuous", "cell_type_score", "minmax"),
}


def export_feature_specs():
    return [
        {
            "name": spec.name,
            "feature_type": spec.feature_type,
            "source_column": spec.source_column,
            "normalization": spec.normalization,
            "vocabulary": spec.vocabulary,
        }
        for spec in FEATURE_SPECS.values()
    ]
