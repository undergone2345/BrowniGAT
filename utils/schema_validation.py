import pandas as pd


SCHEMA_DEFINITIONS = {
    "drug_target": {
        "required_columns": ["drug", "target", "action", "score", "source"],
        "non_null_columns": ["drug", "target", "score"],
        "score_columns": {"score": (0.0, 1.0)},
    },
    "disease_gene": {
        "required_columns": ["disease", "gene", "score", "direction", "source"],
        "non_null_columns": ["disease", "gene", "score"],
        "score_columns": {"score": (0.0, 1.0)},
    },
    "pathway_membership": {
        "required_columns": ["pathway", "gene", "score", "source"],
        "non_null_columns": ["pathway", "gene", "score"],
        "score_columns": {"score": (0.0, 1.0)},
    },
    "spatial_context": {
        "required_columns": ["region", "cell_type", "gene", "region_score", "cell_type_score", "source"],
        "non_null_columns": ["region", "cell_type", "gene", "region_score", "cell_type_score"],
        "score_columns": {
            "region_score": (0.0, 1.0),
            "cell_type_score": (0.0, 1.0),
        },
    },
}


def validate_schema(df, schema_name):
    schema = SCHEMA_DEFINITIONS[schema_name]
    errors = []
    warnings = []

    missing_columns = [column for column in schema["required_columns"] if column not in df.columns]
    if missing_columns:
        errors.append(f"Missing required columns: {missing_columns}")

    for column in schema.get("non_null_columns", []):
        if column in df.columns and df[column].isna().any():
            errors.append(f"Column '{column}' contains null values.")

    for column, (lower, upper) in schema.get("score_columns", {}).items():
        if column not in df.columns:
            continue
        non_numeric_mask = pd.to_numeric(df[column], errors="coerce").isna()
        if non_numeric_mask.any():
            errors.append(f"Column '{column}' contains non-numeric values.")
            continue
        numeric_values = pd.to_numeric(df[column], errors="coerce")
        outside_mask = (numeric_values < lower) | (numeric_values > upper)
        if outside_mask.any():
            errors.append(f"Column '{column}' has values outside [{lower}, {upper}].")

    duplicate_count = int(df.duplicated().sum())
    if duplicate_count > 0:
        warnings.append(f"Detected {duplicate_count} duplicated rows.")

    return {
        "schema": schema_name,
        "rows": int(len(df)),
        "errors": errors,
        "warnings": warnings,
        "is_valid": len(errors) == 0,
    }
