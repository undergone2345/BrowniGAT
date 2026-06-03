import pandas as pd

from utils.baselines import compute_centrality_baseline_targets
from utils.core_target import compute_core_targets


def evaluate_method_recovery(
    method_name,
    embeddings,
    label_encoder,
    graph_metrics_df,
    scoring_cfg,
    benchmark_cfg,
):
    target_genes = [
        gene for gene in scoring_cfg["target_genes"] if gene in set(label_encoder.classes_)
    ]
    records = []
    for held_out_gene in target_genes:
        anchor_genes = [gene for gene in target_genes if gene != held_out_gene]
        if not anchor_genes:
            continue

        if method_name == "centrality":
            ranked_df = compute_centrality_baseline_targets(
                graph_metrics_df=graph_metrics_df,
                target_genes=anchor_genes,
                scoring_cfg=scoring_cfg,
            )
        else:
            ranked_df = compute_core_targets(
                embeddings=embeddings,
                label_encoder=label_encoder,
                target_genes=anchor_genes,
                graph_metrics_df=graph_metrics_df,
                scoring_cfg=scoring_cfg,
            )

        held_out_rank = int(ranked_df.index[ranked_df["Protein"] == held_out_gene][0]) + 1
        records.append(
            {
                "Method": method_name,
                "HeldOutTarget": held_out_gene,
                "AnchorCount": len(anchor_genes),
                "Rank": held_out_rank,
                "ReciprocalRank": 1.0 / held_out_rank,
                "HitAt1": int(held_out_rank <= 1),
                "HitAt5": int(held_out_rank <= 5),
                "HitAt10": int(held_out_rank <= 10),
                "HitAtK": int(held_out_rank <= benchmark_cfg["top_k"]),
            }
        )
    return pd.DataFrame(records)


def summarize_benchmark_results(benchmark_frames):
    benchmark_df = pd.concat(benchmark_frames, ignore_index=True)
    summary_df = (
        benchmark_df.groupby("Method", as_index=False)
        .agg(
            MeanRank=("Rank", "mean"),
            MedianRank=("Rank", "median"),
            MRR=("ReciprocalRank", "mean"),
            HitAt1=("HitAt1", "mean"),
            HitAt5=("HitAt5", "mean"),
            HitAt10=("HitAt10", "mean"),
            HitAtK=("HitAtK", "mean"),
            EvaluatedTargets=("HeldOutTarget", "count"),
        )
        .sort_values(["MRR", "HitAt5", "MeanRank"], ascending=[False, False, True])
        .reset_index(drop=True)
    )
    return benchmark_df, summary_df
