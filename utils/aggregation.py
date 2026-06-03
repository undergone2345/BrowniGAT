import pandas as pd


def aggregate_rankings(result_frames, top_k=20):
    merged = None
    for run_idx, frame in enumerate(result_frames, start=1):
        ranked = frame.copy()
        ranked["Rank"] = ranked.index + 1
        ranked = ranked.rename(
            columns={
                "CompositeScore": f"CompositeScore_run{run_idx}",
                "CoreScore": f"CoreScore_run{run_idx}",
                "Rank": f"Rank_run{run_idx}",
            }
        )

        selected_columns = [
            "Protein",
            "KnownTargetHit",
            "Degree",
            "WeightedDegree",
            "PageRank",
            "Betweenness",
            "Closeness",
            "Eigenvector",
            "MeanTargetSimilarity",
            "MaxTargetSimilarity",
            f"CompositeScore_run{run_idx}",
            f"CoreScore_run{run_idx}",
            f"Rank_run{run_idx}",
        ]
        ranked = ranked[selected_columns]
        if merged is None:
            merged = ranked
        else:
            merged = merged.merge(
                ranked[
                    [
                        "Protein",
                        f"CompositeScore_run{run_idx}",
                        f"CoreScore_run{run_idx}",
                        f"Rank_run{run_idx}",
                    ]
                ],
                on="Protein",
                how="outer",
            )

    score_columns = [col for col in merged.columns if col.startswith("CompositeScore_run")]
    rank_columns = [col for col in merged.columns if col.startswith("Rank_run")]

    merged["CompositeScoreMean"] = merged[score_columns].mean(axis=1)
    merged["CompositeScoreStd"] = merged[score_columns].std(axis=1).fillna(0.0)
    merged["RankMean"] = merged[rank_columns].mean(axis=1)
    merged["TopKHitCount"] = (merged[rank_columns] <= top_k).sum(axis=1)

    return merged.sort_values(
        ["TopKHitCount", "CompositeScoreMean", "RankMean"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def summarize_aggregate_methods(method_rankings):
    frames = []
    for method_name, ranking_df in method_rankings.items():
        top_row = ranking_df.iloc[0].to_dict()
        frames.append(
            {
                "Method": method_name,
                "TopProtein": top_row["Protein"],
                "TopScoreMean": top_row.get("CompositeScoreMean", top_row.get("CompositeScore", 0.0)),
                "TopKHitCount": top_row.get("TopKHitCount", 0),
                "RankMean": top_row.get("RankMean", 0.0),
            }
        )
    return pd.DataFrame(frames).sort_values(
        ["TopKHitCount", "TopScoreMean"],
        ascending=[False, False],
    ).reset_index(drop=True)
