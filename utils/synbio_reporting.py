from pathlib import Path


def _render_table(frame):
    try:
        return frame.to_markdown(index=False)
    except ImportError:
        return frame.to_string(index=False)


def save_synbio_markdown_report(summary, gene_program_df, pathway_df, construct_df, output_path):
    output_path = Path(output_path)
    lines = [
        "# BrowniGAT Synthetic Biology Design Report",
        "",
        f"- Top design target: `{summary['top_design_target']}`",
        f"- Top pathway program: `{summary['top_pathway_program']}`",
        f"- Top construct: `{summary['top_construct']}`",
        "",
        "## Gene Program Design",
        "",
        _render_table(gene_program_df),
        "",
        "## Pathway Rewiring",
        "",
        _render_table(pathway_df),
        "",
        "## Construct Blueprints",
        "",
        _render_table(construct_df),
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
