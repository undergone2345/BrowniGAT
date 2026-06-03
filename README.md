# BrowniGAT

BrowniGAT is a graph representation learning framework for discovering anti-browning core targets from protein-protein interaction networks. The project now supports configuration-driven training, graph topology baselines, repeated experiments, aggregated ranking, network auditing, richer result exports, and direct method comparison across multiple GNN backbones.

The repository now also includes a `vNext` multi-capability stack that moves beyond PPI-only ranking into heterogenous graph reasoning, perturbation forecasting, spatial context prioritization, drug repurposing, and causal target ranking.

## What This Repository Does

- Loads STRING-like PPI interaction tables and builds a graph.
- Learns node embeddings with a configurable GAT encoder.
- Optimizes embeddings with an unsupervised contrastive objective.
- Scores proteins by combining embedding similarity to known targets and network topology statistics.
- Runs repeated experiments across multiple random seeds.
- Aggregates candidate rankings for more stable discovery.
- Exports training history, metadata, markdown summaries, and embedding visualizations.

## Iterative Improvements Added

### Version 1: Pipeline Refactor

- Replaced the monolithic `main.py` workflow with a configuration-driven pipeline.
- Added modular utilities for configuration loading, seeding, training, and reporting.
- Upgraded data loading to support named columns, score thresholds, undirected graphs, and self loops.

### Version 2: Experiment Expansion

- Added graph metrics including degree, weighted degree, PageRank, betweenness, closeness, and eigenvector centrality.
- Introduced multi-run execution with configurable seed stride.
- Added aggregated ranking across repeated runs.

### Version 3: Research Usability

- Added network audit summaries such as density, component counts, clustering, and target coverage.
- Added per-target similarity columns for interpretability.
- Added evidence tags to explain why a protein ranks highly.
- Added richer markdown and JSON outputs for downstream reporting.

### Version 4: Baseline Benchmarking

- Added `GCN` and `GraphSAGE` encoder backbones alongside the original `GAT`.
- Added a `centrality` baseline that ranks proteins directly from topology statistics.
- Added leave-one-target-out recovery benchmarking for method comparison.
- Added per-method output directories and cross-method aggregate summaries.

### Version 5: vNext Multi-Capability Platform

- Added a heterogenous graph pipeline with genes, drugs, disease, pathways, cell types, and spatial regions.
- Added perturbation-aware forecasting based on intervention signatures.
- Added spatial and cell-type aware target scoring.
- Added mechanism-aware drug repurposing outputs.
- Added causal ranking with evidence paths and uncertainty penalties.

## Repository Layout

```text
BrowniGAT/
|-- config/
|   |-- config.yaml
|   `-- toy_config.yaml
|-- data/
|   |-- string_interactions_short.csv
|   |-- toy_string_interactions.tsv
|   `-- vnext_toy/
|-- model/
|   |-- gat_embed.py
|   |-- gcn_embed.py
|   `-- graphsage_embed.py
|-- tests/
|   |-- test_aggregation.py
|   |-- test_benchmark_plotting.py
|   |-- test_config_and_baselines.py
|   `-- test_vnext_pipeline.py
|-- utils/
|   |-- aggregation.py
|   |-- baselines.py
|   |-- benchmark.py
|   |-- benchmark_plotting.py
|   |-- config.py
|   |-- core_target.py
|   |-- data_loader.py
|   |-- graph_metrics.py
|   |-- loss.py
|   |-- model_factory.py
|   |-- network_audit.py
|   |-- reporting.py
|   |-- seed.py
|   |-- trainer.py
|   |-- vnext_config.py
|   |-- vnext_reporting.py
|   `-- visualize.py
|-- results/
|-- notebooks/
|   `-- demo_analysis.ipynb
|-- benchmark_plot.py
|-- main.py
|-- tasks/
|-- vnext_main.py
|-- README.md
`-- requirements.txt
```

## Installation

Install the core dependencies:

```bash
pip install -r requirements.txt
```

Recommended environment:

- Python 3.9+
- PyTorch 2.x
- PyTorch Geometric 2.x

## Input Data Format

The default dataset is a STRING-style interaction table under `data/string_interactions_short.csv`.

The current configuration expects these columns:

- `node1`
- `node2`
- `combined_score`

You can change the column mapping in `config/config.yaml`.

## Quick Start

Run the default experiment:

```bash
python main.py
```

Run with a custom config:

```bash
python main.py --config config/config.yaml
```

Override the device or output directory:

```bash
python main.py --device cpu --output-dir results/demo_run
```

Run the toy example:

```bash
python main.py --config config/toy_config.yaml
```

Regenerate benchmark plots from existing result tables:

```bash
python benchmark_plot.py --output-dir results/method_comparison/plots
```

Run the lightweight test suite:

```bash
python -m unittest discover -s tests
```

Run the vNext multi-capability toy pipeline:

```bash
python vnext_main.py --config config/vnext_toy.yaml
```

## Configuration Overview

Main configurable sections:

- `data`: file path, edge columns, score threshold, graph preprocessing, feature mode.
- `model`: hidden size, output size, attention heads, dropout.
- `training`: epochs, learning rate, weight decay, negative sampling ratio.
- `scoring`: known targets and composite score weights.
- `benchmark`: enabled methods and leave-one-target-out evaluation settings.
- `visualization`: projection method and t-SNE perplexity.
- `runtime`: random seed, repeat count, seed stride, output directory.

Default repeated run behavior:

- `repeats: 3`
- `seed_stride: 17`
- `methods: [gat, gcn, graphsage, centrality]`

This means the project now performs several iterations automatically, trains multiple methods, and aggregates the rankings separately for each method.

## Outputs

Each run is saved under a subdirectory like `results/run_01/`. Inside that run, each method gets its own folder such as `results/run_01/gat/` or `results/run_01/centrality/`.

Each method folder includes:

- `core_targets.tsv`
- `training_history.csv`
- `run_metadata.json`
- `summary.md`
- `embedding_projection.png`
- `benchmark_recovery.tsv`

The root comparison directory also includes:

- `method_comparison/benchmark_details.tsv`
- `method_comparison/benchmark_summary.tsv`
- `method_comparison/benchmark_summary.md`
- `method_comparison/aggregate_overview.tsv`
- `method_comparison/<method>/aggregate_rankings.tsv`
- `method_comparison/plots/benchmark_summary.png`
- `method_comparison/plots/aggregate_overview.png`

The `vNext` pipeline writes a separate result bundle under `results_vnext/`:

- `target_prioritization.tsv`
- `perturbation_forecast.tsv`
- `spatial_targeting.tsv`
- `drug_repurposing.tsv`
- `causal_ranking.tsv`
- `vnext_summary.json`
- `REPORT.md`
- `plots/*.png`

## vNext Capability Map

The new `vNext` stack fills the five big capability gaps that a PPI-only system cannot cover:

1. Heterogenous biomedical graph reasoning
   Genes, pathways, drugs, disease states, cell types, and spatial regions now live in one graph.

2. Perturbation forecasting
   The pipeline estimates which interventions are most likely to reverse disease programs.

3. Spatial and cell-type specificity
   Targets are scored in disease-relevant microenvironments rather than only on a global graph.

4. Drug repurposing
   The system proposes compounds linked to prioritized targets and disease reversal support.

5. Causal ranking
   Final target ranking integrates stability, intervention evidence, context specificity, druggability, and uncertainty.

## Baseline Comparison

The project now compares four methods by default:

- `gat`: attention-based message passing.
- `gcn`: classic graph convolution baseline.
- `graphsage`: neighborhood aggregation baseline.
- `centrality`: non-neural topology baseline using degree and centrality signals only.

To compare them fairly, BrowniGAT now performs leave-one-target-out recovery:

1. Hold out one known anti-browning target.
2. Use the remaining known targets as anchors.
3. Rank all proteins.
4. Measure where the held-out target reappears.

The benchmark summary reports:

- `MeanRank`
- `MedianRank`
- `MRR`
- `HitAt1`
- `HitAt5`
- `HitAt10`
- `HitAtK`

## Ranking Logic

The ranking combines two evidence families:

1. Representation evidence
   Similarity between each protein embedding and the embeddings of known anti-browning targets.

2. Network evidence
   Degree and centrality statistics that capture the structural importance of a protein in the PPI graph.

The final `CompositeScore` is a weighted combination of normalized metrics from `config/config.yaml`.

## Interpretability Features

Each ranked protein now includes:

- `MeanTargetSimilarity`
- `MaxTargetSimilarity`
- `SimilarityTo_<TARGET>`
- `EvidenceTag`

This makes it easier to understand whether a candidate is target-like, hub-like, or both.

## Toy Dataset And Testing

The repository now includes a compact toy interaction network in `data/toy_string_interactions.tsv`.

This toy dataset is designed for:

- quick sanity checks on ranking logic
- small benchmark smoke runs
- local experiments before switching to full STRING-scale data

The `tests/` directory focuses on maintenance-friendly checks that do not require a full GNN runtime:

- ranking aggregation behavior
- centrality baseline scoring
- benchmark summary aggregation
- benchmark plotting output generation
- toy config loading
- vNext heterogenous graph task flow

## Suggested Next Extensions

- Add support for edge-aware objectives using STRING confidence weights directly in loss design.
- Add GO or KEGG enrichment analysis for the top ranked candidates.
- Add more baselines such as `GIN`, `APPNP`, `Node2Vec`, or diffusion-style ranking.
- Add unit tests and a synthetic toy dataset for CI smoke testing.

## Citation

If this repository helps your work, cite it as:

```text
BrowniGAT: Graph Attention based discovery of anti-browning targets from PPI networks.
```
