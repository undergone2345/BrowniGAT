# BrowniGAT

BrowniGAT is an open biomedical AI infrastructure repository that has grown beyond a single PPI target-ranking pipeline.

Today the project spans four connected layers:

- a multimodal biomedical graph platform for ingesting disease-gene, drug-target, pathway, and spatial evidence
- a foundation training engine for workspace building, multimodal task orchestration, checkpointing, and validation
- a scheduler, recovery, and promotion control plane for more engine-like experiment execution
- a synthetic biology design layer that converts ranked targets into intervention programs and construct blueprints

The original anti-browning target-discovery workflow is still present, but it is now only one entry point into a broader systems stack for biomedical graph learning, intervention reasoning, and design-oriented outputs.

## Why BrowniGAT Matters

Many biomedical graph repositories stop at ranking targets.

BrowniGAT is trying to cover a wider loop:

1. normalize real multimodal biological inputs
2. build graph-native and foundation-model-ready training workspaces
3. orchestrate runs with scheduler and recovery semantics
4. generate intervention and synthetic biology design outputs

That broader framing is the repository's real open-source value proposition.

It also now includes a third-layer foundation workspace builder that converts canonical multimodal bundles into manifest-driven pretraining workspaces with modality registries, feature specs, task registries, and sampling plans.

The repository now also includes a fourth-layer pretraining trainer skeleton with multimodal batch construction, task heads, checkpoint schema, and epoch-level training orchestration.

It now also includes a fifth-layer training stack upgrade with configurable backbones, experiment manifests, optimizer and scheduler stubs, gradient accumulation, and richer resume-ready checkpoint metadata.

The repository now also starts to distinguish two different infrastructure layers explicitly:

- `research infra`: flexible, iteration-friendly experimentation utilities
- `engine`: resume-aware, registry-aware, training-runtime oriented infrastructure for scaling

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

### Version 6: Foundation Workspace Layer

- Added canonical-to-foundation workspace conversion for large-scale multimodal pretraining.
- Added modality registry, feature specification registry, and pretraining task registry.
- Added manifest generation and batch sampling plans for multimodal training orchestration.
- Added tests covering workspace building and bundle loading.

### Version 7: Foundation Trainer Skeleton

- Added multimodal pretraining dataset assembly from workspace manifests.
- Added task heads for masked node modeling, masked edge modeling, alignment, perturbation conditioning, and spatial context prediction.
- Added checkpoint schema and per-epoch checkpoint writing.
- Added a trainer skeleton that writes training history and summary metrics.

### Version 8: Foundation Training Stack Upgrade

- Added configurable backbone interfaces for future multimodal encoders.
- Added experiment manifests for reproducible training runs.
- Added optimizer and scheduler stubs to mirror real pretraining systems.
- Added gradient accumulation and richer checkpoint metadata for resume-ready workflows.

### Version 9: Research Infra vs Engine Split

- Added explicit experiment modes for `research` and `engine`.
- Added loss composition, runtime hooks, and run registry for engine-style orchestration.
- Added engine-oriented foundation config with AMP, grad clipping, and richer runtime metadata.

### Version 10: Training Lifecycle Management

- Added engine dataloader abstraction, validation batches, and best-checkpoint selection.
- Added resume-from-checkpoint flow that restores step metadata and appends training history.
- Added runtime logging controls and validation summaries for each epoch.

### Version 11: Artifactized Engine Runtime

- Added task sampling sequence planning so engine runs can control repeat policy independently from dataset layout.
- Added early stopping state tracking with best-epoch metadata persisted into checkpoints.
- Added artifact index generation so downstream tooling can discover summaries, manifests, and checkpoint outputs reliably.
- Added richer training summaries with actual completed epochs, dataloader settings, and final checkpoint resolution.

### Version 12: Scalable Engine Skeleton

- Added runtime topology metadata so single-process and future distributed launches share one schema.
- Added curriculum-phase scheduling to separate graph warmup from later multimodal training phases.
- Added event-log emission for epoch lifecycle, checkpoint saves, and early-stop triggers.
- Added checkpoint catalog indexing so downstream orchestrators can enumerate engine artifacts without scanning the filesystem.

### Version 13: Data Plane For Large-Scale Engine

- Added manifest partition metadata so large canonical bundles can be described as shardable node, edge, and modality slices.
- Added runtime data sharding for task samples, exposing per-task local versus global sample counts.
- Added worker-aware sampling sequences so dataloader workers can consume disjoint task-step schedules.
- Added stage-wise checkpoint retention so engine runs can keep the last checkpoint per curriculum phase plus best checkpoints.

### Version 14: Engine Control Plane

- Added a manifest-aware stage planner that turns curriculum phases and partition summaries into schedulable stages.
- Added a run queue schema with per-stage workspaces, retry budgets, and queue dependencies.
- Added failure recovery policies that decide whether a failed stage should retry and which checkpoint to resume from.
- Added an experiment scheduler entrypoint that writes scheduler manifests, queue files, event logs, and stage summaries.

### Version 15: Resource-Aware Engine And SynBio Design Layer

- Added resource-aware scheduling metadata including CPU slot, GPU slot, concurrency, and stage priority hints.
- Added metric-based promotion policy gates that can halt downstream stages when validation objectives are not met.
- Added a synthetic biology design layer that converts ranked targets into gene programs, pathway rewiring plans, and construct blueprints.
- Added a dedicated `synbio_main.py` entrypoint and tests for synthetic biology design outputs.

## Repository Layout

```text
BrowniGAT/
|-- .github/
|-- config/
|   |-- config.yaml
|   |-- foundation_example.yaml
|   |-- real_data_example.yaml
|   `-- toy_config.yaml
|-- data/
|   |-- real_templates/
|   |-- string_interactions_short.csv
|   |-- toy_string_interactions.tsv
|   `-- vnext_toy/
|-- docs/
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
|-- CHANGELOG.md
|-- CONTRIBUTING.md
|-- CITATION.cff
|-- foundation_main.py
|-- foundation_schedule.py
|-- foundation_train.py
|-- config/foundation_engine_example.yaml
|-- ingest_multimodal_data.py
|-- LICENSE
|-- main.py
|-- ROADMAP.md
|-- synbio_main.py
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

Project metadata and maintenance files:

- contribution guide: `CONTRIBUTING.md`
- roadmap: `ROADMAP.md`
- release history: `CHANGELOG.md`
- citation metadata: `CITATION.cff`
- usage and workflow docs: `docs/`

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

Build a canonical real-data bundle from standardized modality tables:

```bash
python ingest_multimodal_data.py --config config/real_data_example.yaml
```

Build a foundation-model-ready workspace from a canonical bundle:

```bash
python foundation_main.py --config config/foundation_example.yaml
```

Run the foundation trainer skeleton on the workspace:

```bash
python foundation_train.py --config config/foundation_example.yaml
```

Run the more engine-oriented training stack:

```bash
python foundation_train.py --config config/foundation_engine_example.yaml
```

Run the experiment scheduler that materializes a stage plan, run queue, and recovery-aware execution flow:

```bash
python foundation_schedule.py --config config/foundation_engine_example.yaml --workspace-dir results/foundation_scheduler
```

Run the synthetic biology design layer on top of the multimodal toy dataset:

```bash
python synbio_main.py --config config/synbio_toy.yaml --output-dir results_synbio/demo
```

Supporting documents:

- application framing: `docs/APPLICATIONS.md`
- demo command map: `docs/DEMO_WORKFLOWS.md`

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

The synthetic biology design layer writes a separate result bundle under `results_synbio/`:

- `gene_program_design.tsv`
- `pathway_rewiring.tsv`
- `construct_blueprints.tsv`
- `synbio_summary.json`
- `SYNBIO_REPORT.md`

The real-data ingestion pipeline writes a canonical bundle under `results_real_ingestion/`:

- `canonical_nodes.tsv`
- `canonical_edges.tsv`
- `modalities/*.tsv`
- `foundation_manifest.json`
- `ingestion_summary.json`

The foundation workspace builder writes a pretraining-oriented workspace under `results_foundation_workspace/`:

- `pretraining_manifest.json`
- `workspace_summary.json`

The workspace summary includes:

- enabled modalities
- enabled pretraining tasks
- vocabulary sizes
- per-task batch sizes
- a balanced sampling plan for each epoch

The foundation trainer skeleton extends that workspace with:

- `training_history.csv`
- `training_summary.json`
- `checkpoints/epoch_*.json`
- `experiment_manifest.json`
- `run_registry.jsonl`
- `checkpoints/best.json`

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

## Real Data Ingestion Layer

To move toward a real large-scale foundation-model system, the repository now includes a standardized ingestion layer for multimodal biological data.

Supported modality templates:

- `drug-target`
- `disease-gene`
- `pathway-membership`
- `spatial-context`

This ingestion layer adds:

- column mapping from source-specific headers into canonical headers
- schema validation for required fields and score ranges
- entity normalization such as whitespace trimming and uppercase gene symbols
- harmonized node and edge export for downstream heterogenous graph construction
- a foundation-ready manifest describing the canonical bundle

The main example config is `config/real_data_example.yaml`, and template tables live in `data/real_templates/`.

## Foundation Workspace Layer

The third layer of the repository is designed to bridge canonical biological bundles into large-scale foundation-model training infrastructure.

This layer adds:

- a modality registry for pharmacology, disease, pathway, and spatial inputs
- a feature spec registry describing categorical and continuous model features
- a pretraining task registry for multimodal self-supervision and alignment
- a sampling-plan builder that allocates epoch steps across tasks
- a manifest builder that records vocabularies, tasks, and dataset statistics

Core files:

- `foundation_main.py`
- `foundation_train.py`
- `utils/modality_registry.py`
- `utils/feature_specs.py`
- `utils/task_registry.py`
- `utils/pretraining_manifest.py`
- `utils/batch_sampler.py`
- `utils/foundation_workspace.py`

## Foundation Trainer Layer

The fourth layer is a trainer skeleton that can later be upgraded into a full neural foundation model trainer without rewriting the orchestration stack.

This layer adds:

- workspace-backed multimodal dataset loading
- task-specific heads with a shared interface
- batch collation from the sampling plan
- checkpoint payload schemas for reproducible resumes and audits
- training history export for future experiment tracking

Core files:

- `utils/foundation_dataset.py`
- `model/foundation_backbone.py`
- `model/foundation_task_heads.py`
- `utils/checkpoint_schema.py`
- `utils/experiment_registry.py`
- `utils/foundation_trainer.py`
- `utils/training_components.py`

The current stack is still a trainer skeleton rather than a full large-scale neural training system, but it now mirrors the shape of one much more closely:

- backbone configuration
- task-head dispatch
- dataset and collation layer
- optimizer and scheduler state stubs
- checkpoint payloads with resume metadata
- experiment-level manifests

## Research Infra vs Engine

The repository now explicitly separates two operational layers:

### Research Infra

Use this when the goal is to iterate quickly on ideas, tasks, scoring heads, and multimodal schema decisions.

Current examples:

- `vnext_main.py`
- toy multimodal datasets
- heterogenous graph task modules
- lightweight task-head prototypes

### Engine

Use this when the goal is to make training workflows more reproducible, resumable, and eventually scalable.

Current examples:

- `foundation_main.py`
- `foundation_train.py`
- `utils/engine_runtime.py`
- `utils/engine_dataloader.py`
- `utils/loss_composer.py`
- `utils/run_registry.py`
- `utils/validation_registry.py`
- resume-ready checkpoint payloads
- experiment manifests and runtime summaries

The engine layer is still not a production distributed trainer, but it now has the right abstractions for that next jump.

It now also includes:

- validation-time batch construction
- best-checkpoint persistence
- resume-aware training loops
- lifecycle-oriented runtime logging
- sampling-plan driven task sequencing
- early-stopping policies with persisted state
- artifact index generation for training outputs
- curriculum-driven task activation
- runtime topology metadata for future distributed expansion
- event logs and checkpoint catalog files for engine auditing
- manifest partitioning for bundle-scale planning
- data sharding summaries for rank-local task views
- worker-aware sampler sequencing
- stage-wise checkpoint retention policies
- manifest-aware stage planning and queue orchestration
- failure recovery policies and scheduler event logs
- resource-aware slot requests and priority queue metadata
- metric-based promotion gating between stages

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
- real-data ingestion and schema validation
- foundation workspace manifest and sampling-plan generation
- foundation trainer history and checkpoint generation
- advanced foundation training stack orchestration
- explicit research-infra vs engine-layer testing
- validation, best-checkpoint, and resume-flow testing
- artifact index and early-stopping lifecycle testing
- curriculum scheduling, event log, and checkpoint catalog testing
- sharding, manifest partitioning, worker-aware sampling, and retention testing
- scheduler planning, retry, and queue execution testing
- resource-aware scheduling and promotion-gate testing
- synthetic biology design pipeline testing

## Scheduler Layer

The repository now includes a control-plane layer above the foundation trainer.

This layer is intended to look more like a real training engine orchestrator:

- stage planning from the pretraining manifest and curriculum phases
- queue materialization for stage-by-stage execution
- per-stage workspaces and checkpoint handoff between queued runs
- retry-oriented failure recovery using latest or best checkpoints
- resource-aware slot requests, priority queue ordering, and simple concurrency metadata
- metric-based promotion gates for halting or rolling back downstream stages
- scheduler artifacts such as `scheduler_plan.json`, `run_queue.json`, `scheduler_summary.json`, and `scheduler_events.jsonl`

## Synthetic Biology Layer

The repository no longer stops at target ranking alone.

It now also supports a lightweight synthetic biology design workflow that sits on top of the multimodal prioritization stack:

- convert top causal and perturbation-supported targets into CRISPRa or CRISPRi gene programs
- bundle targets into multiplex intervention modules
- summarize pathway rewiring opportunities from pathway-membership edges
- generate construct blueprints with promoter, delivery, and assembly suggestions

This makes the project easier to describe as an intervention-design platform rather than only a target-screening repository.

## Suggested Next Extensions

- Add support for edge-aware objectives using STRING confidence weights directly in loss design.
- Add GO or KEGG enrichment analysis for the top ranked candidates.
- Add more baselines such as `GIN`, `APPNP`, `Node2Vec`, or diffusion-style ranking.
- Add unit tests and a synthetic toy dataset for CI smoke testing.

## Citation

If this repository helps your work, use the citation metadata in `CITATION.cff` or cite it as:

```text
BrowniGAT: Graph Attention based discovery of anti-browning targets from PPI networks.
```
