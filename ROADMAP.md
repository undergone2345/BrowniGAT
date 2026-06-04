# Roadmap

## Current Position

BrowniGAT now spans four public layers:

1. Multimodal biomedical graph platform
2. Foundation training engine
3. Scheduler, recovery, and promotion control plane
4. Synthetic biology design layer

## Near-Term Priorities

### 1. Packaging And Distribution

- package the repository into a reusable Python distribution
- expose stable CLI entrypoints for ingestion, training, scheduling, and synbio design
- publish versioned releases and release notes

### 2. Engine Scalability

- add distributed launch specifications for multi-GPU and multi-node training
- materialize rank-aware configs and launch manifests
- strengthen checkpoint lineage and run provenance graphs

### 3. Evaluation And Adoption

- expand benchmark coverage with more graph and non-graph baselines
- add reproducible demo workflows and richer result examples
- add usage guides for external labs and translational teams

### 4. Biomedical And SynBio Capability Expansion

- add perturbation-program optimization across multiple cell contexts
- add circuit-risk heuristics for multiplex construct complexity
- add pathway intervention scoring for disease-specific rewiring strategies
- add richer intervention design outputs for CRISPR, RNA, and small-molecule hybrid programs

## Longer-Term Direction

- turn BrowniGAT into a reusable biomedical AI systems repo rather than a single-use ranking script
- support real multimodal foundation training workflows with stronger orchestration and observability
- support intervention design loops that connect target discovery, perturbation reasoning, and construct planning
