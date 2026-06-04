# Demo Workflows

This document collects the main public demo commands for BrowniGAT.

## 1. Baseline Ranking

```bash
python main.py --config config/toy_config.yaml
```

## 2. vNext Multimodal Pipeline

```bash
python vnext_main.py --config config/vnext_toy.yaml
```

## 3. Real-Data Ingestion

```bash
python ingest_multimodal_data.py --config config/real_data_example.yaml
```

## 4. Foundation Workspace

```bash
python foundation_main.py --config config/foundation_example.yaml
```

## 5. Foundation Engine

```bash
python foundation_train.py --config config/foundation_engine_example.yaml
```

## 6. Scheduler Control Plane

```bash
python foundation_schedule.py --config config/foundation_engine_example.yaml --workspace-dir results/foundation_scheduler
```

## 7. Synthetic Biology Design Layer

```bash
python synbio_main.py --config config/synbio_toy.yaml --output-dir results_synbio/demo
```
