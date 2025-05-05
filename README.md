# GNN-AntiBrowning-Target

This project builds a graph neural network (GAT) to identify anti-browning core protein targets from STRING PPI networks, combining contrastive learning and GAT embeddings.

## Features
- STRING network ingestion and pre-processing
- GAT-based protein embedding
- Contrastive loss based unsupervised training
- Cosine similarity to known browning-related targets
- Core target ranking and t-SNE visualization

## Installation
```bash
pip install -r requirements.txt
