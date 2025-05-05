🧬 BrowniGAT: A GAT-based Framework for Identifying Anti-Browning Core Targets from PPI Network

### 🔍 Project Overview
**BrowniGAT** is a deep learning framework based on Graph Attention Networks (GAT) for analyzing protein-protein interaction (PPI) networks and identifying **core targets** involved in the anti-browning mechanisms of biological systems.  
It leverages **STRING PPI data**, encodes protein features, learns embeddings via GAT, and ranks core targets based on embedding similarity and topological properties.

---

## 🚀 Features

- ✅ Input: PPI data from STRING database (`.tsv`)
- ✅ Protein encoding via `LabelEncoder` + one-hot
- ✅ Embedding learning with GATConv (2-layer architecture)
- ✅ Contrastive loss for meaningful representation
- ✅ Compute node degree & similarity to known anti-browning targets
- ✅ Identify and rank **core anti-browning candidate targets**
- ✅ Visualize results with t-SNE (colored by core scores)

---

## 🧩 Dependencies

Install the following packages before running:

```bash
pip install pandas torch torch-geometric scikit-learn matplotlib
```

> ✅ Tested on: Python 3.9+, PyTorch 2.x, torch-geometric 2.x

---

## 📂 File Structure

```
BrowniGAT/
├── README.md
├── requirements.txt
├── main.py
├── model/
│   └── gat_embed.py
├── utils/
│   ├── data_loader.py
│   ├── loss.py
│   ├── visualize.py
│   └── core_target.py
├── config/
│   └── config.yaml
├── data/
│   └── string_interactions_short.csv
├── results/
│   └── core_targets.tsv
└── notebooks/
    └── demo_analysis.ipynb

```

---

## 📈 Quick Start

### 1. Prepare STRING PPI Data

Download interaction files from [STRING-db](https://string-db.org/) and place them under `data/`. Ensure it includes at least:

| Protein A | Protein B | ENSEMBL ID A | ENSEMBL ID B | Scores... |

Format example:
```text
ACE    AGTR2    9606.ENSP00000290866 9606.ENSP00000360973 0.00 0.0 ...
```

### 2. Run the Pipeline

```bash
python main.py
```

### 3. Output

- **`core_targets.csv`**: Ranked anti-browning target proteins
- **t-SNE Visualization**: Colored by CoreScore & scaled by degree

---

## 🧠 Methodology

### 🔗 Graph Construction
- Nodes: Proteins
- Edges: STRING interactions
- Features: One-hot encoded (identity)

### 🤖 Model Architecture

```
Input (One-hot) → GATConv (heads=4) → ELU → GATConv → Embedding (64-d)
```

### 🎯 Target Identification
- Define known browning-related targets (e.g., `['TYR', 'MITF', 'UCP1']`)
- Compute **cosine similarity** between each protein and known browning targets
- Multiply similarity by **node degree** to get CoreScore
- Rank proteins by CoreScore

---

## 🖼️ Visualization

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
```

The t-SNE plot highlights:
- **Color**: Similarity to anti-browning targets (CoreScore)
- **Size**: Node degree (importance in PPI network)

Example:

![example_tsne](results/tsne_browning.png)

---

## 🧪 Example Output (Top 5 Targets)

| Protein | Degree | Similarity | CoreScore |
|--------:|--------:|------------:|-----------:|
| TYMS    | 15      | 0.88        | 13.20      |
| MITF    | 12      | 0.90        | 10.80      |
| PPARGC1A| 10      | 0.85        | 8.50       |

---

## 🧾 Citation / Acknowledgment

If you find this useful, please cite this repo:

> *BrowniGAT: GAT-based discovery of anti-browning targets from PPI networks (2025).*

Or acknowledge:

```text
Developed by zhuzi.
Built using PyTorch Geometric and STRING database.
```

---

## 🛠️ Future Work

- [ ] Add support for GO/KEGG annotation of core targets
- [ ] Integrate molecular docking validation results
- [ ] Trainable node features (e.g., using gene expression)

---


