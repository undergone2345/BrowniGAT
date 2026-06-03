from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


def _project_embeddings(embeddings, method, perplexity, random_state):
    if method.lower() != "tsne":
        raise ValueError("Only 'tsne' visualization is currently supported.")

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        max_iter=2000,
        random_state=random_state,
    )
    return tsne.fit_transform(embeddings)


def plot_embedding_projection(
    embeddings,
    node_scores,
    degrees,
    label_encoder,
    highlight_genes,
    method,
    perplexity,
    random_state,
    save_path,
    title="BrowniGAT Embedding Projection",
):
    projected = _project_embeddings(
        embeddings=embeddings,
        method=method,
        perplexity=perplexity,
        random_state=random_state,
    )

    degrees = np.asarray(degrees, dtype=float)
    node_scores = np.asarray(node_scores, dtype=float)
    scaled_sizes = 30 + 120 * (degrees / max(degrees.max(), 1.0))

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(
        projected[:, 0],
        projected[:, 1],
        c=node_scores,
        s=scaled_sizes,
        cmap="YlOrRd",
        alpha=0.8,
        edgecolors="black",
        linewidths=0.2,
    )
    plt.colorbar(scatter, label="Composite Score")
    plt.xlabel(f"{method.upper()} 1")
    plt.ylabel(f"{method.upper()} 2")
    plt.title(title)

    for gene in highlight_genes:
        if gene not in label_encoder.classes_:
            continue
        idx = label_encoder.transform([gene])[0]
        plt.text(
            projected[idx, 0] + 0.7,
            projected[idx, 1] + 0.7,
            gene,
            fontsize=8,
        )

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
