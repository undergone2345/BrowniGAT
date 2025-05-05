from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def plot_tsne(embeddings, core_scores, degrees, title="PPI Core Targets"):
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    emb_2d = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(emb_2d[:, 0], emb_2d[:, 1], 
                c=core_scores, cmap='viridis', 
                s=degrees.numpy()*5, alpha=0.7)
    plt.colorbar(label='Core Score')
    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.show()
