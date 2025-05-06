from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def plot_tsne(embeddings, core_scores, degrees,
              label_encoder=None,
              target_genes=None,
              save_path="../results/tsne_browning.png"):
    """
    embeddings: np.ndarray, shape (num_nodes, embedding_dim)
    core_scores: np.ndarray
    degrees: torch.Tensor
    label_encoder: LabelEncoder，用于将基因名转为索引
    target_genes: list[str]，需要标注的基因名列表
    save_path: 保存路径
    """
    import numpy as np

    # 1. t-SNE 降维
    tsne = TSNE(n_components=2,
                perplexity=5,
                init='pca',
                n_iter=2000,
                random_state=42)
    emb_2d = tsne.fit_transform(embeddings)

    # 2. 颜色处理
    num_nodes = embeddings.shape[0]
    colors = np.full(num_nodes, '#cccccc')  # 默认灰色
    sizes = degrees.numpy() * 5

    if target_genes and label_encoder:
        try:
            browning_ids = label_encoder.transform(target_genes)
            for i in browning_ids:
                colors[i] = 'red'
                sizes[i] = 80  # 放大
        except ValueError as e:
            print("⚠️ 部分 target_genes 未找到对应索引：", e)
            browning_ids = []

    # 3. 绘图
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        emb_2d[:, 0], emb_2d[:, 1],
        c=colors,
        s=sizes,
        alpha=0.7
    )

    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.title("t-SNE of PPI Embeddings (Core Targets Highlighted)")
    
    # 4. 添加标签
    if target_genes and label_encoder:
        for gene in target_genes:
            try:
                idx = label_encoder.transform([gene])[0]
                x, y = emb_2d[idx]
                plt.text(x + 1.5, y, gene, fontsize=8, color='black')
            except ValueError:
                continue

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
