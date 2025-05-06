import os
import torch
from model.gat_embed import GATEmbed
from utils.data_loader import load_ppi_data
from utils.loss import contrastive_loss
from utils.core_target import compute_core_targets
from utils.visualize import plot_tsne

# ─── Step 1: 加载数据 ────────────────────────────
data, le = load_ppi_data("data/string_interactions_short.csv")
num_nodes = data.num_nodes

# ─── Step 2: 构建模型 ────────────────────────────
model = GATEmbed(in_channels=num_nodes, hidden_channels=128, out_channels=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# ─── Step 3: 训练模型 ────────────────────────────
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    embeddings = model(data)
    loss = contrastive_loss(data, embeddings, num_nodes)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# ─── Step 4: 计算节点度数 ──────────────────────────
degrees = torch.zeros(num_nodes)
src, dst = data.edge_index
degrees.scatter_add_(0, src, torch.ones_like(src, dtype=torch.float))
degrees.scatter_add_(0, dst, torch.ones_like(dst, dtype=torch.float))

# ─── Step 5: 提取最终嵌入 ──────────────────────────
model.eval()
with torch.no_grad():
    emb_np = model(data).numpy()

# ─── Step 6: 识别核心靶点并保存 ─────────────────────
browning_genes = ['TYR', 'MITF', 'UCP1', 'PPARGC1A']
result_df = compute_core_targets(emb_np, degrees, le, browning_genes)

# 确保 results 文件夹存在
os.makedirs("results", exist_ok=True)

# 保存全部靶点
result_df.to_csv("results/core_targets.tsv", sep='\t', index=False)


top20_proteins = result_df.head(20)['Protein'].tolist()

# ─── Step 7: 高亮褐变基因的 t-SNE 可视化 ────────────
plot_tsne(
    embeddings    = emb_np,
    core_scores   = result_df["CoreScore"].values,
    degrees       = degrees,
    label_encoder = le,
    target_genes  = top20_proteins,                      
    save_path     = "results/tsne_core20_highlight.png"  
)