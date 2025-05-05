import torch
from model.gat_embed import GATEmbed
from utils.data_loader import load_ppi_data
from utils.loss import contrastive_loss
from utils.core_target import compute_core_targets
from utils.visualize import plot_tsne

# 加载数据
data, le = load_ppi_data("data/string_interactions_short.csv")
num_nodes = data.num_nodes

# 构建模型
model = GATEmbed(in_channels=num_nodes, hidden_channels=128, out_channels=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# 训练模型
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    embeddings = model(data)
    loss = contrastive_loss(data, embeddings, num_nodes)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# 获取结果
degrees = torch.zeros(num_nodes)
src, dst = data.edge_index
degrees.scatter_add_(0, src, torch.ones_like(src, dtype=torch.float))
degrees.scatter_add_(0, dst, torch.ones_like(dst, dtype=torch.float))

with torch.no_grad():
    emb_np = model(data).numpy()

result_df = compute_core_targets(emb_np, degrees, le, target_genes=['TYR', 'MITF', 'UCP1', 'PPARGC1A'])
result_df.to_csv("results/core_targets.tsv", sep='\t', index=False)
plot_tsne(emb_np, result_df["CoreScore"], degrees)
