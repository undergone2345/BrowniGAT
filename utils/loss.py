import torch
import torch.nn.functional as F


def contrastive_loss(data, embeddings, num_nodes, negative_ratio=1, device=None):
    src, dst = data.edge_index

    positive_logits = (embeddings[src] * embeddings[dst]).sum(dim=1)
    positive_loss = F.softplus(-positive_logits).mean()

    sampled_negative_losses = []
    for _ in range(max(1, negative_ratio)):
        negative_dst = torch.randint(
            low=0,
            high=num_nodes,
            size=(src.size(0),),
            device=device or embeddings.device,
        )
        negative_logits = (embeddings[src] * embeddings[negative_dst]).sum(dim=1)
        sampled_negative_losses.append(F.softplus(negative_logits).mean())

    negative_loss = torch.stack(sampled_negative_losses).mean()
    total_loss = positive_loss + negative_loss

    return total_loss, {
        "positive_loss": float(positive_loss.item()),
        "negative_loss": float(negative_loss.item()),
    }
