import torch

def contrastive_loss(data, embeddings, num_nodes):
    src, dst = data.edge_index
    pos_pairs = embeddings[src] * embeddings[dst]
    pos_loss = -torch.log(torch.sigmoid(pos_pairs.sum(dim=1))).mean()
    
    neg_dst = torch.randint(0, num_nodes, (src.size(0),))
    neg_pairs = embeddings[src] * embeddings[neg_dst]
    neg_loss = -torch.log(1 - torch.sigmoid(neg_pairs.sum(dim=1))).mean()

    return pos_loss + neg_loss
