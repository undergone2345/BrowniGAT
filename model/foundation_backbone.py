class BaseFoundationBackbone:
    def __init__(self, backbone_cfg):
        self.backbone_cfg = backbone_cfg
        self.backbone_name = backbone_cfg["name"]
        self.hidden_dim = int(backbone_cfg["hidden_dim"])
        self.num_layers = int(backbone_cfg["num_layers"])

    def encode_batch(self, batch):
        raise NotImplementedError


class TabularFusionBackbone(BaseFoundationBackbone):
    def encode_batch(self, batch):
        batch_size = len(batch)
        token_count = sum(len(str(item)) for item in batch)
        return {
            "batch_size": batch_size,
            "hidden_dim": self.hidden_dim,
            "token_density": token_count / max(batch_size, 1),
            "backbone_name": self.backbone_name,
        }


class HeteroGraphTransformerBackbone(BaseFoundationBackbone):
    def encode_batch(self, batch):
        batch_size = len(batch)
        field_diversity = len(
            {
                key
                for item in batch
                for key in (item.keys() if isinstance(item, dict) else [])
            }
        )
        return {
            "batch_size": batch_size,
            "hidden_dim": self.hidden_dim,
            "field_diversity": field_diversity,
            "backbone_name": self.backbone_name,
        }


def build_foundation_backbone(backbone_cfg):
    name = backbone_cfg["name"].lower()
    if name == "tabular_fusion":
        return TabularFusionBackbone(backbone_cfg)
    if name == "hetero_graph_transformer":
        return HeteroGraphTransformerBackbone(backbone_cfg)
    raise ValueError(f"Unsupported foundation backbone: {backbone_cfg['name']}")
