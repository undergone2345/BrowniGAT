def build_model(model_name, in_channels, model_cfg):
    model_name = model_name.lower()
    hidden_channels = model_cfg["hidden_channels"]
    out_channels = model_cfg["out_channels"]
    dropout = model_cfg["dropout"]

    if model_name == "gat":
        from model.gat_embed import GATEmbed

        return GATEmbed(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            heads=model_cfg["heads"],
            dropout=dropout,
        )

    if model_name == "gcn":
        from model.gcn_embed import GCNEmbed

        return GCNEmbed(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            dropout=dropout,
        )

    if model_name == "graphsage":
        from model.graphsage_embed import GraphSAGEEmbed

        return GraphSAGEEmbed(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            dropout=dropout,
        )

    raise ValueError(f"Unsupported model name: {model_name}")
