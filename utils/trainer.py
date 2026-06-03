import torch

from utils.loss import contrastive_loss


def train_model(model, data, training_cfg, device, run_name="model"):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_cfg["lr"],
        weight_decay=training_cfg["weight_decay"],
    )

    best_loss = float("inf")
    best_state_dict = None
    history = []

    for epoch in range(1, training_cfg["epochs"] + 1):
        model.train()
        optimizer.zero_grad()

        embeddings = model(data)
        loss, diagnostics = contrastive_loss(
            data=data,
            embeddings=embeddings,
            num_nodes=data.num_nodes,
            negative_ratio=training_cfg["negative_ratio"],
            device=device,
        )
        loss.backward()
        optimizer.step()

        epoch_record = {
            "epoch": epoch,
            "loss": float(loss.item()),
            "positive_loss": float(diagnostics["positive_loss"]),
            "negative_loss": float(diagnostics["negative_loss"]),
        }
        history.append(epoch_record)

        if loss.item() < best_loss:
            best_loss = float(loss.item())
            best_state_dict = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }

        if epoch == 1 or epoch % training_cfg["log_every"] == 0:
            print(
                "[{name}] Epoch {epoch:03d} | loss={loss:.4f} | pos={pos:.4f} | neg={neg:.4f}".format(
                    name=run_name,
                    epoch=epoch,
                    loss=epoch_record["loss"],
                    pos=epoch_record["positive_loss"],
                    neg=epoch_record["negative_loss"],
                )
            )

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return {
        "history": history,
        "best_loss": best_loss,
        "epochs": training_cfg["epochs"],
    }
