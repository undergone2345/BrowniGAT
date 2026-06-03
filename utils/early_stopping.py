def update_early_stopping(state, current_score, epoch, mode="min"):
    best_score = state.get("best_score")
    patience = int(state.get("patience", 0))
    min_delta = float(state.get("min_delta", 0.0))
    bad_epochs = int(state.get("bad_epochs", 0))

    improved = False
    if best_score is None:
        improved = True
    elif mode == "min":
        improved = current_score < (best_score - min_delta)
    else:
        improved = current_score > (best_score + min_delta)

    if improved:
        state["best_score"] = float(current_score)
        state["best_epoch"] = int(epoch)
        state["bad_epochs"] = 0
        state["should_stop"] = False
    else:
        bad_epochs += 1
        state["bad_epochs"] = bad_epochs
        state["should_stop"] = bad_epochs > patience

    return state


def initialize_early_stopping(patience, min_delta):
    return {
        "best_score": None,
        "best_epoch": None,
        "bad_epochs": 0,
        "patience": int(patience),
        "min_delta": float(min_delta),
        "should_stop": False,
    }
