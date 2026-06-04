def evaluate_promotion(training_summary, promotion_cfg):
    if not promotion_cfg.get("enabled", False):
        return {
            "enabled": False,
            "passed": True,
            "metric_name": None,
            "metric_value": None,
            "threshold": None,
            "action": "promote",
        }

    metric_name = promotion_cfg.get("metric_name", "best_validation_loss")
    metric_value = training_summary.get(metric_name)
    threshold = promotion_cfg.get("threshold")
    mode = promotion_cfg.get("mode", "min")
    on_fail = promotion_cfg.get("on_fail", "halt")

    if threshold is None or metric_value is None:
        passed = True
    elif mode == "min":
        passed = float(metric_value) <= float(threshold)
    else:
        passed = float(metric_value) >= float(threshold)

    return {
        "enabled": True,
        "passed": bool(passed),
        "metric_name": metric_name,
        "metric_value": metric_value,
        "threshold": threshold,
        "action": "promote" if passed else on_fail,
    }
