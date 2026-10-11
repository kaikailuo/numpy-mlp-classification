from __future__ import annotations

import json

import numpy as np

from a1_mlp.losses import SoftmaxCrossEntropy, SoftmaxMSE
from a1_mlp.model import MLP
from a1_mlp.reporting import write_json


def errors(numeric: np.ndarray, analytic: np.ndarray) -> dict:
    absolute = np.abs(numeric - analytic)
    relative = absolute / np.maximum(1e-8, np.abs(numeric) + np.abs(analytic))
    return {"max_abs_error": float(absolute.max()), "max_relative_error": float(relative.max())}


def check_loss(loss, logits: np.ndarray, labels: np.ndarray, h: float = 1e-5) -> dict:
    _, analytic = loss.value_and_grad(logits, labels)
    numeric = np.empty_like(logits)
    for index in np.ndindex(logits.shape):
        plus = logits.copy()
        minus = logits.copy()
        plus[index] += h
        minus[index] -= h
        numeric[index] = (loss.value_and_grad(plus, labels)[0] - loss.value_and_grad(minus, labels)[0]) / (2 * h)
    return errors(numeric, analytic)


def check_model(h: float = 1e-5) -> dict:
    x = np.asarray([[-1.2, 0.7], [0.3, 1.4], [1.2, -0.8]], dtype=np.float64)
    labels = np.asarray([0, 2, 1])
    model = MLP([2, 3, 3], init_seed=13, dtype=np.float64)
    loss = SoftmaxCrossEntropy()
    _, gradient = loss.value_and_grad(model.forward(x, training=True), labels)
    model.backward(gradient)
    reports = {}
    for parameter in model.parameters():
        analytic = parameter.grad.copy()
        numeric = np.empty_like(parameter.data)
        for index in np.ndindex(parameter.data.shape):
            original = parameter.data[index]
            parameter.data[index] = original + h
            plus = loss.value_and_grad(model.forward(x, training=False), labels)[0]
            parameter.data[index] = original - h
            minus = loss.value_and_grad(model.forward(x, training=False), labels)[0]
            parameter.data[index] = original
            numeric[index] = (plus - minus) / (2 * h)
        reports[parameter.name] = errors(numeric, analytic)
    return reports


def main() -> None:
    rng = np.random.default_rng(8)
    logits = rng.normal(size=(4, 3))
    labels = np.asarray([0, 2, 1, 0])
    result = {
        "dtype": "float64",
        "central_difference_h": 1e-5,
        "cross_entropy": check_loss(SoftmaxCrossEntropy(), logits, labels),
        "softmax_mse": check_loss(SoftmaxMSE(), logits, labels),
        "full_mlp": check_model(),
    }
    thresholds = {"absolute": 1e-7, "relative": 1e-4}
    result["thresholds"] = thresholds
    all_reports = [result["cross_entropy"], result["softmax_mse"], *result["full_mlp"].values()]
    result["status"] = "passed" if all(
        report["max_abs_error"] <= thresholds["absolute"]
        or report["max_relative_error"] <= thresholds["relative"]
        for report in all_reports
    ) else "failed"
    write_json("results/verification/gradient_check.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

