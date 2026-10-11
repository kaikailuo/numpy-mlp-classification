from __future__ import annotations

import argparse
import json

import numpy as np

from src.nn.layers import Parameter
from src.nn.losses import (
    FocalLoss,
    LabelSmoothingCrossEntropy,
    MulticlassHingeLoss,
    SoftmaxCrossEntropy,
    SoftmaxMSE,
)
from src.nn.model import MLP
from src.nn.regularization import add_regularization_grads, regularization_loss
from src.utils.io import write_json


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


def check_regularization(h: float = 1e-6) -> dict:
    values = np.asarray([[-1.2, 0.7], [2.1, -0.4]], dtype=np.float64)
    parameter = Parameter("layers.0.weight", values.copy(), np.zeros_like(values))
    add_regularization_grads([parameter], l1_lambda=0.03, l2_lambda=0.07)
    analytic = parameter.grad.copy()
    numeric = np.empty_like(values)
    for index in np.ndindex(values.shape):
        original = parameter.data[index]
        parameter.data[index] = original + h
        plus = regularization_loss([parameter], l1_lambda=0.03, l2_lambda=0.07)
        parameter.data[index] = original - h
        minus = regularization_loss([parameter], l1_lambda=0.03, l2_lambda=0.07)
        parameter.data[index] = original
        numeric[index] = (plus - minus) / (2 * h)
    return errors(numeric, analytic)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/verification/v1_1/gradient_check.json")
    args = parser.parse_args()
    rng = np.random.default_rng(8)
    logits = rng.normal(size=(4, 3))
    labels = np.asarray([0, 2, 1, 0])
    result = {
        "dtype": "float64",
        "central_difference_h": 1e-5,
        "losses": {
            "cross_entropy": check_loss(SoftmaxCrossEntropy(), logits, labels),
            "softmax_mse": check_loss(SoftmaxMSE(), logits, labels),
            "label_smoothing_cross_entropy": check_loss(LabelSmoothingCrossEntropy(0.1), logits, labels),
            "multiclass_hinge": check_loss(MulticlassHingeLoss(1.0), logits, labels),
            "focal": check_loss(FocalLoss(2.0, 1.0), logits, labels),
        },
        "regularization": check_regularization(),
        "full_mlp": check_model(),
    }
    thresholds = {"absolute": 1e-7, "relative": 1e-4}
    result["thresholds"] = thresholds
    all_reports = [*result["losses"].values(), result["regularization"], *result["full_mlp"].values()]
    result["status"] = "passed" if all(
        report["max_abs_error"] <= thresholds["absolute"]
        or report["max_relative_error"] <= thresholds["relative"]
        for report in all_reports
    ) else "failed"
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

