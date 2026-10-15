"""Compare identical token scores across batch sizes without prescribing a tolerance."""

import json
import math


def load_fixture(path, model):
    fixture = json.loads(path.read_text())
    if model.resolve().name != fixture["model_revision"]:
        raise ValueError("Model snapshot directory must be named for the fixture's pinned revision")
    tokens = fixture["tokens"]
    if len(tokens) < 2 or not all(type(token) is int and token >= 0 for token in tokens):
        raise ValueError("Expected at least two nonnegative integer token IDs")
    return fixture


def compare_rows(reference, actual):
    if not reference or len(reference) != len(actual):
        raise ValueError("Expected nonempty, aligned token scores")
    if not all(math.isfinite(value) for row in (reference, actual) for value in row):
        raise ValueError("Nonfinite token score")
    errors = [abs(left - right) for left, right in zip(reference, actual)]
    return {"positions": len(errors), "mean_abs": sum(errors) / len(errors), "max_abs": max(errors)}


def compare_batches(scores):
    if [len(scores[key]) for key in ("single", "duplicate", "repeat")] != [1, 2, 1]:
        raise ValueError("Expected one row, two duplicate rows, then one repeated row")
    return {
        "batch_size": compare_rows(scores["single"][0], scores["duplicate"][0]),
        "duplicate_rows": compare_rows(scores["duplicate"][0], scores["duplicate"][1]),
        "repeat": compare_rows(scores["single"][0], scores["repeat"][0]),
    }
