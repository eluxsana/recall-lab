"""Independently recompute the published results from the safe tensor checkpoint."""

import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from model import RecallTransformer, dataset, identifier
from train import evaluate, inspect


def close(a, b, tolerance=2e-5):
    if isinstance(a, dict):
        for key in a:
            close(a[key], b[key], tolerance)
    elif isinstance(a, list):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            close(x, y, tolerance)
    elif isinstance(a, float):
        assert abs(a - b) <= tolerance, (a, b)
    else:
        assert a == b, (a, b)


def main():
    root = Path(__file__).parent
    torch.set_num_threads(2)
    data = json.loads((root / "results.json").read_text())
    assert (
        hashlib.sha256((root / "model.safetensors").read_bytes()).hexdigest()
        == data["checkpoint_sha256"]
    )
    paths = [root / "train.py", root / "model.py"]
    digest = hashlib.sha256(b"".join(p.name.encode() + p.read_bytes() for p in paths)).hexdigest()
    assert digest == data["source_sha256"]
    model = RecallTransformer().eval()
    model.load_state_dict(load_file(str(root / "model.safetensors")))
    train_rows = dataset(12000, 17, "train")
    train_ids = {identifier(x) for x, _ in train_rows}
    test_rows = dataset(1024, 9281, "test")
    assert not train_ids & {identifier(x) for x, _ in test_rows}
    close(data["final"], evaluate(model, test_rows))
    for result in data["ablations"]:
        head = list(range(4)) if result["head"] == "all" else result["head"]
        close(
            {k: v for k, v in result.items() if k not in ("layer", "head")},
            evaluate(model, test_rows, (result["layer"], head)),
        )
    for result in data["length_stress"]:
        close(
            {k: v for k, v in result.items() if k != "pairs"},
            evaluate(model, dataset(512, 9281, "test", result["pairs"])),
        )
    for stored, (tokens, target) in zip(data["examples"], test_rows[:8]):
        actual = inspect(model, tokens)
        close(actual, stored)
        assert target == stored["target"]
        for pair, variant in enumerate(stored["variants"]):
            changed = tokens.copy()
            changed[2 + pair * 2] = 10 + ((changed[2 + pair * 2] - 10 + 1) % 8)
            close(inspect(model, changed), variant)
            expected = (
                changed[2 + pair * 2] - 10 if changed[1 + pair * 2] == changed[-1] else target
            )
            assert expected == variant["target"]
            assert variant["seen_in_training"] == (identifier(changed) in train_ids)
    study = json.loads((root / "seed-study.json").read_text())
    for run in study["runs"]:
        if run["seed"] == 42:
            continue
        path = root / f"seed{run['seed']}.safetensors"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == run["checkpoint_sha256"]
        model.load_state_dict(load_file(str(path)))
        close(run["final"], evaluate(model, test_rows))
        for result in run["length_stress"]:
            close(
                {k: v for k, v in result.items() if k != "pairs"},
                evaluate(model, dataset(512, 9281, "test", result["pairs"])),
            )
    print(
        "Verified checkpoint, source hash, test results, all ablations, length stress, and40 recorded forward passes."
    )


if __name__ == "__main__":
    main()
