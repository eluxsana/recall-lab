"""Train, measure and export an attention explorer. No pretrained weights or APIs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch.nn import functional as F

from model import VOCAB, RecallTransformer, dataset, identifier


@torch.no_grad()
def evaluate(model, rows, ablate=None):
    correct, total_loss, last_value = 0, 0.0, 0
    for start in range(0, len(rows), 128):
        chunk = rows[start : start + 128]
        x, y = torch.tensor([r[0] for r in chunk]), torch.tensor([r[1] for r in chunk])
        logits, _ = model(x, ablate)
        correct += (logits[:, -1].argmax(-1) == y).sum().item()
        total_loss += F.cross_entropy(logits[:, -1], y, reduction="sum").item()
        last_value += (x[:, -3] - 10 == y).sum().item()
    return {
        "n": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows),
        "loss": total_loss / len(rows),
        "last_value_accuracy": last_value / len(rows),
    }


@torch.no_grad()
def inspect(model, tokens):
    logits, maps = model(torch.tensor([tokens]))
    p = logits[0, -1].softmax(-1)
    return {
        "tokens": [VOCAB[t] for t in tokens],
        "probabilities": [round(v, 6) for v in p.tolist()],
        "prediction": p.argmax().item(),
        "attention": [
            [[[round(v, 5) for v in row] for row in head] for head in layer[0].tolist()]
            for layer in maps
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("output"))
    parser.add_argument("--steps", type=int, default=1400)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    if args.out.exists():
        parser.error("Choose a new output directory; existing results are never replaced.")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)
    train_rows, test_rows = dataset(12000, 17, "train"), dataset(1024, 9281, "test")
    dev_rows = dataset(256, 99, "dev")
    train_ids = {identifier(x) for x, _ in train_rows}
    test_ids = {identifier(x) for x, _ in test_rows}
    assert not train_ids & test_ids
    model = RecallTransformer()
    before = evaluate(model, test_rows)
    optim = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.01)
    trace, start = [], time.perf_counter()
    x, y = (
        torch.tensor([r[0] for r in train_rows]),
        torch.tensor([r[1] for r in train_rows]),
    )
    for step in range(1, args.steps + 1):
        ix = torch.randint(len(train_rows), (96,))
        optim.zero_grad(set_to_none=True)
        logits, _ = model(x[ix])
        loss = F.cross_entropy(logits[:, -1], y[ix])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        if step == 1 or step % 100 == 0:
            score = evaluate(model, dev_rows)
            trace.append({"step": step, "batch_loss": loss.item(), **score})
            print(
                f"step={step} loss={loss.item():.4f} dev={score['accuracy']:.3f}",
                flush=True,
            )
    seconds = time.perf_counter() - start
    model.eval()
    final = evaluate(model, test_rows)
    ablations = [
        {"layer": layer, "head": head, **evaluate(model, test_rows, (layer, head))}
        for layer in range(2)
        for head in range(4)
    ]
    ablations += [
        {
            "layer": layer,
            "head": "all",
            **evaluate(model, test_rows, (layer, list(range(4)))),
        }
        for layer in range(2)
    ]
    stress = [
        {"pairs": pairs, **evaluate(model, dataset(512, 9281, "test", pairs))}
        for pairs in (2, 3, 4, 5, 6)
    ]
    examples = []
    for tokens, target in test_rows[:8]:
        original = inspect(model, tokens)
        original.update({"target": target, "id": identifier(tokens), "variants": []})
        for pair in range(4):
            changed = tokens.copy()
            position = 2 + pair * 2
            changed[position] = 10 + ((changed[position] - 10 + 1) % 8)
            query_key = tokens[-1]
            new_target = changed[position] - 10 if tokens[position - 1] == query_key else target
            variant = inspect(model, changed)
            variant.update(
                {
                    "pair": pair,
                    "target": new_target,
                    "heldout_hash_partition": int(identifier(changed)[:8], 16) % 10 == 9,
                    "seen_in_training": identifier(changed) in train_ids,
                }
            )
            original["variants"].append(variant)
        examples.append(original)
    paths = [Path(__file__), Path(__file__).with_name("model.py")]
    digest = hashlib.sha256(b"".join(p.name.encode() + p.read_bytes() for p in paths)).hexdigest()
    result = {
        "schema": 1,
        "model": "2 layers, 4 heads, width64, MLP128, learned positions, causal",
        "parameters": sum(p.numel() for p in model.parameters()),
        "seed": args.seed,
        "steps": args.steps,
        "dev_samples": len(dev_rows),
        "train_samples": len(train_rows),
        "test_samples": len(test_rows),
        "train_ids_sha256": hashlib.sha256("".join(sorted(train_ids)).encode()).hexdigest(),
        "test_ids_sha256": hashlib.sha256("".join(sorted(test_ids)).encode()).hexdigest(),
        "exact_train_test_overlap": 0,
        "source_sha256": digest,
        "seconds": seconds,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.system(),
            "threads": 2,
            "device": "cpu",
        },
        "before": before,
        "final": final,
        "trace": trace,
        "ablations": ablations,
        "length_stress": stress,
        "examples": examples,
        "interpretation": "Attention weights show routing, not a causal explanation. Head zeroing is an intervention on this model/task; interacting heads and redundancy limit attribution. One seed, synthetic data, no real-world or language understanding claim.",
    }
    assert all(math.isfinite(v) for v in [final["loss"], final["accuracy"]])
    args.out.mkdir(parents=True)
    save_file(model.state_dict(), str(args.out / "model.safetensors"))
    result["checkpoint_sha256"] = hashlib.sha256(
        (args.out / "model.safetensors").read_bytes()
    ).hexdigest()
    payload = json.dumps(result, separators=(",", ":"), allow_nan=False)
    (args.out / "results.json").write_text(payload + "\n")
    template = Path(__file__).with_name("explorer.html").read_text()
    (args.out / "index.html").write_text(
        template.replace("/*__RESULTS__*/", payload.replace("<", "\\u003c"))
    )
    print(
        f"final={final['accuracy']:.4f}; {seconds:.1f}s training; {result['parameters']} parameters"
    )


if __name__ == "__main__":
    main()
