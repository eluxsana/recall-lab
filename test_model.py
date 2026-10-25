"""Tests for task construction, causal routing, and intervention semantics."""

import unittest

import torch
from safetensors.torch import load_file

from model import RecallTransformer, dataset, identifier


class RecallTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_target_is_value_of_query_key(self):
        for tokens, target in dataset(100, 1, "test"):
            pairs = dict(zip(tokens[1:-2:2], tokens[2:-2:2]))
            self.assertEqual(pairs[tokens[-1]] - 10, target)
            self.assertEqual(len(pairs), 4)

    def test_splits_are_disjoint(self):
        groups = [
            {identifier(x) for x, _ in dataset(200, 2, split)} for split in ("train", "dev", "test")
        ]
        for i in range(3):
            for j in range(i):
                self.assertFalse(groups[i] & groups[j])

    def test_deterministic_data(self):
        self.assertEqual(dataset(15, 23, "test"), dataset(15, 23, "test"))

    def test_causal_prefix_invariance(self):
        torch.manual_seed(8)
        m = RecallTransformer().eval()
        x = torch.tensor([dataset(1, 6, "test")[0][0]])
        altered = x.clone()
        altered[:, 8:] = (altered[:, 8:] + 1) % 18
        with torch.no_grad():
            a, _ = m(x)
            b, _ = m(altered)
        torch.testing.assert_close(a[:, :8], b[:, :8], atol=1e-6, rtol=1e-5)

    def test_attention_is_normalized_and_causal(self):
        m = RecallTransformer().eval()
        with torch.no_grad():
            _, maps = m(torch.tensor([dataset(1, 9, "test")[0][0]]))
        for layer in maps:
            torch.testing.assert_close(layer.sum(-1), torch.ones_like(layer.sum(-1)))
            self.assertEqual(layer.triu(1).count_nonzero().item(), 0)

    def test_zeroed_head_changes_output_without_rewriting_attention(self):
        torch.manual_seed(8)
        m = RecallTransformer().eval()
        x = torch.tensor([dataset(1, 3, "test")[0][0]])
        with torch.no_grad():
            a, maps_a = m(x)
            b, maps_b = m(x, (0, 0))
        self.assertGreater((a - b).abs().max().item(), 1e-5)
        torch.testing.assert_close(maps_a[0], maps_b[0])

    def test_checkpoint_roundtrip(self):
        import tempfile
        from pathlib import Path

        from safetensors.torch import save_file

        torch.manual_seed(8)
        m = RecallTransformer().eval()
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "weights.safetensors")
            save_file(m.state_dict(), path)
            restored = RecallTransformer().eval()
            restored.load_state_dict(load_file(path))
            x = torch.tensor([dataset(1, 3, "test")[0][0]])
            with torch.no_grad():
                torch.testing.assert_close(m(x)[0], restored(x)[0], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
