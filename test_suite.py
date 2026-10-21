"""
STEP 6 - COMPREHENSIVE TEST SUITE
Validates Data Integrity, Model Architectures, Attention Masking, and Metrics

Run with:
    python test_suite.py
"""

import os
import unittest
import numpy as np
import torch
import torch.nn as nn

from cnn_baseline import TextCNN
from cnn_bilstm_attention import CNNBiLSTMAttention, MaskedAttention


class TestDeepLearningProject(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cpu")
        self.vocab_size = 500
        self.batch_size = 8
        self.seq_len = 64

    # 1. Test TextCNN Model Forward Pass
    def test_textcnn_forward_shape(self):
        model = TextCNN(vocab_size=self.vocab_size).to(self.device)
        x = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        out = model(x)
        self.assertEqual(out.shape, (self.batch_size,))
        self.assertFalse(torch.isnan(out).any(), "TextCNN output contains NaNs")

    # 2. Test CNN-BiLSTM-Attention Forward Pass
    def test_bilstm_attention_forward_shape(self):
        model = CNNBiLSTMAttention(vocab_size=self.vocab_size).to(self.device)
        x = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        out = model(x)
        self.assertEqual(out.shape, (self.batch_size,))
        self.assertFalse(torch.isnan(out).any(), "CNN-BiLSTM output contains NaNs")

    # 3. Test Masked Attention Layer
    def test_masked_attention_masking(self):
        hidden_dim = 64
        attn = MaskedAttention(hidden_dim=hidden_dim).to(self.device)
        h = torch.randn(2, 10, hidden_dim)
        mask = torch.ones(2, 10, dtype=torch.bool)
        # Pad last 4 positions of sample 0
        mask[0, 6:] = False

        context, weights = attn(h, mask)
        self.assertEqual(context.shape, (2, hidden_dim))
        self.assertEqual(weights.shape, (2, 10))

        # Padded positions must receive ~0 attention weight
        pad_weights = weights[0, 6:].sum().item()
        self.assertAlmostEqual(pad_weights, 0.0, places=5,
                               msg="Masked attention failed to zero-out padding positions")

        # Active positions must sum to 1.0
        active_weights = weights[0, :6].sum().item()
        self.assertAlmostEqual(active_weights, 1.0, places=5)

    # 4. Test Data Integrity: YOSM Processed Files
    def test_yosm_data_integrity(self):
        yosm_dir = os.path.join("data", "processed", "yosm")
        self.assertTrue(os.path.exists(yosm_dir), "data/processed/yosm does not exist")

        for split in ("train", "val", "test"):
            path = os.path.join(yosm_dir, f"{split}.npz")
            self.assertTrue(os.path.exists(path), f"Missing {path}")
            data = np.load(path)
            self.assertIn("x", data)
            self.assertIn("y", data)
            x, y = data["x"], data["y"]
            self.assertEqual(len(x), len(y))
            self.assertEqual(x.shape[1], 64)
            # Labels must be binary {0, 1}
            unique_labels = set(np.unique(y))
            self.assertTrue(unique_labels.issubset({0, 1}))

    # 5. Test Data Integrity: SST-2 Processed Files
    def test_sst2_data_integrity(self):
        sst2_dir = os.path.join("data", "processed", "sst2")
        self.assertTrue(os.path.exists(sst2_dir), "data/processed/sst2 does not exist")

        for split in ("train", "val", "test"):
            path = os.path.join(sst2_dir, f"{split}.npz")
            self.assertTrue(os.path.exists(path), f"Missing {path}")
            data = np.load(path)
            self.assertIn("x", data)
            self.assertIn("y", data)
            x, y = data["x"], data["y"]
            self.assertEqual(len(x), len(y))
            self.assertEqual(x.shape[1], 64)

    # 6. Test Gradient Propagation
    def test_gradient_flow(self):
        model = CNNBiLSTMAttention(vocab_size=self.vocab_size).to(self.device)
        x = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        y = torch.randint(0, 2, (self.batch_size,)).float()

        loss_fn = nn.BCEWithLogitsLoss()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad and "embedding" not in name:
                self.assertIsNotNone(param.grad, f"Zero gradient in {name}")
                self.assertGreater(param.grad.abs().sum().item(), 0.0, f"Vanishing gradient in {name}")


if __name__ == "__main__":
    print("=" * 60)
    print(" RUNNING DEEP LEARNING VERIFICATION TEST SUITE")
    print("=" * 60)
    unittest.main(verbosity=2)
