import unittest
import torch
from simplevitMoE import (
    VisionTransformerWithMoE,
)  # Ensure to import your model correctly


class TestVisionTransformerWithMoE(unittest.TestCase):
    def setUp(self):
        img_size = 224
        patch_size = 16
        in_channels = 3
        embed_dim = 512
        n_heads = 4
        n_layers = 12
        mlp_ratio = 4
        n_classes = 1000
        dropout = 0.1
        num_experts = 4
        expert_dim = 2048
        self.model = VisionTransformerWithMoE(
            img_size,
            patch_size,
            in_channels,
            embed_dim,
            n_heads,
            n_layers,
            mlp_ratio,
            n_classes,
            dropout,
            num_experts,
            expert_dim,
        )

    def test_forward_pass(self):
        x = torch.rand(
            1, 3, 224, 224
        )  # Batch size of 1, 3 color channels, 224x224 image
        logits = self.model(x)
        self.assertEqual(logits.size(), torch.Size([1, 1000]))  # Assuming 1000 classes


if __name__ == "__main__":
    unittest.main()
