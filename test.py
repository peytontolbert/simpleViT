import unittest
import torch
from simpleViT import (
    PatchEmbedding,
    PositionalEmbedding,
    TransformerEncoder,
    ClassificationHead,
)  # Make sure to import your model correctly


class TestPatchEmbedding(unittest.TestCase):
    def test_patch_embedding_shape(self):
        img_size, patch_size, in_channels, embed_dim = 224, 16, 3, 768
        model = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        x = torch.randn(2, in_channels, img_size, img_size)
        patches = model(x)
        expected_shape = (2, (img_size // patch_size) ** 2, embed_dim)
        self.assertEqual(patches.shape, expected_shape)


# Add more tests as needed
class TestPositionalEmbedding(unittest.TestCase):
    def test_positional_embedding_shape(self):
        n_patches, embed_dim = 196, 768
        model = PositionalEmbedding(n_patches, embed_dim)
        x = torch.randn(2, n_patches, embed_dim)
        pos_embedding = model(x)
        expected_shape = (2, n_patches + 1, embed_dim)  # +1 for cls_token
        self.assertEqual(pos_embedding.shape, expected_shape)


# Add more tests as needed
class TestTransformerEncoder(unittest.TestCase):
    def test_transformer_encoder_output_shape(self):
        embed_dim, n_heads, n_layers, mlp_ratio, dropout = 768, 12, 12, 4, 0.1
        model = TransformerEncoder(embed_dim, n_heads, n_layers, mlp_ratio, dropout)
        x = torch.randn(2, 197, embed_dim)  # 197 = 196 patches + 1 cls_token
        output = model(x)
        self.assertEqual(output.shape, x.shape)


# Add more tests as needed


class TestClassificationHead(unittest.TestCase):
    def test_classification_head_output(self):
        embed_dim, n_classes = 768, 1000
        model = ClassificationHead(embed_dim, n_classes)
        x = torch.randn(2, 197, embed_dim)  # Example input from transformer encoder
        logits = model(x)
        expected_shape = (2, n_classes)
        self.assertEqual(logits.shape, expected_shape)


# Add more tests as needed
