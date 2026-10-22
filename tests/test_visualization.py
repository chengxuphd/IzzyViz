"""Integration tests for the exported visualization entry points.

These drive each public function with small synthetic attention matrices
rather than a real model, so the suite stays fast and needs no network,
no model download, and no GPU. What is asserted is that each entry point
runs end to end and writes a real PDF -- not that the picture is correct.
Rendering correctness is not covered; see tests/README.md.
"""
import os

import pytest
import torch

import izzyviz

PDF_MAGIC = b"%PDF"


def assert_is_pdf(path):
    """A written file exists, is non-empty, and is actually a PDF."""
    assert os.path.exists(path), f"{path} was not created"
    assert os.path.getsize(path) > 0, f"{path} is empty"
    with open(path, "rb") as handle:
        assert handle.read(4) == PDF_MAGIC, f"{path} is not a PDF"


class TestPublicApi:
    """The exported surface is inconsistent -- see the maintenance report, finding M8.

    The same concepts carry different names across entry points:
    labels are x_labels/y_labels, source_tokens/target_tokens, or tokens;
    data is matrix, attention_matrix, attn1/attn2, matrices, or
    attentions_over_time. These tests pin the surface as it is today so
    that any future normalisation is a deliberate, visible change.
    """

    def test_all_exports_are_importable(self):
        for name in izzyviz.__all__:
            assert hasattr(izzyviz, name), f"{name} is exported but missing"

    def test_every_export_is_callable(self):
        for name in izzyviz.__all__:
            assert callable(getattr(izzyviz, name))


class TestVisualizeAttentionMatrix:
    def test_writes_a_pdf(self, attention, tokens, tmp_path):
        out = tmp_path / "matrix.pdf"
        izzyviz.visualize_attention_matrix(
            attention, x_labels=tokens, y_labels=tokens, save_path=str(out)
        )
        assert_is_pdf(str(out))

    def test_accepts_a_numpy_array(self, rng, tokens, tmp_path):
        matrix = rng.random((8, 8))
        out = tmp_path / "numpy.pdf"
        izzyviz.visualize_attention_matrix(
            torch.tensor(matrix, dtype=torch.float32),
            x_labels=tokens,
            y_labels=tokens,
            save_path=str(out),
        )
        assert_is_pdf(str(out))

    def test_non_square_matrix(self, rng, tmp_path):
        matrix = torch.tensor(rng.random((4, 6)), dtype=torch.float32)
        out = tmp_path / "nonsquare.pdf"
        izzyviz.visualize_attention_matrix(
            matrix,
            x_labels=[f"c{i}" for i in range(6)],
            y_labels=[f"r{i}" for i in range(4)],
            save_path=str(out),
        )
        assert_is_pdf(str(out))

    @pytest.mark.parametrize("size", [1, 2, 16])
    def test_various_sizes(self, rng, size, tmp_path):
        matrix = torch.tensor(rng.random((size, size)), dtype=torch.float32)
        labels = [f"t{i}" for i in range(size)]
        out = tmp_path / f"size{size}.pdf"
        izzyviz.visualize_attention_matrix(
            matrix, x_labels=labels, y_labels=labels, save_path=str(out)
        )
        assert_is_pdf(str(out))

    def test_uniform_matrix_does_not_crash(self, tokens, tmp_path):
        """Degenerate input: every cell equal, so there is no maximum."""
        matrix = torch.full((8, 8), 0.125)
        out = tmp_path / "uniform.pdf"
        izzyviz.visualize_attention_matrix(
            matrix, x_labels=tokens, y_labels=tokens, save_path=str(out)
        )
        assert_is_pdf(str(out))


class TestEncoderDecoder:
    DEFAULT_NAMES = {
        "cross_attention": "cross_attention_heatmap.pdf",
        "encoder_self_attention": "encoder_self_attention_heatmap.pdf",
        "decoder_self_attention": "decoder_self_attention_heatmap.pdf",
    }

    @pytest.mark.parametrize("use_case", sorted(DEFAULT_NAMES))
    def test_honours_save_path(self, attention, tokens, use_case, tmp_path, monkeypatch):
        """Regression: each use_case used to overwrite save_path with its own
        default filename, so the caller's path was ignored and a PDF landed in
        the current working directory instead."""
        monkeypatch.chdir(tmp_path)
        out = tmp_path / "chosen" / "encdec.pdf"
        out.parent.mkdir()
        izzyviz.visualize_attention_encoder_decoder(
            attention,
            encoder_tokens=tokens,
            decoder_tokens=tokens,
            use_case=use_case,
            save_path=str(out),
        )
        assert_is_pdf(str(out))
        assert not (tmp_path / self.DEFAULT_NAMES[use_case]).exists()

    def test_default_save_path_is_unchanged(self, attention, tokens, tmp_path, monkeypatch):
        """Callers that never passed save_path still get the same filename."""
        monkeypatch.chdir(tmp_path)
        izzyviz.visualize_attention_encoder_decoder(
            attention, encoder_tokens=tokens, decoder_tokens=tokens
        )
        assert_is_pdf(str(tmp_path / "cross_attention_heatmap.pdf"))


class TestRegionDetection:
    def test_find_attention_regions_returns_rectangles(self, attention):
        regions = izzyviz.find_attention_regions_with_merging(attention.numpy())
        assert regions is not None

    def test_visualize_with_detected_regions(self, attention, tokens, tmp_path):
        out = tmp_path / "regions.pdf"
        izzyviz.visualize_attention_with_detected_regions(
            attention, source_tokens=tokens, target_tokens=tokens, save_path=str(out)
        )
        assert_is_pdf(str(out))


class TestComparison:
    def test_compare_two_attentions_with_circles(self, rng, tokens, tmp_path):
        a = torch.softmax(torch.tensor(rng.random((8, 8)), dtype=torch.float32), -1)
        b = torch.softmax(torch.tensor(rng.random((8, 8)), dtype=torch.float32), -1)
        out = tmp_path / "compare.pdf"
        izzyviz.compare_two_attentions_with_circles(
            a, b, tokens=tokens, save_path=str(out)
        )
        assert_is_pdf(str(out))

    def test_identical_inputs_produce_a_figure(self, attention, tokens, tmp_path):
        """Zero difference everywhere is a boundary case for the colour scale."""
        out = tmp_path / "identical.pdf"
        izzyviz.compare_two_attentions_with_circles(
            attention, attention, tokens=tokens, save_path=str(out)
        )
        assert_is_pdf(str(out))


class TestStabilityAndEvolution:
    def test_stability_heatmap(self, rng, tokens, tmp_path):
        runs = torch.tensor(rng.random((5, 8, 8)), dtype=torch.float32)
        out = tmp_path / "stability.pdf"
        izzyviz.check_stability_heatmap_with_gradient_color(
            runs, x_labels=tokens, y_labels=tokens, save_path=str(out)
        )
        assert_is_pdf(str(out))

    def test_evolution_sparklines(self, rng, tokens, tmp_path):
        """Input is (timesteps, layers, heads, seq, seq); one layer/head is picked."""
        steps = torch.tensor(rng.random((3, 2, 2, 8, 8)), dtype=torch.float32)
        out = tmp_path / "evolution.pdf"
        izzyviz.visualize_attention_evolution_sparklines(
            steps, tokens=tokens, layer=0, head=0, save_path=str(out)
        )
        assert_is_pdf(str(out))
