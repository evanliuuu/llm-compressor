"""Regression tests for HIGGS candidate scheme registration."""

import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    preset_name_to_scheme,
)

from llmcompressor.entrypoints.higgs import HiggsMSECollectorConverter


def _group_w4(group_size):
    return QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=4,
            type="int",
            symmetric=True,
            strategy="group",
            group_size=group_size,
        ),
    )


def _channel_w4(symmetric):
    return QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=4,
            type="int",
            symmetric=symmetric,
            strategy="channel",
        ),
    )


def _sample_tensors():
    torch.manual_seed(0)
    return {
        f"model.layers.{i}.mlp.down_proj.weight": torch.randn(128, 128)
        for i in range(2)
    }


def test_unique_schemes_keep_readable_names():
    """Presets and unique bitwidth schemes stay keyed as W{n}A{m}."""
    custom_w4 = _group_w4(128)
    custom_w8 = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=8, type="int", symmetric=True, strategy="channel"
        ),
    )

    presets = HiggsMSECollectorConverter(
        candidate_schemes=["W4A16", "W8A16"],
        ignore=[],
    )
    readme = HiggsMSECollectorConverter(
        candidate_schemes=[custom_w4, "W8A16"],
        ignore=[],
    )
    objects = HiggsMSECollectorConverter(
        candidate_schemes=[custom_w4, custom_w8],
        ignore=[],
    )

    assert list(presets.candidate_schemes) == ["W4A16", "W8A16"]
    assert presets.candidate_schemes["W4A16"] == preset_name_to_scheme(
        "W4A16", targets=["Linear"]
    )
    assert list(readme.candidate_schemes) == ["W4A16", "W8A16"]
    assert readme.candidate_schemes["W4A16"] is custom_w4
    assert list(objects.candidate_schemes) == ["W4A16", "W8A16"]
    assert objects.candidate_schemes["W8A16"] is custom_w8


def test_keeps_w4a16_schemes_that_differ_by_group_size():
    group_32 = _group_w4(32)
    group_128 = _group_w4(128)
    collector = HiggsMSECollectorConverter(
        candidate_schemes=[group_32, group_128],
        ignore=[],
    )

    assert list(collector.candidate_schemes) == ["W4A16", "W4A16_1"]
    assert collector.candidate_schemes["W4A16"] is group_32
    assert collector.candidate_schemes["W4A16_1"] is group_128


def test_keeps_preset_and_custom_scheme_with_the_same_bitwidth_name():
    custom = _group_w4(32)
    collector = HiggsMSECollectorConverter(
        candidate_schemes=["W4A16", custom],
        ignore=[],
    )

    assert list(collector.candidate_schemes) == ["W4A16", "W4A16_1"]
    assert collector.candidate_schemes["W4A16"] == preset_name_to_scheme(
        "W4A16", targets=["Linear"]
    )
    assert collector.candidate_schemes["W4A16_1"] is custom


def test_keeps_w4a16_schemes_that_differ_by_symmetry():
    symmetric = _channel_w4(True)
    asymmetric = _channel_w4(False)
    collector = HiggsMSECollectorConverter(
        candidate_schemes=[symmetric, asymmetric],
        ignore=[],
    )

    assert list(collector.candidate_schemes) == ["W4A16", "W4A16_1"]
    assert collector.candidate_schemes["W4A16"] is symmetric
    assert collector.candidate_schemes["W4A16_1"] is asymmetric


def test_process_records_mse_for_every_colliding_candidate():
    collector = HiggsMSECollectorConverter(
        candidate_schemes=[_group_w4(32), _group_w4(128)],
        ignore=[],
    )

    collector.process(_sample_tensors())

    assert set(collector.candidate_schemes) == {"W4A16", "W4A16_1"}
    assert len(collector.mse_matrix) == 2
    for layer_mse in collector.mse_matrix.values():
        assert set(layer_mse) == {"W4A16", "W4A16_1"}
        assert layer_mse["W4A16"] < layer_mse["W4A16_1"]


def test_create_config_sees_both_colliding_candidates(require_higgs_cbc):
    collector = HiggsMSECollectorConverter(
        candidate_schemes=[_group_w4(32), _group_w4(128)],
        ignore=[],
    )

    collector.process(_sample_tensors())
    config = collector.create_config()

    assert set(collector.ilp_solution.values()) == {"W4A16"}
    group_sizes = {group.weights.group_size for group in config.config_groups.values()}
    assert group_sizes == {32}
