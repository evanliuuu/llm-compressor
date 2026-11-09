"""
HIGGS: Heuristic ILP-Guided Grouped Scheme Mixed-Precision Quantization

MSE collection and high-level API for ILP-based mixed-precision quantization.
"""

import os
import time
from typing import Dict, List, Optional, Union

import torch
from compressed_tensors.compressors.format import _flatten_formats
from compressed_tensors.config import CompressionFormat
from compressed_tensors.entrypoints.convert import (
    Converter,
    build_inverse_weight_maps,
)
from compressed_tensors.quantization import (
    QuantizationConfig,
    QuantizationScheme,
    QuantizationStatus,
    preset_name_to_scheme,
)
from compressed_tensors.utils import match_quantizable_tensors
from compressed_tensors.utils.safetensors_load import (
    get_checkpoint_files,
    get_weight_map,
    load_tensors_from_inverse_weight_map,
)
from loguru import logger

from llmcompressor.entrypoints.higgs.ilp_solver import (
    solve_ilp_mixed_precision,
)
from llmcompressor.entrypoints.higgs.utils import (
    UNQUANTIZED_SCHEME,
    compute_fused_layer_mse,
    compute_heuristic_alphas,
    compute_layer_mse,
    generate_config_groups,
    infer_scheme_format,
)
from llmcompressor.entrypoints.model_free.converter import split_fused_moe_experts
from llmcompressor.entrypoints.model_free.microscale import (
    get_fused_names,
    is_microscale_scheme,
)

__all__ = [
    "HiggsMSECollectorConverter",
    "get_higgs_config",
]


# ---------------------------------------------------------------------------
# Phase 1: MSE collection and ILP solving
# ---------------------------------------------------------------------------

_SCHEME_FIELDS = (
    "format",
    "weights",
    "input_activations",
    "output_activations",
    "targets",
)


def higgs_distinguishable_key(scheme: QuantizationScheme):
    """Weight args and weight/activation bitwidths, which are all the ILP sees.

    Layer MSE is fake-quantized from ``scheme.weights`` only
    (``compute_fused_layer_mse``). Bitwidth constraints read
    ``weights.num_bits`` and ``input_activations.num_bits`` (``create_config``).
    """
    weights = scheme.weights
    activations = scheme.input_activations
    weight_bits = 16.0 if weights is None else float(weights.num_bits)
    act_bits = 16.0 if activations is None else float(activations.num_bits)
    return (weights, weight_bits, act_bits)


def _indistinguishable_candidate_message(
    left: QuantizationScheme, right: QuantizationScheme
) -> str:
    fields = [
        name for name in _SCHEME_FIELDS if getattr(left, name) != getattr(right, name)
    ]
    differing = ", ".join(fields) if fields else "unlisted fields"
    return (
        "HIGGS candidates have the same weight args and bitwidths, so the ILP "
        f"cannot tell them apart, but they differ in {differing}. "
        "Pass only one of these candidates."
    )


class HiggsMSECollectorConverter(Converter):
    """
    First-pass converter that collects per-layer MSE data, solves the ILP,
    and produces an optimal QuantizationConfig. Does NOT modify tensors.
    """

    def __init__(
        self,
        candidate_schemes: List[Union[str, QuantizationScheme]],
        targets: Union[str, List[str]] = "Linear",
        ignore: List[str] = None,
        device: Union[str, torch.device] = None,
        target_avg_bitwidth: Optional[float] = None,
        target_avg_act_bitwidth: Optional[float] = None,
        allow_unquantized: bool = False,
    ):
        self.targets = targets if isinstance(targets, list) else [targets]
        self.ignore = ignore or ["lm_head"]
        self.device = device or torch.device("cpu")
        self.target_avg_bitwidth = target_avg_bitwidth
        self.target_avg_act_bitwidth = target_avg_act_bitwidth
        self.allow_unquantized = allow_unquantized

        self.candidate_schemes = self._resolve_schemes(candidate_schemes)

        self.mse_matrix: Dict[str, Dict[str, float]] = {}
        self.layer_sizes: Dict[str, int] = {}
        self.ilp_solution: Dict[str, str] = {}
        self.optimal_config: Optional[QuantizationConfig] = None
        self._fused_partners: Optional[Dict[str, set[str]]] = None

    def _canonical_candidate(
        self, candidate: Union[str, QuantizationScheme]
    ) -> tuple[str, QuantizationScheme]:
        """Readable label plus the scheme HIGGS will compare and emit."""
        if isinstance(candidate, str):
            label = candidate
            scheme = preset_name_to_scheme(candidate, targets=self.targets)
        elif isinstance(candidate, QuantizationScheme):
            scheme = candidate
            if scheme.weights:
                label = f"W{scheme.weights.num_bits}"
                label += (
                    f"A{scheme.input_activations.num_bits}"
                    if scheme.input_activations
                    else "A16"
                )
            else:
                label = "scheme"
        else:
            raise ValueError(f"Invalid scheme type: {type(candidate)}")
        scheme = scheme.model_copy(
            update={
                "targets": self.targets,
                "format": infer_scheme_format(scheme),
            }
        )
        return label, scheme

    def _resolve_schemes(
        self, schemes: List[Union[str, QuantizationScheme]]
    ) -> Dict[str, QuantizationScheme]:
        resolved = {}
        for candidate in schemes:
            label, scheme = self._canonical_candidate(candidate)
            incoming = higgs_distinguishable_key(scheme)
            duplicate = False
            label_taken = False
            max_suffix = 0
            prefix = f"{label}_"
            for kept_label, kept in resolved.items():
                if scheme == kept:
                    duplicate = True
                    break
                if incoming == higgs_distinguishable_key(kept):
                    raise ValueError(_indistinguishable_candidate_message(scheme, kept))
                if kept_label == label:
                    label_taken = True
                    continue
                tail = kept_label[len(prefix) :]
                if kept_label.startswith(prefix) and tail.isdigit():
                    max_suffix = max(max_suffix, int(tail))
            if duplicate:
                continue
            unique = f"{label}_{max_suffix + 1}" if label_taken else label
            resolved[unique] = scheme
        return resolved

    def validate(self, tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Validate HIGGS inputs without materializing checkpoint tensors."""
        tensors = split_fused_moe_experts(tensors)
        quantizable_tensors = list(
            match_quantizable_tensors(tensors, self.ignore, self.targets)
        )
        self._validate_tensor_shapes(tensors, quantizable_tensors)

        if not quantizable_tensors:
            logger.warning(
                f"No quantizable tensors. Targets: {self.targets}, "
                f"Ignore: {self.ignore}"
            )
        return tensors

    def _validate_tensor_shapes(
        self,
        tensors: Dict[str, torch.Tensor],
        quantizable_tensors: List[tuple[str, str]],
    ) -> None:
        unsupported_tensors = [
            (tensor_name, tuple(tensors[tensor_name].shape))
            for _, tensor_name in quantizable_tensors
            if tensors[tensor_name].ndim > 2
        ]
        if unsupported_tensors:
            tensor_details = "\n".join(
                f"  - {name}: {shape}" for name, shape in unsupported_tensors
            )
            raise ValueError(
                "HIGGS only supports 2-D weight matrices. The following "
                "3-D or higher-dimensional tensors matched the quantization "
                f"targets:\n{tensor_details}\n"
                "Add these tensors to the HIGGS ignore list if they should not be "
                "quantized, or add their fused MoE layout to "
                "split_fused_moe_experts so HIGGS can handle them as individual "
                "2-D expert weights."
            )

    def process(self, tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute MSE for each candidate scheme; return tensors unchanged."""
        tensors = split_fused_moe_experts(tensors)
        logger.info(f"Collecting MSE data from shard with {len(tensors)} tensors")

        quantizable_tensors = list(
            match_quantizable_tensors(tensors, self.ignore, self.targets)
        )
        tensor_to_module = {
            tensor_name: module_name for module_name, tensor_name in quantizable_tensors
        }
        matrix_tensor_names = {
            tensor_name
            for tensor_name in tensor_to_module
            if tensors[tensor_name].ndim == 2
        }
        fused_groups = get_fused_names(matrix_tensor_names)

        for tensor_name in matrix_tensor_names:
            module_name = tensor_to_module[tensor_name]
            self.layer_sizes[module_name] = tensors[tensor_name].numel()
            self.mse_matrix.setdefault(module_name, {})

        for scheme_name, scheme in self.candidate_schemes.items():
            fused_tensor_names = set()
            if is_microscale_scheme(scheme):
                for fused_group in fused_groups:
                    names = list(fused_group.values())
                    fused_mse = compute_fused_layer_mse(
                        {name: tensors[name] for name in names}, scheme, self.device
                    )
                    for name, mse in fused_mse.items():
                        self.mse_matrix[tensor_to_module[name]][scheme_name] = mse
                    fused_tensor_names.update(names)

            for tensor_name in matrix_tensor_names - fused_tensor_names:
                module_name = tensor_to_module[tensor_name]
                self.mse_matrix[module_name][scheme_name] = compute_layer_mse(
                    tensors[tensor_name], scheme, self.device
                )

        layers_processed = 0
        for module_name, tensor_name in quantizable_tensors:
            # ``Linear`` is an all-inclusive name target for checkpoint-only
            # conversion, so model-specific non-matrix weights can still
            # match it (for example norm vectors). HIGGS MSE is defined for
            # 2-D weight matrices, so leave vectors out of the ILP.
            if tensors[tensor_name].ndim != 2:
                continue

            if self.allow_unquantized:
                self.mse_matrix[module_name][UNQUANTIZED_SCHEME] = 0.0
            layers_processed += 1

        logger.info(f"Processed {layers_processed} layers in this shard")
        return tensors

    def create_config(self) -> QuantizationConfig:
        """Solve ILP after all shards processed and return optimized config."""
        if self.optimal_config is not None:
            return self.optimal_config

        logger.info(
            f"Solving ILP for {len(self.mse_matrix)} layers "
            f"with {len(self.candidate_schemes)} candidate schemes"
        )

        alphas = compute_heuristic_alphas(
            list(self.mse_matrix.keys()), self.layer_sizes
        )

        fused_groups = [
            [tensor_name.removesuffix(".weight") for tensor_name in group.values()]
            for group in get_fused_names(f"{name}.weight" for name in self.mse_matrix)
        ]

        def _bitwidths(attr):
            return {
                name: float(getattr(s, attr).num_bits) if getattr(s, attr) else 16.0
                for name, s in self.candidate_schemes.items()
            }

        has_bw = self.target_avg_bitwidth is not None
        has_act = self.target_avg_act_bitwidth is not None

        ilp_candidate_schemes = list(self.candidate_schemes.keys())
        if self.allow_unquantized:
            ilp_candidate_schemes.append(UNQUANTIZED_SCHEME)

        weight_bitwidths = _bitwidths("weights") if has_bw else None
        activation_bitwidths = _bitwidths("input_activations") if has_act else None
        if self.allow_unquantized:
            if weight_bitwidths is not None:
                weight_bitwidths[UNQUANTIZED_SCHEME] = 16.0
            if activation_bitwidths is not None:
                activation_bitwidths[UNQUANTIZED_SCHEME] = 16.0

        self.ilp_solution = solve_ilp_mixed_precision(
            mse_matrix=self.mse_matrix,
            alphas=alphas,
            candidate_schemes=ilp_candidate_schemes,
            fused_groups=fused_groups,
            target_avg_bitwidth=self.target_avg_bitwidth,
            layer_param_counts=self.layer_sizes if (has_bw or has_act) else None,
            scheme_bitwidths=weight_bitwidths,
            target_avg_act_bitwidth=self.target_avg_act_bitwidth,
            scheme_act_bitwidths=activation_bitwidths,
        )

        config_groups = generate_config_groups(
            self.ilp_solution, self.candidate_schemes
        )
        formats = {
            CompressionFormat(s.format) for s in config_groups.values() if s.format
        }
        self.optimal_config = QuantizationConfig(
            config_groups=config_groups,
            format=_flatten_formats(formats).value,
            quantization_status=QuantizationStatus.COMPRESSED,
            ignore=self.ignore,
        )
        logger.info(
            f"ILP optimization complete: {len(config_groups)} config groups generated"
        )
        return self.optimal_config

    def get_dependencies(self, weight_name: str) -> set[str]:
        if not any(
            is_microscale_scheme(scheme) for scheme in self.candidate_schemes.values()
        ):
            return set()

        if self._fused_partners is None:
            raise ValueError(
                "HIGGS fused dependency resolution requires the complete "
                "checkpoint weight map for microscale schemes. This is supplied "
                "automatically by get_higgs_config(); direct collector use must "
                "provide checkpoint names before building inverse weight maps."
            )
        return set(self._fused_partners.get(weight_name, ()))

    def set_weight_names(self, weight_names: set[str]) -> None:
        """Precompute fused dependencies from the complete checkpoint index."""
        self._fused_partners = {}
        for fused_group in get_fused_names(weight_names):
            primary, *partners = fused_group.values()
            self._fused_partners[primary] = set(partners)


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------


def get_higgs_config(
    model_stub: Union[str, os.PathLike],
    candidate_schemes: List[Union[str, QuantizationScheme]],
    targets: Union[str, List[str]] = "Linear",
    ignore: List[str] = None,
    target_avg_bitwidth: Optional[float] = None,
    target_avg_act_bitwidth: Optional[float] = None,
    device: Optional[Union[str, torch.device]] = None,
    allow_unquantized: bool = False,
    return_collector: bool = False,
) -> QuantizationConfig:
    """
    Compute optimal mixed-precision config via ILP on per-layer MSE.

    Reads model weights (model-free, no GPU model load), evaluates each
    candidate scheme's MSE on every layer, and solves an ILP to minimize
    weighted MSE under a bitwidth budget.

    The returned QuantizationConfig can be applied via oneshot() or
    model_free_ptq().

    Args:
        model_stub: HuggingFace model ID or path to model directory
        candidate_schemes: List of quantization schemes to choose from
        targets: Layer types to quantize (e.g., "Linear")
        ignore: Layers to skip (e.g., ["lm_head"])
        target_avg_bitwidth: Optional constraint on average weight bitwidth
        target_avg_act_bitwidth: Optional constraint on average activation bitwidth
        device: Device for MSE computation (GPU recommended)
        allow_unquantized: Allow the ILP to leave layers unquantized at 16 bits
        return_collector: Return the collector alongside the optimized config.

    Returns:
        QuantizationConfig with optimized config_groups
    """
    if device is None:
        device = torch.device("cuda" if torch.accelerator.is_available() else "cpu")

    if ignore is None:
        ignore = ["lm_head"]

    if (
        len(candidate_schemes) + int(allow_unquantized) > 1
        and target_avg_bitwidth is None
    ):
        logger.warning(
            "Using multiple candidate schemes without target_avg_bitwidth constraint. "
            "ILP will select the scheme with lowest MSE for each layer. "
            "Set target_avg_bitwidth to enable true mixed-precision optimization."
        )

    logger.info("=" * 80)
    logger.info("HIGGS: Collecting MSE data and solving ILP...")
    logger.info("=" * 80)
    start = time.time()

    collector = HiggsMSECollectorConverter(
        candidate_schemes=candidate_schemes,
        targets=targets,
        ignore=ignore,
        device=device,
        target_avg_bitwidth=target_avg_bitwidth,
        target_avg_act_bitwidth=target_avg_act_bitwidth,
        allow_unquantized=allow_unquantized,
    )

    model_files = get_checkpoint_files(model_stub)
    weight_map = get_weight_map(model_files)
    collector.set_weight_names(set(weight_map))
    inverse_weight_maps = build_inverse_weight_maps(
        weight_map=weight_map,
        model_files=model_files,
        converters=[collector],
    )

    shard_names = [f for f in model_files if f.endswith("safetensors")]
    logger.info(f"Processing {len(shard_names)} shards for MSE collection...")

    for shard_name in shard_names:
        if shard_name not in inverse_weight_maps:
            continue
        tensors = load_tensors_from_inverse_weight_map(
            inverse_weight_maps[shard_name], torch.device("meta")
        )
        collector.validate(tensors)

    mse_start = time.time()
    for shard_name in shard_names:
        if shard_name not in inverse_weight_maps:
            logger.warning(f"Shard {shard_name} not in inverse_weight_maps, skipping")
            continue

        logger.info(f"Processing shard: {shard_name}")
        tensors = load_tensors_from_inverse_weight_map(
            inverse_weight_maps[shard_name], device
        )
        collector.process(tensors)

    mse_time = time.time() - mse_start
    logger.info(f"MSE collection completed in {mse_time:.2f}s")

    ilp_start = time.time()
    optimal_config = collector.create_config()
    ilp_time = time.time() - ilp_start

    total_time = time.time() - start
    logger.info(
        f"HIGGS complete in {total_time:.2f}s "
        f"(MSE: {mse_time:.2f}s, ILP: {ilp_time:.2f}s): "
        f"{len(optimal_config.config_groups)} config groups"
    )

    if return_collector:
        return optimal_config, collector
    return optimal_config
