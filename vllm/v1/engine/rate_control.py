# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-request output-rate SLO and paced delivery.

A client declares the output rate it can consume (tokens/s) with
`vllm_xargs={"target_rate": ...}`, which arrives in `SamplingParams.extra_args`.
"""

from typing import Any

from vllm.sampling_params import SamplingParams


def resolve_target_rate(params: Any) -> float | None:
    """Return the declared target output rate in tokens/s, or None.

    Only `SamplingParams` carries `extra_args`, so any other params type, a
    missing key, or a non-positive or non-numeric value gives None.
    """
    if not isinstance(params, SamplingParams):
        return None
    extra = params.extra_args
    if not extra:
        return None
    rate = extra.get("target_rate")
    if isinstance(rate, int | float) and rate > 0:
        return float(rate)
    return None
