# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for `policy="waterfill"` and `preemption_victim_policy="satisfaction"`.

Requests come from `create_requests` without a `target_rate`, so tests set
`slo_rate` and `arrival_time` directly and pass an explicit `now` instead of
reading the wall clock.
"""

import math

import pytest

from vllm.config import SchedulerConfig
from vllm.sampling_params import SamplingParams
from vllm.v1.request import Request

from .utils import create_requests

NOW = 1000.0


def _set(req: Request, *, rate: float, age_s: float, tokens: int) -> None:
    """Pin a request's SLO rate, age at `NOW` and generated-token count."""
    req.slo_rate = rate
    req.arrival_time = NOW - age_s
    while req.num_output_tokens < tokens:
        req.append_output_token_ids(1)


def test_satisfaction_ratio_formula():
    (req,) = create_requests(num_requests=1)
    _set(req, rate=10.0, age_s=2.0, tokens=10)
    # 10 tokens / (2 s * 10 tok/s) = 0.5
    assert req.satisfaction_ratio(NOW) == pytest.approx(0.5)


def test_satisfaction_ratio_unrated_is_inf_and_fresh_is_zero():
    (req,) = create_requests(num_requests=1)
    assert req.slo_rate == -1.0
    assert math.isinf(req.satisfaction_ratio(NOW))

    req.slo_rate = 10.0
    assert req.satisfaction_ratio(req.arrival_time) == 0.0
    assert req.satisfaction_ratio(req.arrival_time - 1.0) == 0.0


def test_slo_rate_read_from_target_rate():
    sampling_params = SamplingParams(max_tokens=8, extra_args={"target_rate": 15})
    req = Request(
        request_id="x",
        prompt_token_ids=[1, 2],
        sampling_params=sampling_params,
        pooling_params=None,
    )
    assert req.slo_rate == 15.0


@pytest.mark.parametrize("target_rate", [0, -3, "fast", None])
def test_invalid_target_rate_is_undeclared(target_rate):
    sampling_params = SamplingParams(
        max_tokens=8, extra_args={"target_rate": target_rate}
    )
    req = Request(
        request_id="x",
        prompt_token_ids=[1, 2],
        sampling_params=sampling_params,
        pooling_params=None,
    )
    assert req.slo_rate == -1.0


def test_waterfill_victim_policy_defaults_to_satisfaction():
    assert SchedulerConfig.default_factory(
        policy="waterfill"
    ).resolved_preemption_victim_policy() == ("satisfaction")
    assert SchedulerConfig.default_factory(
        policy="waterfill", preemption_victim_policy="tail"
    ).resolved_preemption_victim_policy() == ("tail")
