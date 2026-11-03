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
from vllm.v1.core.sched import request_queue
from vllm.v1.core.sched.request_queue import (
    SchedulingPolicy,
    WaterfillRequestQueue,
    create_request_queue,
)
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


def _queue_with(*reqs: Request) -> WaterfillRequestQueue:
    queue = create_request_queue(SchedulingPolicy.WATERFILL)
    assert isinstance(queue, WaterfillRequestQueue)
    queue.set_now(NOW)
    for req in reqs:
        queue.add_request(req)
    return queue


def test_queue_orders_by_ratio_with_unrated_last():
    a, b, c, d = create_requests(num_requests=4)
    _set(a, rate=10.0, age_s=4.0, tokens=30)  # 0.75
    _set(b, rate=10.0, age_s=4.0, tokens=10)  # 0.25
    _set(c, rate=-1.0, age_s=9.0, tokens=0)  # unrated: inf
    _set(d, rate=10.0, age_s=1.0, tokens=0)  # 0.0
    queue = _queue_with(a, b, c, d)

    assert queue.peek_request() is d
    assert list(queue) == [d, b, a, c]
    assert [queue.pop_request() for _ in range(4)] == [d, b, a, c]
    assert not queue


def test_queue_ties_go_to_earlier_arrival():
    a, b = create_requests(num_requests=2)
    _set(a, rate=10.0, age_s=5.0, tokens=0)
    _set(b, rate=10.0, age_s=6.0, tokens=0)
    assert _queue_with(a, b).pop_request() is b


def test_queue_mutations():
    a, b, c = create_requests(num_requests=3)
    for req in (a, b, c):
        _set(req, rate=10.0, age_s=1.0, tokens=0)
    queue = _queue_with(a)
    queue.prepend_request(b)
    queue.prepend_requests([c])
    assert len(queue) == 3

    queue.remove_request(b)
    queue.remove_requests({a})
    assert list(queue) == [c]

    queue.remove_requests([c])
    assert not queue
    with pytest.raises(IndexError):
        queue.pop_request()
    with pytest.raises(IndexError):
        queue.peek_request()


def _crossing_pair() -> tuple[Request, Request]:
    """Two requests at 1 tok/s whose ratios cross at t=10: `a` (10 tokens,
    arrived at t=0) is first before it and `b` (5 tokens, arrived at t=5) after."""
    a, b = create_requests(num_requests=2)
    for req, arrival, tokens in ((a, 0.0, 10), (b, 5.0, 5)):
        req.slo_rate = 1.0
        req.arrival_time = arrival
        while req.num_output_tokens < tokens:
            req.append_output_token_ids(1)
    return a, b


def test_unpinned_queue_peek_and_pop_can_disagree(monkeypatch):
    """Without a pinned clock, the request that `peek` returns need not be the
    one that `pop` removes."""
    a, b = _crossing_pair()
    clock = iter([9.9, 10.1])
    monkeypatch.setattr(request_queue.time, "time", lambda: next(clock))
    queue = WaterfillRequestQueue()
    queue.add_request(a)
    queue.add_request(b)

    assert queue.peek_request() is a
    assert queue.pop_request() is b


def test_pinned_clock_makes_peek_and_pop_agree(monkeypatch):
    a, b = _crossing_pair()
    # The wall clock must not be consulted while the queue is pinned.
    monkeypatch.setattr(request_queue.time, "time", lambda: pytest.fail("clock read"))
    queue = WaterfillRequestQueue()
    queue.add_request(a)
    queue.add_request(b)

    queue.set_now(9.9)
    assert queue.peek_request() is a
    assert queue.pop_request() is a
    assert queue.pop_request() is b

    queue.set_now(None)
    queue.add_request(a)
    monkeypatch.setattr(request_queue.time, "time", lambda: 123.0)
    assert queue.peek_request() is a
