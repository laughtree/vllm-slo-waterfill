# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for `policy="waterfill"` and `preemption_victim_policy="satisfaction"`.

Requests come from `create_requests` without a `target_rate`, so tests set
`slo_rate` and `arrival_time` directly and pass an explicit `now` instead of
reading the wall clock.
"""

import itertools
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
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus

from .utils import create_requests, create_scheduler

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


def _step(scheduler: Scheduler, requests: list[Request]):
    """Run `schedule()` then `update_from_output()`, sampling one token for each
    request that was scheduled."""
    output = scheduler.schedule()
    req_ids = [r.request_id for r in requests]
    scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
            sampled_token_ids=[
                [1] if req_id in output.num_scheduled_tokens else []
                for req_id in req_ids
            ],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )
    return output


@pytest.fixture
def frozen_clock(monkeypatch):
    """Make the scheduler and the queues read `NOW` from `time.time()`."""
    monkeypatch.setattr(request_queue.time, "time", lambda: NOW)


def test_waterfill_scheduler_wiring():
    scheduler = create_scheduler(scheduling_policy="waterfill")
    assert scheduler.preemption_victim_policy == "satisfaction"
    assert isinstance(scheduler.waiting, WaterfillRequestQueue)
    assert isinstance(scheduler.kv_holding_waiting, WaterfillRequestQueue)

    scheduler = create_scheduler()
    assert not isinstance(scheduler.waiting, WaterfillRequestQueue)


def test_scheduler_pins_the_clock_of_both_waiting_queues(frozen_clock):
    scheduler = create_scheduler(scheduling_policy="waterfill")
    assert isinstance(scheduler.waiting, WaterfillRequestQueue)
    assert isinstance(scheduler.kv_holding_waiting, WaterfillRequestQueue)
    assert scheduler.waiting._now is None
    assert scheduler.kv_holding_waiting._now is None

    scheduler.schedule()

    assert scheduler.waiting._now == NOW
    assert scheduler.kv_holding_waiting._now == NOW


def test_running_requests_are_sorted_by_ratio_each_step(frozen_clock):
    scheduler = create_scheduler(scheduling_policy="waterfill")
    reqs = create_requests(num_requests=3, num_tokens=8)
    for req in reqs:
        scheduler.add_request(req)
    _step(scheduler, reqs)
    a, b, c = reqs
    _set(a, rate=10.0, age_s=4.0, tokens=30)  # 0.75
    _set(b, rate=10.0, age_s=4.0, tokens=10)  # 0.25
    _set(c, rate=10.0, age_s=4.0, tokens=20)  # 0.5

    scheduler.schedule()

    assert scheduler.running == [b, c, a]


@pytest.mark.parametrize(
    "ahead, expected_victim",
    [
        # `first` was admitted first and is the one furthest ahead of its SLO.
        ("first", "first"),
        # `second` is the furthest ahead and the most recently admitted.
        ("second", "second"),
        # A request without a declared rate has an infinite ratio.
        ("first-unrated", "first"),
    ],
)
@pytest.mark.parametrize("policy", ["waterfill", "fcfs"])
def test_satisfaction_victim_is_the_request_furthest_ahead(
    frozen_clock, policy, ahead, expected_victim
):
    """`second` needs a new block while none is free, so one running request is
    preempted. Under "tail" that would always be `second`."""
    block_size = 16
    scheduler = create_scheduler(
        scheduling_policy=policy,
        max_num_seqs=3,
        max_num_batched_tokens=200,
        num_blocks=6,  # 5 usable blocks
        block_size=block_size,
    )
    # `create_scheduler` has no victim policy argument.
    scheduler.preemption_victim_policy = "satisfaction"
    first, second = create_requests(
        num_requests=2, num_tokens=2 * block_size, req_ids=["first", "second"]
    )
    # `first` gets a third block on its second step and `second` is admitted
    # next to it, which leaves no free block.
    scheduler.add_request(first)
    _step(scheduler, [first])
    scheduler.add_request(second)
    _step(scheduler, [first, second])
    assert [r.request_id for r in scheduler.running] == ["first", "second"]
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == 0

    behind, front = (
        dict(rate=100.0, age_s=10.0),
        dict(rate=1.0, age_s=1.0),
    )
    rated = {"first": front, "second": behind}
    if ahead == "second":
        rated = {"first": behind, "second": front}
    for req in (first, second):
        _set(req, tokens=req.num_output_tokens, **rated[req.request_id])
    if ahead == "first-unrated":
        first.slo_rate = -1.0

    scheduler.schedule()

    survivor = "second" if expected_victim == "first" else "first"
    assert scheduler.requests[expected_victim].status == RequestStatus.PREEMPTED
    assert scheduler.requests[survivor].status == RequestStatus.RUNNING


def _no_request_in_two_places(scheduler: Scheduler) -> None:
    running = [r.request_id for r in scheduler.running]
    waiting = [r.request_id for r in scheduler.waiting]
    holding = [r.request_id for r in scheduler.kv_holding_waiting]
    everything = running + waiting + holding
    assert len(everything) == len(set(everything)), (running, waiting, holding)


def _admit_crossing_pair(monkeypatch, *, pin_clock: bool) -> set[str]:
    """Admit two waiting requests whose ratios cross at t=10 with a clock that
    alternates between 9.9 and 10.1 on every `time.time()` call.

    Unpinned, the queue sees one request first at peek and the other at pop.
    Returns the ids of the requests that were ever scheduled.
    """
    scheduler = create_scheduler(scheduling_policy="waterfill")
    if not pin_clock:
        monkeypatch.setattr(WaterfillRequestQueue, "set_now", lambda self, now: None)
    a, b = _crossing_pair()  # carry output tokens, like preempted requests
    for req in (a, b):
        scheduler.add_request(req)
    ticks = itertools.cycle([9.9, 10.1])
    monkeypatch.setattr(request_queue.time, "time", lambda: next(ticks))

    scheduled: set[str] = set()
    for _ in range(6):
        output = _step(scheduler, [a, b])
        scheduled.update(output.num_scheduled_tokens)
        _no_request_in_two_places(scheduler)
    return scheduled


def test_scheduler_does_not_double_admit_when_ratios_cross_mid_step(monkeypatch):
    scheduled = _admit_crossing_pair(monkeypatch, pin_clock=True)
    assert scheduled == {"0", "1"}


def test_unpinned_clock_breaks_admission_when_ratios_cross(monkeypatch):
    """Sensitivity check for the test above: without `set_now`, the same
    scenario must fail, otherwise that test proves nothing."""
    with pytest.raises((AssertionError, KeyError)):
        _admit_crossing_pair(monkeypatch, pin_clock=False)
