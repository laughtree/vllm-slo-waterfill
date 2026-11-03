# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for per-request output-rate SLOs and paced delivery.

Time is faked wherever delivery is paced, so the tests neither sleep nor
depend on the speed of the machine.
"""

import asyncio
from types import SimpleNamespace

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.outputs import STREAM_FINISHED, CompletionOutput, RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.v1.engine import rate_control
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.output_processor import (
    RatedRequestOutputCollector,
    RequestOutputCollector,
    create_output_collector,
)
from vllm.v1.engine.rate_control import (
    DEFAULT_CHUNK_SIZE,
    RatePacer,
    resolve_chunk_size,
    resolve_pace,
    resolve_target_rate,
)


def _output(
    tokens: list[int], *, finished: bool = False, request_id: str = "r0"
) -> RequestOutput:
    """Build a delta `RequestOutput` carrying `tokens`."""
    completion = CompletionOutput(
        index=0,
        text="".join(str(t) for t in tokens),
        token_ids=tokens,
        cumulative_logprob=None,
        logprobs=None,
        finish_reason="stop" if finished else None,
    )
    return RequestOutput(
        request_id=request_id,
        prompt=None,
        prompt_token_ids=None,
        prompt_logprobs=None,
        outputs=[completion],
        finished=finished,
    )


def _token_ids(output: RequestOutput) -> list[int]:
    return list(output.outputs[0].token_ids)


def _rated(chunk_size: int = 4, target_rate: float = 10.0):
    return RatedRequestOutputCollector(
        RequestOutputKind.DELTA, "r0", chunk_size, target_rate
    )


def _drain(collector) -> list[list[int]]:
    chunks = []
    while (output := collector.get_nowait()) is not None:
        chunks.append(_token_ids(output))
    return chunks


class FakeClock:
    """Replaces the `time` and `asyncio` modules seen by `rate_control` so that
    sleeping only moves a fake clock."""

    def __init__(self, monkeypatch):
        self.now = 100.0
        self.sleeps: list[float] = []
        monkeypatch.setattr(
            rate_control, "time", SimpleNamespace(monotonic=lambda: self.now)
        )
        monkeypatch.setattr(rate_control, "asyncio", SimpleNamespace(sleep=self.sleep))

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def clock(monkeypatch) -> FakeClock:
    return FakeClock(monkeypatch)


# --- request parameters -------------------------------------------------------


def test_resolve_target_rate():
    assert resolve_target_rate(None) is None
    assert resolve_target_rate(PoolingParams()) is None
    assert resolve_target_rate(SamplingParams()) is None
    assert resolve_target_rate(SamplingParams(extra_args={})) is None
    for invalid in (0, -5, "fast", None):
        params = SamplingParams(extra_args={"target_rate": invalid})
        assert resolve_target_rate(params) is None
    assert resolve_target_rate(SamplingParams(extra_args={"target_rate": 10})) == 10.0
    assert resolve_target_rate(SamplingParams(extra_args={"target_rate": 3.5})) == 3.5


def test_resolve_chunk_size():
    assert resolve_chunk_size(PoolingParams()) == DEFAULT_CHUNK_SIZE
    assert resolve_chunk_size(SamplingParams()) == DEFAULT_CHUNK_SIZE
    assert resolve_chunk_size(SamplingParams(extra_args={})) == DEFAULT_CHUNK_SIZE
    for invalid in (0, -1, "big"):
        params = SamplingParams(extra_args={"chunk_size": invalid})
        assert resolve_chunk_size(params) == DEFAULT_CHUNK_SIZE
    assert resolve_chunk_size(SamplingParams(extra_args={"chunk_size": 8})) == 8


@pytest.mark.parametrize(
    "extra_args, expected",
    [
        (None, True),
        ({}, True),
        ({"pace": True}, True),
        ({"pace": False}, False),
        ({"pace": 0}, False),
        ({"pace": 0.0}, False),
        ({"pace": 1}, True),
        ({"pace": "off"}, True),
    ],
)
def test_resolve_pace(extra_args, expected):
    assert resolve_pace(SamplingParams(extra_args=extra_args)) is expected


def test_resolve_pace_ignores_non_sampling_params():
    assert resolve_pace(PoolingParams()) is True


def test_json_false_reaches_the_engine_as_zero():
    """`vllm_xargs` has no bool type, so `false` arrives as `0`. `resolve_pace`
    must treat that as off."""
    request = ChatCompletionRequest.model_validate(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "vllm_xargs": {"pace": False, "target_rate": 10},
        }
    )
    assert request.vllm_xargs is not None
    assert request.vllm_xargs["pace"] == 0
    assert resolve_pace(SamplingParams(extra_args=request.vllm_xargs)) is False


# --- collector choice ---------------------------------------------------------


def test_collector_without_target_rate_is_the_base_collector():
    collector = create_output_collector(SamplingParams(), "r0")
    assert type(collector) is RequestOutputCollector


def test_collector_with_target_rate_is_rated():
    params = SamplingParams(
        output_kind=RequestOutputKind.DELTA,
        extra_args={"target_rate": 5, "chunk_size": 8},
    )
    collector = create_output_collector(params, "r0")
    assert isinstance(collector, RatedRequestOutputCollector)
    assert collector.target_rate == 5.0
    assert collector._chunk_size == 8


@pytest.mark.parametrize(
    "params",
    [
        SamplingParams(
            n=2, output_kind=RequestOutputKind.DELTA, extra_args={"target_rate": 5}
        ),
        SamplingParams(
            output_kind=RequestOutputKind.CUMULATIVE, extra_args={"target_rate": 5}
        ),
        SamplingParams(
            output_kind=RequestOutputKind.FINAL_ONLY, extra_args={"target_rate": 5}
        ),
        PoolingParams(),
    ],
    ids=["n=2", "cumulative", "final-only", "pooling"],
)
def test_collector_is_the_base_collector_where_pacing_is_unsupported(params):
    assert type(create_output_collector(params, "r0")) is RequestOutputCollector


# --- RatedRequestOutputCollector ------------------------------------------------


def test_chunks_respect_chunk_size():
    collector = _rated(chunk_size=4)
    for token in range(10):
        collector.put(_output([token]))

    chunks = _drain(collector)

    assert [len(c) for c in chunks] == [4, 4, 2]
    assert [t for chunk in chunks for t in chunk] == list(range(10))


def test_oversized_output_is_returned_whole():
    collector = _rated(chunk_size=4)
    collector.put(_output(list(range(7))))

    assert _drain(collector) == [list(range(7))]


def test_finished_output_follows_the_buffered_tokens():
    collector = _rated(chunk_size=4)
    collector.put(_output([1]))
    collector.put(_output([2]))
    collector.put(_output([3], finished=True))

    first = collector.get_nowait()
    assert first is not None
    assert _token_ids(first) == [1, 2]
    assert not first.finished

    last = collector.get_nowait()
    assert last is not None
    assert _token_ids(last) == [3]
    assert last.finished

    assert collector.get_nowait() is None


def test_exception_follows_the_buffered_tokens():
    collector = _rated()
    collector.put(_output([1]))
    collector.put(ValueError("boom"))

    first = collector.get_nowait()
    assert first is not None
    assert _token_ids(first) == [1]
    with pytest.raises(ValueError, match="boom"):
        collector.get_nowait()


def test_stream_finished_sentinel_is_terminal():
    collector = _rated()
    collector.put(_output([1]))
    collector.put(STREAM_FINISHED)

    first = collector.get_nowait()
    assert first is not None
    assert _token_ids(first) == [1]
    assert collector.get_nowait() is STREAM_FINISHED


def test_get_blocks_until_put():
    async def run() -> RequestOutput:
        collector = _rated()

        async def delayed_put() -> None:
            await asyncio.sleep(0)
            collector.put(_output([99]))

        task = asyncio.create_task(delayed_put())
        output = await collector.get()
        await task
        assert isinstance(output, RequestOutput)
        return output

    assert _token_ids(asyncio.run(run())) == [99]


def test_ready_event_follows_the_buffer():
    collector = _rated()
    assert not collector.ready.is_set()

    collector.put(_output([1]))
    assert collector.ready.is_set()
    collector.get_nowait()
    assert not collector.ready.is_set()

    collector.put(_output([2]))
    collector.put(_output([3], finished=True))
    collector.get_nowait()
    assert collector.ready.is_set()  # the finished output is still pending
    collector.get_nowait()
    assert not collector.ready.is_set()


def test_base_collector_still_merges_into_one_output():
    collector = RequestOutputCollector(RequestOutputKind.DELTA, "r0")
    collector.put(_output([1]))
    collector.put(_output([2]))

    output = collector.get_nowait()
    assert output is not None
    assert _token_ids(output) == [1, 2]
    assert collector.get_nowait() is None


# --- RatePacer ----------------------------------------------------------------


def test_first_wait_does_not_sleep(clock):
    pacer = RatePacer(10.0)
    asyncio.run(pacer.wait())
    assert clock.sleeps == []


def test_wait_after_advance_sleeps_for_the_delivered_tokens(clock):
    async def run() -> None:
        pacer = RatePacer(10.0)
        await pacer.wait()
        pacer.advance(8)  # 8 tokens at 10 tok/s: the next delivery is due in 0.8 s
        await pacer.wait()

    asyncio.run(run())
    assert clock.sleeps == [pytest.approx(0.8)]


def test_delivery_schedule_accumulates(clock):
    async def run() -> None:
        pacer = RatePacer(10.0)
        await pacer.wait()
        pacer.advance(4)
        await pacer.wait()
        pacer.advance(6)
        await pacer.wait()

    asyncio.run(run())
    assert clock.sleeps == [pytest.approx(0.4), pytest.approx(0.6)]


def test_no_catch_up_burst_when_behind_schedule(clock):
    async def run() -> None:
        pacer = RatePacer(10.0)
        await pacer.wait()
        pacer.advance(4)  # due in 0.4 s
        clock.now += 1.0  # the consumer arrives late
        await pacer.wait()
        pacer.advance(10)
        await pacer.wait()

    asyncio.run(run())
    # No sleep for the late delivery, and the schedule restarts from that point.
    assert clock.sleeps == [pytest.approx(1.0)]


def test_advance_by_no_tokens_keeps_the_schedule(clock):
    async def run() -> None:
        pacer = RatePacer(10.0)
        await pacer.wait()
        pacer.advance(0)
        await pacer.wait()

    asyncio.run(run())
    assert clock.sleeps == []


# --- AsyncLLM.generate ------------------------------------------------------


def _generate(collector, sampling_params) -> list[list[int]]:
    """Run the real `AsyncLLM.generate` loop over `collector`, whose producer has
    already put all of its output. Returns the delivered chunks."""

    async def add_request(*args, **kwargs):
        return collector

    engine = SimpleNamespace(add_request=add_request)

    async def run() -> list[list[int]]:
        chunks = []
        async for output in AsyncLLM.generate(engine, "prompt", sampling_params, "r0"):
            if output.outputs[0].token_ids:
                chunks.append(_token_ids(output))
        return chunks

    return asyncio.run(run())


def _fill(collector, num_tokens: int) -> None:
    for token in range(num_tokens):
        collector.put(_output([token]))
    collector.put(_output([], finished=True))


def test_generate_delivers_at_the_target_rate(clock):
    rate, chunk_size, num_tokens = 20.0, 4, 40
    collector = _rated(chunk_size, rate)
    _fill(collector, num_tokens)
    start = clock.now

    chunks = _generate(collector, SamplingParams(extra_args={"target_rate": rate}))

    assert [len(c) for c in chunks] == [chunk_size] * (num_tokens // chunk_size)
    assert [t for chunk in chunks for t in chunk] == list(range(num_tokens))
    assert clock.now - start == pytest.approx(num_tokens / rate)


def test_generate_without_pacing_does_not_wait(clock):
    collector = _rated(4, 20.0)
    _fill(collector, 12)
    params = SamplingParams(extra_args={"target_rate": 20.0, "pace": False})

    chunks = _generate(collector, params)

    assert [t for chunk in chunks for t in chunk] == list(range(12))
    assert clock.sleeps == []


def test_generate_without_target_rate_does_not_wait(clock):
    collector = RequestOutputCollector(RequestOutputKind.DELTA, "r0")
    collector.put(_output(list(range(5))))
    collector.put(_output([], finished=True))

    chunks = _generate(collector, SamplingParams())

    assert chunks == [list(range(5))]
    assert clock.sleeps == []
