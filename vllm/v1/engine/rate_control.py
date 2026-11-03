# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-request output-rate SLO and paced delivery.

A client declares the output rate it can consume with `vllm_xargs`, which arrives
in `SamplingParams.extra_args`:

- `target_rate` (tokens/s): the SLO rate. The `waterfill` scheduling policy reads
  it, and delivery is paced to it.
- `chunk_size` (tokens): the most tokens delivered in one chunk when paced.
- `pace` (bool): set to false to declare an SLO but receive output at full speed.
"""

import asyncio
import time
from typing import Any

from vllm.sampling_params import SamplingParams

# Default number of tokens merged into one delivered chunk.
DEFAULT_CHUNK_SIZE = 16


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


def resolve_chunk_size(params: Any) -> int:
    """Return the most tokens delivered per chunk, or the default."""
    if not isinstance(params, SamplingParams):
        return DEFAULT_CHUNK_SIZE
    extra = params.extra_args
    if not extra:
        return DEFAULT_CHUNK_SIZE
    size = extra.get("chunk_size", DEFAULT_CHUNK_SIZE)
    if isinstance(size, int) and size > 0:
        return size
    return DEFAULT_CHUNK_SIZE


def resolve_pace(params: Any) -> bool:
    """Return whether delivery should be paced to the target rate.

    Pacing is on unless `pace` is explicitly false. The OpenAI `vllm_xargs`
    field has no bool type, so a JSON `false` arrives as the int `0`, which
    is why `0` also disables pacing. Any other value keeps pacing on.
    """
    if not isinstance(params, SamplingParams):
        return True
    extra = params.extra_args
    if not extra:
        return True
    pace = extra.get("pace", True)
    return pace is not False and pace != 0


class RatePacer:
    """Spaces delivery of a request's output to a target token rate.

    Delivering `n` tokens pushes the next delivery `n / rate` seconds later. The
    first delivery is never delayed. A consumer that has fallen behind schedule
    is not given a burst to catch up: the schedule restarts from now.
    """

    def __init__(self, rate: float) -> None:
        self._seconds_per_token = 1.0 / rate
        self._next: float | None = None

    async def wait(self) -> None:
        """Sleep until the next delivery is due."""
        now = time.monotonic()
        if self._next is None or self._next <= now:
            self._next = now
            return
        await asyncio.sleep(self._next - now)

    def advance(self, num_tokens: int) -> None:
        """Schedule the next delivery after `num_tokens` were delivered."""
        if num_tokens > 0:
            base = self._next if self._next is not None else time.monotonic()
            self._next = base + num_tokens * self._seconds_per_token
