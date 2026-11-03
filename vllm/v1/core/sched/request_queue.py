# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import heapq
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable, Iterator
from enum import Enum

from vllm.v1.request import Request


class SchedulingPolicy(Enum):
    """Enum for scheduling policies."""

    FCFS = "fcfs"
    PRIORITY = "priority"
    WATERFILL = "waterfill"


class RequestQueue(ABC):
    """Abstract base class for request queues."""

    @abstractmethod
    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to the policy."""
        pass

    @abstractmethod
    def pop_request(self) -> Request:
        """Pop a request from the queue according to the policy."""
        pass

    @abstractmethod
    def peek_request(self) -> Request:
        """Peek at the request at the front of the queue without removing it."""
        pass

    @abstractmethod
    def prepend_request(self, request: Request) -> None:
        """Prepend a request to the front of the queue."""
        pass

    @abstractmethod
    def prepend_requests(self, requests: Iterable[Request]) -> None:
        """Prepend all requests from another queue to the front of this
        queue."""
        pass

    @abstractmethod
    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        pass

    @abstractmethod
    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        pass

    @abstractmethod
    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Get number of requests in queue."""
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to the policy."""
        pass


class FCFSRequestQueue(deque[Request], RequestQueue):
    """A first-come-first-served queue that supports deque operations."""

    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to FCFS policy."""
        self.append(request)

    def pop_request(self) -> Request:
        """Pop a request from the queue according to FCFS policy."""
        return self.popleft()

    def peek_request(self) -> Request:
        """Peek at the next request in the queue without removing it."""
        if not self:
            raise IndexError("peek from an empty queue")
        return self[0]

    def prepend_request(self, request: Request) -> None:
        """Prepend a request to the front of the queue."""
        self.appendleft(request)

    def prepend_requests(self, requests: Iterable[Request]) -> None:
        """Prepend all requests from another queue to the front of this
        queue.

        Note: The requests will be prepended in reverse order of their
        appearance in the `requests` queue.
        """
        self.extendleft(requests)

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        self.remove(request)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        requests_to_remove = set(requests)
        filtered_requests = [req for req in self if req not in requests_to_remove]
        # deque does not support in-place filtering, so we need to clear
        # and extend
        self.clear()
        self.extend(filtered_requests)

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return len(self) > 0

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return super().__len__()

    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to FCFS policy."""
        return super().__iter__()


class PriorityRequestQueue(RequestQueue):
    """A priority queue that supports heap operations.

    Respects the ordering defined in the Request class, where
    requests with a smaller value of `priority` are processed first.
    If multiple requests have the same priority, the one with the earlier
    `arrival_time` is processed first.
    """

    def __init__(self) -> None:
        self._heap: list[Request] = []

    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to priority policy."""
        heapq.heappush(self._heap, request)

    def pop_request(self) -> Request:
        """Pop a request from the queue according to priority policy."""
        if not self._heap:
            raise IndexError("pop from empty heap")
        return heapq.heappop(self._heap)

    def peek_request(self) -> Request:
        """Peek at the next request in the queue without removing it."""
        if not self._heap:
            raise IndexError("peek from empty heap")
        return self._heap[0]

    def prepend_request(self, request: Request) -> None:
        """Add a request to the queue according to priority policy.

        Note: In a priority queue, there is no concept of prepending to the
        front. Requests are ordered by (priority, arrival_time)."""
        self.add_request(request)

    def prepend_requests(self, requests: Iterable[Request]) -> None:
        """Add all requests from another queue according to priority policy.

        Note: In a priority queue, there is no concept of prepending to the
        front. Requests are ordered by (priority, arrival_time)."""
        for request in requests:
            self.add_request(request)

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        self._heap.remove(request)
        heapq.heapify(self._heap)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        requests_to_remove = requests if isinstance(requests, set) else set(requests)
        self._heap = [r for r in self._heap if r not in requests_to_remove]
        heapq.heapify(self._heap)

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return bool(self._heap)

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return len(self._heap)

    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to priority policy."""
        heap_copy = self._heap[:]
        while heap_copy:
            yield heapq.heappop(heap_copy)


class WaterfillRequestQueue(RequestQueue):
    """A queue that admits the request furthest below its SLO first.

    Requests are ordered by ascending `Request.satisfaction_ratio`, with
    `arrival_time` and then `request_id` deciding any ties. Time spent queued
    counts against a request and a preempted request keeps its generated tokens,
    so its ratio carries over. Requests with no SLO rate have an infinite ratio
    and come last.

    The ratio depends on the clock, so the queue is not heap-backed. The
    scheduler peeks a request, allocates blocks for it and only then pops it,
    expecting the same request back. If the clock were re-read on every call,
    two ratios could cross in between, so the wrong request would be removed.
    The scheduler therefore pins the clock once per step with `set_now`. Without
    a pinned clock each call reads `time.time()`.
    """

    def __init__(self) -> None:
        self._requests: list[Request] = []
        self._now: float | None = None

    def set_now(self, now: float | None) -> None:
        """Pin the clock used for ordering, or unpin it with None."""
        self._now = now

    def _clock(self) -> float:
        return self._now if self._now is not None else time.time()

    @staticmethod
    def _key_at(now: float, request: Request) -> tuple[float, float, str]:
        return (
            request.satisfaction_ratio(now),
            request.arrival_time,
            request.request_id,
        )

    def add_request(self, request: Request) -> None:
        """Add a request to the queue."""
        self._requests.append(request)

    def pop_request(self) -> Request:
        """Pop the request with the lowest satisfaction ratio."""
        if not self._requests:
            raise IndexError("pop from an empty queue")
        now = self._clock()
        request = min(self._requests, key=lambda r: self._key_at(now, r))
        self._requests.remove(request)
        return request

    def peek_request(self) -> Request:
        """Peek at the request with the lowest satisfaction ratio."""
        if not self._requests:
            raise IndexError("peek from an empty queue")
        now = self._clock()
        return min(self._requests, key=lambda r: self._key_at(now, r))

    def prepend_request(self, request: Request) -> None:
        """Add a request to the queue.

        Note: There is no front of the queue, requests are ordered by their
        satisfaction ratio."""
        self.add_request(request)

    def prepend_requests(self, requests: Iterable[Request]) -> None:
        """Add all the requests to the queue.

        Note: There is no front of the queue, requests are ordered by their
        satisfaction ratio."""
        self._requests.extend(requests)

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        self._requests.remove(request)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        requests_to_remove = requests if isinstance(requests, set) else set(requests)
        self._requests = [r for r in self._requests if r not in requests_to_remove]

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return bool(self._requests)

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return len(self._requests)

    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue by ascending satisfaction ratio."""
        now = self._clock()
        return iter(sorted(self._requests, key=lambda r: self._key_at(now, r)))


def create_request_queue(policy: SchedulingPolicy) -> RequestQueue:
    """Create request queue based on scheduling policy."""
    if policy == SchedulingPolicy.PRIORITY:
        return PriorityRequestQueue()
    elif policy == SchedulingPolicy.FCFS:
        return FCFSRequestQueue()
    elif policy == SchedulingPolicy.WATERFILL:
        return WaterfillRequestQueue()
    else:
        raise ValueError(f"Unknown scheduling policy: {policy}")
