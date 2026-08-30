import threading
import time
from collections import deque


class SlidingWindowLimiter:
    """Thread-safe limiter allowing at most max_calls within any rolling period_seconds window.

    Unlike a fixed delay between calls, this only blocks when actually near the limit -- so light,
    spaced-out traffic (a normal visitor asking one question) pays no artificial tax, while a burst
    (indexing many chunks back to back, or several visitors arriving together) gets throttled to
    stay within the limit instead of erroring.
    """

    def __init__(self, max_calls: int, period_seconds: float):
        self.max_calls = max_calls
        self.period = period_seconds
        self.calls: deque[float] = deque()
        self.lock = threading.Lock()

    def acquire(self) -> None:
        with self.lock:
            now = time.monotonic()
            self._prune(now)
            if len(self.calls) >= self.max_calls:
                wait = self.period - (now - self.calls[0])
                if wait > 0:
                    time.sleep(wait)
                now = time.monotonic()
                self._prune(now)
            self.calls.append(now)

    def _prune(self, now: float) -> None:
        while self.calls and now - self.calls[0] >= self.period:
            self.calls.popleft()


# Voyage's free tier caps at 3 requests/minute -- shared across every embed/rerank call site.
voyage_limiter = SlidingWindowLimiter(max_calls=3, period_seconds=60)
