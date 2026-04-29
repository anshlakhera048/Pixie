"""Load testing tool for the Pixie API.

Simulates N concurrent users making repeated API calls.
Measures latency distribution, error rate, and throughput.

Can run against a live server or in-process.
"""

from __future__ import annotations

import asyncio
import logging
import statistics
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

log = logging.getLogger(__name__)


@dataclass
class LoadTestConfig:
    """Configuration for a load test run."""

    base_url: str = "http://127.0.0.1:8000"
    concurrent_users: int = 5
    requests_per_user: int = 10
    api_key: str = ""
    endpoint: str = "/chat"
    payload: dict[str, Any] = field(default_factory=lambda: {"message": "hello"})
    timeout_seconds: float = 30.0


@dataclass
class LoadTestResult:
    """Results from a load test run."""

    total_requests: int = 0
    successful: int = 0
    failed: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    status_codes: dict[int, int] = field(default_factory=dict)

    @property
    def error_rate(self) -> float:
        return self.failed / self.total_requests if self.total_requests > 0 else 0.0

    @property
    def throughput_rps(self) -> float:
        return self.total_requests / self.duration_seconds if self.duration_seconds > 0 else 0.0

    @property
    def avg_latency_ms(self) -> float:
        return statistics.mean(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def p50_ms(self) -> float:
        return statistics.median(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def p95_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        sorted_l = sorted(self.latencies_ms)
        idx = int(len(sorted_l) * 0.95)
        return sorted_l[min(idx, len(sorted_l) - 1)]

    @property
    def p99_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        sorted_l = sorted(self.latencies_ms)
        idx = int(len(sorted_l) * 0.99)
        return sorted_l[min(idx, len(sorted_l) - 1)]

    @property
    def max_latency_ms(self) -> float:
        return max(self.latencies_ms) if self.latencies_ms else 0.0

    def summary(self) -> str:
        lines = [
            f"\n  {'='*50}",
            f"  LOAD TEST RESULTS",
            f"  {'='*50}",
            f"  Total requests: {self.total_requests}",
            f"  Successful: {self.successful}",
            f"  Failed: {self.failed}",
            f"  Error rate: {self.error_rate:.1%}",
            f"  Duration: {self.duration_seconds:.1f}s",
            f"  Throughput: {self.throughput_rps:.1f} req/s",
            f"  {'-'*50}",
            f"  Latency:",
            f"    Avg:  {self.avg_latency_ms:.1f} ms",
            f"    P50:  {self.p50_ms:.1f} ms",
            f"    P95:  {self.p95_ms:.1f} ms",
            f"    P99:  {self.p99_ms:.1f} ms",
            f"    Max:  {self.max_latency_ms:.1f} ms",
        ]
        if self.status_codes:
            lines.append(f"  {'-'*50}")
            lines.append(f"  Status codes:")
            for code, count in sorted(self.status_codes.items()):
                lines.append(f"    {code}: {count}")
        if self.errors:
            lines.append(f"  {'-'*50}")
            lines.append(f"  Sample errors (first 5):")
            for err in self.errors[:5]:
                lines.append(f"    - {err[:100]}")
        lines.append(f"  {'='*50}\n")
        return "\n".join(lines)


async def run_load_test(config: LoadTestConfig | None = None) -> LoadTestResult:
    """Execute a load test against the Pixie API.

    Spawns concurrent_users tasks, each making requests_per_user requests.
    """
    if config is None:
        config = LoadTestConfig()

    result = LoadTestResult()
    result.total_requests = config.concurrent_users * config.requests_per_user

    # Prepare headers
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"

    latencies: list[float] = []
    errors: list[str] = []
    status_codes: dict[int, int] = {}
    lock = asyncio.Lock()

    async def _worker(user_id: int) -> None:
        """Simulate a single user making sequential requests."""
        async with httpx.AsyncClient(
            base_url=config.base_url,
            headers=headers,
            timeout=config.timeout_seconds,
        ) as client:
            for req_num in range(config.requests_per_user):
                start = time.perf_counter()
                try:
                    if config.endpoint.startswith("/") and "GET" not in config.endpoint.upper():
                        resp = await client.post(config.endpoint, json=config.payload)
                    else:
                        resp = await client.get(config.endpoint)

                    elapsed_ms = (time.perf_counter() - start) * 1000
                    async with lock:
                        latencies.append(elapsed_ms)
                        status_codes[resp.status_code] = status_codes.get(resp.status_code, 0) + 1
                        if resp.status_code >= 400:
                            errors.append(f"User {user_id} req {req_num}: HTTP {resp.status_code}")

                except Exception as exc:
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    async with lock:
                        latencies.append(elapsed_ms)
                        errors.append(f"User {user_id} req {req_num}: {type(exc).__name__}: {exc}")

    # Run all users concurrently
    start_time = time.perf_counter()
    tasks = [asyncio.create_task(_worker(i)) for i in range(config.concurrent_users)]
    await asyncio.gather(*tasks)
    result.duration_seconds = time.perf_counter() - start_time

    # Aggregate
    result.latencies_ms = latencies
    result.errors = errors
    result.status_codes = status_codes
    result.successful = sum(v for k, v in status_codes.items() if k < 400)
    result.failed = result.total_requests - result.successful

    return result


async def quick_load_test(
    base_url: str = "http://127.0.0.1:8000",
    users: int = 5,
    requests: int = 10,
    api_key: str = "",
) -> LoadTestResult:
    """Convenience wrapper for quick load testing."""
    config = LoadTestConfig(
        base_url=base_url,
        concurrent_users=users,
        requests_per_user=requests,
        api_key=api_key,
    )
    return await run_load_test(config)
