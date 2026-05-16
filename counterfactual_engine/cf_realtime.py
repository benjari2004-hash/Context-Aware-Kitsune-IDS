"""
cf_realtime.py — Latency-bounded CF generation for real-time IDS use.

Wraps cf_generator passes with wall-clock time budgets.  Four modes:
    STRICT   2 ms  — Pass 1 only within budget
    NORMAL  10 ms  — Pass 1 + 2 within budget
    RELAXED 50 ms  — all three passes within budget
    BATCH   None   — no time limit, same three passes, no early exits
"""
import collections
import time

from .cf_generator import _pass1_search, _pass2_search, _pass3_temporal

BUDGETS = {
    "STRICT":  0.002,   # seconds
    "NORMAL":  0.010,
    "RELAXED": 0.050,
    "BATCH":   None,
}


class LatencyBoundedCFGenerator:
    """Generate counterfactual explanations under a wall-clock time budget."""

    def __init__(self, mode: str = "NORMAL"):
        if mode not in BUDGETS:
            raise ValueError(f"mode must be one of {list(BUDGETS)}")
        self.mode   = mode
        self.budget = BUDGETS[mode]     # seconds, or None for BATCH
        self.stats  = {
            "pass1_used": 0,
            "pass2_used": 0,
            "pass3_used": 0,
            "pass4_used": 0,    # always 0 after BUG3A fix; kept for schema compat
            "timeout":    0,
            "total":      0,
        }
        self._latencies: collections.deque = collections.deque(maxlen=10_000)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _record_latency(self, elapsed_ms: float) -> None:
        self._latencies.append(elapsed_ms)

    def _over_budget(self, t0: float) -> bool:
        if self.budget is None:
            return False
        return (time.perf_counter() - t0) >= self.budget

    def _run_pass1(self, score, risk, attack_type, freq, current_action):
        return _pass1_search(score, risk, attack_type, freq, current_action)

    def _run_pass2(self, score, risk, attack_type, freq, current_action):
        return _pass2_search(score, risk, attack_type, freq, current_action)

    def _run_pass3(self, score, risk, attack_type, freq, current_action):
        return _pass3_temporal(score, risk, attack_type, freq, current_action)

    # ------------------------------------------------------------------
    # Public generate
    # ------------------------------------------------------------------

    def generate(self, score: float, risk: float, attack_type: str,
                 freq: float, current_action: str):
        """
        Run passes in order, stopping at the first hit or budget expiry.

        Returns (cf_dict, pass_label, elapsed_ms).
        cf_dict["found"] is False when no flip was found within budget.
        """
        t0 = time.perf_counter()
        self.stats["total"] += 1

        # Pass 1 — single-feature flip
        cf = self._run_pass1(score, risk, attack_type, freq, current_action)
        elapsed = (time.perf_counter() - t0) * 1000

        if cf["found"]:
            self.stats["pass1_used"] += 1
            self._record_latency(elapsed)
            return cf, "pass1", elapsed

        # Check budget before Pass 2
        if self._over_budget(t0):
            self.stats["timeout"] += 1
            self._record_latency(elapsed)
            return cf, "timeout", elapsed

        # Pass 2 — override decomposition
        cf = self._run_pass2(score, risk, attack_type, freq, current_action)
        elapsed = (time.perf_counter() - t0) * 1000

        if cf["found"]:
            self.stats["pass2_used"] += 1
            self._record_latency(elapsed)
            return cf, "pass2", elapsed

        # Check budget before Pass 3
        if self._over_budget(t0):
            self.stats["timeout"] += 1
            self._record_latency(elapsed)
            return cf, "timeout", elapsed

        # Pass 3 — temporal CF
        cf = self._run_pass3(score, risk, attack_type, freq, current_action)
        elapsed = (time.perf_counter() - t0) * 1000

        # BUG3A FIX: only credit pass3_used when a CF was actually found.
        # The original incremented unconditionally for all bounded modes,
        # inflating coverage stats whenever Pass 3 returned found=False.
        self._record_latency(elapsed)
        if cf["found"]:
            self.stats["pass3_used"] += 1
        else:
            self.stats["timeout"] += 1
        return cf, "pass3", elapsed

    # ------------------------------------------------------------------
    # Stats / reporting
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        total = self.stats["total"]
        if total == 0:
            return {**self.stats, "coverage_pct": 0.0, "avg_latency_ms": 0.0}

        found = (
            self.stats["pass1_used"] +
            self.stats["pass2_used"] +
            self.stats["pass3_used"] +
            self.stats["pass4_used"]
        )
        coverage = found / total * 100
        avg_lat  = (sum(self._latencies) / len(self._latencies)
                    if self._latencies else 0.0)
        return {
            **self.stats,
            "coverage_pct":   round(coverage, 2),
            "avg_latency_ms": round(avg_lat, 4),
        }

    def reset_stats(self) -> None:
        for key in self.stats:
            self.stats[key] = 0
        self._latencies.clear()
