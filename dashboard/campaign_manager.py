"""
Session-scoped campaign manager.
Coordinates the launch, monitoring, and cancellation of experiment campaigns.

Uses a module-level singleton (not st.session_state) to store thread state,
because background threads cannot safely access st.session_state.
"""

import threading
import time
from typing import Optional, List

from dashboard.log_capture import ThreadSafeLogBuffer
from dashboard.experiment_wrappers import run_multiple_experiments


class Campaign:
    """
    Represents a single campaign (one or more experiments run serially).
    """

    def __init__(self, codes: List[str]):
        self.codes = list(codes)
        self.log_buffer = ThreadSafeLogBuffer(max_lines=5000)
        self.stop_flag = threading.Event()
        self.thread: Optional[threading.Thread] = None

        self._lock = threading.Lock()
        self._state = "idle"                # idle | running | done | aborted | failed
        self._started_at: Optional[float] = None
        self._finished_at: Optional[float] = None
        self._summary: Optional[dict] = None
        self._current_experiment: Optional[str] = None
        self._error: Optional[str] = None

    def start(self):
        """Starts the campaign in a background thread."""
        with self._lock:
            if self._state == "running":
                return
            self._state = "running"
            self._started_at = time.time()
            self._finished_at = None
            self._summary = None
            self.stop_flag.clear()
            self.log_buffer.clear()

        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        """Internal thread target."""
        try:
            summary = run_multiple_experiments(
                self.codes,
                self.log_buffer,
                stop_flag=self.stop_flag,
                also_print=False,
            )
            with self._lock:
                self._summary = summary
                self._finished_at = time.time()
                if summary.get("aborted"):
                    self._state = "aborted"
                elif summary["failed"] > 0 and summary["succeeded"] == 0:
                    self._state = "failed"
                else:
                    self._state = "done"
        except Exception as e:
            import traceback
            with self._lock:
                self._state = "failed"
                self._finished_at = time.time()
                self._error = f"{type(e).__name__}: {e}"
                self._summary = {
                    "total": len(self.codes),
                    "completed": 0,
                    "succeeded": 0,
                    "failed": len(self.codes),
                    "aborted": False,
                    "per_experiment": [],
                    "total_duration_seconds": 0.0,
                    "fatal_error": self._error,
                    "traceback": traceback.format_exc(),
                }

    def request_stop(self):
        """Requests cancellation. Current experiment finishes; subsequent ones abort."""
        self.stop_flag.set()
        self.log_buffer.write("\n[UTILISATEUR] Demande d'annulation recue.\n")

    def get_snapshot(self) -> dict:
        """Returns a UI-friendly snapshot of the campaign state."""
        with self._lock:
            elapsed = 0.0
            if self._started_at is not None:
                end = self._finished_at or time.time()
                elapsed = end - self._started_at

            return {
                "state": self._state,
                "codes": list(self.codes),
                "elapsed_seconds": elapsed,
                "summary": self._summary,
                "error": self._error,
                "started_at": self._started_at,
                "finished_at": self._finished_at,
            }

    def get_logs(self, tail: Optional[int] = None) -> List[str]:
        """Returns log lines (all or last N)."""
        if tail:
            return self.log_buffer.get_tail(tail)
        return self.log_buffer.get_all()

    def is_running(self) -> bool:
        with self._lock:
            return self._state == "running"


# ---------------------------------------------------------------------------
# Global campaign singleton (survives Streamlit reruns)
# ---------------------------------------------------------------------------

_current_campaign: Optional[Campaign] = None
_registry_lock = threading.Lock()


def get_current_campaign() -> Optional[Campaign]:
    """Returns the currently registered campaign, or None."""
    with _registry_lock:
        return _current_campaign


def start_new_campaign(codes: List[str]) -> Campaign:
    """
    Starts a new campaign. If one is already running, raises RuntimeError.
    """
    global _current_campaign
    with _registry_lock:
        if _current_campaign is not None and _current_campaign.is_running():
            raise RuntimeError(
                "Une campagne est deja en cours d'execution. "
                "Attendez qu'elle se termine ou annulez-la."
            )
        _current_campaign = Campaign(codes)
    _current_campaign.start()
    return _current_campaign


def clear_finished_campaign():
    """Clears the reference to a finished campaign."""
    global _current_campaign
    with _registry_lock:
        if _current_campaign is not None and not _current_campaign.is_running():
            _current_campaign = None