"""
Asynchronous experiment execution manager.
Wraps blocking experiment functions in a background thread with
progress reporting via a thread-safe status dict.

Prevents the Streamlit UI from freezing during long campaigns (E1-E10).
"""

import threading
import time
import traceback
from typing import Callable, Optional


class ExperimentJob:
    """
    Represents a single asynchronous experiment execution.
    Thread-safe status tracking.
    """

    def __init__(self, name: str, target: Callable, args: tuple = (), kwargs: dict = None):
        self.name = name
        self.target = target
        self.args = args
        self.kwargs = kwargs or {}
        self._lock = threading.Lock()
        self._status = {
            "state": "idle",         # idle | running | done | failed
            "message": "",
            "started_at": None,
            "finished_at": None,
            "elapsed": 0.0,
            "error": None,
            "traceback": None,
            "logs": [],
        }
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    def start(self):
        """Starts the experiment in a background thread."""
        with self._lock:
            if self._status["state"] == "running":
                return
            self._status["state"] = "running"
            self._status["started_at"] = time.time()
            self._status["message"] = f"Lancement de {self.name}..."
            self._status["logs"] = []
            self._status["error"] = None

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------
    def _run(self):
        """Internal thread target."""
        try:
            self._log(f"Debut de l'execution : {self.name}")
            self.target(*self.args, **self.kwargs)
            with self._lock:
                self._status["state"] = "done"
                self._status["finished_at"] = time.time()
                self._status["elapsed"] = self._status["finished_at"] - self._status["started_at"]
                self._status["message"] = f"{self.name} termine avec succes"
            self._log(f"[OK] {self.name} termine")
        except Exception as e:
            tb = traceback.format_exc()
            with self._lock:
                self._status["state"] = "failed"
                self._status["finished_at"] = time.time()
                self._status["elapsed"] = self._status["finished_at"] - (self._status["started_at"] or time.time())
                self._status["error"] = str(e)
                self._status["traceback"] = tb
                self._status["message"] = f"Echec: {type(e).__name__}: {e}"
            self._log(f"[ERREUR] {type(e).__name__}: {e}")

    # ------------------------------------------------------------------
    def _log(self, message: str):
        """Thread-safe log append."""
        with self._lock:
            ts = time.strftime("%H:%M:%S")
            self._status["logs"].append(f"[{ts}] {message}")
            # Cap log length to prevent memory bloat
            if len(self._status["logs"]) > 500:
                self._status["logs"] = self._status["logs"][-500:]

    # ------------------------------------------------------------------
    def get_status(self) -> dict:
        """Returns a snapshot of the current status."""
        with self._lock:
            return {
                "name": self.name,
                "state": self._status["state"],
                "message": self._status["message"],
                "elapsed": self._compute_elapsed(),
                "error": self._status["error"],
                "traceback": self._status["traceback"],
                "logs": list(self._status["logs"]),
            }

    def _compute_elapsed(self) -> float:
        if self._status["started_at"] is None:
            return 0.0
        end = self._status["finished_at"] or time.time()
        return end - self._status["started_at"]

    def is_running(self) -> bool:
        with self._lock:
            return self._status["state"] == "running"

    def is_done(self) -> bool:
        with self._lock:
            return self._status["state"] in ("done", "failed")


class JobRegistry:
    """
    Session-scoped registry of experiment jobs.
    """

    def __init__(self):
        self._jobs = {}
        self._lock = threading.Lock()

    def create_job(self, key: str, name: str, target: Callable,
                   args: tuple = (), kwargs: dict = None) -> ExperimentJob:
        """Creates or replaces a job under the given key."""
        with self._lock:
            job = ExperimentJob(name=name, target=target, args=args, kwargs=kwargs)
            self._jobs[key] = job
            return job

    def get_job(self, key: str) -> Optional[ExperimentJob]:
        with self._lock:
            return self._jobs.get(key)

    def list_active(self) -> list:
        with self._lock:
            return [
                (key, job.get_status())
                for key, job in self._jobs.items()
                if job.is_running()
            ]

    def clear_finished(self):
        with self._lock:
            self._jobs = {k: j for k, j in self._jobs.items() if j.is_running()}