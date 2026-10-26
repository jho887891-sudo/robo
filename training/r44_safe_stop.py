"""Safe stop and atomic checkpoint helpers for the R44 training loop.

This module intentionally has no PyTorch dependency. The trainer supplies a
writer callback (normally torch.save(state, fp)), which makes its safety
behavior unit-testable on CPU without loading a model or CUDA.
"""
from __future__ import annotations

import os
import signal
import tempfile
import threading
from pathlib import Path
from types import FrameType
from typing import BinaryIO, Callable, Iterable, Optional

CheckpointWriter = Callable[[BinaryIO], None]


def configure_torch_memory_fraction(
    max_process_memory_gb: Optional[float],
    *,
    device: Optional[int] = None,
) -> dict:
    """Optionally cap this process's PyTorch CUDA caching allocator.

    This is NOT GPU compute isolation and NOT a hard cap on allocations made
    outside the PyTorch caching allocator. It must be called before model/CUDA
    allocations. No limit is applied when max_process_memory_gb is None.
    """
    if max_process_memory_gb is None:
        return {
            "enabled": False,
            "scope": "none",
            "compute_isolation": False,
            "hard_gpu_isolation": False,
        }
    if max_process_memory_gb <= 0:
        raise ValueError("max_process_memory_gb must be positive")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; cannot configure a process memory fraction")
    index = torch.cuda.current_device() if device is None else int(device)
    total_bytes = int(torch.cuda.get_device_properties(index).total_memory)
    requested_bytes = int(max_process_memory_gb * (1024**3))
    fraction = requested_bytes / total_bytes
    if not (0.0 < fraction <= 1.0):
        raise ValueError(
            "max_process_memory_gb %.3f exceeds visible device memory %.3f GiB"
            % (max_process_memory_gb, total_bytes / (1024**3))
        )
    torch.cuda.set_per_process_memory_fraction(fraction, device=index)
    return {
        "enabled": True,
        "requested_gb": float(max_process_memory_gb),
        "visible_device_total_gb": total_bytes / (1024**3),
        "fraction_of_visible_device": fraction,
        "device_index": index,
        "scope": "PyTorch CUDA caching allocator only",
        "compute_isolation": False,
        "hard_gpu_isolation": False,
    }


class R44StopController:
    """Observe manual stop requests and optional GPU free-memory guardrails.

    Signal handlers only set a plain flag. Checkpointing is deliberately left
    to the training loop at a safe optimizer-step boundary.

    If stop_file exists, the request remains active. Remove or rename that
    file explicitly before a deliberate resume; this prevents an accidental
    immediate restart after a stop request.
    """

    def __init__(
        self,
        stop_file: os.PathLike[str] | str,
        *,
        min_free_gpu_gb: Optional[float] = None,
        consecutive_low_memory_samples: int = 2,
    ) -> None:
        if min_free_gpu_gb is not None and min_free_gpu_gb <= 0:
            raise ValueError("min_free_gpu_gb must be positive")
        if consecutive_low_memory_samples < 1:
            raise ValueError("consecutive_low_memory_samples must be >= 1")
        self.stop_file = Path(stop_file)
        self.min_free_gpu_bytes = (
            None if min_free_gpu_gb is None else int(min_free_gpu_gb * (1024**3))
        )
        self.consecutive_low_memory_samples = consecutive_low_memory_samples
        self._event = threading.Event()
        self._reason: Optional[str] = None
        self._signal_number: Optional[int] = None
        self._old_handlers: dict[int, object] = {}
        self._low_memory_count = 0
        self._lock = threading.Lock()

    @property
    def stop_requested(self) -> bool:
        return self._event.is_set() or self._signal_number is not None

    @property
    def reason(self) -> Optional[str]:
        if self._reason is not None:
            return self._reason
        if self._signal_number is not None:
            try:
                return f"signal:{signal.Signals(self._signal_number).name}"
            except ValueError:
                return f"signal:{self._signal_number}"
        return None

    def request_stop(self, reason: str) -> None:
        """Idempotently request a stop; safe to call outside signal handlers."""
        if not self._event.is_set():
            with self._lock:
                if not self._event.is_set():
                    self._reason = reason
                    self._event.set()

    def _signal_handler(self, signum: int, _frame: Optional[FrameType]) -> None:
        # Do not acquire locks or perform I/O in a signal handler. The main
        # training loop converts this flag into a normal stop request later.
        self._signal_number = signum

    def install_signal_handlers(
        self, signals: Iterable[int] = (signal.SIGTERM, signal.SIGINT)
    ) -> None:
        """Install process signal handlers; must be called from main thread."""
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("signal handlers must be installed from the main thread")
        for signum in signals:
            if signum in self._old_handlers:
                continue
            self._old_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, self._signal_handler)

    def restore_signal_handlers(self) -> None:
        """Restore handlers previously replaced by this controller."""
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("signal handlers must be restored from the main thread")
        for signum, handler in self._old_handlers.items():
            signal.signal(signum, handler)
        self._old_handlers.clear()

    def poll_stop_file(self) -> bool:
        if self._signal_number is not None and not self._event.is_set():
            self.request_stop(self.reason or "signal")
        if self.stop_file.exists():
            self.request_stop(f"stop_file:{self.stop_file}")
        return self.stop_requested

    def observe_gpu_free_memory(self, free_bytes: int) -> bool:
        """Feed a fresh free-memory sample; returns whether stopping is requested.

        The caller controls sampling cadence (for example every 10 optimizer
        steps) and must not infer compute utilization from memory utilization.
        """
        if free_bytes < 0:
            self.request_stop("gpu_memory_probe_invalid")
            return True
        if self.min_free_gpu_bytes is None:
            return self.stop_requested
        if free_bytes < self.min_free_gpu_bytes:
            self._low_memory_count += 1
            if self._low_memory_count >= self.consecutive_low_memory_samples:
                self.request_stop(
                    "gpu_free_memory_below_threshold:"
                    f"{free_bytes / (1024**3):.3f}GiB<"
                    f"{self.min_free_gpu_bytes / (1024**3):.3f}GiB"
                )
        else:
            self._low_memory_count = 0
        return self.stop_requested

    def observe_cuda_free_memory(self) -> bool:
        """Query the current CUDA device and apply the configured memory guard.

        Call only from the actual training process, at a configurable safe
        polling cadence. A failed probe fails closed by requesting a safe stop.
        This guards memory availability, not GPU compute utilization.
        """
        if self.should_stop():
            return True
        if self.min_free_gpu_bytes is None:
            return False
        try:
            import torch

            if not torch.cuda.is_available():
                self.request_stop("cuda_unavailable_for_resource_guard")
                return True
            free_bytes, _total_bytes = torch.cuda.mem_get_info()
        except Exception as exc:
            self.request_stop(f"gpu_memory_probe_failed:{type(exc).__name__}")
            return True
        return self.observe_gpu_free_memory(int(free_bytes))

    def should_stop(self) -> bool:
        self.poll_stop_file()
        return self.stop_requested


def atomic_save_checkpoint(
    destination: os.PathLike[str] | str,
    writer: CheckpointWriter,
) -> None:
    """Write a checkpoint atomically without destroying the last good file.

    The writer writes to an already-open binary file. The temporary file is
    created in the destination directory so os.replace stays on one filesystem.
    The previous checkpoint remains untouched unless serialization and file
    fsync both succeed.
    """
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, target)
        # Persist the directory entry where supported. Some filesystems do not
        # allow directory fsync; the checkpoint file itself has already been
        # fsynced and atomically replaced at this point.
        try:
            dir_fd = os.open(target.parent, os.O_RDONLY)
        except OSError:
            dir_fd = None
        if dir_fd is not None:
            try:
                try:
                    os.fsync(dir_fd)
                except OSError:
                    pass
            finally:
                os.close(dir_fd)
    except BaseException:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def stop_and_checkpoint_if_requested(
    controller: R44StopController,
    *,
    destination: os.PathLike[str] | str,
    state_factory: Callable[[], object],
    writer_factory: Callable[[object], CheckpointWriter],
) -> bool:
    """At a safe training-loop boundary, checkpoint and return True on stop.

    Call this after optimizer.step() and at the top of the next iteration.
    state_factory must capture the current complete resumable state; writer
    adapts that state to the serialization backend.
    """
    if not controller.should_stop():
        return False
    state = state_factory()
    atomic_save_checkpoint(destination, writer_factory(state))
    return True
