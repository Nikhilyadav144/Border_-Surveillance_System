"""The gate that stops concurrent Metal encoding aborting the process.

These tests cannot reproduce the original failure. It was not a Python
exception: PyTorch's MPS backend hit a Metal assertion and called abort(),
so the process died with `Abort trap: 6` and no traceback. What can be
pinned is the behaviour that prevents it - that MPS serialises, that nothing
else does, and that the lock is shared by every caller in the process.
"""

from __future__ import annotations

import threading
import time
from contextlib import nullcontext

from core.inference_gate import inference_gate, needs_serialising


class TestDeviceSelection:
    def test_mps_needs_serialising(self):
        assert needs_serialising("mps") is True
        assert needs_serialising("mps:0") is True
        assert needs_serialising("MPS") is True

    def test_nothing_else_does(self):
        # On CPU torch releases the GIL, so camera threads genuinely run in
        # parallel and a lock would halve multi-camera throughput. CUDA
        # serialises per stream in the driver and is thread-safe.
        for device in ("cpu", "cuda", "cuda:0", "", None):
            assert needs_serialising(device) is False

    def test_non_mps_gets_a_no_op_context(self):
        gate = inference_gate("cpu")
        assert isinstance(gate, type(nullcontext()))


class TestSerialisation:
    def test_two_threads_never_hold_the_mps_gate_at_once(self):
        """The property the crash needed violated.

        Metal aborted because a second thread began encoding while the first
        was still inside a command buffer. Overlap is the thing to forbid.
        """
        overlaps = []
        inside = []
        lock = threading.Lock()

        def worker():
            with inference_gate("mps"):
                with lock:
                    inside.append(1)
                    if len(inside) > 1:
                        overlaps.append(len(inside))
                time.sleep(0.02)          # stand-in for an inference call
                with lock:
                    inside.pop()

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert overlaps == [], f"threads overlapped inside the gate: {overlaps}"

    def test_cpu_work_does_overlap(self):
        """The other half: the gate must not serialise what it need not."""
        peak = 0
        inside = []
        lock = threading.Lock()

        def worker():
            nonlocal peak
            with inference_gate("cpu"):
                with lock:
                    inside.append(1)
                    peak = max(peak, len(inside))
                time.sleep(0.05)
                with lock:
                    inside.pop()

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert peak > 1, "CPU inference must still run in parallel"

    def test_the_gate_is_one_lock_for_the_whole_process(self):
        """Every camera builds its own detector and its own pose estimator.

        A per-instance lock would let two cameras encode Metal work at the
        same time, which is exactly the crash. The lock is module level, so
        separate calls must hand back the same object.
        """
        assert inference_gate("mps") is inference_gate("mps")

    def test_the_gate_is_reentrant(self):
        # The detector's tracking path falls back to detect() on error, and
        # both are gated. A plain Lock would deadlock there.
        with inference_gate("mps"):
            with inference_gate("mps"):
                pass


class TestCallSitesAreGated:
    def test_detector_and_pose_both_use_the_gate(self):
        """Both models run from camera threads, so both need the gate.

        Gating only the detector would have left the pose estimator free to
        collide with it, and the crash would have survived the fix.
        """
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        for name in ("core/detector.py", "core/pose.py"):
            source = (root / name).read_text()
            assert "inference_gate(" in source, f"{name} is not gated"
