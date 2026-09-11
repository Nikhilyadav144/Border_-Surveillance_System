"""One process-wide gate for model inference, needed only on Apple's GPU.

WHY THIS EXISTS
---------------
Running two cameras at once killed the whole backend on an M-series Mac. Not
a slow response, not a dropped frame: the process aborted. The evidence, from
``backend.log`` immediately after the second camera was started:

    INFO: 127.0.0.1 - "POST /api/cameras/CAM-02/start HTTP/1.1" 200 OK
    -[AGXG15GFamilyCommandBuffer tryCoalescingPreviousComputeCommandEncoderWithConfig:
      nextEncoderClass:]:1094: failed assertion
      `A command encoder is already encoding to this command buffer'

and from the shell that launched it:

    ./start.sh: line 24: 29796 Abort trap: 6   uvicorn api.server:app ...

That is Metal, not Python. PyTorch's MPS backend keeps per-process Metal
command buffers, and it is not safe for two threads to encode work into them
at the same time. When the second camera thread began inference while the
first was mid-encode, Metal failed an internal assertion and called abort(),
taking the API, every camera and the database session with it. No Python
exception handler can catch that, which is why the failure looked like "the
backend vanished" rather than "a camera errored".

WHAT THIS DOES
--------------
Serialises every model call in the process behind one lock, but ONLY when the
resolved device is MPS. Elsewhere it is a no-op context manager:

* on CPU, torch releases the GIL during inference, so several camera threads
  genuinely do run in parallel and a lock would halve the throughput of a
  multi-camera setup for no reason;
* on CUDA, the driver serialises per-stream work itself and is thread-safe.

WHAT IT COSTS
-------------
On MPS the cameras share the GPU sequentially instead of colliding on it.
There is no throughput lost that was ever really there - one GPU executes one
kernel at a time regardless. What changes is that the work is queued in an
orderly way instead of racing.

The lock is deliberately as narrow as possible: it wraps the model call and
nothing else. Frame decoding, tracking bookkeeping, zone tests, behaviour
classification and JPEG encoding all stay outside it and stay parallel.
"""

from __future__ import annotations

import logging
import threading
from contextlib import nullcontext
from typing import ContextManager

logger = logging.getLogger(__name__)

#: The one lock. Module level, so every detector and every pose estimator in
#: the process shares it no matter how many cameras construct their own.
_GPU_LOCK = threading.RLock()

#: Set once, the first time anything asks. Kept so the log line explaining
#: the choice is printed once rather than per frame.
_announced = False


def needs_serialising(device: str | None) -> bool:
    """Is this device one that cannot take concurrent calls from threads?"""
    return bool(device) and str(device).lower().startswith("mps")


def inference_gate(device: str | None) -> ContextManager:
    """Return the lock for MPS, or a no-op for every other device.

    Usage at a call site::

        with inference_gate(self.device):
            results = self._model.track(image, **kwargs)
    """
    global _announced
    if not needs_serialising(device):
        return nullcontext()
    if not _announced:
        _announced = True
        logger.info(
            "Model inference is serialised across threads because the device "
            "is MPS. Concurrent Metal encoding aborts the process; see "
            "core/inference_gate.py."
        )
    return _GPU_LOCK
