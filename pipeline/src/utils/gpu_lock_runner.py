import os
from pathlib import Path
import torch
import logging
from filelock import FileLock, Timeout
from contextlib import contextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LOCK_DIR = Path(os.environ.get("GPU_LOCK_DIR", "~/dvc_gpu_locks"))
LOCK_DIR = LOCK_DIR.expanduser()
LOCK_DIR.mkdir(parents=True, exist_ok=True)
logger.info(f"GPU's lock directory: {LOCK_DIR}")

def get_available_devices() -> list[int]:
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if devices is not None:
        return list(map(int, devices.split(",")))
    else:
        return list(range(torch.cuda.device_count()))

DEVICES = get_available_devices()
logger.info(f"Available GPUs: {DEVICES}")

@contextmanager
def lock_free_gpu():
    """
    Context manager that finds the first free GPU (based on lock files),
    acquires a lock for it, and yields the GPU ID.
    Automatically releases the lock when the context exits.
    """
    for gpu_id in DEVICES:
        lock_path = os.path.join(LOCK_DIR, f"gpu{gpu_id}.lock")
        lock = FileLock(lock_path, timeout=0.1)
        try:
            lock.acquire()
            logger.info(f"Acquired lock for GPU {gpu_id}")
            yield gpu_id
            return
        except Timeout:
            continue
        finally:
            if lock.is_locked:
                lock.release()
                logger.info(f"Released lock for GPU {gpu_id}")
    logger.error("No free GPU available")
    raise RuntimeError("All GPUs are currently locked")
