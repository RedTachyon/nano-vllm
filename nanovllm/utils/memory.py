import os
import torch

def get_gpu_memory():
    if torch.cuda.is_available():
        from pynvml import (nvmlInit, nvmlShutdown,
                            nvmlDeviceGetHandleByIndex,
                            nvmlDeviceGetMemoryInfo)

        torch.cuda.synchronize()
        nvmlInit()
        visible = list(map(int,
                           os.getenv("CUDA_VISIBLE_DEVICES",
                                     "0,1,2,3,4,5,6,7").split(',')))
        handle = nvmlDeviceGetHandleByIndex(visible[torch.cuda.current_device()])
        mem = nvmlDeviceGetMemoryInfo(handle)
        nvmlShutdown()
        return mem.total, mem.used, mem.free

    # ----------  Apple-Silicon / MPS  ----------
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        # These APIs landed in PyTorch 2.2 – guard for older builds
        if not all(hasattr(torch.mps, f) for f in
                   ("recommended_max_memory",
                    "driver_allocated_memory")):
            raise RuntimeError("Update PyTorch ≥2.2 for full MPS memory stats")

        total = torch.mps.recommended_max_memory()          # bytes
        used  = torch.mps.driver_allocated_memory()         # bytes
        free  = max(total - used, 0)
        return total, used, free

    # ----------  No supported accelerator  ----------
    return None, None, None
