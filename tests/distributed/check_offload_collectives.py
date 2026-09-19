"""Run with torchrun and the matching torch-memory-saver LD_PRELOAD library."""
import os

import torch
import torch.distributed as dist
from torch_memory_saver import torch_memory_saver as saver

from rl_engine.distributed.collectives import DeterministicCollective

rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(rank)
with saver.disable():
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    dist.barrier()
collective = DeterministicCollective(max_size_bytes=1024 * 1024)
x = torch.full((256,), rank + 1.0, device="cuda")
for phase in range(2):
    if phase:
        torch.cuda.synchronize()
        saver.pause()
        saver.resume()
    result = collective.all_reduce(x)
    assert torch.equal(result, torch.full_like(x, 36.0))
    dist.barrier()
if rank == 0:
    print("OFFLOAD_IPC_RESULT=PASS before and after pause/resume", flush=True)
collective.close()
dist.destroy_process_group()
