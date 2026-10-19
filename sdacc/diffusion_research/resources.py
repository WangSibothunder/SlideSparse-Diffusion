"""Shared read-only GPU/RAM/IO resource handling; never signals other jobs."""
import os,time,threading,shutil
from pathlib import Path
from . import _nvml as nvml

class PhysicalGPU:
    def __init__(self,index):
        nvml.nvmlInit();self.nvml=nvml;self.handle=nvml.nvmlDeviceGetHandleByIndex(int(index))
        self.uuid=nvml.nvmlDeviceGetUUID(self.handle)
        if isinstance(self.uuid,bytes):self.uuid=self.uuid.decode()
    def pids(self):
        return {int(x.pid) for x in (*nvml.nvmlDeviceGetComputeRunningProcesses(self.handle),*nvml.nvmlDeviceGetGraphicsRunningProcesses(self.handle))}
    def wait_clean(self):
        while self.pids()-{os.getpid()}:
            import torch
            if not torch.cuda.is_initialized():
                for index in range(nvml.nvmlDeviceGetCount()):
                    handle=nvml.nvmlDeviceGetHandleByIndex(index)
                    busy=[*nvml.nvmlDeviceGetComputeRunningProcesses(handle),*nvml.nvmlDeviceGetGraphicsRunningProcesses(handle)]
                    if busy or nvml.nvmlDeviceGetMemoryInfo(handle).free<(8<<30) or nvml.nvmlDeviceGetTemperature(handle,0)>=85:continue
                    uuid=nvml.nvmlDeviceGetUUID(handle)
                    if isinstance(uuid,bytes):uuid=uuid.decode()
                    self.handle=handle;self.uuid=uuid;os.environ.update(SDACC_PHYSICAL_GPU=str(index),CUDA_VISIBLE_DEVICES=uuid)
                    return
            time.sleep(15)
    def healthy(self):
        m=nvml.nvmlDeviceGetMemoryInfo(self.handle)
        return m.free>=4<<30 and nvml.nvmlDeviceGetTemperature(self.handle,0)<85
    def close(self):nvml.nvmlShutdown()

class Memory:
    def __init__(self,model):self.model=model
    def sample(self):
        import torch
        p=self.model.physical;n=p.nvml;m=n.nvmlDeviceGetMemoryInfo(p.handle);own=[v for v in n.nvmlDeviceGetComputeRunningProcesses(p.handle) if v.pid==os.getpid()]
        if len(own)!=1 or not p.healthy():raise RuntimeError('GPU health or 4GiB free margin unavailable')
        self.external=max(self.external,max(0,int(own[0].usedGpuMemory)-torch.cuda.memory_reserved()))
        self.last=dict(total_bytes=int(m.total),free_bytes=int(m.free),own_NVML_bytes=int(own[0].usedGpuMemory))
    def begin(self):
        import torch
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();self.external=0;self.sample()
    def finish(self):
        import torch
        torch.cuda.synchronize();self.sample();peak=max(torch.cuda.max_memory_allocated(),torch.cuda.max_memory_reserved())+self.external
        if peak+(4<<30)>self.last['total_bytes']:raise RuntimeError('configuration own peak plus4GiB exceeds device')
        return dict(self.last,own_peak_bytes=peak,margin_bytes=4<<30)

def host_available():
    return int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024

def resource_snapshot(output):
    disk=shutil.disk_usage(output)
    return dict(RAM_available_bytes=host_available(),disk_free_bytes=disk.free,process_IO={a:int(b) for a,b in (x.split(':') for x in Path('/proc/self/io').read_text().splitlines())})

class Isolation:
    def __init__(self,physical):self.physical=physical;self.foreign=set();self.error=None;self.stop=threading.Event()
    def __enter__(self):
        def sample():
            while not self.stop.is_set():
                try:self.foreign.update(self.physical.pids()-{os.getpid()})
                except Exception as e:self.error=repr(e)
                self.stop.wait(.25)
        self.thread=threading.Thread(target=sample,daemon=True);self.thread.start();return self
    def __exit__(self,*args):self.stop.set();self.thread.join()
