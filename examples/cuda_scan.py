"""Optional NVRTC accelerator for teacher-forced evaluation on NVIDIA GPUs."""
import ctypes as ct
from ctypes.util import find_library
import os
from pathlib import Path
import torch

P = ct.c_void_p
I = ct.c_int
S = ct.c_char_p


def checked(lib, name, arguments, *values):
    fn = getattr(lib, name)
    fn.argtypes = arguments
    fn.restype = I
    status = fn(*values)
    if status:
        raise RuntimeError(f"{name} returned {status}")


def nvrtc_library():
    torch_dir = Path(torch.__file__).resolve().parent
    paths = list((torch_dir/'lib').glob('nvrtc64_*.dll'))
    for directory in (torch_dir/'lib', torch_dir.parent/'nvidia'/'cuda_nvrtc'/'lib'):
        paths.extend(sorted(directory.glob('libnvrtc.so*')))
    for candidate in [*paths, find_library('nvrtc')]:
        if candidate:
            try:
                return ct.CDLL(str(candidate))
            except OSError:
                pass
    raise RuntimeError("Install an NVIDIA CUDA-enabled PyTorch and matching NVRTC, or use the torch backend.")


class Scan:
    def __init__(self):
        torch.cuda.init()
        self.device = torch.cuda.current_device()
        driver = ct.WinDLL('nvcuda') if os.name == 'nt' else ct.CDLL('libcuda.so.1')
        compiler = nvrtc_library()
        source = Path(__file__).with_name('scan.cu').read_bytes()
        prog = P()
        checked(compiler,'nvrtcCreateProgram',[ct.POINTER(P),S,S,I,P,P],ct.byref(prog),source,b'scan.cu',0,None,None)
        try:
            major,minor = torch.cuda.get_device_capability()
            options = (S*2)(f'--gpu-architecture=compute_{major}{minor}'.encode(),b'--fmad=false')
            try:
                checked(compiler,'nvrtcCompileProgram',[P,I,ct.POINTER(S)],prog,2,options)
            except RuntimeError as err:
                size = ct.c_size_t()
                checked(compiler,'nvrtcGetProgramLogSize',[P,ct.POINTER(ct.c_size_t)],prog,ct.byref(size))
                log = ct.create_string_buffer(size.value)
                checked(compiler,'nvrtcGetProgramLog',[P,P],prog,log)
                raise RuntimeError(log.value.decode()) from err
            size = ct.c_size_t()
            checked(compiler,'nvrtcGetPTXSize',[P,ct.POINTER(ct.c_size_t)],prog,ct.byref(size))
            ptx = ct.create_string_buffer(size.value)
            checked(compiler,'nvrtcGetPTX',[P,P],prog,ptx)
        finally:
            checked(compiler,'nvrtcDestroyProgram',[ct.POINTER(P)],ct.byref(prog))
        self.module, self.kernel = P(), P()
        checked(driver,'cuModuleLoadData',[ct.POINTER(P),P],ct.byref(self.module),ptx)
        checked(driver,'cuModuleGetFunction',[ct.POINTER(P),P,S],ct.byref(self.kernel),self.module,b'scan')
        self.driver = driver

    def __call__(self, u, dt, b, c, a, modes, boundary, policy):
        batch,length,width = u.shape
        if width % 8 or dt.shape != u.shape or b.shape != (batch,length,16) or c.shape != b.shape:
            raise ValueError("unsupported Mamba scan geometry")
        if a.shape != (width,16) or modes.shape != (width,) or modes.dtype != torch.uint8:
            raise ValueError("invalid transition or block modes")
        arrays = [t.contiguous() for t in (u,dt,b,c,a,modes)]
        if any(t.device != torch.device('cuda', self.device) for t in arrays) or any(t.dtype != torch.float32 for t in arrays[:5]):
            raise ValueError("expected FP32 tensors and uint8 modes on the active CUDA device")
        state = u.new_zeros(batch,width,16)
        output = torch.empty_like(arrays[0])
        pointers = [P(t.data_ptr()) for t in [*arrays,state,output]]
        stream = torch.cuda.current_stream(self.device).cuda_stream
        for first in range(0,length,128):
            values = pointers + [I(v) for v in (width,length,first,min(first+128,length),boundary,policy)]
            params = (P*len(values))(*(ct.addressof(v) for v in values))
            checked(self.driver,'cuLaunchKernel',[P]+[ct.c_uint]*7+[P,ct.POINTER(P),P],
                    self.kernel,batch*width//8,1,1,128,1,1,0,stream,params,None)
        return output
