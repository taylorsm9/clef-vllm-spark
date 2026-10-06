"""Microbenchmarks on the GB10: BF16 vs FP8 GEMM, and SDPA flash / efficient / explicit-mask kernels at head_dim 256.
Run with any torch+CUDA Python, e.g. inside the vLLM image or the reference env."""
import torch, time
import torch.nn.functional as F
torch.cuda.set_per_process_memory_fraction(0.12)
def t(fn, n=10):
    fn(); torch.cuda.synchronize(); s=time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-s)/n
M,K,N=8192,5120,17408
a=torch.randn(M,K,device='cuda',dtype=torch.bfloat16); b=torch.randn(N,K,device='cuda',dtype=torch.bfloat16)
dt=t(lambda: a@b.t()); print(f"bf16 gemm {2*M*K*N/dt/1e12:.1f} TFLOPS")
a8=a.to(torch.float8_e4m3fn); b8=b.to(torch.float8_e4m3fn); one=torch.ones((),device='cuda')
try:
    dt=t(lambda: torch._scaled_mm(a8,b8.t(),scale_a=one,scale_b=one,out_dtype=torch.bfloat16)); print(f"fp8 gemm {2*M*K*N/dt/1e12:.1f} TFLOPS")
except Exception as e: print("fp8 fail", e)
del a,b,a8,b8
L=16384; q=torch.randn(1,24,L,256,device='cuda',dtype=torch.bfloat16); k=torch.randn(1,24,L,256,device='cuda',dtype=torch.bfloat16); v=torch.randn_like(k)
fl=4*24*256*L*L/2
from torch.nn.attention import sdpa_kernel, SDPBackend
for be in [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION]:
    try:
        with sdpa_kernel(be): dt=t(lambda: F.scaled_dot_product_attention(q,k,v,is_causal=True),3)
        print(f"sdpa causal {be.name}: {fl/dt/1e12:.1f} TFLOPS")
    except Exception as e: print(be.name,"fail",str(e)[:120])
mask=torch.ones(L,L,dtype=torch.bool,device='cuda').tril()[None,None]
try:
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION): dt=t(lambda: F.scaled_dot_product_attention(q,k,v,attn_mask=mask),3)
    print(f"sdpa explicit-mask EFFICIENT: {fl/dt/1e12:.1f} TFLOPS")
except Exception as e: print("mask fail", str(e)[:120])
