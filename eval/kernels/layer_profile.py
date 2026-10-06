"""Profile 4 Clef decoder layers (3 linear-attention + 1 full-attention) with random weights; x16 estimates the full 64-layer forward.
Usage: python layer_profile.py <clef release dir> <seq_len> <batch> <pad 0|1>   (needs transformers + fla, e.g. reference/ env)"""
import torch, time, sys, json
torch.cuda.set_per_process_memory_fraction(0.3)
from transformers import AutoConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
cfg = AutoConfig.from_pretrained(sys.argv[1]).text_config
cfg.num_hidden_layers = 4; cfg.layer_types = cfg.layer_types[:4]; cfg.vocab_size = 1024; cfg.use_cache=False
cfg._attn_implementation = "sdpa"
m = Qwen3_5TextModel(cfg).to("cuda", torch.bfloat16).eval()
from transformers.models.qwen3_5 import modeling_qwen3_5 as mq
L = int(sys.argv[2]); B = int(sys.argv[3]); pad = int(sys.argv[4])
ids = torch.randint(0, 1024, (B, L), device="cuda"); am = torch.ones(B, L, dtype=torch.long, device="cuda")
if pad: am[0, L//2:] = 0
@torch.inference_mode()
def run(): return m(input_ids=ids, attention_mask=am, use_cache=False)
run(); torch.cuda.synchronize()
s=time.perf_counter(); run(); torch.cuda.synchronize(); dt=time.perf_counter()-s
print(f"L={L} B={B} pad={pad}: 4 layers {dt*1000:.0f} ms -> full 64-layer est {dt*16:.1f}s, {B*L/(dt*16):.0f} tok/s")
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CUDA]) as p: run(); torch.cuda.synchronize()
print(p.key_averages().table(sort_by="cuda_time_total", row_limit=12, max_name_column_width=70))
