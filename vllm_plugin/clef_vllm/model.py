"""Clef-Flash as a vLLM pooling model.

The Qwen3.5 backbone runs in vLLM; the release's JointSchemaHead runs in the pooler on the
full prompt's final hidden states. Each request carries its schema layout in
PoolingParams.extra_kwargs["clef_questions"]: a list of [type_id, [q_start, q_end],
[[opt_start, opt_end], ...]] built by joint_schema_model.encode_record on the client.
The output is one flat float32 tensor: every option logit of every question, in order.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Iterable, Set
from pathlib import Path

import torch
from safetensors.torch import load_file
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.model_executor.layers.pooler import Pooler, PoolingParamsUpdate
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
    Qwen3_5ProcessingInfo,
)
from vllm.model_executor.models.qwen3_vl import (
    Qwen3VLDummyInputsBuilder,
    Qwen3VLMultiModalProcessor,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.tasks import PoolingTask
from vllm.v1.pool.metadata import PoolingMetadata

from clef_vllm.schema import TASK


def _import_release_module(model_dir: Path):
    spec = importlib.util.spec_from_file_location("joint_schema_model", model_dir / "joint_schema_model.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve annotations via sys.modules
    spec.loader.exec_module(module)
    return module


class ClefPooler(Pooler):
    def __init__(self, model: "ClefFlashForDecision") -> None:
        super().__init__()
        # Not a submodule: the head and lm_head are owned (and loaded) by the model.
        self._model = [model]
        scheduler_config = get_current_vllm_config().scheduler_config
        self.enable_chunked_prefill = scheduler_config.enable_chunked_prefill

    def get_supported_tasks(self) -> Set[PoolingTask]:
        return {TASK}

    def get_pooling_updates(self, task: PoolingTask) -> PoolingParamsUpdate:
        return PoolingParamsUpdate(requires_token_ids=True)

    def forward(self, hidden_states: torch.Tensor, pooling_metadata: PoolingMetadata):
        model = self._model[0]
        cursor = pooling_metadata.get_pooling_cursor()
        chunks = list(torch.split(hidden_states, cursor.num_scheduled_tokens_cpu.tolist()))
        if self.enable_chunked_prefill:
            finished = cursor.is_finished().tolist()
            full: list[torch.Tensor | None] = []
            for state, chunk, done in zip(pooling_metadata.pooling_states, chunks, finished):
                state.hidden_states_cache.append(chunk if done and not state.hidden_states_cache else chunk.clone())
                if done:
                    cache = state.hidden_states_cache
                    full.append(cache[0] if len(cache) == 1 else torch.cat(cache, dim=0))
                    state.clean()
                else:
                    full.append(None)
        else:
            full = chunks

        token_ids = pooling_metadata.get_prompt_token_ids_cpu()
        outputs: list[torch.Tensor | None] = []
        for hidden, ids, params in zip(full, token_ids, pooling_metadata.pooling_params):
            if hidden is None:
                outputs.append(None)
                continue
            layout = (params.extra_kwargs or {}).get("clef_questions")
            if layout is None or model.lexical_weight is None:
                # vLLM's profiling dummy run (no schema), or a TP rank that does not pool.
                layout = [[1, [0, 1], [[0, 1], [0, 1]]]]
                if model.lexical_weight is None:
                    outputs.append(torch.zeros(2, device=hidden.device))
                    continue
            outputs.append(model.score(hidden, ids, layout))
        return outputs


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Qwen3_5ProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class ClefFlashForDecision(Qwen3_5ForConditionalGeneration):
    is_pooling_model = True
    default_seq_pooling_type = "LAST"
    default_tok_pooling_type = "ALL"

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.model_dir = Path(vllm_config.model_config.model)
        self.release = _import_release_module(self.model_dir)
        head_config = json.loads((self.model_dir / "clef_head" / "joint_head_config.json").read_text())
        self.head = self.release.JointSchemaHead(**head_config)
        self.pooler = ClefPooler(self)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded = super().load_weights(weights)
        state = load_file(self.model_dir / "clef_head" / "joint_head.safetensors")
        self.head.load_state_dict(state, strict=True)
        lm_head = self.language_model.lm_head.weight
        self.head.to(device=lm_head.device, dtype=lm_head.dtype).eval()
        # The head indexes lm_head rows by token id; under TP the weight is vocab-sharded,
        # so gather it once. The pooler only runs on TP rank 0, so only rank 0 keeps it, and
        # in host memory: a request needs only its option tokens' rows (see score), and a full
        # GPU copy would cost rank 0 ~vocab x hidden x 2 bytes of KV cache (2.5 GB for Clef 27B).
        self.lexical_weight = lm_head.data  # plain tensor, not a registered Parameter
        if get_tensor_model_parallel_world_size() > 1:
            gathered = tensor_model_parallel_all_gather(lm_head.data.contiguous(), dim=0)
            self.lexical_weight = gathered.cpu().pin_memory() if get_tensor_model_parallel_rank() == 0 else None
            del gathered
            torch.cuda.empty_cache()
        return loaded | {f"head.{name}" for name in state}

    @torch.inference_mode()
    def score(self, hidden: torch.Tensor, ids: torch.Tensor, layout: list) -> torch.Tensor:
        release = self.release
        questions = tuple(
            release.EncodedQuestion(
                question_id=str(index),
                question_type=int(type_id),
                question_span=tuple(question_span),
                option_spans=tuple(tuple(span) for span in option_spans),
                option_ids=tuple(str(i) for i in range(len(option_spans))),
            )
            for index, (type_id, question_span, option_spans) in enumerate(layout)
        )
        record = release.EncodedRecord(input_ids=tuple(), questions=questions, record_id="vllm")
        device = hidden.device
        length = hidden.shape[0]
        lexical = self.lexical_weight
        if lexical.device.type == "cpu":
            # The head reads lexical rows only at option-span tokens: copy just those rows to the
            # GPU and remap the span token ids into the compact table.
            ids = ids.cpu()
            positions = torch.cat([torch.arange(start, end) for _, _, spans in layout for start, end in spans])
            unique, inverse = torch.unique(ids[positions], return_inverse=True)
            lexical = lexical[unique].to(device, non_blocking=True)
            ids = torch.zeros(ids.shape, dtype=torch.long)
            ids[positions] = inverse
        logits = self.head(
            hidden.unsqueeze(0),
            ids.to(device).unsqueeze(0),
            torch.ones((1, length), dtype=torch.long, device=device),
            [record],
            lexical,
        )[0]
        return torch.cat([question_logits.float() for question_logits in logits])
