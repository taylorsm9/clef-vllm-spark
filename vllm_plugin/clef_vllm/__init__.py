"""vLLM plugin for Clef decision models (Cloudflare/clef-flash and its quantizations)."""

from clef_vllm.schema import ARCHITECTURE, TASK, schema_layout

__all__ = ["ARCHITECTURE", "TASK", "schema_layout", "register"]


def register() -> None:
    from vllm import ModelRegistry

    if ARCHITECTURE not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(ARCHITECTURE, "clef_vllm.model:ClefFlashForDecision")
