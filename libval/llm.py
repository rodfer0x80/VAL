from __future__ import annotations
import logging
from pathlib import Path
from typing import Any


class LLM:
    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logging.getLogger(__name__)

        self.MODEL_NAME: Final = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        self.tokenizer = None
        self.model = None
        self.device = None

        self.torch_available: bool = False

        self.maybe_load_torch_backend()


    def maybe_load_torch_backend(self) -> None:
        try:
            import torch  # type: ignore
            import transformers # type: ignore
            self.torch_available = True
            self.logger.info("torch and transformers backend ready [model=%s]", self.MODEL_NAME)
        except ModuleNotFoundError as exc:
            self.torch_available = False
            self.logger.warning(
                "torch or transformers not found - limited functionality (%s)", exc
            )

