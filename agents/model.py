from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .utils import count_tokens, Timer


@dataclass
class GenerationConfig:
    max_new_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.95


class HFModel:
    def __init__(self, model_id: str, load_in_4bit: bool = True):
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        use_cuda = torch.cuda.is_available()
        if load_in_4bit and not use_cuda:
            load_in_4bit = False
            print("Warning: CUDA not available; disabling 4-bit quantization.")

        quant_config = None
        if load_in_4bit:
            quant_config = BitsAndBytesConfig(load_in_4bit=True)

        dtype = torch.float16 if use_cuda else torch.float32
        device_map = "auto" if use_cuda else "cpu"

        if use_cuda:
            print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA not available; running on CPU.")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device_map,
            dtype=dtype,
            quantization_config=quant_config,
        )

    def generate(self, prompt: str, config: Optional[GenerationConfig] = None) -> dict:
        if config is None:
            config = GenerationConfig()

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        tokens_in = inputs.input_ids.shape[-1]

        with Timer() as timer:
            output = self.model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                do_sample=config.temperature > 0,
            )

        text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        tokens_out = count_tokens(self.tokenizer, text)

        return {
            "text": text,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "latency_ms": timer.elapsed_ms,
        }
