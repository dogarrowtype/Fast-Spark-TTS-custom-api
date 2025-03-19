# -*- coding: utf-8 -*-
# Time      :2025/3/19 12:30
# Author    :Hui Huang
from typing import Optional, Literal

import torch

from .generator import Generator
from transformers import AutoModelForCausalLM, GenerationConfig


class TorchGenerator(Generator):
    def __init__(self,
                 model_path: str,
                 max_length: int = 32768,
                 device: str = "cpu",
                 attn_implementation: Optional[Literal["sdpa", "flash_attention_2", "eager"]] = None,
                 torch_dtype: Literal['float16', "bfloat16", 'float32', 'auto'] = "auto",
                 cache_implementation: Optional[str] = None,
                 **kwargs):
        self.device = torch.device(device)
        self.cache_implementation = cache_implementation
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=getattr(torch, torch_dtype, "auto"),
            attn_implementation=attn_implementation,
            **kwargs
        )
        self.model.eval().to(self.device)

        super(TorchGenerator, self).__init__(
            tokenizer=model_path,
            max_length=max_length,
        )

    async def async_generate(
            self,
            prompt: str,
            max_tokens: int = 1024,
            temperature: float = 0.6,
            top_p: float = 0.9,
            top_k: int = 50,
            **kwargs
    ) -> str:
        """
        如果使用动态批处理，hf代码无法为每一个请求单独设置generation 参数，所以还是逐个处理吧。
        如果想要动态批处理，建议使用vllm、sglang
        Args:
            prompt:
            max_tokens:
            temperature:
            top_p:
            top_k:
            **kwargs:

        Returns:

        """
        input_ids = self.tokenize(prompt, self.max_length - max_tokens)

        input_ids = torch.LongTensor([input_ids]).to(self.device)
        generated_ids = self.model.generate(
            input_ids,
            generation_config=GenerationConfig(
                max_length=self.max_length,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
                cache_implementation=self.cache_implementation,
                **kwargs
            ),
        )
        prompt_length = input_ids.size(1)
        completion_ids = generated_ids[:, prompt_length:]
        completions_text = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        return completions_text[0]
