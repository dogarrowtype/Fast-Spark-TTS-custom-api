# -*- coding: utf-8 -*-
# Time      :2025/3/29 10:52
# Author    :Hui Huang
import os
from typing import AsyncIterator, Optional, List
from ..logger import get_logger
from .base_llm import BaseLLM

logger = get_logger()

__all__ = ["LlamaCppGenerator"]


class LlamaCppGenerator(BaseLLM):
    def __init__(
            self,
            model_path: str,
            max_length: int = 32768,
            stop_tokens: Optional[list[str]] = None,
            stop_token_ids: Optional[List[int]] = None,
            **kwargs
    ):
        from llama_cpp import Llama

        model_files = []
        for filename in os.listdir(model_path):
            if filename.endswith(".gguf"):
                model_files.append(filename)
        if len(model_files) == 0:
            logger.error("No gguf file found in the model directory")
            raise ValueError("No gguf file found in the model directory")
        else:
            if len(model_files) > 1:
                logger.warning(
                    f"Multiple gguf files found in the model directory, using the first one: {model_files[0]}")
            model_file = os.path.join(model_path, model_files[0])
        self.model = Llama(
            model_file,
            n_ctx=max_length,
            **kwargs
        )
        # 不使用llama cpp 的 tokenizer
        super(LlamaCppGenerator, self).__init__(
            tokenizer=model_path,
            max_length=max_length,
            stop_tokens=stop_tokens,
            stop_token_ids=stop_token_ids,
        )

    async def async_generate(
            self,
            prompt: str,
            max_tokens: int = 1024,
            temperature: float = 0.9,
            top_p: float = 0.9,
            top_k: int = 50,
            **kwargs
    ) -> str:
        prompt_tokens = self.tokenize(prompt, self.max_length - max_tokens)
        completion_tokens = []
        for token in self.model.generate(
                prompt_tokens,
                top_k=top_k,
                top_p=top_p,
                temp=temperature,
                **kwargs
        ):
            if token in self.stop_token_ids:
                break
            if len(completion_tokens) + len(prompt_tokens) >= self.max_length:
                break
            completion_tokens.append(token)

        # Decode the generated tokens into text
        output = self.tokenizer.decode(completion_tokens)
        return output

    async def async_stream_generate(
            self,
            prompt: str,
            max_tokens: int = 1024,
            temperature: float = 0.9,
            top_p: float = 0.9,
            top_k: int = 50,
            **kwargs) -> AsyncIterator[str]:
        prompt_tokens = self.tokenize(prompt, self.max_length - max_tokens)
        completion_tokens = []
        previous_texts = ""
        for token in self.model.generate(
                prompt_tokens,
                top_k=top_k,
                top_p=top_p,
                temp=temperature,
                **kwargs
        ):
            if token in self.stop_token_ids:
                break
            if len(completion_tokens) + len(prompt_tokens) >= self.max_length:
                break
            completion_tokens.append(token)

            text = self.tokenizer.decode(completion_tokens)

            delta_text = text[len(previous_texts):]
            previous_texts = text

            yield delta_text
