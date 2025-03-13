# -*- coding: utf-8 -*-
# Time      :2025/3/13 20:31
# Author    :Hui Huang
from .generator import Generator


class VllmGenerator(Generator):
    def __init__(self, model_path: str, max_length: int = 32768, **kwargs):
        from vllm import AsyncEngineArgs, AsyncLLMEngine

        engine_kwargs = dict(
            model=model_path,
            max_model_len=max_length,
            **kwargs
        )
        async_args = AsyncEngineArgs(**engine_kwargs)

        self.model = AsyncLLMEngine.from_engine_args(async_args)

        super(VllmGenerator, self).__init__(
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
        from vllm import SamplingParams

        prompt_tokens = self.tokenize(prompt, self.max_length - max_tokens)
        inputs = {"prompt_token_ids": prompt_tokens}
        sampling_params = SamplingParams(
            n=1,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop_token_ids=self.stop_token_ids,
            **kwargs)
        results_generator = self.model.generate(
            prompt=inputs,
            sampling_params=sampling_params.to_vllm())
        final_res = None

        async for res in results_generator:
            final_res = res
        assert final_res is not None
        choices = []
        for output in final_res.outputs:
            choices.append(output.text)
        return choices[0]
