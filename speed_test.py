# -*- coding: utf-8 -*-
# Project : Fast-Spark-TTS
# Time    : 2025/3/14 15:36
# Author  : Hui Huang
from typing import Literal
from fast_sparktts import AsyncFastSparkTTS
import asyncio
import time


async def run(
        model_path: str,
        num_infer: int = 5,
        engine: Literal["vllm", "llama-cpp", "sglang", "torch"] = "torch",
        device: Literal["cpu", "cuda", "auto"] | str = "auto",
        warmup: bool = False
):
    if warmup:
        num_infer = num_infer + 1

    model_kwargs = {
        "model_path": model_path,
        "max_length": 32768,
        "llm_device": device,
        "tokenizer_device": device,
        "detokenizer_device": device,
        "engine": engine
    }
    if engine in ["vllm", "sglang", "torch"]:
        model_kwargs["wav2vec_attn_implementation"] = "sdpa"
        model_kwargs["llm_gpu_memory_utilization"] = 0.6
    if engine == "torch":
        model_kwargs["torch_dtype"] = "bfloat16"
        model_kwargs["llm_attn_implementation"] = 'sdpa'
    model = AsyncFastSparkTTS(**model_kwargs)

    target_text = "身临其境，换新体验。塑造开源语音合成新范式，让智能语音更自然。"
    reference_wav = "data/roles/赞助商/reference_audio.wav"

    start_time = time.perf_counter()
    for i in range(num_infer):
        audio = await model.clone_voice_async(
            text=target_text,
            reference_audio=reference_wav,
            reference_text=None,
            temperature=0.6,
            top_p=0.95,
            top_k=40,
            max_tokens=2048
        )
        if warmup and i == 0:
            start_time = time.perf_counter()
    end_time = time.perf_counter()
    if warmup:
        num_infer = num_infer - 1

    import soundfile as sf
    sf.write("result.wav", audio, 16000, "PCM_16")
    print(f"inference time: {end_time - start_time}")
    print(f"Average time: {(end_time - start_time) / num_infer}")


if __name__ == '__main__':
    asyncio.run(run(engine="vllm", warmup=True, num_infer=5, model_path="Spark-TTS-0.5B"))
