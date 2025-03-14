# -*- coding: utf-8 -*-
# Project : Fast-Spark-TTS
# Time    : 2025/3/14 15:36
# Author  : Hui Huang
from typing import Literal
from fast_sparktts import AsyncFastSparkTTS
import asyncio
from time import time


async def run(
        model_path: str,
        num_infer: int = 5,
        engine: Literal["vllm", "llama-cpp", "sglang"] = "llama-cpp",
        warmup: bool = False
):
    if warmup:
        num_infer = num_infer + 1
    if engine == "llama-cpp":
        device = 'cpu'
    else:
        device = 'cuda'

    model_kwargs = {
        "model_path": model_path,
        "max_length": 32768,
        "llm_device": device,
        "audio_device": device,
        "vocoder_device": device,
        "engine": engine
    }
    if engine in ["vllm", "sglang"]:
        model_kwargs["wav2vec_attn_implementation"] = "sdpa"
        model_kwargs["llm_gpu_memory_utilization"] = 0.6

    model = AsyncFastSparkTTS(**model_kwargs)

    target_text = "身临其境，换新体验。塑造开源语音合成新范式，让智能语音更自然。"
    reference_text = "吃燕窝就选燕之屋，本节目由26年专注高品质燕窝的燕之屋冠名播出。豆奶牛奶换着喝，营养更均衡，本节目由豆本豆豆奶特约播出。"
    reference_wav = "example/prompt_audio.wav"

    t1 = time()
    for i in range(num_infer):
        audio = await model.async_clone_voice(
            reference_wav_path=reference_wav,
            reference_text=reference_text,
            target_text=target_text,
            temperature=0.2,
            top_p=0.95,
            top_k=40,
            max_tokens=1024
        )
        if warmup and i == 0:
            t1 = time()
    t2 = time()
    if warmup:
        num_infer = num_infer - 1

    import soundfile as sf
    sf.write("result.wav", audio, 16000, "PCM_16")
    print(f"inference time: {t2 - t1}")
    print(f"Average time: {(t2 - t1) / num_infer}")


if __name__ == '__main__':
    asyncio.run(run(engine="vllm", warmup=True, num_infer=5, model_path="Spark-TTS-0.5B"))
