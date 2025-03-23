# -*- coding: utf-8 -*-
# Project : Fast-Spark-TTS
# Time    : 2025/3/13 12:32
# Author  : Hui Huang
from fast_sparktts import AsyncFastSparkTTS
import soundfile as sf


def prepare_engine():
    # vllm
    # engine = AsyncFastSparkTTS(
    #     model_path="Spark-TTS-0.5B",
    #     max_length=32768,
    #     llm_device="cuda:0",
    #     tokenizer_device="cuda:0",
    #     detokenizer_device="cuda:0",
    #     engine="vllm",
    #     wav2vec_attn_implementation="sdpa",  # 使用flash attn加速wav2vec
    #     llm_gpu_memory_utilization=0.6,
    #     seed=0
    # )

    # sglang
    # engine = AsyncFastSparkTTS(
    #     model_path="Spark-TTS-0.5B",
    #     max_length=32768,
    #     llm_device="cuda",  # sglang没办法指定gpu id，需要使用CUDA_VISIBLE_DEVICES=0设置。
    #     tokenizer_device="cuda:0",
    #     detokenizer_device="cuda:0",
    #     engine="sglang",
    #     wav2vec_attn_implementation="sdpa",  # 使用flash attn加速wav2vec
    #     llm_gpu_memory_utilization=0.6,
    #     seed=0
    # )

    # llama-cpp
    # engine = AsyncFastSparkTTS(
    #     model_path="Spark-TTS-0.5B",
    #     max_length=32768,
    #     llm_device="cpu",
    #     tokenizer_device="cpu",
    #     detokenizer_device="cpu",
    #     engine="llama-cpp",
    #     wav2vec_attn_implementation="eager"
    # )

    # torch
    engine = AsyncFastSparkTTS(
        model_path="Spark-TTS-0.5B",
        max_length=32768,
        llm_device="cuda",
        tokenizer_device="cuda",
        detokenizer_device="cuda",
        engine="torch",
        wav2vec_attn_implementation="sdpa",
        llm_attn_implementation="sdpa",
        torch_dtype="bfloat16",
        seed=0
    )
    return engine


def generate_voice(engine: AsyncFastSparkTTS):
    wav = engine.generate_voice(
        "我是无敌的小可爱。",
        gender="female",
        temperature=0.6,
        top_p=0.95,
        top_k=50,
        max_tokens=512
    )
    return wav


async def async_generate_voice(engine: AsyncFastSparkTTS):
    wav = await engine.generate_voice_async(
        "我是无敌的小可爱。",
        gender="female",
        temperature=0.6,
        top_p=0.95,
        top_k=50,
        max_tokens=512
    )
    return wav


def clone_voice(engine: AsyncFastSparkTTS):
    text = "身临其境，换新体验。塑造开源语音合成新范式，让智能语音更自然。"

    wav = engine.clone_voice(
        text=text,
        reference_audio="data/roles/赞助商/reference_audio.wav",
        reference_text=None,
        temperature=0.6,
        top_p=0.95,
        top_k=50,
        max_tokens=512
    )
    return wav


async def async_clone_voice(engine: AsyncFastSparkTTS):
    text = "身临其境，换新体验。塑造开源语音合成新范式，让智能语音更自然。"

    wav = await engine.clone_voice_async(
        text=text,
        reference_audio="data/roles/赞助商/reference_audio.wav",
        temperature=0.6,
        top_p=0.95,
        top_k=50,
        max_tokens=512
    )
    return wav


def main():
    engine = prepare_engine()
    # audio = generate_voice(engine)
    audio = clone_voice(engine)
    sf.write("result.wav", audio, 16000, "PCM_16")


async def run():
    engine = prepare_engine()
    audio = await async_generate_voice(engine)
    # audio = await async_clone_voice(engine)
    sf.write("result.wav", audio, 16000, "PCM_16")


if __name__ == '__main__':
    main()
    # asyncio.run(run())
