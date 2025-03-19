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
    #     audio_device="cuda:0",
    #     vocoder_device="cuda:0",
    #     engine="vllm",
    #     wav2vec_attn_implementation="sdpa",  # 使用flash attn加速wav2vec
    #     llm_gpu_memory_utilization=0.6
    # )

    # sglang
    # engine = AsyncFastSparkTTS(
    #     model_path="Spark-TTS-0.5B",
    #     max_length=32768,
    #     llm_device="cuda",  # sglang没办法指定gpu id，需要使用CUDA_VISIBLE_DEVICES=0设置。
    #     audio_device="cuda:0",
    #     vocoder_device="cuda:0",
    #     engine="sglang",
    #     wav2vec_attn_implementation="sdpa",  # 使用flash attn加速wav2vec
    #     llm_gpu_memory_utilization=0.6
    # )

    # llama-cpp
    # engine = AsyncFastSparkTTS(
    #     model_path="Spark-TTS-0.5B",
    #     max_length=32768,
    #     llm_device="cpu",
    #     audio_device="cpu",
    #     vocoder_device="cpu",
    #     engine="llama-cpp",
    #     wav2vec_attn_implementation="eager"
    # )

    # torch
    engine = AsyncFastSparkTTS(
        model_path="SparkTTS-0.5B",
        max_length=32768,
        llm_device="cpu",
        audio_device="cpu",
        vocoder_device="cpu",
        engine="torch",
        wav2vec_attn_implementation="eager",
        llm_attn_implementation="eager"
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
    wav = await engine.async_generate_voice(
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
    reference_text = "吃燕窝就选燕之屋，本节目由26年专注高品质燕窝的燕之屋冠名播出。豆奶牛奶换着喝，营养更均衡，本节目由豆本豆豆奶特约播出。"

    wav = engine.clone_voice(
        text=text,
        reference_audio="data/roles/赞助商/reference_audio.wav",
        reference_text=reference_text,
        temperature=0.6,
        top_p=0.95,
        top_k=50,
        max_tokens=512
    )
    return wav


async def async_clone_voice(engine: AsyncFastSparkTTS):
    text = "身临其境，换新体验。塑造开源语音合成新范式，让智能语音更自然。"
    reference_text = "吃燕窝就选燕之屋，本节目由26年专注高品质燕窝的燕之屋冠名播出。豆奶牛奶换着喝，营养更均衡，本节目由豆本豆豆奶特约播出。"

    wav = await engine.async_clone_voice(
        text=text,
        reference_audio="data/roles/赞助商/reference_audio.wav",
        reference_text=reference_text,
        temperature=0.6,
        top_p=0.95,
        top_k=50,
        max_tokens=512
    )
    return wav


def main():
    engine = prepare_engine()
    audio = generate_voice(engine)
    # audio = clone_voice(engine)
    sf.write("result.wav", audio, 16000, "PCM_16")


async def run():
    engine = prepare_engine()
    audio = await async_generate_voice(engine)
    # audio = await async_clone_voice(engine)
    sf.write("result.wav", audio, 16000, "PCM_16")


if __name__ == '__main__':
    main()
    # asyncio.run(run())
