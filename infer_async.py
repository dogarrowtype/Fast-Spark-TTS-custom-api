# -*- coding: utf-8 -*-
# Time      :2025/3/13 20:54
# Author    :Hui Huang
import asyncio
from runtime import AsyncFastSparkTTS
import soundfile as sf

async def run():
    # vllm
    # predictor = AsyncFastSparkTTS(
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
    # predictor = AsyncFastSparkTTS(
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
    predictor = AsyncFastSparkTTS(
        model_path="Spark-TTS-0.5B",
        max_length=32768,
        llm_device="cpu",
        audio_device="cpu",
        vocoder_device="cpu",
        engine="llama-cpp",
        wav2vec_attn_implementation="eager"
    )
    target_text = "身临其境，换新体验。塑造开源语音合成新范式，让智能语音更自然。"
    reference_text = "吃燕窝就选燕之屋，本节目由26年专注高品质燕窝的燕之屋冠名播出。豆奶牛奶换着喝，营养更均衡，本节目由豆本豆豆奶特约播出。"

    audio = await predictor.inference(
        reference_wav_path="example/prompt_audio.wav",
        reference_text=reference_text,
        target_text=target_text,
        temperature=0.2,
        top_p=0.95,
        top_k=40,
        max_tokens=1024
    )
    sf.write("result.wav", audio, 16000, "PCM_16")


if __name__ == '__main__':
    asyncio.run(run())
