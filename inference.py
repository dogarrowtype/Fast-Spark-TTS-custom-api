# -*- coding: utf-8 -*-
# Project : Fast-Spark-TTS
# Time    : 2025/3/13 12:32
# Author  : Hui Huang
import asyncio
import numpy as np
from fast_tts import AsyncSparkEngine, SparkAcousticTokens

long_text = ("今日是二零二五年三月十九日，国内外热点事件聚焦于国际局势、经济政策及社会民生领域。"
             "国际局势中，某国领导人围绕地区冲突停火问题展开对话，双方同意停止攻击对方能源设施并推动谈判，但对全面停火提议的落实仍存分歧。"
             "某地区持续军事行动导致数百人伤亡，引发民众抗议，质疑冲突背后的政治动机。另有一方宣称对连续袭击军事目标负责，称此为对前期打击的回应。"
             "欧洲某国通过争议性财政草案，计划放宽债务限制以支持国防与环保项目，引发经济政策讨论。 "
             "国内动态方面，新修订的市场竞争管理条例将于四月二十日施行，重点规范市场秩序。"
             "多部门联合推出机动车排放治理新规，加强对高污染车辆的监管。"
             "社会层面，某地涉及非法集资的大案持续引发关注，受害人数以万计，涉案金额高达数百亿元，暴露出特定领域投资风险。"
             "经济与科技领域，某科技企业公布年度营收突破三千六百五十九亿元，并上调智能汽车交付目标至三十五万台。"
             "另一巨头宣布全面推动人工智能转型，要求各部门绩效与人工智能应用深度绑定，计划年内推出多项相关产品。"
             "充电基础设施建设加速，公共充电桩总量已接近四百万个，同比增长超六成。 "
             "民生政策方面，多地推出新举措：某地限制顺风车单日接单次数以规范运营，另一地启动职工数字技能培训计划，目标三年内覆盖十万女性从业者。"
             "整体来看，今日热点呈现国际博弈复杂化、国内经济科技加速转型、民生政策精准化调整的特点。")


def prepare_engine():
    # vllm
    # engine = AsyncSparkEngine(
    #     model_path="Spark-TTS-0.5B",
    #     max_length=32768,
    #     llm_device="cuda:0",
    #     tokenizer_device="cuda:0",
    #     detokenizer_device="cuda:0",
    #     backend="vllm",
    #     wav2vec_attn_implementation="sdpa",  # 使用flash attn加速wav2vec
    #     llm_gpu_memory_utilization=0.6,
    #     seed=0
    # )

    # sglang
    # engine = AsyncSparkEngine(
    #     model_path="Spark-TTS-0.5B",
    #     max_length=32768,
    #     llm_device="cuda",  # sglang没办法指定gpu id，需要使用CUDA_VISIBLE_DEVICES=0设置。
    #     tokenizer_device="cuda:0",
    #     detokenizer_device="cuda:0",
    #     backend="sglang",
    #     wav2vec_attn_implementation="sdpa",  # 使用flash attn加速wav2vec
    #     llm_gpu_memory_utilization=0.6,
    #     seed=0
    # )

    # llama-cpp
    # engine = AsyncSparkEngine(
    #     model_path="Spark-TTS-0.5B",
    #     max_length=32768,
    #     llm_device="cpu",
    #     tokenizer_device="cpu",
    #     detokenizer_device="cpu",
    #     backend="llama-cpp",
    #     wav2vec_attn_implementation="eager"
    # )

    # torch
    engine = AsyncSparkEngine(
        model_path="Spark-TTS-0.5B",
        max_length=32768,
        llm_device="cuda",
        tokenizer_device="cuda",
        detokenizer_device="cuda",
        backend="torch",
        wav2vec_attn_implementation="sdpa",
        llm_attn_implementation="sdpa",
        torch_dtype="bfloat16",
        seed=0
    )
    return engine


async def generate_voice(engine: AsyncSparkEngine):
    """
    异步语音合成示例
    """
    wav = await engine.generate_voice_async(
        "我是无敌的小可爱。",
        gender="female",
        temperature=0.9,
        top_p=0.95,
        top_k=50,
        max_tokens=512
    )
    return wav


async def clone_voice(engine: AsyncSparkEngine):
    """
        异步语音克隆示例
        """
    text = "身临其境，换新体验。塑造开源语音合成新范式，让智能语音更自然。"

    wav = await engine.clone_voice_async(
        text=text,
        reference_audio="data/roles/赞助商/reference_audio.wav",
        temperature=0.9,
        top_p=0.95,
        top_k=50,
        max_tokens=512
    )
    return wav


async def generate_long_voice(engine: AsyncSparkEngine):
    """
        异步长文本语音合成示例，split表示开启句子切分，window_size为句子窗口大小
        """
    wav = await engine.generate_voice_async(
        text=long_text,
        split=True,
        length_threshold=50,
        window_size=50
    )
    return wav


async def clone_long_voice(engine: AsyncSparkEngine):
    """
    异步长文本语音克隆示例
    """
    wav = await engine.clone_voice_async(
        text=long_text,
        reference_audio="data/roles/赞助商/reference_audio.wav",
        length_threshold=50,
        window_size=50,
    )
    return wav


async def generate_voice_stream(engine: AsyncSparkEngine):
    """
    流式音频合成示例
    """
    audios = []
    async for chunk in engine.generate_voice_stream_async(
            text=long_text,
            split=True,
            length_threshold=50,
            window_size=50
    ):
        audios.append(chunk)

    audio = np.concatenate(audios)
    return audio


async def clone_voice_stream(engine: AsyncSparkEngine):
    """
    异步流式语音克隆示例
    """
    audios = []
    async for chunk in engine.clone_voice_stream_async(
            text=long_text,
            reference_audio="data/roles/赞助商/reference_audio.wav",
            split=True,
            length_threshold=50,
            window_size=50
    ):
        audios.append(chunk)
    audio = np.concatenate(audios)
    return audio


async def retain_acoustic_example(engine: AsyncSparkEngine):
    """
    复用音色例子
    """
    # 1. 随机测试一句话，设置return_acoustic_tokens为True
    wav, tokens = await engine.generate_voice_async(
        text="今日是二零二五年三月十九日，国内外热点事件聚焦于国际局势、经济政策及社会民生领域。",
        return_acoustic_tokens=True
    )
    # 2. 真巧，这是我想要的音色，直接保存为txt
    tokens.save("acoustic_tokens.txt")
    # 同时保存第一次生成的音频，以便对比
    engine.write_audio(wav, "first.wav")

    # 3. 加载保存的音色，生成第二个音频
    wav = await engine.generate_voice_async(
        text="国际局势中，某国领导人围绕地区冲突停火问题展开对话，双方同意停止攻击对方能源设施并推动谈判，但对全面停火提议的落实仍存分歧。",
        acoustic_tokens=SparkAcousticTokens.load("acoustic_tokens.txt"),
    )
    engine.write_audio(wav, "second.wav")
    # 4. 试听first.wav和second.wav，惊奇发现，这两个音频的音色是一致的


async def retain_acoustic_stream_example(engine: AsyncSparkEngine):
    """
    流式输出时，复用音色例子
    """
    # 1. 随机测试一句话，设置return_acoustic_tokens为True
    audios = []
    acoustic_tokens = None
    async for chunk in engine.generate_voice_stream_async(
            text="身临其境，换新体验。塑造开源语音合成新范式，让智能语音更自然。",
            return_acoustic_tokens=True
    ):
        if isinstance(chunk, SparkAcousticTokens):
            acoustic_tokens = chunk
        else:
            audios.append(chunk)
    audio = np.concatenate(audios)

    # 2. 真巧，这是我想要的音色，直接保存为txt
    engine.write_audio(audio, "first.wav")
    acoustic_tokens.save("acoustic_tokens.txt")

    # 3. 加载保存的音色，生成第二个音频
    audios = []
    async for chunk in engine.generate_voice_stream_async(
            text="今日是二零二五年三月十九日，国内外热点事件聚焦于国际局势、经济政策及社会民生领域。",
            acoustic_tokens=SparkAcousticTokens.load("acoustic_tokens.txt")
    ):
        audios.append(chunk)

    # 4. 试听first.wav和second.wav，惊奇发现，这两个音频的音色是一致的
    audio = np.concatenate(audios)
    engine.write_audio(audio, "second.wav")


async def run():
    engine = prepare_engine()
    audio = await generate_voice(engine)
    # audio = await clone_voice(engine)
    # audio = await generate_long_voice(engine)
    engine.write_audio(audio, "result.wav")


async def run_orpheus():
    """
    Orpheus目前仅支持英文
    """
    from fast_tts import AsyncOrpheusEngine

    engine = AsyncOrpheusEngine(
        model_path="orpheus-3b-0.1-ft",
        snac_path="snac_24khz",
        max_length=8192,
        llm_device="cuda",
        detokenizer_device="cuda",
        backend="vllm",
        torch_dtype='bfloat16'
    )
    wav = await engine.speak_async(
        name='tara',
        text="Hey there guys. It's, <giggle> Tara here, and let me introduce you to Zac.. who seems to asleep.",
    )
    engine.write_audio(wav, "result.wav")


if __name__ == '__main__':
    asyncio.run(run())
