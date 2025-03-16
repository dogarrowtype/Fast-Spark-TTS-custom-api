# -*- coding: utf-8 -*-
# Time      :2025/3/15 11:37
# Author    :Hui Huang
import argparse
import base64
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import io
import uvicorn
import soundfile as sf
from starlette.middleware.cors import CORSMiddleware

from fast_sparktts.runtime import AsyncFastSparkTTS

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 定义 TTS 合成请求体（JSON 格式）
class TTSRequest(BaseModel):
    text: str
    gender: Literal["female", "male"] = "female"
    pitch: Literal["very_low", "low", "moderate", "high", "very_high"] = "moderate"
    speed: Literal["very_low", "low", "moderate", "high", "very_high"] = "moderate"
    temperature: float = 0.6
    top_k: int = 50
    top_p: float = 0.95
    max_tokens: int = 2048


# 定义支持多种方式传入参考音频的请求协议
class CloneRequest(BaseModel):
    text: str
    reference_text: str
    # reference_audio 字段既可以是一个 URL，也可以是 base64 编码的音频数据
    reference_audio: str
    temperature: float = 0.6
    top_k: int = 50
    top_p: float = 0.95
    max_tokens: int = 2048


# TTS 合成接口：接收 JSON 请求，返回合成语音（wav 格式）
@app.post("/generate_voice")
async def generate_voice(req: TTSRequest):
    try:
        audio = await engine.async_generate_voice(
            req.text,
            gender=req.gender,
            pitch=req.pitch,
            speed=req.speed,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            max_tokens=req.max_tokens
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    audio_io = io.BytesIO()
    sf.write(audio_io, audio, 16000, format="WAV", subtype="PCM_16")
    audio_io.seek(0)
    return StreamingResponse(audio_io, media_type="audio/wav")


async def get_audio_bytes_from_url(url: str) -> bytes:
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="无法从指定 URL 下载参考音频")
        return response.content


# 克隆语音接口：接收 multipart/form-data，上传参考音频和其它表单参数
@app.post("/clone_voice")
async def clone_voice(
        req: CloneRequest
):
    # 根据 reference_audio 内容判断读取方式
    if req.reference_audio.startswith("http://") or req.reference_audio.startswith("https://"):
        audio_bytes = await get_audio_bytes_from_url(req.reference_audio)
    else:
        try:
            audio_bytes = base64.b64decode(req.reference_audio)
        except Exception as e:
            raise HTTPException(status_code=400, detail="无效的 base64 音频数据: " + str(e))
    # 利用 BytesIO 包装字节数据，然后使用 soundfile 读取为 numpy 数组
    try:
        bytes_io = io.BytesIO(audio_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail="读取参考音频失败: " + str(e))

    try:
        inputs = await engine.prepare_clone_inputs(
            text=req.text,
            reference_audio=bytes_io,
            reference_text=req.reference_text,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            max_tokens=req.max_tokens
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail="读取上传的参考文件失败：" + str(e))

    try:
        audio = await engine.async_clone_voice_from_ndarray(
            **inputs
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail="生成克隆语音失败：" + str(e))

    audio_io = io.BytesIO()
    sf.write(audio_io, audio, 16000, format="WAV", subtype="PCM_16")
    audio_io.seek(0)
    return StreamingResponse(audio_io, media_type="audio/wav")


if __name__ == '__main__':
    # 使用 argparse 获取启动参数
    parser = argparse.ArgumentParser(description="FastAPI TTS 部署服务")
    parser.add_argument("--model_path", type=str, required=True,
                        help="模型路径")
    parser.add_argument("--engine", type=str, required=True,
                        choices=["llama-cpp", "vllm", "sglang"],
                        help="引擎类型，如 llama-cpp、vllm 或 sglang")
    parser.add_argument("--llm_device", type=str, default="auto",
                        help="llm 设备，例如 cpu 或 cuda:0")
    parser.add_argument("--audio_device", type=str, default="auto",
                        help="audio 设备")
    parser.add_argument("--vocoder_device", type=str, default="auto",
                        help="vocoder 设备")
    parser.add_argument("--wav2vec_attn_implementation", type=str, default="eager",
                        choices=["sdpa", "flash_attention_2", "eager"],
                        help="wav2vec 的 attn_implementation 方式")
    parser.add_argument("--max_length", type=int, default=32768,
                        help="最大生成长度")
    parser.add_argument("--llm_gpu_memory_utilization", type=float, default=0.6,
                        help="vllm和sglang暂用显存比例，单卡可降低该参数")
    parser.add_argument("--batch_size", type=int, default=32, help="音频处理组件单批次处理的最大请求数。")
    parser.add_argument("--wait_timeout", type=float, default=0.01, help="动态批处理请求超时阈值，单位为秒。")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务监听地址")
    parser.add_argument("--port", type=int, default=8000, help="服务监听端口")
    args = parser.parse_args()

    # 使用解析到的参数初始化全局 TTS 引擎
    engine = AsyncFastSparkTTS(
        model_path=args.model_path,
        max_length=args.max_length,
        llm_device=args.llm_device,
        audio_device=args.audio_device,
        vocoder_device=args.vocoder_device,
        engine=args.engine,
        wav2vec_attn_implementation=args.wav2vec_attn_implementation,
        llm_gpu_memory_utilization=args.llm_gpu_memory_utilization,
        batch_size=args.batch_size,
        wait_timeout=args.wait_timeout
    )

    uvicorn.run(app, host=args.host, port=args.port)
